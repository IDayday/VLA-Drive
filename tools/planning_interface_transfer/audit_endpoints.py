"""S100k train4096/dev1696, current-only W cache and frozen label-side diagnostics."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from torch.nn import functional as F
from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,decode_ego
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from tools.foresight.checkpoints import checkpoint_identity,load_student
from tools.full_foresight.evaluate_auxiliary import FixedPositionMoments,frozen_mae_decode
from tools.action_video_foresight.audit_existing_auxiliary import trajectory_error
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def read(path):return json.loads(Path(path).read_text())


def dataset(plan,split,arm):
    oldroot=Path(plan['campaign_root']);clip=oldroot/('targets_dino_sequence_'+split+'_v1' if arm in ('S0','S1') else 'targets_video_clip_'+split+'_v1')
    interaction=Path(plan['interaction_root']) if split=='train' else Path(plan['interaction_root']).parent/'interaction_dev_v1'
    ci=read(clip/'identity.json');di=read(Path(plan['dino_root'])/'identity.json');ii=read(interaction/'identity.json')
    ds=ActionVideoForesightDataset(plan[split+'_data'],candidate='C1',current=True,future=True,
        dino_root=plan['dino_root'],dino_index=plan['dino_index'],expected_dino=di['identity'],
        interaction_root=str(interaction),expected_interaction=ii['identity'],clip_root=str(clip),
        expected_clip=ci['identity'],future_type=ci['future_target_type'],expected_teacher=identity_hash(ci['recipe']),allow_partial=False)
    return ds


def queries(a,split):
    return read(Path(a.reference_representations)/(split+'_representation_queries.json'))


def fit(a,out,plan):
    ds=dataset(plan,'train','S0');population=queries(a,'train');mapping={r['token']:i for i,r in enumerate(ds.index)}
    cur=curcount=zraw=znorm=None;zn=0
    for j,row in enumerate(population):
        i=mapping[row['token']];refs=ds.dino_scenes[i]['images'][0];pairs=[ds.read_image(x) for x in refs]
        values=torch.stack([p[0] for p in pairs]).double();valid=torch.stack([p[1] for p in pairs])[:,None]
        if not torch.isfinite(values[valid.expand_as(values)]).all():raise ValueError('Illegal current label')
        if cur is None:cur=torch.zeros_like(values);curcount=torch.zeros_like(valid,dtype=torch.float64)
        cur+=torch.where(valid,values,0);curcount+=valid
        label=torch.load(ds.interaction_root/'targets'/(row['token']+'.pt'),weights_only=True)
        if label['identity']!=ds.interaction_identity['identity']:raise ValueError('Foreign interaction label')
        if label['interaction_target_valid']:
            z=label['latent'].double()
            if zraw is None:zraw=torch.zeros_like(z);znorm=torch.zeros_like(z)
            zraw+=z;znorm+=F.layer_norm(z.float(),(512,)).double();zn+=1
        if j%100==0:atomic_json(out/'progress.json',{'fitted_training_scenes':j,'total':len(population)})
    result={'current_mean':(cur/curcount.clamp_min(1)).float(),'current_valid':curcount[:,0]>0,
        'interaction_raw_mean':(zraw/max(zn,1)).float(),'interaction_normalized_mean':(znorm/max(zn,1)).float(),
        'identity':{'train_queries':identity_hash(population),'scenes':len(population),'interaction_valid_scenes':zn,
        'current_space':'raw postnorm pooled DINO, arithmetic mean','interaction_space':'mean raw Z and mean LN(Z) separately; no second LN on arithmetic LN mean'}}
    torch.save(result,out/'templates.pt');atomic_json(out/'COMPLETE.json',result['identity'])


def encode(a,out,plan):
    run=Path(plan['campaign_root'])/'students'/plan['runs'][a.arm]['run_id'];training,cp=checkpoint_identity(run,'milestone_100000')
    model=load_student(run,'milestone_100000',training,strip=False);dest=out/'representations';dest.mkdir(exist_ok=True)
    if cp['completed']!=100000:raise ValueError('Only common100k endpoint')
    for split in ('train','dev'):
        ds=ForesightCurrentDataset(plan[split+'_data']);mapping={r['token']:i for i,r in enumerate(ds.index)};population=queries(a,split)
        if split=='dev' and population!=ds.index:raise ValueError('Full development population required')
        selected=[r for i,r in enumerate(population) if i%a.shards==a.shard]
        for j,row in enumerate(selected):
            path=dest/(row['token']+'.pt')
            if path.exists():
                if torch.load(path,weights_only=True)['checkpoint']!=cp['sha256']:raise ValueError('Foreign representation')
                continue
            with torch.inference_mode():encoded=model.encode_current([ds[mapping[row['token']]]])
            temp=path.with_suffix('.'+str(a.shard)+'.tmp');torch.save({'token':row['token'],'checkpoint':cp['sha256'],
                'W':encoded['W'][0].cpu().clone(),'H_A':encoded['action_queries'][0].cpu().clone(),'split':split},temp);temp.replace(path)
            if j%25==0:atomic_json(out/f'encode_{split}_{a.shard}.json',{'status':'RUNNING','completed':j,'requested':len(selected),'checkpoint':cp})
        atomic_json(out/f'encode_{split}_{a.shard}.json',{'status':'COMPLETE','requested':len(selected),'checkpoint':cp})


def mse(p,t,v):
    mask=torch.broadcast_to(v,p.shape)
    if not torch.isfinite(p[mask]).all() or not torch.isfinite(t[mask]).all():raise ValueError('Nonfinite valid feature')
    return float((p[mask].float()-t[mask].float()).square().mean()) if mask.any() else None


def evaluate(a,out,plan):
    run=Path(plan['campaign_root'])/'students'/plan['runs'][a.arm]['run_id'];training,cp=checkpoint_identity(run,'milestone_100000')
    model=load_student(run,'milestone_100000',training,strip=False)
    template=torch.load(Path(a.templates)/'templates.pt',weights_only=True)
    if template['identity']['train_queries']!=identity_hash(queries(a,'train')):raise ValueError('Template population changed')
    teacher_root=Path(a.teacher_root);frozen=read(a.frozen_teacher);tid=read(teacher_root/'identity.json');weight=teacher_root/frozen['checkpoint']
    if file_sha256(weight)!=frozen['checkpoint_sha256']:raise ValueError('Frozen teacher weight changed')
    teacher=TrajectoryMAE(**tid['model']).cuda().eval().requires_grad_(False);teacher.load_state_dict(torch.load(weight,weights_only=False)['model'],strict=True)
    for split in ('train','dev'):
        ds=dataset(plan,split,a.arm);population=queries(a,split);mapping={r['token']:i for i,r in enumerate(ds.index)}
        order=sorted(range(len(population)),key=lambda i:hashlib.sha256(('W-swap-v1:'+population[i]['token']).encode()).hexdigest())
        paired={order[i]:order[(i+len(order)//2)%len(order)] for i in range(len(order))}
        if {r['log'] for r in queries(a,'train')}&{r['log'] for r in queries(a,'dev')}:raise ValueError('Log leakage')
        meanfile=Path(plan['campaign_root'])/('future_dino_sequence_normalized_train_mean_v1.pt' if a.arm in ('S0','S1') else 'future_video_clip_normalized_train_mean_v1.pt')
        futuremean=torch.load(meanfile,weights_only=True)
        if futuremean['identity']['train_queries']!=template['identity']['train_queries']:raise ValueError('Future mean population mismatch')
        rows=[];moments={}
        path=out/(split+f'_scenes_{a.shard}.jsonl')
        if path.exists():raise FileExistsError('Use a new audit attempt; partial rows retained')
        def encoded(row):
            z=torch.load(out/'representations'/(row['token']+'.pt'),weights_only=True)
            if z['checkpoint']!=cp['sha256']:raise ValueError('Wrong W checkpoint')
            return {'W':z['W'][None].cuda(),'action_queries':z['H_A'][None].cuda()}
        for i,row in enumerate(population):
            if i%a.shards!=a.shard:continue
            r={**row,'failure':None}
            try:
                enc=encoded(row);swap=encoded(population[paired[i]]);_,t=ds[mapping[row['token']]]
                target=t['current_dino'].cuda().float();cv=t['current_dino_valid'].cuda();ft=t['future_clip'].cuda().float();fv=t['future_clip_valid'].cuda()
                with torch.inference_mode():
                    cur=model.dino_head(enc['W'],torch.zeros(1,device='cuda'),(6,8))[0];cs=model.dino_head(swap['W'],torch.zeros(1,device='cuda'),(6,8))[0]
                    z=model.predict_interaction(enc)[0];zs=model.predict_interaction(swap)[0]
                    action=None;swapped_action=None
                    if model.foresight_config.future_action_condition=='gt_ego':
                        decoded=decode_ego(t['ego'].cuda());action=torch.cat((decoded[:,:2],decoded[:,2:3].sin(),decoded[:,2:3].cos()),-1)[None]
                        _,other=ds[mapping[population[paired[i]]['token']]];d=decode_ego(other['ego'].cuda());swapped_action=torch.cat((d[:,:2],d[:,2:3].sin(),d[:,2:3].cos()),-1)[None]
                    future=model.spatiotemporal_head(enc['W'],(9,12),gt_action=action)[0]
                    fs=model.spatiotemporal_head(swap['W'],(9,12),gt_action=action)[0]
                    fa=model.spatiotemporal_head(enc['W'],(9,12),gt_action=swapped_action)[0] if action is not None else None
                r['current']={};r['future']={}
                for v in range(3):
                    mask=cv[v][None]
                    r['current'][str(v)]={name:mse(value,target[v],mask) for name,value in [('model',cur[v]),('training_mean',template['current_mean'][v].cuda()),('shuffled_W',cs[v])]}
                    r['current'][str(v)].update(prediction_norm=float(cur[v,:,cv[v]].norm(dim=0).mean()),target_norm=float(target[v,:,cv[v]].norm(dim=0).mean()))
                for name,value in [('current_prediction',cur),('current_target',target)]:moments.setdefault(name,FixedPositionMoments()).add(value,cv[:,None])
                norm=lambda x:F.layer_norm(x,(x.shape[-1],),eps=model.foresight_config.normalization_eps)
                nt=norm(torch.where(fv[...,None],ft,0.));nf=norm(future);nfs=norm(fs)
                for v in range(3):
                    for tt in range(ft.shape[1]):
                        mask=fv[v,tt,:, :, None];key=str(v)+'_'+str(tt)
                        r['future'][key]={name:mse(val[v,tt],nt[v,tt],mask) for name,val in [('model',nf),('training_mean',futuremean['mean'].cuda()),('unit_training_mean',norm(futuremean['mean'].cuda())),('shuffled_W',nfs)]}
                        if fa is not None:r['future'][key]['shuffled_action']=mse(norm(fa)[v,tt],nt[v,tt],mask)
                for name,value in [('future_prediction',nf),('future_target',nt)]:moments.setdefault(name,FixedPositionMoments()).add(value,fv[...,None])
                iv=bool(t['interaction_valid']);r['interaction_valid']=iv
                zt=t['interaction_latent'].cuda().float()
                r['interaction']={name:mse(value,norm(zt),torch.ones_like(zt,dtype=torch.bool)) if iv else None for name,value in [('model',norm(z)),('training_mean',template['interaction_normalized_mean'].cuda()),('unit_training_mean',norm(template['interaction_normalized_mean'].cuda())),('shuffled_memory',norm(zs))]}
                if iv:
                    for name,value in [('interaction_prediction',norm(z)),('interaction_target',norm(zt))]:moments.setdefault(name,FixedPositionMoments()).add(value,torch.ones_like(value,dtype=torch.bool))
                    rec=torch.load(Path(a.teacher_data)/'records'/(row['token']+'.pt'),weights_only=True)['record']
                    anchor=rec['current'][0,:2].cuda();truth=rec['future'][0].cuda();valid=rec['point_valid'][0].cuda()
                    with torch.inference_mode():r['decoded_ego']={name:trajectory_error(frozen_mae_decode(teacher,value,anchor),truth,valid) for name,value in [('teacher',zt),('student',z),('training_mean',template['interaction_raw_mean'].cuda()),('shuffled_student',zs)]}
                else:r['decoded_ego']=None
            except Exception as error:r['failure']=repr(error)
            with path.open('a') as f:f.write(json.dumps(r)+'\n')
            rows.append(r)
            if len(rows)%25==0:atomic_json(out/f'eval_{split}_{a.shard}.json',{'status':'RUNNING','scenes':len(rows),'failed':sum(x['failure'] is not None for x in rows)})
        torch.save({k:vars(v) for k,v in moments.items()},out/f'{split}_moments_{a.shard}.pt')
        atomic_json(out/f'eval_{split}_{a.shard}.json',{'status':'COMPLETE' if not any(r['failure'] for r in rows) else 'FAILED','scenes':len(rows),'failed':sum(x['failure'] is not None for x in rows),'checkpoint':cp,'precision':'FP32 master / FP32 compute / TF32 off'})
        if any(r['failure'] for r in rows):raise RuntimeError('Audit failed rows preserved')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('fit','encode','evaluate'))
    for k in ('plan','reference-representations','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--arm',default='S0',choices=('S0','S1','S2','S3','S4'));p.add_argument('--templates');p.add_argument('--teacher-root');p.add_argument('--teacher-data');p.add_argument('--frozen-teacher')
    p.add_argument('--shards',type=int,default=8);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);plan=read(a.plan)
    with metered_run(a.campaign_root,a.run_id,0 if a.mode=='fit' else 1,{'kind':'S100k_endpoint_'+a.mode,'real_optimizer_updates':0}) as (meter,_,save):
        {'fit':fit,'encode':encode,'evaluate':evaluate}[a.mode](a,out,plan);save()

if __name__=='__main__':main()
