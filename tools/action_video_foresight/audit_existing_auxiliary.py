"""Frozen100k: full training templates, current-only representation cache, full dev audit."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tools.foresight.checkpoints import checkpoint_identity,load_student
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from tools.full_foresight.evaluate_auxiliary import FixedPositionMoments,masked_feature_error,frozen_mae_decode
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run

H=[0,1,2,4]


def dataset(a,split):
    di=json.loads((Path(a.dino_root)/'identity.json').read_text());ii=json.loads((Path(getattr(a,split+'_interactions'))/'identity.json').read_text())
    return FullForesightDataset(getattr(a,split+'_data'),candidate=a.candidate,current=True,future=True,
        dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=di['identity'],allow_partial=False,
        interaction_root=getattr(a,split+'_interactions'),expected_interaction=ii['identity'])


def selected_train(index,limit):
    key=lambda r:hashlib.sha256(('action-video-fixed-train-v1:'+r['token']).encode()).hexdigest()
    groups={}
    for row in index:groups.setdefault(row['log'],[]).append(row)
    first=[min(rows,key=key) for rows in groups.values()]
    tokens={r['token'] for r in first}
    return sorted(first+sorted([r for r in index if r['token'] not in tokens],key=key)[:max(0,limit-len(first))],key=lambda r:r['token'])


def mean_templates(a,out):
    train=dataset(a,'train');dev=dataset(a,'dev')
    if {r['log'] for r in train.index}&{r['log'] for r in dev.index}:raise ValueError('Log leakage')
    refs=train.dino_scenes;cache=train.dino_identity
    weights=torch.zeros(4,3,cache['image_count'],dtype=torch.float64)
    for scene in refs:
        for h,row in enumerate(scene['images']):
            for v,image in enumerate(row):
                if image>=0:weights[h,v,image]+=1
    sums=torch.zeros(4,3,1024,*cache['grid_hw'],dtype=torch.float64);counts=torch.zeros(4,3,1,*cache['grid_hw'],dtype=torch.float64)
    chunk_size=cache['chunk_size'];chunks=(cache['image_count']+chunk_size-1)//chunk_size
    for c in range(chunks):
        w=weights[:,:,c*chunk_size:(c+1)*chunk_size]
        if not w.any():continue
        with safe_open(str(Path(a.dino_root)/f'chunk_{c:06d}.safetensors'),framework='pt',device='cpu') as f:
            if f.metadata()['identity']!=cache['identity']:raise ValueError('Template cache identity')
            z=f.get_tensor('features').double();valid=f.get_tensor('valid')
        if not torch.isfinite(z[valid[:,None].expand_as(z)]).all():raise ValueError('Invalid train target')
        z=torch.where(valid[:,None],z,0.)
        sums+=torch.einsum('hvn,ncxy->hvcxy',w,z)
        counts+=torch.einsum('hvn,nxy->hvxy',w,valid.double())[:,:,None]
        if c%200==0:atomic_json(out/'template_progress.json',{'dino_chunk':c,'dino_chunks':chunks})
    means=(sums/counts.clamp_min(1)).float();zraw=zsum=None;n=0
    root=Path(a.train_interactions);iid=train.interaction_identity['identity']
    def read(row):
        label=torch.load(root/'targets'/(row['token']+'.pt'),map_location='cpu',weights_only=True)
        if label['identity']!=iid or label['token']!=row['token']:raise ValueError('Interaction identity')
        if not label['interaction_target_valid']:return None
        z=label['latent'].float().clone()
        if z.shape!=(8,512) or not torch.isfinite(z).all():raise ValueError('Invalid raw teacher Z')
        return z
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,z in enumerate(pool.map(read,train.index)):
            if z is not None:
                norm=F.layer_norm(z,(512,))
                if zsum is None:zsum=torch.zeros_like(z,dtype=torch.float64);zraw=torch.zeros_like(z,dtype=torch.float64)
                zsum+=norm.double();zraw+=z.double();n+=1
            if i%2000==0:atomic_json(out/'template_progress.json',{'dino_complete':True,'interaction_scenes':i,'valid_interaction_scenes':n})
    torch.save({'visual_mean':means,'visual_valid':counts[:, :, 0]>0,
        'interaction_normalized_mean':(zsum/max(n,1)).float(),'interaction_raw_mean':(zraw/max(n,1)).float(),
        'interaction_valid_scenes':n},out/'templates.pt')
    record={'train_identity':train.identity['identity'],'dino_identity':cache['identity'],'interaction_identity':iid,
        'training_scenes':len(train),'training_logs':len({r['log'] for r in train.index}),
        'dino_training_definition':'raw post-norm patch MSE, no extra normalization',
        'interaction_training_definition':'mean LayerNorm(raw Z) per fixed time/channel; raw Z mean separately for reconstruct',
        'template_file_sha256':file_sha256(out/'templates.pt'),'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    atomic_json(out/'TEMPLATES_COMPLETE.json',record)


def encode(a,out):
    training,cp=checkpoint_identity(a.training_run,a.checkpoint_tag)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    model=load_student(a.training_run,a.checkpoint_tag,training,strip=False)
    root=out/'representations';root.mkdir(exist_ok=True)
    for split in ('train','dev'):
        ds=ForesightCurrentDataset(getattr(a,split+'_data'))
        chosen=selected_train(ds.index,a.train_limit) if split=='train' else ds.index
        mapping={r['token']:i for i,r in enumerate(ds.index)}
        path=out/(split+'_representation_queries.json')
        if path.exists() and json.loads(path.read_text())!=chosen:raise ValueError('Fixed representation population changed')
        atomic_json(path,chosen)
        for j,row in enumerate(chosen):
            if j%a.shards!=a.shard:continue
            dest=root/(row['token']+'.pt')
            if dest.exists():
                saved=torch.load(dest,weights_only=True)
                if saved['checkpoint']!=cp['sha256']:raise ValueError('Foreign W checkpoint')
                continue
            with torch.inference_mode():z=model.encode_current([ds[mapping[row['token']]]])
            record={'checkpoint':cp['sha256'],'token':row['token'],'split':split,'W':z['W'][0].cpu().clone(),
                'H_A':z['action_queries'][0].cpu().clone()}
            tmp=dest.with_suffix(f'.{a.shard}.tmp');torch.save(record,tmp);tmp.replace(dest)
        atomic_json(out/f'encode_{split}_shard_{a.shard:02d}.json',{'checkpoint':cp,'shard':a.shard,'shards':a.shards,
            'queries':len(chosen),'represented':sum((root/(r['token']+'.pt')).exists() for r in chosen),
            'pure_current_forward':True,'real_optimizer_updates':0})


def trajectory_error(p,t,mask):
    if not torch.isfinite(p[mask]).all() or not torch.isfinite(t[mask]).all():raise ValueError('Nonfinite decoded xy')
    d=(p[mask]-t[mask]).norm(dim=-1)
    return {'points':int(mask.sum()),'ADE':float(d.mean()) if len(d) else None,
        'FDE':float((p[-1]-t[-1]).norm()) if mask[-1] else None}


def evaluate(a,out):
    ds=dataset(a,'dev');templates=torch.load(Path(a.templates)/'templates.pt',weights_only=True)
    tr,cp=checkpoint_identity(a.training_run,a.checkpoint_tag);model=load_student(a.training_run,a.checkpoint_tag,tr,strip=False)
    frozen=json.loads(Path(a.frozen_teacher).read_text());tid=json.loads((Path(a.teacher_root)/'identity.json').read_text())
    weight=Path(a.teacher_root)/frozen['checkpoint']
    if tid['identity']!=frozen['teacher_run_identity'] or file_sha256(weight)!=frozen['checkpoint_sha256']:raise ValueError('Frozen MAE identity')
    teacher=TrajectoryMAE(**tid['model']).cuda().eval();state=torch.load(weight,map_location='cpu',weights_only=False)
    teacher.load_state_dict(state['model'],strict=True);teacher.requires_grad_(False)
    queries=json.loads((Path(a.representations)/'dev_representation_queries.json').read_text())
    if queries!=ds.index:raise ValueError('Audit must cover complete dev')
    # Fixed derangement based only on token, locked before computing errors.
    order=sorted(range(len(queries)),key=lambda i:hashlib.sha256(('W-swap-v1:'+queries[i]['token']).encode()).hexdigest())
    paired={order[i]:order[(i+len(order)//2)%len(order)] for i in range(len(order))}
    atomic_json(out/'query_manifest.json',[{'token':r['token'],'log':r['log'],'swap_token':queries[paired[i]]['token']} for i,r in enumerate(queries)])
    root=Path(a.representations)/'representations';variances={};summed={};rows=[]
    def load_world(token):
        record=torch.load(root/(token+'.pt'),weights_only=True)
        if record['checkpoint']!=cp['sha256']:raise ValueError('Frozen representation mismatch')
        return record['W'][None].cuda()
    for i,row in enumerate(ds.index):
        result={**row,'failure':None,'swap_token':queries[paired[i]]['token']}
        try:
            w=load_world(row['token']);sw=load_world(result['swap_token'])
            # No current-image encoder rerun is needed for label-side head diagnostics.
            image_features=[];image_valid=[]
            for refs in ds.dino_scenes[i]['images']:
                pairs=[ds.read_image(j) for j in refs]
                image_features.append(torch.stack([p[0] for p in pairs]).cuda().float())
                image_valid.append(torch.stack([p[1] for p in pairs]).cuda())
            targets=torch.stack(image_features);valid=torch.stack(image_valid)
            with torch.inference_mode():
                pred=torch.stack([model.dino_head(w,torch.tensor([float(h)],device='cuda'),targets.shape[-2:])[0] for h in H])
                shuffle=torch.stack([model.dino_head(sw,torch.tensor([float(h)],device='cuda'),targets.shape[-2:])[0] for h in H])
                if model.foresight_config.interaction_readout_source != 'world':
                    raise ValueError('Legacy W-only audit cannot evaluate an action readout; use planning_interface_transfer.audit_endpoints')
                z=model.predict_interaction({'W':w})[0];zs=model.predict_interaction({'W':sw})[0]
            for h in range(4):
                for v in range(3):
                    mask=valid[h,v][None];key=f'h{H[h]}_v{v}'
                    for name,value,extra in [('model',pred[h,v],mask),('train_mean',templates['visual_mean'][h,v].cuda(),mask & templates['visual_valid'][h,v][None].cuda()),
                        ('shuffled_W',shuffle[h,v],mask),('copy_current',targets[0,v],mask&valid[0,v][None]),
                        ('copy_predicted_current',pred[0,v],mask&valid[0,v][None])]:
                        error=masked_feature_error(value,targets[h,v],extra);result[key+'_'+name]=error
                        sums=summed.setdefault(key+'_'+name,{'squared_error':0.,'elements':0,'scenes':0})
                        for k in ('squared_error','elements'):sums[k]+=error[k]
                        sums['scenes']+=int(error['elements']>0)
                    for name,value in [('prediction',pred[h,v]),('target',targets[h,v])]:
                        variances.setdefault(key+'_'+name,FixedPositionMoments()).add(value,mask)
            label=torch.load(Path(a.dev_interactions)/'targets'/(row['token']+'.pt'),weights_only=True)
            iv=bool(label['interaction_target_valid']);result['with_valid_peer']=iv
            zt=label['latent'].cuda().float();norm=lambda x:F.layer_norm(x,(512,),eps=model.foresight_config.normalization_eps)
            if iv:
                for name,value in [('student',norm(z)),('teacher',norm(zt)),('shuffled',norm(zs))]:
                    variances.setdefault('interaction_'+name,FixedPositionMoments()).add(value,torch.ones_like(value,dtype=torch.bool))
                for name,value in [('model',norm(z)),('train_mean',templates['interaction_normalized_mean'].cuda()),('shuffled_W',norm(zs))]:
                    result['interaction_'+name]=float((value-norm(zt)).square().mean())
            gt=torch.load(Path(a.teacher_data)/'records'/(row['token']+'.pt'),weights_only=True)['record']
            current=gt['current'][0,:2].cuda();truth=gt['future'][0].cuda();point=gt['point_valid'][0].cuda()
            if not iv:
                from tools.foresight.teacher_runtime import inputs
                batch={k:v[None].cuda() for k,v in gt.items()}
                visible=batch['point_valid'].clone();visible[:,0]=False
                with torch.inference_mode():zt=teacher(inputs(batch,torch.zeros(1,dtype=torch.long,device='cuda'),visible))['latent'][0]
            with torch.inference_mode():
                result['decoded_ego']={name:trajectory_error(frozen_mae_decode(teacher,value,current),truth,point) for name,value in
                    [('teacher',zt),('student',z),('train_mean',templates['interaction_raw_mean'].cuda()),('shuffled_student',zs)]}
        except Exception as error:result['failure']=repr(error)
        with (out/'scenes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        rows.append(result)
        if i%50==0:atomic_json(out/'progress.json',{'scenes':len(rows),'requested':len(ds),'failures':sum(r['failure'] is not None for r in rows)})
    summary={'scenes':len(rows),'requested':len(ds),'logs':len({r['log'] for r in ds.index}),
        'failures':sum(r['failure'] is not None for r in rows),'visual':{k:{**v,'mse':v['squared_error']/v['elements'] if v['elements'] else None} for k,v in summed.items()},
        'fixed_position_cross_scene_variance':{k:v.result() for k,v in variances.items()},'interaction':{},'decoded_ego':{},
        'checkpoint':cp,'precision':'FP32 master, FP32 compute, TF32 off','real_optimizer_updates':0}
    for group in ('with_peer','without_peer'):
        selected=[r for r in rows if r['failure'] is None and r['with_valid_peer']==(group=='with_peer')]
        summary['interaction'][group]={'scenes':len(selected),**{name:sum(r['interaction_'+name] for r in selected)/len(selected) if selected and group=='with_peer' else None for name in ('model','train_mean','shuffled_W')}}
        summary['decoded_ego'][group]={name:{metric:sum(r['decoded_ego'][name][metric] for r in selected if r['decoded_ego'][name][metric] is not None)/max(1,sum(r['decoded_ego'][name][metric] is not None for r in selected)) for metric in ('ADE','FDE')} for name in ('teacher','student','train_mean','shuffled_student')}
    atomic_json(out/'SUMMARY.json',summary)
    if summary['failures']:raise RuntimeError('Audit failures retained; no complete valid result')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('templates','encode','evaluate'))
    for k in ('train-data','dev-data','dino-root','dino-index','train-interactions','dev-interactions','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    for k in ('training-run','checkpoint-tag','teacher-root','teacher-data','frozen-teacher','templates','representations'):p.add_argument('--'+k)
    p.add_argument('--candidate',default='C1');p.add_argument('--train-limit',type=int,default=4096)
    p.add_argument('--shards',type=int,default=8);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze audit source')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with metered_run(a.campaign_root,a.run_id,0 if a.mode=='templates' else 1,{'kind':'frozen100k_'+a.mode,'real_optimizer_updates':0}) as (meter,_,save):
        if a.mode=='templates':mean_templates(a,out)
        elif a.mode=='encode':encode(a,out)
        else:evaluate(a,out)
        save()

if __name__=='__main__':main()
