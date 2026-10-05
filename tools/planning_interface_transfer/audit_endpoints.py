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


def static_dino_dataset(plan,split,root):
    ref=FullForesightDataset(plan[split+'_data'],candidate='C3',current=True,future=False,
        dino_root=root,dino_index=plan['dino_index'],expected_dino=read(Path(root)/'identity.json')['identity'],
        interaction_root=None,expected_interaction=None,allow_partial=False)
    return ref


def current_dino_reference(ref,i,times):
    pairs=[ref.read_image(x) for x in ref.dino_scenes[i]['images'][0]]
    features=torch.stack([v[0] for v in pairs]).float().permute(0,2,3,1)[:,None]
    valid=torch.stack([v[1] for v in pairs])[:,None]
    return features.expand(-1,times,-1,-1,-1),valid.expand(-1,times,-1,-1)


def fit_change(a,out,plan):
    """Training-only quartiles of normalized future minus genuine static reference."""
    ds=dataset(plan,'train',a.arm);population=queries(a,'train');lookup={r['token']:i for i,r in enumerate(ds.index)}
    ref=static_dino_dataset(plan,'train',a.dino_sequence_current_root) if a.arm in ('S0','S1') else None
    if ref is None:
        sr=Path(a.static_train);sid=read(sr/'identity.json')
        if sid['recipe']!=ds.clip_identity['recipe'] or sid.get('diagnostic_query_hash')!=identity_hash(population):raise ValueError('Wrong static training reference')
        if read(sr/'COMPLETE.json')['scenes']!=len(population):raise ValueError('Static reference incomplete')
    values=[]
    for j,row in enumerate(population):
        i=lookup[row['token']];_,targets=ds[i];future=targets['future_clip'].float();valid=targets['future_clip_valid']
        if ref:static,sv=current_dino_reference(ref,i,future.shape[1])
        else:
            item=torch.load(sr/'targets'/(row['token']+'.pt'),weights_only=True)
            if item['identity']!=sid['identity']:raise ValueError('Foreign static feature')
            static=item['features'].float();sv=torch.ones_like(valid)
        good=valid&sv
        if not torch.isfinite(future[good]).all() or not torch.isfinite(static[good]).all():raise ValueError('Invalid active target')
        norm=lambda x:F.layer_norm(torch.where(good[...,None],x,0.),(x.shape[-1],))
        change=(norm(future)-norm(static)).square().mean(-1)
        values.append(change[good])
        if j%100==0:atomic_json(out/'progress.json',{'fitted_training_scenes':j,'total':len(population)})
    values=torch.cat(values)
    result={'schema':'training_static_change_quartiles_v1','target_type':ds.clip_identity['future_target_type'],
        'train_queries':identity_hash(population),'target_identity':ds.clip_identity['identity'],
        'static_identity':ref.dino_identity['identity'] if ref else sid['identity'],
        'definition':'channel mean squared LN(future)-LN(static current); valid views/positions only',
        'quantiles':[.25,.75],'low_threshold':float(values.quantile(.25)),
        'high_threshold':float(values.quantile(.75)),'valid_patches':len(values),'scenes':len(population)}
    atomic_json(out/'thresholds.json',result)


def summarize(a,out,plan):
    """Merge complete shards with fixed-position moments; retain full denominators."""
    result={'evaluation_source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'frozen endpoint auxiliary diagnostics; no planning score or optimizer update',
        'arm':a.arm,'splits':{},'scene_averaging':'mean over valid per-scene queries, with coverage separately reported',
        'means':'raw current arithmetic template; normalized future/interaction arithmetic and unit-renormalized references both shown'}
    def flatten(record,prefix=''):
        for key,value in record.items():
            name=prefix+'/'+key if prefix else key
            if isinstance(value,dict):yield from flatten(value,name)
            elif isinstance(value,(int,float)) and not isinstance(value,bool):yield name,float(value)
    for split in ('train','dev'):
        records=[];merged={}
        for shard in range(a.shards):
            receipt=read(out/f'eval_{split}_{shard}.json')
            if receipt['status']!='COMPLETE':raise ValueError('Failed/incomplete audit shard')
            records.extend(json.loads(line) for line in (out/f'{split}_scenes_{shard}.jsonl').read_text().splitlines())
            states=torch.load(out/f'{split}_moments_{shard}.pt',weights_only=True)
            for name,state in states.items():
                if name not in merged:merged[name]=state
                else:
                    for k in ('count','total','square'):merged[name][k]+=state[k]
        population=queries(a,split)
        if {r['token'] for r in records}!={r['token'] for r in population} or len(records)!=len(population):raise ValueError('Missing/duplicate requested scene')
        if any(r['failure'] for r in records):raise ValueError('Failed rows cannot become complete summary')
        values={};decoded={}
        for r in records:
            for section in ('current','future','interaction'):
                for key,value in flatten(r[section]):values.setdefault(section+'/'+key,[]).append(value)
            if r['decoded_ego']:
                for name,row in r['decoded_ego'].items():
                    for key,value in row.items():
                        if isinstance(value,(float,int)):decoded.setdefault(name+'/'+key,[]).append(value)
                        elif key=='point_error':
                            for t,error in enumerate(value):
                                if error is not None:decoded.setdefault(name+'/time_'+str(t),[]).append(error)
        statistics={key:{'mean':sum(v)/len(v),'valid_scenes':len(v)} for key,v in values.items()}
        variance={}
        for name,state in merged.items():
            obj=FixedPositionMoments();obj.__dict__.update(state);variance[name]=obj.result()
        result['splits'][split]={'requested':len(population),'scenes':len(records),'failed':0,
            'valid_interaction':sum(r['interaction_valid'] for r in records),
            'no_valid_interaction':sum(not r['interaction_valid'] for r in records),
            'statistics':statistics,'fixed_position_cross_scene_variance':variance,
            'frozen_mae_functional_decode':{k:{'mean':sum(v)/len(v),'valid_scenes':len(v)} for k,v in decoded.items()}}
    result['train_development_gap']={key:{'train':tr['mean'],'development':result['splits']['dev']['statistics'][key]['mean'],
        'difference':result['splits']['dev']['statistics'][key]['mean']-tr['mean']} for key,tr in result['splits']['train']['statistics'].items()
        if key in result['splits']['dev']['statistics']}
    atomic_json(out/'SUMMARY.json',result)


def evaluate(a,out,plan):
    run=Path(plan['campaign_root'])/'students'/plan['runs'][a.arm]['run_id'];training,cp=checkpoint_identity(run,'milestone_100000')
    model=load_student(run,'milestone_100000',training,strip=False)
    change_rule=read(a.change_thresholds)
    if change_rule['train_queries']!=identity_hash(queries(a,'train')):raise ValueError('Foreign change threshold population')
    expected_type='dino_sequence' if a.arm in ('S0','S1') else 'video_clip'
    if change_rule['target_type']!=expected_type:raise ValueError('Threshold target family mismatch')
    template=torch.load(Path(a.templates)/'templates.pt',weights_only=True)
    if template['identity']['train_queries']!=identity_hash(queries(a,'train')):raise ValueError('Template population changed')
    teacher_root=Path(a.teacher_root);frozen=read(a.frozen_teacher);tid=read(teacher_root/'identity.json');weight=teacher_root/frozen['checkpoint']
    if file_sha256(weight)!=frozen['checkpoint_sha256']:raise ValueError('Frozen teacher weight changed')
    teacher=TrajectoryMAE(**tid['model']).cuda().eval().requires_grad_(False);teacher.load_state_dict(torch.load(weight,weights_only=False)['model'],strict=True)
    atomic_json(out/'identity.json',{'training_source':training['source_sha'] if 'source_sha' in training else training.get('source'),
        'evaluation_source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'checkpoint':cp,'train_queries':identity_hash(queries(a,'train')),'dev_queries':identity_hash(queries(a,'dev')),
        'template_sha256':file_sha256(Path(a.templates)/'templates.pt'),'change_rule':change_rule,
        'precision':'FP32 master / FP32 compute / TF32 off','main_optimizer_updates':0,
        'predicted_current_reference':'C1 h0 prediction resampled bilinearly6x8 to9x12; diagnostic only; teacher copy-current uses native384x288 encoding'})
    for split in ('train','dev'):
        ds=dataset(plan,split,a.arm);population=queries(a,split);mapping={r['token']:i for i,r in enumerate(ds.index)}
        representation_root=Path(a.representation_root) if a.representation_root else out
        static_root=Path(a.static_train if split=='train' else a.static_dev) if a.arm not in ('S0','S1') else None
        if static_root:
            static_identity=read(static_root/'identity.json')
            if static_identity['recipe']!=ds.clip_identity['recipe']:raise ValueError('Static video reference uses another encoder')
            if split=='train' and static_identity.get('diagnostic_query_hash')!=identity_hash(population):raise ValueError('Static training reference population mismatch')
            if read(static_root/'COMPLETE.json')['scenes']!=len(population):raise ValueError('Incomplete static video diagnostic')
        current_reference=None
        if a.arm in ('S0','S1'):
            ref_root=Path(a.dino_sequence_current_root)
            current_reference=static_dino_dataset(plan,split,str(ref_root))
            # Explicit different schema implementations can share the exact teacher/preprocessing.
            rec=current_reference.dino_identity['recipe'];clip_rec=ds.clip_identity['recipe']
            for key in ('weight_sha256','revision','feature','extra_norm','mean','std','pool'):
                if key=='pool':
                    if current_reference.dino_identity['candidate']['pool']!=clip_rec['pool']:raise ValueError('Copy-current pool differs')
                elif rec[key]!=clip_rec[key]:raise ValueError('Copy-current teacher recipe differs: '+key)
            if list(current_reference.dino_identity['grid_hw'])!=ds.clip_identity['target_shape'][2:4]:raise ValueError('Copy-current native grid differs')
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
            z=torch.load(representation_root/'representations'/(row['token']+'.pt'),weights_only=True)
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
                    r['current'][str(v)]['normalized']={name:mse(F.layer_norm(value.permute(1,2,0),(value.shape[0],)),F.layer_norm(target[v].permute(1,2,0),(target.shape[1],)),cv[v][...,None]) for name,value in [('model',cur[v]),('training_mean',template['current_mean'][v].cuda()),('shuffled_W',cs[v])]}
                for name,value in [('current_prediction',cur),('current_target',target)]:moments.setdefault(name,FixedPositionMoments()).add(value,cv[:,None])
                norm=lambda x:F.layer_norm(x,(x.shape[-1],),eps=model.foresight_config.normalization_eps)
                nt=norm(torch.where(fv[...,None],ft,0.));nf=norm(future);nfs=norm(fs)
                if current_reference:
                    static,static_valid=current_dino_reference(current_reference,mapping[row['token']],ft.shape[1])
                    static=static.cuda();static_valid=static_valid.cuda()
                    predicted_static=F.interpolate(cur,size=ft.shape[2:4],mode='bilinear',align_corners=False).permute(0,2,3,1)[:,None].expand_as(ft)
                else:
                    reference=torch.load(static_root/'targets'/(row['token']+'.pt'),weights_only=True)
                    if reference['identity']!=static_identity['identity']:raise ValueError('Foreign static video tensor')
                    static=reference['features'].cuda().float();static_valid=torch.ones_like(fv)
                    predicted_static=None
                normalized_static=norm(torch.where(static_valid[...,None],static,0.))
                change=(nt-normalized_static).square().mean(-1)
                for v in range(3):
                    for tt in range(ft.shape[1]):
                        mask=fv[v,tt,:, :, None];key=str(v)+'_'+str(tt)
                        r['future'][key]={name:mse(val[v,tt],nt[v,tt],mask) for name,val in [('model',nf),('training_mean',futuremean['mean'].cuda()),('unit_training_mean',norm(futuremean['mean'].cuda())),('shuffled_W',nfs)]}
                        reference_mask=(fv&static_valid)[v,tt,...,None]
                        r['future'][key]['static_current']=mse(normalized_static[v,tt],nt[v,tt],reference_mask)
                        r['future'][key]['static_comparison_model']=mse(nf[v,tt],nt[v,tt],reference_mask)
                        r['future'][key]['static_comparison_patches']=int(reference_mask.sum())
                        if predicted_static is not None:r['future'][key]['copy_resampled_predicted_current']=mse(norm(predicted_static)[v,tt],nt[v,tt],mask)
                        r['future'][key]['prediction_norm']=float(future[v,tt].norm(dim=-1)[fv[v,tt]].mean()) if fv[v,tt].any() else None
                        r['future'][key]['target_norm']=float(ft[v,tt].norm(dim=-1)[fv[v,tt]].mean()) if fv[v,tt].any() else None
                        r['future'][key]['raw_model_mse']=mse(future[v,tt],ft[v,tt],mask)
                        r['future'][key]['change_groups']={}
                        for group,selected in [('low',change[v,tt]<=change_rule['low_threshold']),('high',change[v,tt]>=change_rule['high_threshold'])]:
                            active=reference_mask&selected[...,None]
                            r['future'][key]['change_groups'][group]={'patches':int(active.sum()),
                                'model':mse(nf[v,tt],nt[v,tt],active),'static':mse(normalized_static[v,tt],nt[v,tt],active),
                                'training_mean':mse(futuremean['mean'][v,tt].cuda(),nt[v,tt],active)}
                        if fa is not None:r['future'][key]['shuffled_action']=mse(norm(fa)[v,tt],nt[v,tt],mask)
                for name,value in [('future_prediction',nf),('future_target',nt)]:moments.setdefault(name,FixedPositionMoments()).add(value,fv[...,None])
                for name,value in [('future_raw_prediction',future),('future_raw_target',ft)]:moments.setdefault(name,FixedPositionMoments()).add(value,fv[...,None])
                iv=bool(t['interaction_valid']);r['interaction_valid']=iv
                zt=t['interaction_latent'].cuda().float()
                r['interaction']={name:mse(value,norm(zt),torch.ones_like(zt,dtype=torch.bool)) if iv else None for name,value in [('model',norm(z)),('training_mean',template['interaction_normalized_mean'].cuda()),('unit_training_mean',norm(template['interaction_normalized_mean'].cuda())),('shuffled_memory',norm(zs))]}
                if iv:
                    for name,value in [('interaction_prediction',norm(z)),('interaction_target',norm(zt))]:moments.setdefault(name,FixedPositionMoments()).add(value,torch.ones_like(value,dtype=torch.bool))
                    rec=torch.load(Path(a.teacher_data)/'records'/(row['token']+'.pt'),weights_only=True)['record']
                    anchor=rec['current'][0,:2].cuda();truth=rec['future'][0].cuda();valid=rec['point_valid'][0].cuda()
                    with torch.inference_mode():
                        r['decoded_ego']={}
                        for name,value in [('teacher',zt),('student',z),('training_mean',template['interaction_raw_mean'].cuda()),('shuffled_student',zs)]:
                            trajectory=frozen_mae_decode(teacher,value,anchor)
                            r['decoded_ego'][name]=trajectory_error(trajectory,truth,valid)
                            distances=(trajectory-truth).norm(dim=-1)
                            r['decoded_ego'][name]['point_error']=[float(d) if bool(ok) else None for d,ok in zip(distances,valid)]
                else:r['decoded_ego']=None
            except Exception as error:r['failure']=repr(error)
            with path.open('a') as f:f.write(json.dumps(r)+'\n')
            rows.append(r)
            if len(rows)%25==0:atomic_json(out/f'eval_{split}_{a.shard}.json',{'status':'RUNNING','scenes':len(rows),'failed':sum(x['failure'] is not None for x in rows)})
        torch.save({k:vars(v) for k,v in moments.items()},out/f'{split}_moments_{a.shard}.pt')
        atomic_json(out/f'eval_{split}_{a.shard}.json',{'status':'COMPLETE' if not any(r['failure'] for r in rows) else 'FAILED','scenes':len(rows),'failed':sum(x['failure'] is not None for x in rows),'checkpoint':cp,'precision':'FP32 master / FP32 compute / TF32 off'})
        if any(r['failure'] for r in rows):raise RuntimeError('Audit failed rows preserved')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('fit','fit-change','encode','evaluate','summarize'))
    for k in ('plan','reference-representations','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--arm',default='S0',choices=('S0','S1','S2','S3','S4'));p.add_argument('--templates');p.add_argument('--teacher-root');p.add_argument('--teacher-data');p.add_argument('--frozen-teacher')
    p.add_argument('--representation-root',help='Immutable previously encoded endpoint cache')
    p.add_argument('--dino-sequence-current-root',help='Same384x288 DINO encoder/pool cache, used only as static reference')
    p.add_argument('--static-train');p.add_argument('--static-dev')
    p.add_argument('--change-thresholds')
    p.add_argument('--shards',type=int,default=8);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.mode=='evaluate':
        needed=('templates','teacher_root','teacher_data','frozen_teacher','change_thresholds')
        if any(not getattr(a,k) for k in needed):raise ValueError('Evaluation requires genuine fixed teacher and training templates')
        if a.arm in ('S0','S1') and not a.dino_sequence_current_root:raise ValueError('True current384x288 DINO reference required')
        if a.arm not in ('S0','S1') and not (a.static_train and a.static_dev):raise ValueError('Same-video-encoder static references required')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);plan=read(a.plan)
    with metered_run(a.campaign_root,a.run_id,0 if a.mode in ('fit','fit-change','summarize') else 1,{'kind':'S100k_endpoint_'+a.mode,'real_optimizer_updates':0}) as (meter,_,save):
        {'fit':fit,'fit-change':fit_change,'encode':encode,'evaluate':evaluate,'summarize':summarize}[a.mode](a,out,plan);save()

if __name__=='__main__':main()
