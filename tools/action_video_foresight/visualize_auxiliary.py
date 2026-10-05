"""Frozen current/future visual audit, fixed scene gallery, no optimizer updates.

prepare locks attribute selection and fits display-only PCA on training labels.
infer loads FP32 optimizer masters and preserves every development failure.
render writes private source pictures ONLY to the external artifact directory.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time
import shutil

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from torch.nn import functional as F

from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset, decode_ego
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.foresight.checkpoints import checkpoint_identity, load_student
from tools.full_foresight.evaluate_auxiliary import FixedPositionMoments
from .visualization_metrics import (clean_features, reference_metrics, patch_error,
    affinity_map, select_representatives, projected_vehicle_centres, relative_gain)


def read_json(path): return json.loads(Path(path).read_text())


def visual_data(root, cfg, split='dev', candidate='C1'):
    dino = Path(cfg['dino_root']).parent/candidate
    ident = read_json(dino/'identity.json')
    return FullForesightDataset(cfg[split+'_data'], candidate=candidate, current=True, future=True,
        dino_root=dino, dino_index=cfg['dino_index'], expected_dino=ident['identity'], allow_partial=False)


def images(ds, i):
    pairs = [[ds.read_image(j) for j in row] for row in ds.dino_scenes[i]['images']]
    features = torch.stack([torch.stack([z for z,_ in row]) for row in pairs]).permute(0,1,3,4,2)
    valid = torch.stack([torch.stack([m for _,m in row]) for row in pairs])
    return features.float(), valid


def clip(cache, i):
    ident = read_json(Path(cache)/'identity.json')
    chunk, at = divmod(i, ident['chunk_size'])
    with safe_open(str(Path(cache)/f'chunk_{chunk:06d}.safetensors'), framework='pt') as f:
        if f.metadata()['identity'] != ident['identity']: raise ValueError('Foreign clip shard')
        z, m = f.get_slice('features')[at], f.get_slice('valid')[at]
    if list(z.shape) != ident['target_shape']: raise ValueError('Native clip geometry changed')
    return z.float(), m


def fit_display_basis(values):
    x = torch.cat(values).float().cuda(); mean = x.mean(0)
    centered = x-mean; covariance = centered.T@centered/max(1,len(x)-1)
    _, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:,-3:].flip(-1)
    # Fix otherwise arbitrary PCA signs for reproducible display.
    for c in range(3):
        j = basis[:,c].abs().argmax()
        if basis[j,c]<0: basis[:,c] *= -1
    reduced = centered@basis
    low, high = torch.quantile(reduced,.01,dim=0), torch.quantile(reduced,.99,dim=0)
    return {k:v.cpu() for k,v in dict(mean=mean,basis=basis,low=low,high=high).items()}


def prepare(args):
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    if args.reuse_preparation:
        parent=Path(args.reuse_preparation)
        candidates=read_json(parent/'representatives.json')
        ids=[int(x) for x in args.representative_indices.split(',')]
        if len(ids)!=args.gallery_size or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=len(candidates):raise ValueError('Explicit representative subset')
        selected=[candidates[i] for i in ids]
        for name in ('config.json','population.json','display_projection.pt'):shutil.copyfile(parent/name,out/name)
        atomic_json(out/'representatives.json',selected)
        identity=read_json(parent/'identity.json')
        identity.update(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            gallery_scenes=len(selected),selection_sha256=file_sha256(out/'representatives.json'),
            selection='10 cases chosen from the already attribute-selected24 after viewing CURRENT RGB only; no predictions/errors used',
            parent_preparation_identity_sha256=file_sha256(parent/'identity.json'),parent_representative_indices=ids,
            scope='10 representative current/future auxiliary cases ONLY; not a complete-development statistical conclusion')
        atomic_json(out/'identity.json',identity)
        print('Reused fixed TRAIN-only display projection; selected',len(selected),'cases',flush=True)
        return
    cfg = read_json(args.plan)
    cfg.update(teacher_data=args.teacher_data, c1_run=args.c1_run, legacy_representations=args.legacy_representations)
    train, dev = [ForesightCurrentDataset(cfg[s+'_data']) for s in ('train','dev')]
    if {r['log'] for r in train.index}&{r['log'] for r in dev.index}: raise ValueError('Train/dev log overlap')
    rows=[]
    clip_rows=read_json(Path(cfg['campaign_root'])/'clip_index_v1/dev_scenes.json')
    if [r['token'] for r in clip_rows]!=[r['token'] for r in dev.index]: raise ValueError('Clip population/order')
    for i,row in enumerate(dev.index):
        current=read_json(dev.root/'current'/(row['token']+'.json'))
        label=torch.load(Path(args.teacher_data)/'records'/(row['token']+'.pt'),weights_only=True)['record']
        moves=[]
        for j in range(1,len(label['active'])):
            m=label['point_valid'][j]
            if label['active'][j] and m.any():
                moves.append(float((label['future'][j,m]-label['current'][j,:2]).norm(dim=-1).max()))
        speed=current['ego_speed']
        row={**row,'index':i,'navigation':current['navigation'],'ego_speed':speed,
             'ego_motion':'stopped' if speed<.3 else 'slow' if speed<3 else 'moving',
             'peer_motion':'moving_peer' if moves and max(moves)>2 else 'stationary_peer' if moves else 'no_valid_peer',
             'peer_count':int(label['active'][1:].sum()),'clip_valid':all(clip_rows[i]['clip_view_valid'])}
        row['anchors']=projected_vehicle_centres(current,label['current'].numpy(),label['active'].numpy())
        rows.append(row)
    selected=select_representatives(rows,args.gallery_size)
    atomic_json(out/'representatives.json',selected)
    atomic_json(out/'population.json',rows)
    atomic_json(out/'config.json',cfg)
    # Display projection is fitted to training LABELS only, never a model input,
    # target compressor, prediction decoder or additional learned objective.
    order=sorted(range(len(train.index)),key=lambda i:hashlib.sha256(('viz-train-v1:'+train.index[i]['token']).encode()).hexdigest())[:64]
    c1=visual_data(out,cfg,'train'); values={'current':[],'dino_sequence':[],'video_clip':[]}
    thresholds={'dino_sequence':[],'video_clip':[]}
    for i in order:
        f,m=images(c1,i);values['current'].append(f[0][m[0]])
        for kind in ('dino_sequence','video_clip'):
            cache=Path(cfg['campaign_root'])/('targets_'+kind+'_train_v1')
            f,m=clip(cache,i);z=clean_features(f,m,True)
            # Deterministic row-major sparse sample controls display fitting cost.
            values[kind].append(z[m][::16])
            delta=(z[:,-1]-z[:,0]).square().mean(-1); valid=m[:,-1]&m[:,0]
            thresholds[kind].append(delta[valid])
    projections={k:fit_display_basis(v) for k,v in values.items()}
    torch.save(projections,out/'display_projection.pt')
    identity={'schema':'auxiliary_visualization_v1','source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'plan_sha256':file_sha256(args.plan),'dev_identity':dev.identity['identity'],
        'dev_scenes':len(dev),'dev_logs':len({r['log'] for r in dev.index}),
        'selection':'round-robin navigation/ego-speed/GT-peer-motion/clip-valid strata; hashed within stratum, prefer unrepresented logs; no model errors',
        'gallery_scenes':len(selected),'selection_sha256':file_sha256(out/'representatives.json'),
        'projection_training_scenes':[train.index[i] for i in order],
        'projection_sha256':file_sha256(out/'display_projection.pt'),
        'display':'3-channel PCA for visualization ONLY; same train-fit basis/scales for target/prediction/references; not RGB reconstruction',
        'temporal_change_thresholds':{k:float(torch.quantile(torch.cat(v),.75)) for k,v in thresholds.items()},
        'real_optimizer_updates':0,'precision':'FP32 master/compute, TF32 off',
        'scope':'current/future auxiliary diagnostics on dev, no Navtest selection or new PDMS evaluation'}
    atomic_json(out/'identity.json',identity)
    print(json.dumps({'selected':len(selected),'strata':dict(Counter(str((r['navigation'],r['ego_motion'],r['peer_motion'])) for r in selected))},ensure_ascii=False),flush=True)


def infer(args):
    root=Path(args.output);cfg=read_json(root/'config.json');pop=read_json(root/('representatives.json' if args.scope=='representatives' else 'population.json'))
    out=root/args.arm;out.mkdir(exist_ok=False);(out/'features').mkdir()
    run=Path(cfg['c1_run']) if args.arm=='C1' else Path(cfg['campaign_root'])/'students'/f'formal_{args.arm}_seed42_full100k_v2'
    start=time.time();training,cp=checkpoint_identity(run,'milestone_100000')
    model=load_student(run,'milestone_100000',training,strip=False);model.requires_grad_(False)
    if cp['completed']!=100000:raise ValueError('Only common 100k endpoint')
    current=ForesightCurrentDataset(cfg['dev_data']);ds=visual_data(root,cfg)
    kinds=cfg['campaign_root'];selection={r['token'] for r in read_json(root/'representatives.json')}
    templates=torch.load(Path(kinds)/'training_templates_C1_v1/templates.pt',weights_only=True)
    current_mean=templates['visual_mean'][0].permute(0,2,3,1)
    kind='legacy_single_frame' if args.arm=='C1' else training['config']['foresight']['future_target_type']
    if kind!='legacy_single_frame':
        cache=Path(kinds)/('targets_'+kind+'_dev_v1'); ci=read_json(cache/'identity.json')
        if ci['scene_index_hash']!=current.identity['index_sha256']:raise ValueError('Future dev population')
        if identity_hash(ci['recipe'])!=training['config']['foresight']['video_teacher_identity']:raise ValueError('Teacher recipe differs')
        means=torch.load(Path(kinds)/('future_'+kind+'_normalized_train_mean_v1.pt'),weights_only=True)
        train_ci=read_json(Path(kinds)/('targets_'+kind+'_train_v1/identity.json'))
        if means['identity']['train_targets']!=train_ci['identity'] or means['identity']['split']!='train':raise ValueError('Mean not train-only')
        future_mean=means['mean'];intervals=ci['time_intervals_s']
        c3=visual_data(root,cfg,candidate='C3') if kind=='dino_sequence' else None
        static_identity=read_json(Path(kinds)/'static_video_dev_reference_v1/identity.json') if kind=='video_clip' else None
        if static_identity and static_identity['future_target_identity']!=ci['identity']:raise ValueError('Wrong static encoder')
    else:intervals=[[h,h] for h in (1,2,4)]
    # One current forward per scene. Stored only in this diagnostic directory.
    worlds=[]
    with torch.inference_mode():
        for i,row in enumerate(pop):
            if args.arm=='C1':
                z=torch.load(Path(cfg['legacy_representations'])/'representations'/(row['token']+'.pt'),weights_only=True)
                if z['checkpoint']!=cp['sha256']:raise ValueError('Foreign cached W')
                w=z['W']
            else:w=model.encode_current([current[row['index']]])['W'][0].cpu()
            if not torch.isfinite(w).all():raise ValueError('Invalid W')
            worlds.append(w)
            if i%10==0:print(f'{args.arm} encode {i}/{len(pop)}',flush=True)
    world=torch.stack(worlds);del worlds
    # A fixed identity-only derangement, identical for all arms.
    order=sorted(range(len(pop)),key=lambda i:hashlib.sha256(('viz-W-swap-v1:'+pop[i]['token']).encode()).hexdigest())
    swap={order[i]:order[(i+len(order)//2)%len(order)] for i in range(len(order))}
    rows=[];moments={};cfg_head=model.foresight_config
    def add_moment(key,z,m):moments.setdefault(key,FixedPositionMoments()).add(z,m[...,None])
    with torch.inference_mode():
        for i,row in enumerate(pop):
            source_index=row['index']
            result={**{k:v for k,v in row.items() if k!='anchors'},'failure':None,'scores':{},'swap_token':pop[swap[i]]['token']}
            try:
                w=world[i:i+1].cuda();sw=world[swap[i]:swap[i]+1].cuda()
                f,m=images(ds,source_index);target=f[0];valid=m[0]
                pred=model.dino_head(w,torch.zeros(1,device='cuda'),(6,8))[0].permute(0,2,3,1).cpu()
                shuffled=model.dino_head(sw,torch.zeros(1,device='cuda'),(6,8))[0].permute(0,2,3,1).cpu()
                pack={'current_target':target,'current_pred':pred,'current_shuffled':shuffled,'current_mean':current_mean,'current_valid':valid}
                for v in range(3):
                    for name,z in [('model',pred),('train_mean',current_mean),('shuffled_W',shuffled)]:
                        result['scores'][f'current/v{v}/{name}']=reference_metrics(z[v],target[v],valid[v])
                    for name,z in [('model',pred),('target',target)]:add_moment('current/v'+str(v)+'/'+name,z[v],valid[v])
                if kind=='legacy_single_frame':
                    ft=f[1:];fv=m[1:];fp=torch.stack([model.dino_head(w,torch.tensor([float(h)],device='cuda'),(6,8))[0].permute(0,2,3,1).cpu() for h in (1,2,4)]).transpose(0,1)
                    fs=torch.stack([model.dino_head(sw,torch.tensor([float(h)],device='cuda'),(6,8))[0].permute(0,2,3,1).cpu() for h in (1,2,4)]).transpose(0,1)
                    ft=ft.transpose(0,1);fv=fv.transpose(0,1)
                    fm=templates['visual_mean'][1:].permute(1,0,3,4,2)
                    static=target[:,None].expand_as(ft);static_mask=valid[:,None].expand_as(fv)&fv
                else:
                    ft,fv=clip(cache,source_index)
                    action=None
                    if cfg_head.future_action_condition=='gt_ego':
                        ego=torch.load(current.root/'ego'/(row['token']+'.pt'),weights_only=True)['ego']
                        decoded=decode_ego(ego.cuda());action=torch.cat((decoded[:,:2],decoded[:,2:3].sin(),decoded[:,2:3].cos()),-1)[None]
                    fp=model.spatiotemporal_head(w,(9,12),gt_action=action)[0].cpu()
                    fs=model.spatiotemporal_head(sw,(9,12),gt_action=action)[0].cpu()
                    ft=clean_features(ft,fv,True);fp=clean_features(fp,fv,True);fs=clean_features(fs,fv,True);fm=future_mean
                    if c3:
                        source,source_valid=images(c3,source_index);static=source[0][:,None].expand_as(ft);static_mask=source_valid[0][:,None].expand_as(fv)&fv
                        static=clean_features(static,static_mask,True)
                    else:
                        value=torch.load(Path(kinds)/'static_video_dev_reference_v1/targets'/(row['token']+'.pt'),weights_only=True)
                        if value['identity']!=static_identity['identity']:raise ValueError('Foreign static clip')
                        static=clean_features(value['features'].float(),fv,True);static_mask=fv
                if not torch.equal(static_mask,fv):raise ValueError('Static reference cannot shrink evaluation denominator')
                pack.update(future_target=ft,future_pred=fp,future_shuffled=fs,future_mean=fm,future_static=static,future_valid=fv)
                for v in range(3):
                    for t in range(ft.shape[1]):
                        prefix=f'future/v{v}/t{t}'
                        for name,z in [('model',fp),('train_mean',fm),('shuffled_W',fs),('static',static)]:
                            result['scores'][prefix+'/'+name]=reference_metrics(z[v,t],ft[v,t],fv[v,t])
                        for name,z in [('model',fp),('target',ft)]:add_moment(prefix+'/'+name,z[v,t],fv[v,t])
                        # Upper quartile of target feature change within the clip,
                        # fixed from training labels; not object motion segmentation.
                        if kind!='legacy_single_frame':
                            delta=(ft[v,-1]-ft[v,0]).square().mean(-1)
                            cutoff=read_json(root/'identity.json')['temporal_change_thresholds'][kind]
                            for group,mask in [('high_change',fv[v,t]&(delta>cutoff)),('low_change',fv[v,t]&(delta<=cutoff))]:
                                for name,z in [('model',fp),('static',static)]:
                                    result['scores'][prefix+'/'+group+'/'+name]=reference_metrics(z[v,t],ft[v,t],mask)
                # Semantic affinity at projected current vehicle centres. Ground
                # truth centres are strictly analysis anchors, not student inputs.
                affinity=[]
                for anchor in row['anchors']:
                    v,y,x=anchor['view'],anchor['row'],anchor['column'];source=target[v,y,x]
                    gtmap=affinity_map(target[v],source);pmap=affinity_map(pred[v],source);meanmap=affinity_map(current_mean[v],source)
                    affinity.append({**anchor,'map_rmse':float((pmap-gtmap).square().mean().sqrt()),
                        'mean_map_rmse':float((meanmap-gtmap).square().mean().sqrt()),
                        'anchor_rank':int((pmap>pmap[y,x]).sum())+1,'patches':gtmap.numel(),
                        'target_anchor_similarity':float(F.cosine_similarity(pred[v,y,x],source,dim=0)),
                        'same_position_mean_similarity':float(F.cosine_similarity(current_mean[v,y,x],source,dim=0))})
                result['vehicle_anchor_diagnostics']=affinity
                if row['token'] in selection:
                    # Quantization is display-only; all above metrics remain FP32.
                    for k,z in pack.items():pack[k]=z if z.dtype==torch.bool else z.half()
                    torch.save(pack,out/'features'/(row['token']+'.pt'))
            except Exception as error:result['failure']=repr(error)
            rows.append(result)
            with (out/'scenes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if i%50==0:
                atomic_json(out/'progress.json',{'scenes':len(rows),'requested':len(pop),'failures':sum(r['failure'] is not None for r in rows)})
                print(f'{args.arm} heads {i}/{len(pop)}',flush=True)
    sums={};failed=sum(r['failure'] is not None for r in rows)
    for row in rows:
        for key,score in row['scores'].items():
            total=sums.setdefault(key,dict(patches=0,squared_channel_mean_sum=0.,scenes=0))
            for k in ('patches','squared_channel_mean_sum'):total[k]+=score[k]
            total['scenes']+=score['patches']>0
    for total in sums.values():total['mse']=total['squared_channel_mean_sum']/total['patches'] if total['patches'] and not failed else None
    anchors=[a for r in rows for a in r.get('vehicle_anchor_diagnostics',[])]
    summary={'arm':args.arm,'checkpoint':cp,'requested':len(pop),'scenes':len(rows),'logs':len({r['log'] for r in rows}),
        'failures':failed,'kind':kind,'time_intervals_s':intervals,'precision':model.deployment_precision,'scope':args.scope,
        'scores':sums,'fixed_position_variance':{k:v.result() for k,v in moments.items()},
        'vehicle_anchors':{'count':len(anchors),'scenes':sum(bool(r.get('vehicle_anchor_diagnostics')) for r in rows),
            **{k:float(np.mean([r[k] for r in anchors])) if anchors else None for k in ('map_rmse','mean_map_rmse','anchor_rank','target_anchor_similarity','same_position_mean_similarity')}},
        'selection_sha256':file_sha256(root/'representatives.json'),'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'allocation_seconds':time.time()-start,'allocation_gpu_hours':(time.time()-start)/3600,'real_optimizer_updates':0}
    atomic_json(out/'SUMMARY.json',summary)
    if failed:raise RuntimeError('Failed rows retained; no complete result')


def display_rgb(value, projection):
    z=(value.float()-projection['mean'])@projection['basis']
    return ((z-projection['low'])/(projection['high']-projection['low']).clamp_min(1e-8)).clamp(0,1).numpy()


def source_picture(path):
    with Image.open(path) as image:return np.asarray(image.convert('RGB').resize((512,288)))


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.output); cfg=read_json(root/'config.json'); reps=read_json(root/'representatives.json')
    bases=torch.load(root/'display_projection.pt',weights_only=True)
    clip_rows=read_json(Path(cfg['campaign_root'])/'clip_index_v1/dev_scenes.json')
    arms=[a for a in ('C1','S0','S1','S2','S3','S4') if (root/a/'SUMMARY.json').exists()]
    out=root/'gallery';out.mkdir(exist_ok=True);pages=[]
    for number,row in enumerate(reps):
        token=row['token']; alias=f'scene_{number:02d}'; scene=out/alias;scene.mkdir(exist_ok=True)
        packs={a:torch.load(root/a/'features'/(token+'.pt'),weights_only=True) for a in arms}
        current=read_json(Path(cfg['dev_data'])/'current'/(token+'.json'))
        for view in range(3):
            # One train-fit PCA scale for all models, targets and references.
            reference=packs[arms[0]];p=bases['current'];plots=[('Current RGB',source_picture(current['image_paths'][view]))]
            plots += [('Target feature (not RGB)',display_rgb(reference['current_target'][view],p)),('Train mean',display_rgb(reference['current_mean'][view],p))]
            plots += [(a+' predicted feature',display_rgb(packs[a]['current_pred'][view],p)) for a in arms]
            fig,axes=plt.subplots(2,len(plots),figsize=(3*len(plots),5.3),squeeze=False)
            errors=[patch_error(packs[a]['current_pred'][view].float(),reference['current_target'][view].float(),reference['current_valid'][view]) for a in arms]
            vmax=max(float(torch.quantile(torch.cat([e.flatten() for e in errors]),.98)),1e-6)
            for j,(name,value) in enumerate(plots):axes[0,j].imshow(value,interpolation='nearest');axes[0,j].set_title(name,fontsize=9);axes[0,j].axis('off');axes[1,j].axis('off')
            for j,(a,e) in enumerate(zip(arms,errors),3):
                im=axes[1,j].imshow(e.numpy(),cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');axes[1,j].set_title(f'MSE {float(e.mean()):.4f}',fontsize=9)
            fig.colorbar(im,ax=list(axes[1]),shrink=.6,label='FP16 display pack MSE; statistics use FP32')
            fig.suptitle(f'{alias} / view {view}: current grid6x8, common PCA colour basis; no RGB reconstruction')
            fig.savefig(scene/f'current_v{view}.png',dpi=100,bbox_inches='tight');plt.close(fig)
            if row['anchors']:
                anchors=[a for a in row['anchors'] if a['view']==view]
                if anchors:
                    anchor=anchors[0];y,x=anchor['row'],anchor['column'];source=reference['current_target'][view,y,x].float()
                    fig,axes=plt.subplots(1,len(arms)+2,figsize=(3*(len(arms)+2),3))
                    axes[0].imshow(source_picture(current['image_paths'][view]));axes[0].scatter(anchor['u']*512,anchor['v']*288,c='red');axes[0].set_title('GT vehicle centre anchor')
                    for ax,name,feat in zip(axes[1:],['Target']+arms,[reference['current_target'][view]]+[packs[a]['current_pred'][view] for a in arms]):
                        ax.imshow(affinity_map(feat,source),vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest');ax.scatter(x,y,c='black',s=15);ax.set_title(name)
                    for ax in axes:ax.axis('off')
                    fig.suptitle('Same TARGET patch query in every map; coarse vehicle/background mixtures, not segmentation')
                    fig.savefig(scene/f'affinity_v{view}.png',dpi=100,bbox_inches='tight');plt.close(fig)
        future_files=[]
        for a,pack in packs.items():
            summary=read_json(root/a/'SUMMARY.json');kind=summary['kind'];p=bases['current' if kind=='legacy_single_frame' else kind]
            intervals=summary['time_intervals_s'];time_count=len(intervals)
            for view in range(3):
                names=('Actual future RGB','Target feature','Model feature','Static reference','Train mean','Shuffled W','Model MSE','Static MSE','GT deviation from static','Predicted deviation from static')
                fig,axes=plt.subplots(len(names),time_count,figsize=(3*time_count,15.5),squeeze=False)
                error=patch_error(pack['future_pred'][view].float(),pack['future_target'][view].float(),pack['future_valid'][view])
                se=patch_error(pack['future_static'][view].float(),pack['future_target'][view].float(),pack['future_valid'][view])
                vmax=max(float(torch.quantile(torch.cat((error.flatten(),se.flatten())),.98)),1e-6)
                predicted_change=patch_error(pack['future_pred'][view].float(),pack['future_static'][view].float(),pack['future_valid'][view])
                change_vmax=max(float(torch.quantile(torch.cat((predicted_change.flatten(),se.flatten())),.98)),1e-6)
                for t,(lo,hi) in enumerate(intervals):
                    idx=int(round(hi*2))-1;frame=clip_rows[row['index']]['frames_by_view'][view][idx]
                    picture=source_picture(frame['path']) if frame else np.zeros((288,512,3),dtype=np.uint8)
                    values=[picture]+[display_rgb(pack[k][view,t],p) for k in ('future_target','future_pred','future_static','future_mean','future_shuffled')]+[error[t].numpy(),se[t].numpy(),se[t].numpy(),predicted_change[t].numpy()]
                    for y,value in enumerate(values):
                        if y>0 and not pack['future_valid'][view,t].any():
                            axes[y,t].text(.5,.5,'NO VALID CLIP TARGET',ha='center',va='center',fontsize=8,transform=axes[y,t].transAxes)
                        else:axes[y,t].imshow(value,**({'cmap':'magma','vmin':0,'vmax':change_vmax if y>=8 else vmax} if y>=6 else {}),interpolation='nearest')
                        axes[y,t].set_xticks([]);axes[y,t].set_yticks([])
                        if t==0:axes[y,t].set_ylabel(names[y],fontsize=8)
                    axes[0,t].set_title(f'{lo:g}-{hi:g}s'+(' (interval end RGB)' if lo!=hi else ''),fontsize=9)
                condition='GT ego action in AUXILIARY head only' if a in ('S1','S3','S4') else 'no GT action condition'
                fig.suptitle(f'{alias} {a} view{view}; {kind}; {condition}; '+('whole-clip bidirectional teacher tokens' if kind=='video_clip' else 'independent frame features'))
                fig.tight_layout();file=f'{a}_future_v{view}.png';fig.savefig(scene/file,dpi=95);plt.close(fig);future_files.append(file)
                if view==0:
                    # An actual sequence of feature panels, never generated RGB.
                    frames=[]
                    for t in range(time_count):
                        fs=[Image.fromarray((display_rgb(pack[k][view,t],p)*255).astype('uint8')).resize((288,162),Image.Resampling.NEAREST) for k in ('future_target','future_pred','future_static')]
                        if not pack['future_valid'][view,t].any():fs=[Image.new('RGB',(288,162),'gray') for _ in fs]
                        canvas=Image.new('RGB',(864,162));[canvas.paste(z,(j*288,0)) for j,z in enumerate(fs)];frames.append(canvas)
                    frames[0].save(scene/f'{a}_features.gif',save_all=True,append_images=frames[1:],duration=650,loop=0)
        body=f'<h1>{alias}</h1><p>Navigation {row["navigation"]}; ego {row["ego_motion"]}; {row["peer_motion"]}; {row["peer_count"]} selected GT peers. Selection precedes prediction.</p>'
        body+='<p>Colours: shared train-fit PCA display only, NOT reconstructed RGB. All teacher/GT paths are label-side. Hover/click to zoom.</p>'
        body+='<h2>Current alignment</h2>'+''.join(f'<a href="current_v{v}.png"><img src="current_v{v}.png"></a>' for v in range(3))
        body+='<h2>Vehicle-centre feature affinity (diagnostic anchors only)</h2>'+''.join(f'<a href="{f.name}"><img src="{f.name}"></a>' for f in sorted(scene.glob('affinity*.png')))
        body+='<h2>Future feature sequence</h2><label>Model <select id="arm">'+''.join(f'<option>{a}</option>' for a in arms)+'</select></label>'
        body+=''.join(f'<div class="future" data-arm="{a}"><p>{a}: target | prediction | static reference</p><img src="{a}_features.gif">'+''.join(f'<a href="{a}_future_v{v}.png"><img src="{a}_future_v{v}.png"></a>' for v in range(3))+'</div>' for a in arms)
        script='<script>function choose(){document.querySelectorAll(".future").forEach(x=>x.hidden=x.dataset.arm!==document.querySelector("#arm").value)}document.querySelector("#arm").onchange=choose;choose();</script>'
        (scene/'index.html').write_text('<meta charset="utf-8"><style>body{font-family:sans-serif;margin:24px}img{max-width:100%;display:block;margin-bottom:20px}a{color:#0369a1}</style>'+body+script)
        pages.append({'alias':alias,'token':token,'navigation':row['navigation'],'ego_motion':row['ego_motion'],'peer_motion':row['peer_motion'],'peer_count':row['peer_count']})
    atomic_json(root/'GALLERY_INDEX.json',pages)
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Frozen100k auxiliary visual audit</h1><p>Private sensor images: local analysis only. No optimizer updates. Scenes chosen from data attributes, before model results.</p>'+''.join(f'<p><a href="{r["alias"]}/index.html">{r["alias"]}</a> nav{r["navigation"]} / {r["ego_motion"]} / {r["peer_motion"]}</p>' for r in pages))
    print(out/'index.html')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','infer','render'))
    p.add_argument('--output',required=True);p.add_argument('--plan');p.add_argument('--teacher-data');p.add_argument('--c1-run');p.add_argument('--legacy-representations')
    p.add_argument('--gallery-size',type=int,default=10);p.add_argument('--arm',choices=('C1','S0','S1','S2','S3','S4'))
    p.add_argument('--scope',choices=('representatives','full_dev'),default='representatives')
    p.add_argument('--reuse-preparation');p.add_argument('--representative-indices')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    if a.mode=='prepare':prepare(a)
    elif a.mode=='infer':infer(a)
    else:render(a)


if __name__=='__main__':main()
