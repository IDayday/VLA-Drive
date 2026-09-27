"""Budgeted two-forward mechanism trainer; review ledger forbids real updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from starVLA.model.modules.joint_scene.contracts import SCHEMA_VERSION
from starVLA.model.modules.joint_scene.graphs import GraphConfig
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss
from starVLA.model.modules.joint_scene.masks import RoleScheduler
from starVLA.model.modules.joint_scene.masks import role_completion_mask
from tools.joint_local_scene_v3.data import AnnotatedCorpus
from tools.joint_local_scene_v3.runtime import batch_scenes,evaluate,build_queries
from tools.joint_local_scene_v3.budget import BudgetRun,atomic_json


CODE_FILES=('starVLA/model/modules/joint_scene/contracts.py','starVLA/model/modules/joint_scene/graphs.py',
    'starVLA/model/modules/joint_scene/masks.py','starVLA/model/modules/joint_scene/flow.py',
    'starVLA/model/modules/joint_world/flow.py','starVLA/model/modules/action_model/flow_matching_head/action_encoder.py',
    'tools/joint_local_scene_v3/data.py','tools/joint_local_scene_v3/runtime.py','tools/joint_local_scene_v3/train_mechanism.py','tools/joint_local_scene_v3/budget.py')


def code_identity():
    root=Path(__file__).resolve().parents[2]
    return {'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'files_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in CODE_FILES}}


def validate_config(cfg):
    required={'schema_version','model','graph','training_seed','sampling_seed','sampling_steps','role_loss_weight','forwards_per_batch','initial_lr','weight_decay','warmup_steps'}
    metadata={'current_neighbor_velocity','stage','condition_dim_source'}
    if required-set(cfg) or set(cfg)-required-metadata:raise ValueError('Missing/unknown training config keys')
    if cfg['schema_version']!=SCHEMA_VERSION:raise ValueError('Wrong training schema')
    if cfg['forwards_per_batch']!=2:raise ValueError('Only two matched forwards per batch are supported')
    for key in ('training_seed','sampling_seed','sampling_steps','warmup_steps'):
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<(1 if key.endswith('steps') else 0):raise ValueError('Invalid '+key)
    for key in ('role_loss_weight','initial_lr','weight_decay'):
        if not isinstance(cfg[key],(int,float)) or not math.isfinite(cfg[key]) or cfg[key]<(1e-15 if key=='initial_lr' else 0):raise ValueError('Invalid '+key)
    allowed={'dim','heads','layers','steps','condition_dim','xy_scale','state_policy'}
    if set(cfg['model'])!=allowed:raise ValueError('Model settings must be explicit, including actual condition_dim/state_policy')
    GraphConfig(**cfg['graph']).validate()
    return cfg


def forward_weights(mode,role_weight):
    return [.5,.5] if mode=='all' else [1/(1+role_weight),role_weight/(1+role_weight)]


def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'],device)


def deterministic_settings():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train','holdout','config','output','ledger','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=['all','mask'],required=True);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--schedule-updates',type=int,required=True);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--limit',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--eval-train',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--acknowledge-stop',action='store_true',help='Explicitly archive an existing STOP_REQUESTED before resume')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--deterministic',action='store_true')
    return p


def validate_arguments(a):
    if not str(a.output).strip():raise ValueError('Nonempty output directory is required')
    if min(a.updates,a.schedule_updates,a.batch,a.eval_every)<1 or a.updates>a.schedule_updates:raise ValueError('Invalid update/schedule/batch/evaluation interval')
    if a.limit is not None and a.limit<1:raise ValueError('Invalid dataset limit')
    if a.stop_after is not None and not 1<=a.stop_after<=a.updates:raise ValueError('Invalid stop boundary')
    out=Path(a.output)
    if not a.resume and out.exists() and any(out.iterdir()):raise FileExistsError('Nonempty output requires explicit resume; no overwrite')
    if a.resume and not (out/'checkpoint.pt').exists():raise FileNotFoundError('Resume checkpoint missing')
    if a.acknowledge_stop and not a.resume:raise ValueError('Stop acknowledgment is only for resume')
    if (out/'STOP_REQUESTED').exists() and not a.acknowledge_stop:raise ValueError('STOP_REQUESTED exists; acknowledge explicitly before any update')
    return out


def run(a):
    out=validate_arguments(a);cfg=validate_config(json.loads(Path(a.config).read_text()))
    if a.seed is not None and a.seed!=cfg['training_seed']:raise ValueError('CLI seed differs from declared training_seed')
    seed=cfg['training_seed'];source=code_identity();device=torch.device(a.device)
    run_identity={'source':source,'output':str(out.resolve()),'config':cfg,'train':str(Path(a.train).resolve()),'holdout':str(Path(a.holdout).resolve()),
        'mode':a.mode,'batch':a.batch,'schedule_updates':a.schedule_updates,'seed':seed,'device':str(device),'deterministic':a.deterministic,'limit':a.limit,'eval_every':a.eval_every,'eval_train':a.eval_train}
    with BudgetRun(a.ledger,a.run_id,run_identity,int(device.type=='cuda'),a.resume) as budget:
        try:
            out.mkdir(parents=True,exist_ok=True)
            train=AnnotatedCorpus(a.train,a.limit);holdout=AnnotatedCorpus(a.holdout)
            data_kind=train.manifest['data_kind']
            if data_kind!=holdout.manifest['data_kind']:raise ValueError('Training/holdout provenance mismatch')
            budget.require_updates(data_kind) # review ledger hard-forbids all real updates
            for corpus in (train,holdout):
                if corpus.manifest['horizon_steps']!=cfg['model']['steps'] or corpus.manifest['time_step_s']!=.5:raise ValueError('Data time horizon differs from model')
                if corpus.manifest['graph_config']!=cfg['graph']:raise ValueError('Corpus graph rule differs from configuration')
                for i in range(len(corpus)):
                    if corpus[i].future.shape[2]!=cfg['model']['steps']:raise ValueError('A scene has a different label horizon')
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if a.deterministic:deterministic_settings()
            if device.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=JointSceneFlow(**cfg['model']).to(device);model.condition.requires_grad_(False)
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=cfg['initial_lr'],weight_decay=cfg['weight_decay'])
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/cfg['warmup_steps'],1)*(.1+.9*.5*(1+math.cos(math.pi*min(s,a.schedule_updates)/a.schedule_updates))))
            noise_rng=torch.Generator(device=device).manual_seed(seed+101);time_rng=torch.Generator(device=device).manual_seed(seed+201);roles=RoleScheduler(seed+301)
            # Provenance/counting hooks also include every integration forward in evaluation.
            hook=model.register_forward_pre_hook(lambda *unused:budget.note(data_kind,forwards=1))
            identity={'schema_version':SCHEMA_VERSION,'source':source,'config':cfg,'mode':a.mode,'batch':a.batch,'schedule_updates':a.schedule_updates,
                'training_seed':seed,'train_identity':train.manifest['identity_sha256'],'holdout_identity':holdout.manifest['identity_sha256'],
                'limit':a.limit,'device':str(device),'deterministic':a.deterministic,'data_kind':data_kind,
                'trainable_parameters':sum(p.numel() for p in parameters),'all_parameters':sum(p.numel() for p in model.parameters()),
                'forwards_per_batch':2,'forward_weights':forward_weights(a.mode,cfg['role_loss_weight']),'resume_boundary':'optimizer step, same code/config/data/device and deterministic settings; no cross-device bitwise claim'}
            step=epoch=offset=presentations=forward_count=0;supervised={};task_totals={}
            if a.resume:
                saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
                if saved['identity']!=identity:raise ValueError('Checkpoint data/schema/graph/code/config identity differs')
                if saved['step']>=a.updates:raise ValueError('Resume has no requested remaining updates')
                if a.stop_after is not None and a.stop_after<=saved['step']:raise ValueError('Stop boundary is not ahead of resume')
                model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
                step,epoch,offset,presentations,forward_count=[saved[k] for k in ('step','epoch','offset','presentations','forward_count')]
                supervised=saved['supervised'];task_totals=saved['task_totals'];restore_rng(saved['rng'],device)
                noise_rng.set_state(saved['noise_rng']);time_rng.set_state(saved['time_rng']);roles.load_state_dict(saved['role_scheduler'])
                if (out/'STOP_REQUESTED').exists():
                    archive=out/f'STOP_REQUESTED.acknowledged_step{step}'
                    if archive.exists():raise FileExistsError('Stop archive exists; inspect instead of overwriting')
                    (out/'STOP_REQUESTED').rename(archive)
            atomic_json(out/'manifest.json',identity)
            query_manifests={name:build_queries(c) for name,c in [('holdout',holdout)]+([('train',train)] if a.eval_train else [])}
            for name,q in query_manifests.items():
                qpath=out/(name+'_queries.json')
                if qpath.exists() and json.loads(qpath.read_text())!=q:raise ValueError('Fixed query identity changed')
                if not qpath.exists():atomic_json(qpath,q)
            def save():
                payload={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'forward_count':forward_count,'supervised':supervised,'task_totals':task_totals,
                    'rng':rng_state(device),'noise_rng':noise_rng.get_state(),'time_rng':time_rng.get_state(),'role_scheduler':roles.state_dict()}
                torch.save(payload,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
            def assess(tag):
                for name,corpus in [('holdout',holdout)]+([('train',train)] if a.eval_train else []):
                    budget.check();report=evaluate(model,corpus,query_manifests[name],cfg['sampling_seed'],cfg['sampling_steps'],device=str(device),output=out/f'{name}_{tag}')
                    if not report['summary']['aggregate_valid']:raise RuntimeError('Evaluation failure retained in complete query/scene tables')
            if not a.resume:assess('0')
            model.train();paused=False
            while step<a.updates:
                try:budget.check();budget.require_updates(data_kind)
                except RuntimeError:paused=True;save();break
                if (out/'STOP_REQUESTED').exists():paused=True;save();break
                order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(train)).tolist()
                ids=order[offset:offset+a.batch];scenes=[train[i] for i in ids]
                if not scenes:raise ValueError('Invalid saved data offset')
                start=time.monotonic();g,y,valid=batch_scenes(scenes,device);optimizer.zero_grad(set_to_none=True)
                counts_parts=[];role_stats={};losses=[];weights=identity['forward_weights']
                for pass_index in range(2):
                    hidden=g.active_actor_mask
                    if pass_index==1 and a.mode=='mask':
                        hidden,statistics=role_completion_mask(valid,g.active_actor_mask,roles)
                        role_stats={k:v.tolist() for k,v in statistics.items()}
                        for key in ('neighbor_requested','neighbor_actual','ego_only_fallback','ego_valid_coordinates','neighbor_valid_coordinates'):
                            task_totals[key]=task_totals.get(key,0)+int(statistics[key].sum())
                        task_totals['nominal_role_tasks']=task_totals.get('nominal_role_tasks',0)+len(scenes)
                    noise=torch.randn(y.shape,device=device,generator=noise_rng);tau=torch.rand(len(scenes),device=device,generator=time_rng)
                    sums,counts=flow_loss_sums(model,y,valid,hidden,noise,tau,g);loss=normalized_loss(sums,counts)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint loss')
                    (loss*weights[pass_index]).backward();budget.note(data_kind,backwards=1);losses.append(float(loss.detach()));counts_parts.append(counts)
                    for k,v in counts.items():
                        name=('all_hidden' if pass_index==0 or a.mode=='all' else 'role')+'_'+k;supervised[name]=supervised.get(name,0)+v
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
                budget.claim_update(data_kind);optimizer.step();budget.completed_update(data_kind);scheduler.step()
                step+=1;offset+=len(scenes);presentations+=len(scenes);forward_count+=2
                if offset==len(train):epoch+=1;offset=0
                if device.type=='cuda':torch.cuda.synchronize()
                row={'step':step,'epoch':epoch,'offset':offset,'presentations':presentations,'forward_scene_presentations':presentations*2,'forward_count':forward_count,
                    'weighted_loss':sum(x*w for x,w in zip(losses,weights)),'forward_losses':losses,'forward_weights':weights,
                    'valid_coordinates_per_forward':counts_parts,'supervised':dict(supervised),'role_tasks':role_stats,'task_totals':dict(task_totals),
                    'gradient_before_clip':norm,'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'lr':optimizer.param_groups[0]['lr'],'data_kind':data_kind}
                with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(out/'progress.json',{'step':step,'epoch':epoch,'offset':offset,'presentations':presentations})
                paused=(out/'STOP_REQUESTED').exists() or (a.stop_after is not None and step>=a.stop_after and step<a.updates)
                if step%a.eval_every==0 or step==a.updates:assess(str(step))
                save()
                if paused:
                    if not (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').write_text('Explicit requested synthetic/test pause boundary.\n')
                    break
            save();hook.remove();budget.terminal='paused' if paused else 'complete'
            atomic_json(out/'status.json',{'status':budget.terminal,'step':step,'epochs':epoch,'presentations':presentations,'data_kind':data_kind,'forward_count':forward_count,'supervised_coordinates':supervised,'task_totals':task_totals})
        except BaseException as exc:
            if out.exists():
                atomic_json(out/(f'resume_failure_{time.time_ns()}.json' if a.resume else 'status.json'),{'status':'failed','error':str(exc),'run_id':a.run_id})
            raise


def main():run(parser().parse_args())
if __name__=='__main__':main()
