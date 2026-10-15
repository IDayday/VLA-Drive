"""One conditional, nondeployable current-state diagnostic with fit-only scaling."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import io
import os
import subprocess
import numpy as np
import torch
from torch import nn
from iqe.contracts import require,TrajectoryContract
from iqe.io import atomic_json,atomic_bytes,read_json,file_hash,digest,complete,lock
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.objectives import SceneSampler
from iqe.specialist_study.metrics import report_metrics
from iqe.specialist_study.training import runtime
from iqe.training.trainer import train
from .binding import verify_binding
from .data import read_split,canonical_bank,FrozenContexts
from .models import SetSelector,choose
from .objectives import fixed_normalizer,cost_ce,denominators
from .training import recipe,resource,take,create,predict

KEYS={'agents','agent_valid','map_points','map_valid','current_ego'}


def trigger(config):
    split=read_split(config);plan=read_json(config.root/'REFIT_PLAN.json')
    metrics={k:v['inner_metrics'] for k,v in plan['models'].items()}
    fits={};success={}
    for kind in ('M_scene','M_set'):
        p=config.root/'overfit'/kind/'FIT_REPORT.json'
        fits[kind]=read_json(p) if p.exists() else None
        # A bounded fitted model must capture >=80% of its available mean gain,
        # with <=.01 regret. This decides only whether to trust the implementation diagnostic.
        success[kind]=bool(fits[kind] and fits[kind]['regret_01']<=.01 and fits[kind]['oracle_capture'] is not None and fits[kind]['oracle_capture']>=.8)
    both_negative=all(v['selected_gain_01']<=0 for v in metrics.values())
    tail=all(v['high_to_zero']>0 for v in metrics.values()) and all(success.values())
    r=dict(status='TRIGGERED' if both_negative or tail else 'NOT_TRIGGERED',
        registered_rule=split['probe_trigger'],both_no_internal_gain=both_negative,both_tail_after_fitting=tail,
        overfit_success=success,overfit_metrics=fits,internal_metrics=metrics,
        internal_plan_hash=plan['hash'],dev_read=False)
    atomic_json(config.root/'current_probe/TRIGGER.json',r,immutable=True)
    return r


def plan_probe(config):
    source=config.source();split=read_split(config);bound=verify_binding(config);reason=trigger(config)
    if reason['status']!='TRIGGERED':return reason
    meta,bank,scenes=canonical_bank(config,'scorer_fit');lookup={s.scene_id:s for s in scenes}
    parts={}
    for role,ids,limit in [('internal_fit',split['fit']['scene_ids'],2048),('internal_val',split['validation']['scene_ids'],512)]:
        chosen=sorted(ids,key=lambda s:digest([config.partition_seed,'P_current',s]))[:limit]
        parts[role]=dict(scene_ids=chosen,scenes=len(chosen),logs=len({lookup[s].source_log_id for s in chosen}))
    rows=[]
    for role,part in parts.items():
        for sid in part['scene_ids']:
            s=lookup[sid];rows.append({k:getattr(s,k) for k in ('scene_id','observation_ref','observation_hash','metric_context_ref','metric_context_hash')}|dict(role=role))
    r=dict(schema='one_bounded_current_probe_v1',binding=bound['hash'],split_hash=split['hash'],bank_hash=meta['hash'],parts=parts,
        records=rows,updates=2000,global_batch=128,optimizer=dict(lr=1e-4,weight_decay=.01,warmup=100),
        structured_schema='t0_timestamp_attested_agents_and_static_map_v1',max_agents=64,max_map_points=256,radius_m=50,
        shape='same candidate-time/set decoder as M_set; current structured encoder replaces W/A only',
        normalization='internal_fit_valid_tokens_only',forbidden_export=True,init_seed=config.seed+300,
        main_models_changed=False,velocity='provided current annotation, never future finite difference',
        visibility='restricted to current camera frusta; occlusion not verified; map/objects remain privileged',
        checkpoint_selection='fixed last 2000, one internal_val report, no dev')
    r['hash']=digest(r);atomic_json(config.root/'current_probe/PLAN.json',r,immutable=True);return r


def prepare(config,workers=4):
    resource(config,'prepare-current-state-probe');require(1<=workers<=4,'bounded shared IO/CPU')
    plan=plan_probe(config)
    if plan.get('status')=='NOT_TRIGGERED':return plan
    root=config.root/'current_probe';repo=Path(__file__).parents[2]
    with lock(root/'PREPARE.lock'):
        requests=[]
        for shard in range(workers):
            path=root/f'worker_{shard:02d}_REQUEST.json'
            request=dict(navsim_root=config.source().navsim_root,output=str(root),plan_hash=plan['hash'],
                         scenes=plan['records'][shard::workers],receipt=str(root/f'worker_{shard:02d}_COMPLETE.json'))
            atomic_json(path,request,immutable=True);requests.append(path)
        def run(path):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONPATH=str(repo)+os.pathsep+str(repo/'nuplan-devkit'))
            with open(path.with_suffix('.log'),'ab',buffering=0) as f:
                result=subprocess.run([config.source().scoring_python,'-m','iqe.decision_selection.current_state','--request',str(path)],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            require(result.returncode==0,'current-state worker failed: '+str(path.with_suffix('.log')))
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,requests))
        outputs=[]
        for row in plan['records']:
            f=root/'records'/row['scene_id'];r=read_json(f/'COMPLETE.json')
            require(r['inputs']['plan_hash']==plan['hash'],'probe record belongs to another plan')
            for path,h in r['outputs'].items():require(file_hash(path)==h,'probe record corrupt')
            outputs.append(f/'COMPLETE.json')
        complete(root/'DATA_COMPLETE.json',dict(plan_hash=plan['hash']),outputs,scenes=len(outputs),official_score_calls=0)
        return dict(status='COMPLETE',scenes=len(outputs),plan_hash=plan['hash'])


def normalization(values,fit_indices):
    result={}
    for name,mask,columns in [('agents','agent_valid',[0,1,4,5,6,7,8]),('map_points','map_valid',[0,1]),('current_ego',None,list(range(4)))]:
        x=values[name][fit_indices].astype(np.float64)
        if mask:x=x[values[mask][fit_indices]]
        else:x=x.reshape(-1,x.shape[-1])
        require(len(x)>0,'no valid fit inputs for '+name)
        mean=np.zeros(x.shape[-1],np.float32);scale=np.ones(x.shape[-1],np.float32)
        mean[columns]=x[:,columns].mean(0);scale[columns]=np.maximum(x[:,columns].std(0),1e-3)
        result[name]=dict(mean=mean.tolist(),scale=scale.tolist(),valid_fit_tokens=len(x))
    return result


class StructuredContext(nn.Module):
    def __init__(self,statistics):
        super().__init__();self.projections=nn.ModuleDict()
        for name,width in [('agents',19),('map_points',24),('current_ego',4)]:
            self.register_buffer(name+'_mean',torch.tensor(statistics[name]['mean']))
            self.register_buffer(name+'_scale',torch.tensor(statistics[name]['scale']))
            self.projections[name]=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Linear(256,256),nn.LayerNorm(256))
        self.types=nn.Parameter(torch.randn(3,256)*.02)
    def forward(self,context):
        require(set(context)==KEYS,'privileged current input whitelist')
        tokens=[]
        for i,name in enumerate(('agents','map_points','current_ego')):
            x=context[name].detach().float()
            require(torch.isfinite(x).all(),'nonfinite current input')
            z=self.projections[name]((x-getattr(self,name+'_mean'))/getattr(self,name+'_scale'))+self.types[i]
            tokens.append(z[:,None] if name=='current_ego' else z)
        valid=torch.cat((context['agent_valid'].detach().bool(),context['map_valid'].detach().bool(),
                         torch.ones(len(tokens[0]),1,dtype=torch.bool,device=tokens[0].device)),1)
        return torch.cat(tokens,1),valid


class CurrentSelector(SetSelector):
    """Nondeployable; no W/A or external encoder, no future/reference state."""
    deployment_allowed=False
    def __init__(self,contract,statistics):
        super().__init__(1,1,contract)
        self.context=StructuredContext(statistics)


def run(config,device='cuda:0',resume=None,workers=4):
    resource(config,'bounded-current-state-probe',training=True);device=runtime(device)
    result=prepare(config,workers)
    if result['status']=='NOT_TRIGGERED':return result
    root=config.root/'current_probe';plan=read_json(root/'PLAN.json');binding=verify_binding(config)
    with lock(root/'TRAIN.lock'):
        meta,bank,scenes=canonical_bank(config,'scorer_fit');lookup={s.scene_id:i for i,s in enumerate(scenes)}
        ids=plan['parts']['internal_fit']['scene_ids']+plan['parts']['internal_val']['scene_ids']
        require(len(ids)==len(set(ids)),'probe fit/val overlap')
        arrays=[]
        for sid in ids:
            folder=root/'records'/sid;r=read_json(folder/'COMPLETE.json')
            for p,h in r['outputs'].items():require(file_hash(p)==h,'probe feature checksum')
            with np.load(folder/'current.npz',allow_pickle=False) as f:arrays.append({k:f[k].copy() for k in f.files})
        values={k:np.stack([a[k] for a in arrays]) for k in arrays[0]}
        index={sid:i for i,sid in enumerate(ids)};nf=plan['parts']['internal_fit']['scenes'];fit_ids=ids[:nf]
        stats=normalization(values,np.arange(nf));atomic_json(root/'FIT_NORMALIZATION.json',stats,immutable=True)
        with initialization_rng(plan['init_seed']):model=CurrentSelector(TrajectoryContract(**binding['trajectory']),stats)
        init=dict(initial_state_hash=module_hash(model),init_seed=plan['init_seed'],external_weights_loaded=False,
                  export_allowed=False,parameters=sum(p.numel() for p in model.parameters()))
        atomic_json(root/'INITIALIZATION.json',init,immutable=True)
        data={k:v[[lookup[sid] for sid in ids]] for k,v in bank.items()}
        z=fixed_normalizer(torch.from_numpy(data['scores'][:nf]),torch.from_numpy(data['valid'][:nf]))
        require(z>0,'no current-state decision signal')
        sampler=SceneSampler(fit_ids,config.sampler_seed,digest([plan['hash'],'current_natural_flow']))
        def fetch(entries,device):
            ix=[index[e['scene_id']] for e in entries]
            t=torch.from_numpy(data['trajectories'][ix]).to(device);role=torch.zeros(t.shape[:2],device=device,dtype=torch.bool);role[:,0]=True
            return dict(context={k:torch.from_numpy(v[ix]).to(device) for k,v in values.items()},trajectories=t,base_role=role,
                scores=torch.from_numpy(data['scores'][ix]).to(device),label_valid=torch.from_numpy(data['valid'][ix]).to(device),
                scene_valid=torch.tensor([not e.get('_padding',False) for e in entries],device=device))
        def objective(m,b):return dict(decision=cost_ce(m(b['context'],b['trajectories'],b['base_role'])['decision_logits'],b['scores'],b['label_valid'],z,b['scene_valid']))
        cfg=recipe(config,2000,validation=False)
        dependencies=dict(plan_hash=plan['hash'],Z_train=z,normalization=digest(stats),initialization=init['initial_state_hash'],
                          code=file_hash(__file__),extractor=file_hash(Path(__file__).with_name('current_state.py')))
        train(model,objective,fetch,sampler,cfg,dependencies,root/'training',device=device,seed=config.training_seed,
              resume=resume,denominator_function=denominators)
        state=read_json(root/'training/result.json')
        if state['status']!='COMPLETE':return state
        model.eval();reports={};preds={}
        for role_,role_ids in [('internal_fit',fit_ids),('internal_val',ids[nf:])]:
            logits=[]
            with torch.no_grad():
                for at in range(0,len(role_ids),32):
                    b=fetch([dict(scene_id=sid) for sid in role_ids[at:at+32]],device)
                    logits.append(model(b['context'],b['trajectories'],b['base_role'])['decision_logits'].cpu().numpy())
            logits=np.concatenate(logits);ix=[index[s] for s in role_ids];selected=choose(torch.from_numpy(logits),meta['expert_ids']).numpy()
            comps=np.where(data['component_valid'][ix],data['components'][ix],np.nan)
            reports[role_]=report_metrics(data['scores'][ix],comps,selected,[scenes[lookup[s]].source_log_id for s in role_ids],
                                         seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
            preds[role_]=logits
        b=io.BytesIO();np.savez(b,**preds);atomic_bytes(root/'PREDICTIONS.npz',b.getvalue(),immutable=True)
        # Compare the frozen internally selected main heads on exactly these
        # probe scenes. Full-D_S refits would have seen inner_val and are excluded.
        comparators={}
        for role_,role_ids in [('internal_fit',fit_ids),('internal_val',ids[nf:])]:
            pm,pb,ps=take(meta,bank,scenes,role_ids);cache=FrozenContexts(config,ps)
            comparators[role_]={}
            for kind in ('M_scene','M_set'):
                main,identity=create(config,kind,'diagnostic',cache.read(ps[0]))
                result=read_json(config.root/'diagnostic'/kind/'result.json')
                path=config.root/'diagnostic'/kind/'best.pt'
                require(file_hash(path)==result['best_checkpoint_hash'],'diagnostic comparator checkpoint changed')
                main.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                logits=predict(main.to(device).eval(),kind,ps,pb,cache,device)
                chosen=choose(torch.from_numpy(logits),meta['expert_ids']).numpy()
                stats_=report_metrics(pb['scores'],np.where(pb['component_valid'],pb['components'],np.nan),chosen,pm['source_logs'],
                    seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
                comparators[role_][kind]=dict(checkpoint_hash=result['best_checkpoint_hash'],metrics=stats_,
                    training_scenes=read_split(config)['fit']['scenes'],comparison_not_equal_data_or_parameter_budget=True)
                del main
        out=dict(status='COMPLETE',plan_hash=plan['hash'],metrics=reports,optimizer_updates=2000,
             checkpoint_hash=state['best_checkpoint_hash'],only_current_privileged_information=True,
             visibility_limitations=plan['visibility'],export_allowed=False,dev_accessed=False,
             normalization_fit_scenes=nf,main_weights_modified=False,matched_subset_comparators=comparators,
             interpretation='privileged information diagnostic; extra visibility/annotation precision, model capacity and sample count differ')
        atomic_json(root/'RESULTS.json',out,immutable=True);return out
