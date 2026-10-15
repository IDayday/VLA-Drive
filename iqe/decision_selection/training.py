"""One natural scene stream, existing IQE atomic trainer and exact state resume."""
from pathlib import Path
import io
import json
import math
import socket
import numpy as np
import torch
from iqe.contracts import TrajectoryContract, require
from iqe.io import atomic_json, atomic_bytes, read_json, file_hash, digest, lock
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.objectives import SceneSampler
from iqe.specialist_study.training import runtime
from iqe.specialist_study.metrics import report_metrics
from iqe.training.trainer import train
from .binding import verify_binding
from .data import canonical_bank, read_split, FrozenContexts
from .models import SceneRouter, SetSelector, choose
from .objectives import fixed_normalizer, cost_ce, denominators

KINDS = ('M_scene', 'M_set')


def resource(config, command, training=False):
    policy = Path('/mnt/project/server_dispatch_policy.json')
    if policy.exists():
        d = read_json(policy); hosts = set(d['allowed_hosts'])
        for row in d['allowed_hosts'].values(): hosts.update(row.get('aliases', []))
        require(socket.gethostname() in hosts, 'decision study requires permanently authorized host')
        if training: require(not socket.gethostname().startswith('training-vla-zt2'), 'preserve G2/Base host')
    from iqe.specialist_study.resource_guard import check
    return check(config, command)


def create(config, kind, phase, example=None):
    require(kind in KINDS and phase in ('diagnostic','refit','overfit','resume_audit'), 'model/run phase')
    binding = verify_binding(config)
    if example is None:
        scene = scenes_by_role(config.source())['scorer_fit'][0]
        example = ReusedContext(config.source(),capacity=0).read(scene)
    dims = [example['world'].shape[-1], example['action'].shape[-1]]
    contract = TrajectoryContract(**binding['trajectory'])
    with initialization_rng(config.seed):
        module = (SceneRouter(*dims,binding['expert_ids']) if kind == 'M_scene'
                  else SetSelector(*dims,contract))
    identity = dict(schema='scratch_cost_sensitive_decision_v1', kind=kind,phase=phase,
                    experiment=config.experiment_id,init_seed=config.seed,initial_state_hash=module_hash(module),
                    common_context_hash=module_hash(module.context),world_dim=dims[0],action_dim=dims[1],
                    parameters={n:dict(shape=list(p.shape),count=p.numel()) for n,p in module.named_parameters()},
                    parameter_count=sum(p.numel() for p in module.parameters()),
                    external_scorer_weights_loaded=False, own_pretrained_scorer_weights_loaded=False,
                    allowed_resume_lineage=f'{config.experiment_id}/{phase}/{kind}',
                    Base_hash=binding['Base']['checkpoint_hash'],pool_hash=binding['pool_hash'])
    module._iqe_initialization_identity = identity
    return module, identity


def forward(module, kind, batch):
    if kind == 'M_scene': return module(batch['context'])
    return module(batch['context'],batch['trajectories'],batch['base_role'],batch['available'])


def make_fetch(config, scenes, bank, cache, kind):
    lookup = {s.scene_id:(s,i) for i,s in enumerate(scenes)}
    def fetch(entries, device):
        rows = [lookup[e['scene_id']] for e in entries]
        indices = [i for _,i in rows]
        out = dict(context=cache.batch([s for s,_ in rows],device),
                   scores=torch.from_numpy(bank['scores'][indices]).to(device),
                   label_valid=torch.from_numpy(bank['valid'][indices]).to(device),
                   scene_valid=torch.tensor([not e.get('_padding',False) for e in entries],device=device))
        if kind == 'M_set':
            trajectory = torch.from_numpy(bank['trajectories'][indices]).to(device)
            role = torch.zeros(trajectory.shape[:2],dtype=torch.bool,device=device); role[:,0] = True
            out.update(trajectories=trajectory,base_role=role,available=torch.isfinite(trajectory).all((-1,-2)))
        return out
    return fetch


@torch.no_grad()
def predict(module, kind, scenes, bank, cache, device, batch_size=32):
    module.eval(); out=[]
    fetch = make_fetch(None,scenes,bank,cache,kind)
    for at in range(0,len(scenes),batch_size):
        entries = [{'scene_id':s.scene_id} for s in scenes[at:at+batch_size]]
        result = forward(module,kind,fetch(entries,device))
        out.append(result['decision_logits'].detach().float().cpu().numpy())
    return np.concatenate(out)


def selection_metrics(config, meta, data, logits):
    # Labels never enter inference. Finite trajectory availability is deployment-observable.
    available = np.isfinite(data['trajectories']).all((-1,-2))
    selected = choose(torch.as_tensor(logits),meta['expert_ids'],torch.from_numpy(available)).numpy()
    components = np.where(data['component_valid'],data['components'],np.nan)
    result = report_metrics(data['scores'],components,selected,meta['source_logs'],
                            seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
    result['selection_counts'] = {eid:int((selected==i).sum()) for i,eid in enumerate(meta['expert_ids'])}
    result['selection_key'] = [result['selected_01'],-sum(result['safety'][c]['new_failures'] for c in ('no_at_fault_collisions','drivable_area_compliance'))]
    result['output_semantics'] = 'decision logits, not predicted PDMS'
    return result


def take(meta, data, scenes, ids):
    wanted = set(ids); indices=[i for i,s in enumerate(scenes) if s.scene_id in wanted]
    require(len(indices)==len(wanted),'subset missing scenes')
    m=dict(meta);m['scene_ids']=[scenes[i].scene_id for i in indices];m['source_logs']=[scenes[i].source_log_id for i in indices]
    return m,{k:v[indices] for k,v in data.items()},[scenes[i] for i in indices]


def recipe(config, updates, *, validation=True):
    return dict(lr=config.lr,weight_decay=config.weight_decay,global_batch_size=config.global_batch,
                micro_batch_size=config.micro_batch,max_optimizer_steps=updates,warmup_steps=config.warmup,
                lr_min_ratio=1.,schedule='warmup_constant',grad_clip_norm=config.grad_clip,
                validation_steps=list(config.validation_steps) if validation else [],
                validation_every_steps=500,early_stop_patience=10**9,checkpoint_every_steps=config.checkpoint_interval)


def train_model(config, kind, phase='diagnostic', device='cuda:0', resume=None, stop_after=None):
    require(kind in KINDS and phase in ('diagnostic','refit','overfit'), 'bounded study model/phase')
    resource(config,'train-decision-'+phase,training=True);device=runtime(device)
    binding=verify_binding(config); split=read_split(config); folder=config.root/phase/kind
    with lock(folder/'RUN.lock'):
        meta,data,scenes=canonical_bank(config,'scorer_fit')
        require(meta['hash']==binding['banks']['scorer_fit']['hash'],'training bank changed')
        if phase=='refit':
            plan=read_json(config.root/'REFIT_PLAN.json')
            require(plan['hash']==digest({k:v for k,v in plan.items() if k!='hash'}) and plan['split_hash']==split['hash'],'refit plan changed')
            updates=plan['models'][kind]['updates']; ids=split['full_scene_ids']
        else:
            ids=split['fit']['scene_ids'];updates=config.diagnostic_updates
            if phase=='overfit':
                # Fixed hash non-tie support; selection uses only legal fit rewards.
                index={s.scene_id:i for i,s in enumerate(scenes)}
                ids=sorted([sid for sid in ids if np.ptp(data['scores'][index[sid]])>.01],key=lambda sid:digest([config.partition_seed,'overfit64',sid]))[:64]
                require(ids,'no non-tie fit scenes');updates=300
                atomic_json(folder/'OVERFIT_SCENES.json',dict(ids=ids,split_hash=split['hash'],rule='fixed hash among fit non-ties >.01'),immutable=True)
        fm,fb,fs=take(meta,data,scenes,ids)
        vm,vb,vs=take(meta,data,scenes,split['validation']['scene_ids'])
        cache=FrozenContexts(config,fs+(vs if phase=='diagnostic' else []))
        module,identity=create(config,kind,phase,cache.read(fs[0]));module.to(device)
        atomic_json(folder/'INITIALIZATION.json',identity,immutable=True)
        z=fixed_normalizer(torch.from_numpy(fb['scores']),torch.from_numpy(fb['valid']))
        require(z>0,'NO_DECISION_SIGNAL: no cost differences in this fit population')
        sampling_identity=digest([meta['hash'],split['hash'],ids])
        sampler=SceneSampler(ids,config.sampler_seed,sampling_identity)
        cfg=recipe(config,updates,validation=phase=='diagnostic')
        if phase=='overfit':cfg['checkpoint_every_steps']=100
        deps=dict(experiment=binding['hash'],inner_split=split['hash'],bank=meta['hash'],phase=phase,kind=kind,
                  sampler=sampler.identity,Z_train=z,initialization=identity['initial_state_hash'],
                  code={name:file_hash(Path(__file__).with_name(name)) for name in ('training.py','models.py','objectives.py','data.py')},
                  trainer=file_hash(Path(__file__).parents[1]/'training/trainer.py'))
        if phase=='refit':deps['refit_plan']=plan['hash']
        atomic_json(folder/'RUN_IDENTITY.json',dict(dependencies=deps,training=cfg,fit_scene_ids=ids,
                    training_seed=config.training_seed,sampler_seed=config.sampler_seed,init_seed=config.seed,
                    label_normalization='fixed full-fit mean sum_i(r_i-min_j r_j)',Z_train=z,
                    sampling='uniform_scene_with_replacement; no cost/pair weighting',no_VLM_parameters=True),immutable=True)
        fetch=make_fetch(config,fs,fb,cache,kind)
        def loss(m,b):
            output=forward(m,kind,b)
            return {'decision':cost_ce(output['decision_logits'],b['scores'],b['label_valid'],z,b['scene_valid'])}
        def validate(m,step):
            logits=predict(m,kind,vs,vb,cache,device)
            result=selection_metrics(config,vm,vb,logits)
            buf=io.BytesIO();np.savez(buf,decision_logits=logits)
            path=folder/f'predictions_inner_val_{step:06d}.npz';atomic_bytes(path,buf.getvalue(),immutable=True)
            result.update(predictions_hash=file_hash(path),model_hash=module_hash(m),role='D_router_val',
                          validation_scene_hash=digest(vm['scene_ids']),pool_hash=binding['pool_hash'])
            return result
        train(module,loss,fetch,sampler,cfg,deps,folder,device=device,resume=resume,stop_after=stop_after,
              validate=validate if phase=='diagnostic' else None,seed=config.training_seed,
              denominator_function=denominators)
        result=read_json(folder/'result.json')
        require(file_hash(result['checkpoint'])==read_json(result['checkpoint']+'.COMPLETE.json')['checksum'],'saved final checkpoint checksum')
        evidence=dict(result,kind=kind,phase=phase,Z_train=z,unique_fit_scenes=len(fs),unique_fit_logs=len({s.source_log_id for s in fs}),
                      consumed_scenes=sampler.consumed,VLM_calls=0,generator_updates=0,
                      source_initialization_hash=identity['initial_state_hash'],new_parameters=identity['parameter_count'],
                      external_scorer_weights_loaded=False,own_pretrained_scorer_weights_loaded=False)
        atomic_json(folder/'TRAINING_EVIDENCE.json',evidence)
        if result['status']=='COMPLETE':
            # Frozen endpoint fit reporting cannot select another training endpoint.
            best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);module.load_state_dict(best,strict=True)
            logits=predict(module,kind,fs,fb,cache,device)
            buf=io.BytesIO();np.savez(buf,decision_logits=logits)
            atomic_bytes(folder/'fit_predictions.npz',buf.getvalue(),immutable=True)
            report=selection_metrics(config,fm,fb,logits)
            report.update(endpoint='internal_best' if phase=='diagnostic' else 'fixed_last',
                          model_hash=module_hash(module),prediction_hash=file_hash(folder/'fit_predictions.npz'))
            atomic_json(folder/'FIT_REPORT.json',report,immutable=True)
        return evidence


def lock_refit_plan(config):
    split=read_split(config);binding=verify_binding(config);models={}
    for kind in KINDS:
        folder=config.root/'diagnostic'/kind;result=read_json(folder/'result.json')
        require(result['status']=='COMPLETE' and result['optimizer_steps']==config.diagnostic_updates,'complete diagnostic required')
        require(file_hash(folder/'best.pt')==result['best_checkpoint_hash'],'diagnostic selected checkpoint corrupted')
        validations=[(step,read_json(folder/f'validation_{step:06d}.json')) for step in config.validation_steps]
        step,stats=max(validations,key=lambda row:(row[1]['selection_key'],-row[0]))
        require(stats['selection_key']==result['best_selection_key'],'best validation state mismatch')
        updates=math.ceil(step*len(split['full_scene_ids'])/split['fit']['scenes'])
        require(updates<=config.refit_limit,'refit budget exceeded')
        models[kind]=dict(inner_endpoint=step,inner_metrics=stats,updates=updates,
                          diagnostic_checkpoint_hash=result['best_checkpoint_hash'],
                          initialization_hash=read_json(folder/'INITIALIZATION.json')['initial_state_hash'],
                          starts_from='original_random_initialization_NOT_diagnostic_weights')
        atomic_json(config.root/(kind.upper()+'_DIAGNOSTIC_RESULTS.json'),dict(validations=validations,selected_step=step,
                    fit=read_json(folder/'FIT_REPORT.json'),R00_internal_val='in-sample_background_only'),immutable=True)
    r=dict(schema='locked_full_DS_refit_v1',binding_hash=binding['hash'],split_hash=split['hash'],models=models,
           selection_role='internal_D_router_val_only',source_stage_val_or_dev_used=False,
           formula='ceil(T_inner*N_full/N_inner)',init_seed=config.seed,training_seed=config.training_seed,
           sampler_seed=config.sampler_seed,full_DS_scenes=len(split['full_scene_ids']),
           terminal_checkpoint_selection=False,external_evaluation='same fixed last endpoint, no reselection')
    r['hash']=digest(r);atomic_json(config.root/'REFIT_PLAN.json',r,immutable=True)
    return r
