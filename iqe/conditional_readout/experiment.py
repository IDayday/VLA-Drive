"""One new readout experiment; reuse IQE native training, banks and samplers."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import io
import subprocess
import numpy as np
import torch
from iqe.contracts import require, TrajectoryContract
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, atomic_bytes, digest, file_hash, read_json, lock
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.metrics import metrics, select
from iqe.specialist_study.objectives import StudySampler, terms, denominators
from iqe.specialist_study.scorer_training import batch_fetch, fetch_context
from iqe.specialist_study.training import preload, runtime
from iqe.training.trainer import train
from .model import ConditionalReadoutScorer


def recipe(config):
    source = config.source()
    receipt = read_json(source.root/'resolved_recipes/scorer_2x2.json')
    require(file_hash(receipt['source']) == receipt['checksum'], 'reference recipe changed')
    value = receipt['resolved']
    require(value['max_optimizer_steps'] == config.updates == 6000 and
            value['global_batch_size'] == 128 and value['phase_switch_step'] == 3000 and
            value['dropout'] == 0. and value['validation_steps'] == [3000,4000,5000,6000],
            'not the registered original S11 control')
    return value


def bind(config):
    source=config.source();r=recipe(config);roles=scenes_by_role(source)
    fit=roles['scorer_fit']
    require(all(s.split_role=='incremental_fit' for s in fit), 'D_S role violation')
    logs={s.source_log_id for s in fit}
    for role in ('expert_fit','stage_val','selector_cal','dev_report'):
        require(not logs & {s.source_log_id for s in roles[role]}, 'source log leakage')
    artifacts={name:file_hash(source.root/name) for name in
        ('BASE_BINDING.json','OFFLINE_BASE_CONTRACT.json','DATA_PARTITION.json','scenes.json',
         'SERVING_CANDIDATE_POOL.json','SCORE_PROTOCOL.json','scorers/S11/SCORER_INITIALIZATION.json',
         'scorers/S11/result.json','resolved_recipes/scorer_2x2.json')}
    for kind,role in (('deploy','scorer_fit'),('rich','scorer_fit'),('deploy','stage_val')):
        meta,_=load_bank(source,kind,role)
        require(meta['scene_ids']==[s.scene_id for s in roles[role]], 'bank order mismatch')
        artifacts[kind+'/'+role]=meta['hash']
    identity=dict(schema='conditional_visual_readout_followup_v1',config_hash=config.identity,
        experiment_id=config.experiment_id,source_campaign=str(source.root),artifacts=artifacts,
        control='completed S11; reused as a control, no trained weights loaded into new model',
        formal_new_scorers=1,formal_optimizer_update_cap=6000,diagnostics=2,diagnostic_update_cap_each=300,
        unchanged=['Base','expert_pool','D_S','inference_precision','training_seed','sample_streams','recipe','parameter_shapes'],
        changed='each private visual query additionally reads interpolated candidate time-token embeddings',
        prior_dev_results_informed_research_direction=True,dev_used_for_this_checkpoint_selection=False,
        final_test_access=False,external_scorer_weights_loaded=False,ACTIVE_updated=False,
        authorization='active user goal: research and practice a robust internally trained Scorer; old matrices remain closed')
    with lock(config.root/'BIND.lock'):
        path=config.root/'EXPERIMENT_BINDING.json'
        if path.exists():
            old=read_json(path);require(old['identity']==identity,'experiment dependencies changed');return old
        result=dict(identity=identity,registered_utc=datetime.now(timezone.utc).isoformat(),
                    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        atomic_json(path,result,immutable=True);return result


def create(config, conditioned=True):
    source=config.source();scene=scenes_by_role(source)['scorer_fit'][0]
    x=ReusedContext(source,1).read(scene);v=VerifiedSpatialReader(source,1).read(scene)
    contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json')
    with initialization_rng(source.seed):
        model=ConditionalReadoutScorer(x['world'].shape[-1],x['action'].shape[-1],
            TrajectoryContract(**contract['trajectory']),visual_dim=v['visual'].shape[-1],
            views=len(contract['input']['views']),conditioned=conditioned)
    control=read_json(source.root/'scorers/S11/SCORER_INITIALIZATION.json')
    require(module_hash(model)==control['initial_state_hash'], 'random initialization does not match S11 control')
    identity=dict(experiment_id=config.experiment_id,conditioned=conditioned,init_seed=source.seed,
        initial_state_hash=module_hash(model),parameter_count=sum(p.numel() for p in model.parameters()),
        parameters={n:dict(shape=list(p.shape),count=p.numel()) for n,p in model.named_parameters()},
        external_scorer_weights_loaded=False,loaded_scorer_weights=[],
        allowed_resume_lineage=config.experiment_id,control_initialization_checksum=file_hash(source.root/'scorers/S11/SCORER_INITIALIZATION.json'),
        Base_binding_hash=file_hash(source.root/'BASE_BINDING.json'),pool_hash=pool(source)['hash'])
    model._iqe_initialization_identity=identity
    return model,identity


@torch.no_grad()
def predict(config,model,scenes,bank,device,*,cache=None,spatial=None,wrong_visual=False):
    source=config.source();cache=cache or ReusedContext(source,128)
    spatial=spatial or VerifiedSpatialReader(source,64)
    model.eval();rows=[]
    for start in range(0,len(scenes),16):
        batch=scenes[start:start+16];x=fetch_context(source,batch,device,cache,spatial)
        if wrong_visual:
            shifted=[scenes[(i+1)%len(scenes)] for i in range(start,start+len(batch))]
            x.update(spatial.batch(shifted,device))
        t=torch.from_numpy(bank['trajectories'][start:start+len(batch)]).to(device)
        p=model(x,t)['values'];require(torch.isfinite(p).all(),'nonfinite scorer')
        rows.append(p.cpu().numpy())
    return np.concatenate(rows)


def train_cell(config,device='cuda:0',resume=None,stop_after=None,diagnostic=None):
    require(diagnostic in (None,'global','conditioned'),'unknown diagnostic')
    bound=bind(config);source=config.source();r=recipe(config);device=runtime(device)
    folder=config.root/('diagnostics/'+diagnostic if diagnostic else 'scorer')
    with lock(folder/'RUN.lock'):
        if diagnostic is None:
            for kind in ('global','conditioned'):
                check=read_json(config.root/'diagnostics'/kind/'DIAGNOSTIC.json')
                require(check['status']=='COMPLETE' and check['optimizer_steps']==300,
                        'complete the bounded real diagnostics before the new formal cell')
            decision=read_json(config.root/'TRAINING_DECISION.json')
            require(file_hash(config.root/'DIAGNOSTIC_COMPARISON.json')==decision['evidence_hash'],
                    'diagnostic decision evidence changed')
            require(decision['proceed'], 'registered follow-up stopped: '+decision['reason'])
        dm,deploy=load_bank(source,'deploy','scorer_fit');rm,rich=load_bank(source,'rich','scorer_fit')
        vm,vb=load_bank(source,'deploy','stage_val');roles=scenes_by_role(source);scenes=roles['scorer_fit']
        require(dm['scene_ids']==rm['scene_ids']==[s.scene_id for s in scenes], 'fit bank join')
        if diagnostic:
            valid=rich['comparison_valid'];scores=rich['scores']
            non_tie=np.where(valid,scores,-np.inf).max(1)-np.where(valid,scores,np.inf).min(1)>.01
            ix=sorted(np.flatnonzero(non_tie),key=lambda i:digest([source.seed,scenes[int(i)].scene_id]))[:64]
            require(len(ix)==64,'64 real non-tie D_S scenes required')
            scenes=[scenes[int(i)] for i in ix]
            deploy={k:v[ix] for k,v in deploy.items()};rich={k:v[ix] for k,v in rich.items()}
            atomic_json(config.root/'DIAGNOSTIC_DATA.json',dict(scene_ids=[s.scene_id for s in scenes],
                rich_bank_hash=rm['hash'],rule='fixed seeded-hash non-tie D_S scenes',
                final_test_or_dev_access=False,used_for_formal_model_selection=False),immutable=True)
        cache=preload(replace(source,feature_memory_gib=config.feature_memory_gib),scenes)
        spatial=VerifiedSpatialReader(source,64)
        model,initial=create(config,conditioned=diagnostic!='global')
        atomic_json(folder/'SCORER_INITIALIZATION.json',initial,immutable=True)
        ids=[s.scene_id for s in scenes]
        bank_identity=digest([dm['hash'],rm['hash'],ids])
        sampler=StudySampler(ids,deploy['scores'],deploy['comparison_valid'],seed=source.seed+10,
                             bank_hash=bank_identity,phase_steps=3000)
        fetch=batch_fetch(source,scenes,cache,spatial,deploy,rich,True)
        keys=('lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps',
              'lr_min_ratio','grad_clip_norm','schedule','validation_steps','checkpoint_every_steps')
        cfg={k:r[k] for k in keys}|dict(micro_batch_size=config.micro_batch,validation_every_steps=1000,early_stop_patience=10**9)
        if diagnostic:cfg.update(max_optimizer_steps=300,validation_steps=[],checkpoint_every_steps=100)
        weights={'value':1.,'components':.2,'ranking':.2}
        dependencies=dict(experiment=digest(bound),diagnostic=diagnostic,deploy_bank=dm['hash'],rich_bank=rm['hash'],
            validation=vm['hash'],sampler=sampler.identity,init=initial['initial_state_hash'],
            source={name:file_hash(Path(__file__).with_name(name)) for name in ('experiment.py','model.py','config.py')},
            training_rng_seed=source.seed+20,weights=weights)
        identity_path=folder/'RUN_IDENTITY.json'
        run_identity=dict(dependencies=dependencies,training=cfg)
        if identity_path.exists():
            saved=read_json(identity_path)
            require(all(saved[key]==value for key,value in run_identity.items()), 'run dependencies changed')
        else:
            atomic_json(identity_path,run_identity|dict(
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),immutable=True)
        atomic_json(folder/'COMPARISON_DISTRIBUTION.json',dict(identity=sampler.identity,
            pairs=sampler.pairs.tolist(),weights=sampler.weights.tolist(),
            q_hash=digest(sampler.q.tolist()) if sampler.q is not None else None,
            gap_applied_once=True),immutable=True)
        val_cache=ReusedContext(source,128);val_spatial=VerifiedSpatialReader(source,64)
        def loss(m,b):return terms(m(b['context'],b['trajectories']),b)
        def validation(m,step):
            p=predict(config,m,roles['stage_val'],vb,device,cache=val_cache,spatial=val_spatial)
            selected=select(p,vb['valid'],expert_ids=vm['provenance']['sources'])
            result=metrics(vb['scores'],vb['components'],selected,vm['source_logs'],
                           seed=source.bootstrap_seed,repetitions=source.bootstrap_repetitions)
            result['selection_key']=[result['selected_01'],-sum(result['safety'][k]['new_failures'] for k in
                ('no_at_fault_collisions','drivable_area_compliance'))]
            result.update(scorer_state_hash=module_hash(m),bank_hash=vm['hash'],pool_hash=pool(source)['hash'])
            data=io.BytesIO();np.savez(data,values=p)
            path=folder/f'predictions_stage_val_{step:06d}.npz'
            atomic_bytes(path,data.getvalue(),immutable=True);result['prediction_hash']=file_hash(path)
            return result
        train(model,loss,fetch,sampler,cfg,dependencies,folder,device=device,weights=weights,
              resume=resume,stop_after=stop_after,seed=source.seed+20,
              validate=None if diagnostic else validation,denominator_function=denominators)
        result=read_json(folder/'result.json')
        if diagnostic and result['status']=='COMPLETE':
            p=predict(config,model,scenes,deploy,device,cache=cache,spatial=spatial)
            wrong=predict(config,model,scenes,deploy,device,cache=cache,spatial=spatial,wrong_visual=True)
            def summary(values):
                choice=select(values,deploy['valid'],expert_ids=dm['provenance']['sources'])
                return dict(selected_01=float(deploy['scores'][np.arange(len(scenes)),choice].mean()),
                    base_01=float(deploy['scores'][:,0].mean()),oracle_01=float(deploy['scores'].max(1).mean()),
                    mse=float(((values-deploy['scores'])**2)[deploy['valid']].mean()))
            atomic_json(folder/'DIAGNOSTIC.json',dict(status='COMPLETE',optimizer_steps=300,scenes=64,
                correct=summary(p),wrong_visual=summary(wrong),
                changed_selections=int((p.argmax(1)!=wrong.argmax(1)).sum()),
                maximum_prediction_change=float(np.abs(p-wrong).max()),
                scientific_generalization_result=False,weights_eligible_for_deployment=False,
                final_state_hash=module_hash(model),parameters_changed=result['parameters_changed']),immutable=True)
        return result
