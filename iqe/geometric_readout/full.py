"""One bounded, from-scratch geometric Scorer trial on the locked S11 pool.

Reuses IQE's actual bank, loss, two-flow sampler and optimizer-boundary trainer.
No expert/Base updates, external scoring weights, navtest, or ACTIVE mutation.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import io
import json
import os
import socket
import subprocess
import numpy as np
import torch
import torch.distributed as dist
import yaml
from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_json, atomic_bytes, digest, file_hash, lock, read_json
from iqe.losses import collective_error
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.config import StudyConfig
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.scorer_training import batch_fetch, fetch_context
from iqe.specialist_study.training import preload, runtime
from iqe.specialist_study.objectives import StudySampler, terms, denominators
from iqe.specialist_study.metrics import metrics, select
from iqe.training.trainer import train, init_distributed
from .calibration_bank import PreparedCalibrationReader
from .model import GeometricReadoutScorer


@dataclass(frozen=True)
class Config:
    experiment_id: str
    source_campaign: str
    output_root: str
    calibration_bank: str
    geometry_audit: str
    diagnostic_root: str

    def __post_init__(self):
        require(self.experiment_id and all(Path(x).is_absolute() for x in
            (self.source_campaign,self.output_root,self.calibration_bank,self.geometry_audit,self.diagnostic_root)),
            'explicit absolute source/output identities required')
        require(Path(self.output_root).resolve() not in
            (Path(self.source_campaign).resolve(),Path(self.diagnostic_root).resolve()), 'isolated full experiment')

    @property
    def root(self): return Path(self.output_root)

    def source(self): return StudyConfig(**read_json(Path(self.source_campaign)/'resolved_config.json'))


def load_config(path):
    return Config(**yaml.safe_load(Path(path).read_text()))


def training_config(source):
    # Read the parent's actual resolved recipe. The generic recipe loader uses
    # its own worktree path and would attempt to rewrite the historical receipt.
    receipt=read_json(source.root/'resolved_recipes/scorer_2x2.json')
    require(file_hash(receipt['source'])==receipt['checksum'],'parent resolved recipe changed')
    recipe=receipt['resolved']
    require(recipe['max_optimizer_steps']==6000 and recipe['global_batch_size']==128
            and recipe['validation_steps']==[3000,4000,5000,6000], 'fixed matched S11 budget')
    keys=('lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps',
          'lr_min_ratio','grad_clip_norm','schedule','validation_steps','checkpoint_every_steps')
    return {k:recipe[k] for k in keys}|dict(micro_batch_size=source.micro_batch,
        validation_every_steps=1000,early_stop_patience=10**9)


def resource_check(config, training=False):
    from iqe.specialist_study.resource_guard import check
    policy=read_json('/mnt/project/server_dispatch_policy.json')
    permanent=set(policy['allowed_hosts'])
    for row in policy['allowed_hosts'].values(): permanent.update(row.get('aliases',[]))
    require(socket.gethostname() in permanent, 'follow-up runs on permanent authorized hosts only')
    check(config, 'train-scorer-cell' if training else 'research-preflight')


def binding(config):
    source=config.source(); roles=scenes_by_role(source)
    require(all(s.split_role=='incremental_fit' for s in roles['scorer_fit']), 'D_S fit role only')
    fit_logs={s.source_log_id for s in roles['scorer_fit']}
    for role in ('expert_fit','stage_val','selector_cal','dev_report'):
        require(not fit_logs & {s.source_log_id for s in roles[role]}, 'Scorer fit log leakage')
    files=('BASE_BINDING.json','CAMPAIGN_BINDING.json','DATA_PARTITION.json','scenes.json',
           'OFFLINE_BASE_CONTRACT.json','SERVING_CANDIDATE_POOL.json','SCORE_PROTOCOL.json',
           'resolved_config.json','resolved_recipes/scorer_2x2.json')
    artifacts={name:file_hash(source.root/name) for name in files}
    # Bind evaluation data contents without consuming development labels here.
    for role in ('scorer_fit','stage_val','selector_cal','dev_report'):
        meta=read_json(source.root/f'banks/deploy/{role}/BANK.json')
        require(meta['scene_ids']==[s.scene_id for s in roles[role]], 'fixed role/bank scene order')
        artifacts[f'deploy/{role}']=meta['hash']
    rm,_=load_bank(source,'rich','scorer_fit')
    artifacts['rich/scorer_fit']=rm['hash']
    audit=read_json(Path(config.geometry_audit)/'REPORT.json')
    require(audit['all_calibrations_match'] and audit['all_lidar2ego_identity'] and
            audit['official_projection_max_abs']<1e-8, 'actual geometry audit required')
    registration=read_json(Path(config.geometry_audit)/'REGISTRATION.json')
    require(registration['base_binding_hash']==artifacts['BASE_BINDING.json'], 'geometry Base mismatch')
    diagnostic=read_json(Path(config.diagnostic_root)/'DIAGNOSTIC.json')
    require(diagnostic['status']=='COMPLETE' and diagnostic['optimizer_steps']==300 and
            diagnostic['parameters_changed'], 'completed real bounded diagnostic required')
    bank=PreparedCalibrationReader(source,config.calibration_bank)
    del bank
    recipe=read_json(source.root/'resolved_recipes/scorer_2x2.json')
    require(file_hash(recipe['source'])==recipe['checksum'], 'inherited recipe source changed')
    require(recipe['resolved']['phase_switch_step']==3000 and
            recipe['resolved']['natural_phase2']==recipe['resolved']['comparison_phase2']==64,
            'matched original two-stream exposure')
    code={name:file_hash(Path(__file__).with_name(name)) for name in
          ('full.py','model.py','geometry.py','calibration_bank.py','calibration.py')}
    for name in ('iqe/specialist_study/objectives.py','iqe/training/trainer.py',
                 'iqe/specialist_study/context_reader.py','iqe/internal_baseline/scorer.py'):
        code[name]=file_hash(Path(__file__).parents[2]/name)
    identity=dict(schema='geometric_scorer_full_v1',configuration=asdict(config),artifacts=artifacts,
        calibration_hash=file_hash(Path(config.calibration_bank)/'MANIFEST.json'),
        geometry_audit_hash=file_hash(Path(config.geometry_audit)/'REPORT.json'),
        diagnostic_hash=file_hash(Path(config.diagnostic_root)/'DIAGNOSTIC.json'),code=code,
        max_new_scorer_runs=1,max_optimizer_steps=6000,new_experts=0,new_Base_updates=0,
        training=training_config(source),recipe=recipe['resolved'],
        candidate_pool=pool(source)['hash'],external_scorer_weights_loaded=False,
        changed='candidate-local geometric visual readout only; original S11 objective and scene streams',
        input_geometry='locked NAVSIM pinhole, z=0/1m hypotheses, 3x3 neighborhoods',
        precision='unchanged FP32 Scorer and inherited frozen context runtime',
        distributed_semantics='same global128 denominators; floating reduction order may differ by world size',
        prior_dev_results_seen=True,exploratory_followup=True,new_navtest=False,ACTIVE_updated=False,
        endpoint_selection='stage_val at3000/4000/5000/6000; max argmax PDMS, fewer new NC+DAC, earlier step',
        stop_rule='fixed6000 unless numerical/data/resource failure; no automatic extension')
    with lock(config.root/'BIND.lock'):
        atomic_json(config.root/'resolved_config.json',asdict(config),immutable=True)
        path=config.root/'EXPERIMENT_BINDING.json'
        if path.exists():
            old=read_json(path);require(old['identity']==identity,'immutable full trial/source changed');return old
        result=dict(identity=identity,registered_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        atomic_json(path,result,immutable=True);return result


def create(config):
    source=config.source(); scene=scenes_by_role(source)['scorer_fit'][0]
    x=ReusedContext(source,1).read(scene);v=VerifiedSpatialReader(source,1).read(scene)
    contract=TrajectoryContract(**read_json(source.root/'OFFLINE_BASE_CONTRACT.json')['trajectory'])
    with initialization_rng(source.seed):
        model=GeometricReadoutScorer(x['world'].shape[-1],x['action'].shape[-1],contract,
                                    visual_dim=v['visual'].shape[-1])
    reference=read_json(source.root/'scorers/S11/SCORER_INITIALIZATION.json')
    require(module_hash(model)==reference['initial_state_hash'], 'common scratch initialization differs from S11')
    identity=dict(experiment_id=config.experiment_id,module_init_seed=source.seed,
        initial_state_hash=module_hash(model),parameter_count=sum(p.numel() for p in model.parameters()),
        parameters={n:list(p.shape) for n,p in model.named_parameters()},
        external_scorer_weights_loaded=False,loaded_scorer_weights=[],
        diagnostic_weights_loaded=False,allowed_resume_lineage=str(config.root/'scorer'),
        Base_binding_hash=file_hash(source.root/'BASE_BINDING.json'),pool_hash=pool(source)['hash'])
    model._iqe_initialization_identity=identity
    return model,identity


@contextmanager
def collective_run_lock(path,rank):
    handle=None;error=None
    if rank==0:
        try:
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            handle=open(path,'a+b');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except Exception as exc:error=repr(exc)
    try:
        collective_error(error)
        yield
    finally:
        if handle is not None:handle.close()


def together(fn):
    result=None;error=None
    try:result=fn()
    except Exception as exc:error=repr(exc)
    collective_error(error)
    return result


def calibrated_fetch(ordinary,scenes,reader):
    lookup={s.scene_id:s for s in scenes}
    def fetch(entries,device):
        batch=ordinary(entries,device)
        batch['context'].update(reader.batch([lookup[e['scene_id']] for e in entries],device))
        return batch
    return fetch


@torch.no_grad()
def predict(config,model,role,device,*,cache=None,spatial=None,calibration=None,diagnostic=None):
    require(role in ('scorer_fit','stage_val','selector_cal','dev_report'),'test access forbidden')
    source=config.source();meta,bank=load_bank(source,'deploy',role);scenes=scenes_by_role(source)[role]
    require(meta['scene_ids']==[s.scene_id for s in scenes], 'prediction scene join')
    cache=cache or ReusedContext(source,128)
    spatial=spatial or VerifiedSpatialReader(source,64)
    calibration=calibration or PreparedCalibrationReader(source,config.calibration_bank)
    model.eval();pred=[]
    for at in range(0,len(scenes),16):
        sub=scenes[at:at+16];context=fetch_context(source,sub,device,cache,spatial)
        context.update(calibration.batch(sub,device))
        trajectory=torch.from_numpy(bank['trajectories'][at:at+len(sub)]).to(device)
        if diagnostic=='wrong_visual':
            wrong=[scenes[(i+1)%len(scenes)] for i in range(at,at+len(sub))]
            context.update(spatial.batch(wrong,device))
        elif diagnostic=='mask_visual':context['visual']=torch.zeros_like(context['visual'])
        elif diagnostic is not None:raise ValueError('unsupported geometric information diagnostic')
        p=model(context,trajectory)['values']
        require(torch.isfinite(p).all().item(),'nonfinite valid inference')
        pred.append(p.cpu().numpy())
    return meta,bank,np.concatenate(pred)


def train_full(config,device='cuda:0',resume=None,stop_after=None):
    require(stop_after is None or 0<stop_after<=6000,'bounded optimizer stop')
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        device='cuda:'+os.environ['LOCAL_RANK'] if device.startswith('cuda') else device
    device=runtime(device);rank,world=init_distributed(device)
    resource_check(config,True)
    folder=config.root/'scorer'
    if resume not in (None,'auto'):
        require(Path(resume).resolve().parent==folder.resolve(),'same experimental Scorer resume only')
    with collective_run_lock(folder/'RUN.lock',rank):
        bound=together(lambda:binding(config));source=config.source();roles=scenes_by_role(source)
        meta,deploy=together(lambda:load_bank(source,'deploy','scorer_fit'))
        rm,rich=together(lambda:load_bank(source,'rich','scorer_fit'))
        require(meta['scene_ids']==rm['scene_ids']==[s.scene_id for s in roles['scorer_fit']], 'stream scene identity')
        cache=together(lambda:preload(source,roles['scorer_fit']))
        spatial=together(lambda:VerifiedSpatialReader(source,64))
        calibration=together(lambda:PreparedCalibrationReader(source,config.calibration_bank))
        model,initial=together(lambda:create(config))
        sampler=StudySampler(meta['scene_ids'],deploy['scores'],deploy['comparison_valid'],seed=source.seed+10,
            bank_hash=digest([meta['hash'],rm['hash'],meta['scene_ids']]),phase_steps=3000)
        cfg=training_config(source)
        deps=dict(experiment=digest(bound),sampler=sampler.identity,initial=initial['initial_state_hash'],
                  training_rng_seed=source.seed+20,calibration=bound['identity']['calibration_hash'])
        def receipts():
            if rank==0:
                atomic_json(folder/'SCORER_INITIALIZATION.json',initial,immutable=True)
                atomic_json(folder/'RUN_IDENTITY.json',dict(dependencies=deps,training=cfg),immutable=True)
                atomic_json(folder/'COMPARISON_DISTRIBUTION.json',dict(sampler_identity=sampler.identity,
                    pairs=sampler.pairs.tolist(),weights=sampler.weights.tolist(),
                    q_hash=digest(sampler.q.tolist()) if sampler.q is not None else None,
                    gap_weight_applied_once=True),immutable=True)
        together(receipts)
        fetch=calibrated_fetch(batch_fetch(source,roles['scorer_fit'],cache,spatial,deploy,rich,True),
                               roles['scorer_fit'],calibration)
        val_cache=ReusedContext(source,1024)
        def validate(m,step):
            vm,vb,p= predict(config,m,'stage_val',device,cache=val_cache,spatial=spatial,calibration=calibration)
            selected=select(p,vb['valid'],expert_ids=vm['provenance']['sources'])
            result=metrics(vb['scores'],vb['components'],selected,vm['source_logs'],
                           seed=source.bootstrap_seed,repetitions=source.bootstrap_repetitions)
            result['selection_key']=[result['selected_01'],-sum(result['safety'][c]['new_failures'] for c in
                                      ('no_at_fault_collisions','drivable_area_compliance'))]
            result.update(scorer_state_hash=module_hash(m),bank_hash=vm['hash'],pool_hash=pool(source)['hash'])
            buf=io.BytesIO();np.savez(buf,values=p)
            path=folder/f'predictions_stage_val_{step:06d}.npz';atomic_bytes(path,buf.getvalue(),immutable=True)
            result['prediction_hash']=file_hash(path);return result
        train(model,lambda m,b:terms(m(b['context'],b['trajectories']),b),fetch,sampler,cfg,deps,folder,
            device=device,resume=resume,stop_after=stop_after,validate=validate,seed=source.seed+20,
            weights={'value':1.,'components':.2,'ranking':.2},denominator_function=denominators)
        if rank==0:
            result=read_json(folder/'result.json')
            result.update(natural_consumed=sampler.natural_consumed,comparison_consumed=sampler.comparison_consumed,
                training_phase=1 if sampler.step<3000 else 2,world_size=world,
                spatial_reader_statistics=spatial.statistics,external_scorer_weights_loaded=False,
                VLM_calls=0,Base_updates=0,expert_updates=0)
            atomic_json(folder/'SCORER_TRAINING_REPORT.json',result)
        if dist.is_initialized():dist.barrier()
        return read_json(folder/'result.json')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('command',choices=('bind','train'))
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',nargs='?',const='auto');p.add_argument('--stop-after',type=int)
    a=p.parse_args();config=load_config(a.config)
    if a.command=='bind':
        require(a.resume is None and a.stop_after is None,'resume/stop only apply to training')
        resource_check(config);result=binding(config)
        result=dict(status='REGISTERED',identity_hash=digest(result),updates=6000,new_Scorers=1)
    else:result=train_full(config,a.device,a.resume,a.stop_after)
    if not dist.is_initialized() or dist.get_rank()==0:print(json.dumps(result,allow_nan=False))
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
