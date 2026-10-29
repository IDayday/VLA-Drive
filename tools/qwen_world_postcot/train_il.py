"""Full clean-x imitation + representation + action-metric supervision, ZeRO2.

Formal budget includes1000 interface warmup updates and keeps all five losses.
Qualification/profile changes only execution limits, never the100k LR horizon.
"""
import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
import yaml
from starVLA.rl.qwen_world_postcot.contracts import ROOT,require_dispatch,boundary_stop_due,file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.batching import BatchBuilder,to_device,update_denominators
from starVLA.rl.qwen_world_postcot import checkpoint as cp
from starVLA.rl.qwen_world_postcot.bank_versions import BankVersions,validate_refresh_config
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.flow_grpo.reproducibility import configure_numerics

QWEN='/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct'
QWEN_IDENTITY=Path('/mnt/project/qwen-sourceflow-native-20261018/assets/qwen_identity.json')
STOP=False
def stop(*_):
    global STOP
    STOP=True

def validate(cfg):
    keys={'schema','seed','uses_metric_supervision','uses_rl','global_batch','micro_batch','gradient_accumulation','FM_repeat',
        'max_updates','interface_updates','aux_warmup_updates','save_every','dev_every','rolling_checkpoints','optimizer',
        'learning_rates','scheduler','precision','distributed','model','flow','loss_weights','teacher_cache','candidate_bank','candidate_refresh','training_data'}
    if set(cfg)!=keys:raise ValueError('Unknown/missing IL recipe fields: '+repr(sorted(set(cfg)^keys)))
    nested=dict(optimizer={'name','betas','eps','weight_decay','gradient_clip','fused'},learning_rates={'qwen_language_merger','qwen_vision','action_memory_aux_heads'},
        scheduler={'type','warmup_fraction'},distributed={'backend','offload','overlap_comm','reduce_bucket_size','allgather_bucket_size','qwen_activation_checkpointing'},
        flow={'prediction_type','training_loss','sigma_formula','inference_steps','shift'},model={'memory','predictive','action','dino_readout','future_readout','object_readout','post_head'})
    for name,fields in nested.items():
        if set(cfg[name])!=fields:raise ValueError('Unknown/missing '+name+' recipe fields')
    validate_refresh_config(cfg['candidate_refresh'])
    if cfg['schema']!='qwen_world_postcot_full_IL_v1' or cfg['seed']!=42 or cfg['FM_repeat']!=1:raise ValueError('Fixed full-framework seed42/FMrepeat1 recipe changed')
    if cfg['micro_batch'] not in (2,4) or cfg['gradient_accumulation']!=4//cfg['micro_batch']:raise ValueError('Main topology is micro4/accum1 or micro2/accum2 on8GPU')
    if cfg['scheduler']['type']!='cosine' or cfg['optimizer']['name']!='AdamW':raise ValueError('Only the wired cosine/AdamW recipe is supported')
    if cfg['scheduler']['warmup_fraction']!=.03:raise ValueError('Main100k recipe uses the fixed3percent warmup')
    if cfg['flow']['training_loss']!='valid_mean_L1_FP32' or cfg['flow']['sigma_formula']!='5*u/(1+4*u), u~Uniform(0,1)':raise ValueError('Unsupported clean-x loss or sigma formula')
    if cfg['dev_every']!=5000 or cfg['save_every']!=1000:raise ValueError('Fixed DEV/recovery cadence changed')
    if cfg['global_batch']!=32 or cfg['max_updates']!=100000 or cfg['interface_updates']!=1000:raise ValueError('Formal batch/budget/stage contract changed')
    if cfg['uses_metric_supervision'] is not True or cfg['uses_rl'] is not False:raise ValueError('Mislabelled metric-supervised imitation stage')
    if cfg['precision']!={'forward':'bfloat16','parameters':'float32','optimizer_master_states':'float32','action_state_sigma_losses':'float32'}:raise ValueError('Unsupported precision')
    if cfg['distributed']['backend']!='zero2' or cfg['distributed']['offload']:raise ValueError('Only qualified GPU ZeRO2 supported')
    if cfg['flow']['prediction_type']!='x0' or cfg['flow']['inference_steps']!=10 or cfg['flow']['shift']!=5:raise ValueError('Clean-x K10shift5 is required')
    if set(cfg['loss_weights'])!={'x','dino','future','object','post'}:raise ValueError('All five losses must be configured')
    if any(not math.isfinite(v) or v<=0 for v in list(cfg['loss_weights'].values())+list(cfg['learning_rates'].values())):raise ValueError('All full-framework losses and trainable groups require finite positive weights/LRs')
    if cfg['aux_warmup_updates']<1 or cfg['rolling_checkpoints']<1:raise ValueError('Positive auxiliary warmup/recovery retention required')
    expected_memory=dict(input_dim=2048,hidden_dim=1024,num_layers=2,num_heads=16,selected_layers=[8,16,28],
        camera_count=3,spatial_grid=[6,8],global_tokens=16,ego_dim=8,use_merged_visual=True)
    expected_predictive=dict(input_dim=1024,hidden_dim=512,num_layers=2,num_heads=8,intervals=4,tokens_per_interval=8)
    expected_action=dict(hidden_dim=1024,num_layers=20,num_heads=16,ffn_dim=4096,action_tokens=8,action_dim=3,ego_dim=8,dropout=0.)
    action=dict(cfg['model']['action']);checkpointing=action.pop('activation_checkpointing',None)
    if (cfg['model']['memory']!=expected_memory or cfg['model']['predictive']!=expected_predictive
            or action!=expected_action or type(checkpointing) is not bool):
        raise ValueError('Main recipe requires the fixed20layer467M actor and144/16/32 selected8/16/28 memory')
    return cfg

def groups(model,cfg):
    order=('qwen_language_merger','qwen_vision','action_memory_aux_heads')
    buckets={k:[] for k in order}
    for name,p in model.named_parameters():
        key='action_memory_aux_heads'
        if name.startswith('qwen.'):
            key='qwen_vision' if name.startswith('qwen.model.visual.') and 'merger' not in name else 'qwen_language_merger'
        buckets[key].append(p)
    return [dict(params=ps,lr=cfg['learning_rates'][name],name=name) for name,ps in buckets.items()]

def optimizer_group_contract(model,parameter_groups):
    names={id(parameter):name for name,parameter in model.named_parameters()}
    return [dict(name=group['name'],parameters=[dict(name=names[id(parameter)],shape=list(parameter.shape),
        dtype=str(parameter.dtype)) for parameter in group['params']]) for group in parameter_groups]

def learning_scale(step,cfg):
    warm=int(cfg['max_updates']*cfg['scheduler']['warmup_fraction'])
    return (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/(cfg['max_updates']-warm))))

def learning_scales(parameter_groups,cfg,*,qualification_joint=False,interface_updates=None):
    # ZeRO partitions flatten even unused Qwen parameters. A zero gradient in
    # flattened AdamW is not a freeze: decoupled decay can still change weights.
    # Keep the full optimizer from update0, but use exactly zero Qwen LR for the
    # interface stage. Joint updates continue the original100k schedule.
    result=[];interface_updates=cfg['interface_updates'] if interface_updates is None else interface_updates
    for group in parameter_groups:
        qwen=group['name'].startswith('qwen_')
        result.append(lambda step,qwen=qwen: 0. if qwen and not qualification_joint and step<interface_updates else learning_scale(step,cfg))
    return result

def initialize_joint_optimizer(engine,state):
    """Initialize newly unfrozen Qwen Adam groups once; preserve actor/schedule.

    ZeRO2 flattened zero gradients can advance Adam's step during interface
    warmup. Merely setting LR0 preserves weights but not first-update bias
    correction. At the explicitly recorded stage boundary, start Qwen moments
    and counters at0. Native checkpoints carry the completed-transition flag.
    """
    if state.get('qwen_optimizer_initialized',False):return None
    wrapped=engine.optimizer
    optimizer=getattr(wrapped,'optimizer',wrapped)
    groups=[]
    for group in optimizer.param_groups:
        if not group['name'].startswith('qwen_'):continue
        previous=[]
        for parameter in group['params']:
            moments=optimizer.state.get(parameter,{})
            step=moments.get('step',0)
            previous.append(float(step.item() if isinstance(step,torch.Tensor) else step))
            for key,value in moments.items():
                if key in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    if isinstance(value,torch.Tensor):value.zero_()
                    else:moments[key]=0
                else:raise ValueError('Unrecognized Qwen Adam state at joint initialization: '+key)
        groups.append(dict(name=group['name'],previous_partition_steps=previous))
    if {group['name'] for group in groups}!={'qwen_vision','qwen_language_merger'}:raise ValueError('Cannot identify both Qwen optimizer groups at stage boundary')
    event=dict(kind='explicit_Qwen_Adam_initialization_at_joint_boundary',after_updates=state['updates'],
        groups=groups,actor_optimizer_preserved=True,scheduler_global_timeline_preserved=True)
    state.update(qwen_optimizer_initialized=True,joint_optimizer_initialization=event)
    return event

def source_identity():
    paths=set(Path('starVLA/model/modules/qwen_world_postcot').glob('*.py'))
    paths.update(Path(name) for name in (
        'tools/qwen_world_postcot/train_il.py','starVLA/rl/qwen_world_postcot/batching.py',
        'starVLA/rl/qwen_world_postcot/checkpoint.py','starVLA/rl/qwen_world_postcot/loading.py',
        'starVLA/rl/qwen_world_postcot/bank_versions.py',
        'starVLA/rl/qwen_world_postcot/coordinate_objective.py',
        'starVLA/rl/qwen_world_postcot/loss_designs.py',
        'starVLA/rl/qwen_world_postcot/fresh_loss.py',
        'starVLA/rl/qwen_world_postcot/data.py','starVLA/rl/qwen_world_postcot/contracts.py',
        'starVLA/model/modules/qwen_sourceflow/action_codec.py',
        'starVLA/model/modules/qwen_sourceflow/qwen_encoder.py',
        'starVLA/rl/flow_grpo/reproducibility.py'))
    return {str(path):file_sha(path) for path in sorted(paths)}

def consistent_cpu_data_error(error,device):
    flag=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if not bool(flag):return
    errors=[error]
    if dist.is_initialized():
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    raise RuntimeError('CPU data preparation failed on at least one rank; no targets/cursor substituted: '+repr(errors))

def completed_update_guard(engine,before_steps,before_skipped,device):
    steps=int(engine.global_steps);skipped=int(getattr(engine,'skipped_steps',0))
    optimizer=engine.optimizer;overflow=bool(getattr(optimizer,'overflow',False))
    norm=engine.get_global_grad_norm() if hasattr(engine,'get_global_grad_norm') else getattr(optimizer,'_global_grad_norm',None)
    if norm is None:norm=getattr(optimizer,'_global_grad_norm',None)
    value=None if norm is None else float(norm)
    error=None
    if steps!=before_steps+1 or skipped!=before_skipped or overflow or value is None or not math.isfinite(value) or value<0:
        error=dict(global_steps_before=before_steps,global_steps_after=steps,skipped_before=before_skipped,
            skipped_after=skipped,overflow=overflow,grad_norm=repr(value))
    flag=torch.tensor(int(error is not None),device=device,dtype=torch.int32)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if bool(flag):
        errors=[error]
        if dist.is_initialized():errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
        raise FloatingPointError('Optimizer update was skipped/nonfinite/incomplete; cursor remains unconsumed: '+repr(errors))
    return dict(grad_norm=value,optimizer_global_steps=steps,optimizer_skipped_steps=skipped)

def verify_generic_qwen_identity(identity_path=QWEN_IDENTITY,model_path=QWEN):
    path=Path(identity_path);receipt=json.loads(path.read_text());root=Path(model_path).resolve()
    if (receipt['repository']!='Qwen/Qwen3-VL-2B-Instruct' or not receipt.get('revision')
            or Path(receipt['path']).resolve()!=root or receipt.get('official_weight_sha_verified') is not True
            or not any(name.endswith('.safetensors') for name in receipt['files'])):
        raise ValueError('Unverified generic Qwen initialization provenance')
    for name,expected in receipt['files'].items():
        target=root/name
        if not target.resolve().is_relative_to(root) or file_sha(target)!=expected:
            raise ValueError('Generic Qwen initialization file changed: '+name)
    return dict(receipt=receipt,identity_path=str(path.resolve()),identity_file_sha256=file_sha(path),
                all_files_reverified=True,loading='strict official Qwen; vocabulary output head discarded')

class RuntimeDtypeAudit:
    """Bounded hooks record actual tensors, without retaining their graphs."""
    def __init__(self,model):
        self.handles=[];self.result=dict(memory={},projected_KV={},action_state_input=[],incoming_gradients={})
        def memory_hook(module,args,value):
            self.result['memory']={name:str(getattr(value,name).dtype) for name in ('spatial','global_memory','predictive')}
            self.result['memory']['actor_concatenation']=str(value.actor_memory.dtype)
        self.handles.append(model.predictive_provider.register_forward_hook(memory_hook))
        for i,block in enumerate(model.actor.blocks):
            for name in ('k','v'):
                key=f'block{i}.{name}'
                self.result['projected_KV'][key]=[]
                def kv_hook(module,args,value,key=key):self.result['projected_KV'][key].append(str(value.dtype))
                self.handles.append(getattr(block.cross_attn,name).register_forward_hook(kv_hook))
        def action_hook(module,args,value):self.result['action_state_input'].append(str(args[0].dtype))
        self.handles.append(model.actor.action_input.register_forward_hook(action_hook))
        prefixes=('qwen.model.visual.','qwen.model.language_model.','scene_provider.','predictive_provider.',
                  'policy.actor.action_input.','policy.actor.blocks.0.cross_attn.k.','dino_readout.','future_readout.','object_readout.','post_head.')
        names=set()
        for prefix in prefixes:
            match=next(((name,p) for name,p in model.named_parameters() if name.startswith(prefix) and p.requires_grad),None)
            if match is None:raise ValueError('Dtype audit cannot identify module: '+prefix)
            name,p=match
            if name in names:continue
            names.add(name);self.result['incoming_gradients'][name]=None
            def gradient_hook(value,name=name):self.result['incoming_gradients'][name]=str(value.dtype)
            self.handles.append(p.register_hook(gradient_hook))
        self.result['parameter_dtypes']=sorted({str(p.dtype) for p in model.parameters()})
    def finish(self,engine):
        for handle in self.handles:handle.remove()
        wrapped=engine.optimizer;optimizer=getattr(wrapped,'optimizer',wrapped)
        self.result['zero_flat_master_dtypes']=sorted({str(p.dtype) for p in getattr(wrapped,'single_partition_of_fp32_groups',[])})
        states={}
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor):states.setdefault(key,set()).add(str(value.dtype))
        self.result['Adam_state_dtypes']={key:sorted(value) for key,value in states.items()}
        return self.result

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/qwen_world_postcot/il.yaml')
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--qualification',action='store_true')
    p.add_argument('--coordinate-registration');p.add_argument('--coordinate-arm',choices=('CONTROL','PHYSICAL','TIME_RESIDUAL','DELTA','HUBER'))
    p.add_argument('--coordinate-qualification-updates',type=int,default=0)
    p.add_argument('--coordinate-pause-after',type=int,default=0)
    p.add_argument('--fresh-loss-registration');p.add_argument('--fresh-loss-arm',choices=('PHYSICAL','TIME_RESIDUAL','DELTA','HUBER'))
    p.add_argument('--fresh-loss-stage',choices=('pilot','full'),default='pilot');p.add_argument('--fresh-selection')
    p.add_argument('--qualification-joint',action='store_true',help='Bounded engineering profile with all joint gradients; never formal IL')
    p.add_argument('--qualification-interface-updates',type=int,help='Bounded stage-transition diagnostic only; formal interface remains1000')
    p.add_argument('--qualification-save-at',nargs='+',type=int,default=[],help='Extra boundary checkpoints in bounded engineering qualification only')
    p.add_argument('--stop-after',type=int);p.add_argument('--micro-batch',type=int);p.add_argument('--teacher-cache');p.add_argument('--candidate-bank')
    p.add_argument('--profile-start',type=int,default=0);p.add_argument('--profile-steps',type=int,default=0)
    p.add_argument('--profile-no-final-checkpoint',action='store_true',help='Explicit disposable bounded performance/trace only; reports no recoverable training checkpoint')
    a=p.parse_args();host=require_dispatch(admit_long_job=True);cfg=validate(yaml.safe_load(Path(a.config).read_text()))
    study=None;fresh=None
    if a.fresh_loss_registration:
        from starVLA.rl.qwen_world_postcot.fresh_loss import read_registration as read_fresh,objective as fresh_objective,student_initialization_digest,verify_selection
        fresh=read_fresh(a.fresh_loss_registration,a.fresh_loss_arm,a.config)
        if a.coordinate_registration or a.micro_batch or a.teacher_cache or a.candidate_bank or (a.stop_after and not a.qualification):
            raise ValueError('Fresh screen preserves the complete common recipe and100k schedule')
        if a.fresh_loss_stage=='full':
            if a.qualification:raise ValueError('Full continuation cannot use qualification weights')
            verify_selection(a.fresh_selection,fresh,a.fresh_loss_arm,a.resume)
        elif a.fresh_selection:raise ValueError('Pilot does not accept a continuation selection')
    elif a.fresh_loss_arm or a.fresh_selection or a.fresh_loss_stage!='pilot':
        raise ValueError('Fresh options require a sealed registration')
    if a.coordinate_registration:
        from starVLA.rl.qwen_world_postcot.coordinate_objective import read_registration,CoordinateObjective,restore_fork
        study=read_registration(a.coordinate_registration,a.coordinate_arm)
        if (a.qualification or a.stop_after or a.micro_batch or a.teacher_cache or a.candidate_bank
                or not 0<=a.coordinate_qualification_updates<=16):raise ValueError('Study may not override the common parent recipe')
        if a.coordinate_pause_after and not 0<a.coordinate_pause_after<a.coordinate_qualification_updates:
            raise ValueError('Artificial pause is qualification-only and must precede its endpoint')
    elif a.coordinate_arm or a.coordinate_qualification_updates or a.coordinate_pause_after:
        raise ValueError('Coordinate options require the sealed study registration')
    if a.qualification_joint and not a.qualification:raise ValueError('Joint-stage override is qualification-only')
    if a.qualification_interface_updates is not None and (not a.qualification or a.qualification_joint or not 1<=a.qualification_interface_updates<=250):raise ValueError('Short interface override is only for an explicit bounded qualification transition')
    if a.qualification_save_at and (not a.qualification or any(not 1<=step<=250 for step in a.qualification_save_at)):raise ValueError('Extra save boundaries are bounded qualification-only execution settings')
    interface_updates=a.qualification_interface_updates or cfg['interface_updates']
    if a.micro_batch is not None:
        if a.micro_batch not in (2,4):raise ValueError('Only micro4/accum1 or micro2/accum2 preserves the main batch')
        cfg['micro_batch']=a.micro_batch
    for key,value in [('teacher_cache',a.teacher_cache),('candidate_bank',a.candidate_bank)]:
        if value:cfg[key]=value
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if not a.qualification and world!=8:raise ValueError('Formal main experiment requires8GPU/global32')
    if a.qualification and a.stop_after is None:raise ValueError('Qualification requires an explicit bounded --stop-after')
    if a.stop_after and (not a.qualification or not 1<=a.stop_after<=250):raise ValueError('Only bounded qualification/profile may stop before100k')
    if a.profile_start<0 or a.profile_steps<0 or a.profile_steps>10 or (a.profile_steps and (not a.qualification or a.profile_start+a.profile_steps>a.stop_after)):raise ValueError('Profiler requires a bounded qualification window of at most10 steps')
    if a.profile_no_final_checkpoint and (not a.qualification or a.resume or a.qualification_save_at):raise ValueError('Skipping a final checkpoint is only supported for an explicit bounded fresh performance/trace invocation')
    micro=cfg['micro_batch'];accum=cfg['global_batch']//(world*micro)
    if accum<1 or world*micro*accum!=32:raise ValueError('Micro topology does not yield global32')
    cfg['gradient_accumulation']=accum
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise ValueError('Main numerical contract requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    configure_numerics()
    torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    qwen_initialization=cp._rank_zero_call(verify_generic_qwen_identity)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lk=(out/f'rank{rank}.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    data=WorldSceneData(cfg['training_data']);processor=QwenObservationProcessor(QWEN)
    # A bounded qualification may consume committed immutable records while the
    # full builder is running. It never treats that growing index as COMPLETE.
    teachers=TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN',allow_building=a.qualification)
    bank=CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',),
        expected_identity=dict(source_data_manifest_sha256=data.sha,source_records_sha256=data.manifest['records_sha256']))
    # Formal coverage is an input prerequisite, never silently train a tiny pilot.
    teacher_tokens=set(teachers.records)
    if teachers.identity['records_sha256']!=data.manifest['records_sha256']:raise ValueError('Teacher cache belongs to another current-observation dataset')
    if not a.qualification and teacher_tokens!=set(data.tokens):raise ValueError('Teacher TRAIN targets are not complete')
    if not a.qualification and set(bank.manifest['scene_ids'])!=set(data.tokens):raise ValueError('Candidate TRAIN bank is not complete')
    builder=BatchBuilder(data,processor,teachers,bank,cfg['seed']+rank,require_full_bank=not a.qualification,
        max_candidates=cfg['candidate_refresh']['max_candidates'])
    model=QwenWorldPostCoT(QWEN,config=cfg['model']).to(device)
    if fresh is not None:
        initialization_digest=cp._rank_zero_call(lambda:student_initialization_digest(model))
        if initialization_digest!=fresh['initialization_student_sha256']:
            raise ValueError('Fresh student differs from the common generic initialization')
        model.policy.coordinate_objective=fresh_objective(fresh['arms'][a.fresh_loss_arm])
    if study is not None:
        if study['schema']=='world_IL_loss_designs_v1':
            from starVLA.rl.qwen_world_postcot.loss_designs import LossDesign
            model.policy.coordinate_objective=LossDesign(**study['arms'][a.coordinate_arm])
        else:model.policy.coordinate_objective=CoordinateObjective(**study['arms'][a.coordinate_arm])
    if cfg['distributed']['qwen_activation_checkpointing']:model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameter_groups=groups(model,cfg);group_contract=optimizer_group_contract(model,parameter_groups)
    optcfg=cfg['optimizer'];optimizer=torch.optim.AdamW(parameter_groups,betas=tuple(optcfg['betas']),eps=optcfg['eps'],
        weight_decay=optcfg['weight_decay'],fused=optcfg['fused'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,learning_scales(optimizer.param_groups,cfg,qualification_joint=a.qualification_joint,interface_updates=interface_updates))
    import deepspeed
    dscfg=dict(train_batch_size=32,train_micro_batch_size_per_gpu=micro,gradient_accumulation_steps=accum,
        gradient_clipping=optcfg['gradient_clip'],bf16=dict(enabled=False),fp16=dict(enabled=False),
        zero_optimization=dict(stage=2,overlap_comm=cfg['distributed']['overlap_comm'],contiguous_gradients=True,
            reduce_bucket_size=cfg['distributed']['reduce_bucket_size'],allgather_bucket_size=cfg['distributed']['allgather_bucket_size'],
            ignore_unused_parameters=True),steps_per_print=1000000,wall_clock_breakdown=False)
    engine,optimizer,_,scheduler=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=dscfg,dist_init_required=False)
    identity=dict(schema='world_postcot_full_IL_v1',uses_metric_supervision=True,uses_rl=False,qualification=a.qualification,
        model_config=cfg['model'],qwen_config=model.qwen.model.config.to_dict(),config=cfg,zero_config=dscfg,
        data_sha256=data.sha,teacher_identity_sha256=file_sha(Path(cfg['teacher_cache'])/'IDENTITY.json'),
        optimizer_ordered_parameter_groups=group_contract,
        teacher_manifest_sha256=None if a.qualification else file_sha(Path(cfg['teacher_cache'])/'MANIFEST.json'),
        teacher_partial_admission=a.qualification,candidate_bank_hash=bank.bank_hash,
        source_files=source_identity(),diagnostic_stage_override='joint' if a.qualification_joint else None,
        diagnostic_interface_updates=a.qualification_interface_updates,
        initialization=dict(qwen=qwen_initialization,new_student_seed=cfg['seed'],
            new_student_scheme='Torch default linear/readouts; memory queries Normal(0,.02); action position Normal(0,.02); AdaLN modulation zeros; action output Xavier',
            qwen_joint_optimizer='explicit new Qwen Adam step/moments initialization after interface; actor/scheduler history preserved'),
        numerics=dict(deterministic=torch.are_deterministic_algorithms_enabled(),tf32=False,bf16_reduced_precision_reduction=False,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG']),
        torch=torch.__version__,deepspeed=deepspeed.__version__,world=world,train_heads=True)
    if study is not None:
        identity['coordinate_experiment']=dict(registration_sha256=study['sha256'],arm=a.coordinate_arm,
            objective=study['arms'][a.coordinate_arm],freeze_parent_candidate_bank=True,
            qualification_updates=a.coordinate_qualification_updates)
        identity['qualification']=bool(a.coordinate_qualification_updates)
    if fresh is not None:
        identity['fresh_loss_experiment']=dict(registration_sha256=fresh['sha256'],arm=a.fresh_loss_arm,
            objective=fresh['arms'][a.fresh_loss_arm],initialization_student_sha256=initialization_digest,
            initial_driving_checkpoint=None,total_schedule_updates=100000,pilot_updates=10000)
    identity=cp.canonical_identity(identity)
    def publish_identity():
        existing=out/'IDENTITY.json'
        if existing.exists() and json.loads(existing.read_text())!=identity:
            cp.atomic_json(out/'IDENTITY_MISMATCH.json',dict(differences=cp.identity_diff(json.loads(existing.read_text()),identity)))
            raise ValueError('Output already belongs to another run; see exact IDENTITY_MISMATCH.json')
        cp.atomic_json(existing,identity);cp.atomic_json(out/'RESOLVED_CONFIG.json',cfg)
    cp._rank_zero_call(publish_identity)
    versions=BankVersions(out,cfg['candidate_refresh'],builder,identity)
    random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);torch.manual_seed(cfg['seed']+rank)
    state=dict(updates=0,cursor=0,phase='interface',qwen_optimizer_initialized=a.qualification_joint);last=None
    if a.resume:state=cp.restore(a.resume,engine,identity,builder);last=a.resume
    elif study is not None:
        state=restore_fork(study['parent_checkpoint'],engine,identity,builder,study,out)
    if study is None:versions.adopt(builder,state['updates'])
    start=state['updates'];limit=min(a.stop_after or 100000,100000)
    if fresh is not None and not a.qualification and a.fresh_loss_stage=='pilot':
        limit=fresh['pilot_updates']
    if study is not None:
        limit=study['parent_updates']+(a.coordinate_qualification_updates or study['added_updates'])
    profiler=None
    try:
        with (out/f'metrics_rank{rank}.jsonl').open('a',buffering=1) as log:
            while state['updates']<limit:
                if a.profile_steps and state['updates']==a.profile_start:
                    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True);profiler.__enter__()
                tick=time.monotonic();phase='interface' if state['updates']<interface_updates and not a.qualification_joint else 'joint';model.configure_stage(phase=='interface');engine.train()
                if phase=='joint' and not state['qwen_optimizer_initialized']:
                    event=initialize_joint_optimizer(engine,state)
                    cp.atomic_json(out/f'JOINT_OPTIMIZER_INITIALIZATION_rank{rank}.json',event)
                bank_used=builder.bank.bank_hash
                batch_tokens=[[data.tokens[data.index(state['cursor']+(m*world+rank)*micro+j)] for j in range(micro)] for m in range(accum)]
                data_error=None
                try:batches=[builder.build(ts) for ts in batch_tokens]
                except Exception as exc:data_error=f'{type(exc).__name__}: {exc}'[:2000]
                if data_error is None and int(engine.global_steps)!=state['updates']:data_error='Native engine and consumed update cursor disagree before execution'
                consistent_cpu_data_error(data_error,device);data_seconds=time.monotonic()-tick
                denoms=update_denominators(batches,device);weights=dict(cfg['loss_weights']);ramp=min(1.,(state['updates']+1)/cfg['aux_warmup_updates'])
                for key in weights:
                    if key!='x':weights[key]*=ramp
                sums={};coordinate_sums=None;engine.zero_grad();torch.cuda.reset_peak_memory_stats()
                dtype_audit=RuntimeDtypeAudit(model) if state['updates']==start or state['phase']!=phase else None
                lr_used=[g['lr'] for g in engine.optimizer.param_groups]
                before_steps=int(engine.global_steps);before_skipped=int(getattr(engine,'skipped_steps',0))
                for batch in batches:
                    with torch.profiler.record_function('world/H2D'):batch=to_device(batch,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=engine(batch,weights=weights,denominators=denoms)
                    if result['loss'].dtype!=torch.float32 or any(v.dtype!=torch.float32 for v in result['losses'].values()):raise TypeError('All complete-model losses and reductions must remain FP32')
                    if dtype_audit is not None:
                        dtype_audit.result['total_loss_dtype']=str(result['loss'].dtype)
                        dtype_audit.result['loss_scalar_dtypes']={key:str(value.dtype) for key,value in result['losses'].items()}
                    with torch.profiler.record_function('world/backward'):engine.backward(result['loss'],scale_wrt_gas=False)
                    for key,value in result['losses'].items():sums[key]=sums.get(key,0.)+float(value)/(accum if key=='sigma_mean' else 1)
                    if study is not None or fresh is not None:
                        diag=result['trajectory_diagnostics']
                        value=torch.stack((diag['normalized_sum'],diag['valid_count']))
                        coordinate_sums=value if coordinate_sums is None else coordinate_sums+value
                    with torch.profiler.record_function('world/optimizer'):engine.step()
                update_health=completed_update_guard(engine,before_steps,before_skipped,device)
                state.update(updates=state['updates']+1,cursor=state['cursor']+32,phase=phase)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                maxwall=torch.tensor(elapsed,device=device);dist.all_reduce(maxwall,op=dist.ReduceOp.MAX)
                # Loss contributions already use the full-update denominator.
                # Sum microbatches; only average the rank-scaled sums over ranks.
                names=list(sums);metrics=torch.tensor([sums[k] for k in names],device=device)
                dist.all_reduce(metrics);sums={k:float(v)/world for k,v in zip(names,metrics)}
                row=dict(updates=state['updates'],cursor=state['cursor'],phase=phase,losses=sums,seconds=elapsed,
                    slowest_rank_seconds=float(maxwall),data_seconds=data_seconds,peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),sample_ids=batch_tokens,lr_used=lr_used,lr=[g['lr'] for g in engine.optimizer.param_groups])
                row.update(update_health,candidate_bank_hash_used=bank_used)
                if coordinate_sums is not None:
                    dist.all_reduce(coordinate_sums)
                    count=coordinate_sums[1];norm=coordinate_sums[0]/count.clamp_min(1)
                    physical=norm*norm.new_tensor((33.37*100,21*100,1.765*180/math.pi))
                    row['trajectory_geometry']=dict(normalized_MAE_by_time_channel=norm.cpu().tolist(),
                        physical_MAE_by_time_channel=physical.cpu().tolist(),units=['cm','cm','degree_unwrapped'],
                        valid_counts=count.cpu().tolist(),kind='TRAIN_noisy_clean_x_reconstruction_not_ODE')
                    row.update(invocation_start_update=start)
                    if study is not None:row['study_new_updates']=state['updates']-study['parent_updates']
                    if fresh is not None:row.update(fresh_loss_arm=a.fresh_loss_arm,fresh_loss_stage=a.fresh_loss_stage,
                        fresh_registration_sha256=fresh['sha256'])
                adopt_tick=time.monotonic();binding=versions.adopt(builder,state['updates']) if study is None else None
                if binding is not None:row.update(bank_boundary_seconds=time.monotonic()-adopt_tick,candidate_bank_hash_next=builder.bank.bank_hash)
                row['valid_supervision_counts']=dict(action_elements=float(denoms['action_elements']),
                    dino_scenes=float(denoms['dino']),future_scenes=float(denoms['future']),
                    **{key:float(value) for key,value in denoms['audit_counts'].items()})
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if dtype_audit is not None:
                    cp.atomic_json(out/f'DTYPE_update{state["updates"]:06d}_rank{rank}.json',dtype_audit.finish(engine))
                if rank==0 and (state['updates']<=5 or state['updates']%20==0):print(json.dumps(row),flush=True)
                if profiler is not None and state['updates']==a.profile_start+a.profile_steps:
                    profiler.__exit__(None,None,None);profiler.export_chrome_trace(str(out/f'trace_rank{rank}.json'));profiler=None
                requested=torch.tensor(int(STOP or boundary_stop_due(host)),device=device);dist.all_reduce(requested,op=dist.ReduceOp.MAX)
                terminal=state['updates']==limit or bool(requested) or (study is not None and a.coordinate_pause_after>0 and state['updates']==study['parent_updates']+a.coordinate_pause_after)
                if (terminal or state['updates']%cfg['save_every']==0 or state['updates'] in a.qualification_save_at) and not a.profile_no_final_checkpoint:
                    save_start=time.monotonic();last=cp.save(out,engine,state,identity,builder)
                    if rank==0:
                        status=('PILOT_COMPLETE' if fresh is not None and a.fresh_loss_stage=='pilot' and not a.qualification and state['updates']==limit else
                            'STUDY_COMPLETE' if study is not None and state['updates']==limit else
                            'COMPLETE' if state['updates']==100000 else 'STOPPED' if terminal else 'RUNNING')
                        cp.atomic_json(out/'progress.json',dict(state,status=status,checkpoint=last,save_seconds=time.monotonic()-save_start))
                        cp.retain_rolling(out,cfg['rolling_checkpoints'])
                elif terminal and rank==0:
                    cp.atomic_json(out/'progress.json',dict(state,status='DIAGNOSTIC_COMPLETE',checkpoint=None,
                        recoverable_training_checkpoint=False,reason='Explicit fresh disposable bounded performance/trace invocation'))
                if terminal:break
    except BaseException:
        cp.atomic_json(out/f'FAILURE_rank{rank}.json',dict(error=traceback.format_exc(),state=state,last_complete_checkpoint=last));raise
    finally:
        if profiler is not None:profiler.__exit__(None,None,None)
        builder.bank.close();teachers.close();dist.destroy_process_group()

if __name__=='__main__':main()
