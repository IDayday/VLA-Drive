"""Complete100k shared-objective IL; finite qualification never initializes it."""
import argparse
import copy
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
import sys
if __package__ in (None,''):
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or os.curdir).resolve()!=script_dir]
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from starVLA.model.modules.qwen_world_compare.factory import load_config,build_model
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot import checkpoint as cp
from starVLA.rl.qwen_world_compare.contracts import require_dispatch,boundary_stop_due,file_sha,fingerprint
from starVLA.rl.qwen_world_compare.track_targets import FutureTrackStore
from starVLA.rl.qwen_world_compare.evidence import EvidenceBank
from starVLA.rl.qwen_world_compare.batching import BatchBuilder,to_device,update_denominators
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from tools.qwen_world_postcot.train_il import (initialize_joint_optimizer,completed_update_guard,
    consistent_cpu_data_error,verify_generic_qwen_identity,optimizer_group_contract)

STOP=False
def stop(*_):
    global STOP
    STOP=True

class GradientRolesAudit:
    def __init__(self,model):
        self.roles=model.parameter_roles();self.values={};self.seen=set();self.positive=set();self.handles=[]
        for name,p in model.named_parameters():
            def capture(g,name=name):
                self.values[name]=g.detach().float().norm();return g
            self.handles.append(p.register_hook(capture))
    def finish_update(self):
        names=list(self.values)
        if names:
            values=torch.stack([self.values[n]for n in names]).cpu().tolist()
            self.seen.update(names);self.positive.update(n for n,v in zip(names,values)if v>0)
            if not all(math.isfinite(v)for v in values):raise FloatingPointError('Nonfinite parameter gradient')
        self.values.clear()
    def result(self):
        return dict(seen_names=sorted(self.seen),nonzero_names=sorted(self.positive),roles={r:dict(seen=sum(self.roles[n]==r for n in self.seen),nonzero=sum(self.roles[n]==r for n in self.positive))for r in set(self.roles.values())})

def source_files():
    roots=['starVLA/model/modules/qwen_world_compare','starVLA/rl/qwen_world_compare','tools/qwen_world_compare',
           'starVLA/model/modules/qwen_world_postcot','starVLA/model/modules/qwen_sourceflow']
    paths={p for root in roots for p in Path(root).glob('*.py')}
    paths.update(Path(p)for p in ['tools/qwen_world_postcot/train_il.py','starVLA/rl/qwen_world_postcot/checkpoint.py','starVLA/rl/qwen_world_postcot/loading.py','starVLA/rl/qwen_world_postcot/data.py','starVLA/rl/flow_grpo/reproducibility.py'])
    return {str(p):file_sha(p)for p in sorted(paths)}

def parameter_groups(model,cfg):
    roles=model.parameter_roles();buckets={k:[]for k in ['qwen_language_merger','qwen_vision','new_modules']}
    for name,p in model.named_parameters():
        role=roles[name];key={'pretrained_language_merger':'qwen_language_merger','pretrained_vision':'qwen_vision'}.get(role,'new_modules')
        buckets[key].append(p)
    rates=cfg['il']['learning_rates'];return [dict(params=v,name=k,lr=rates[{'qwen_language_merger':'pretrained_language_merger','qwen_vision':'pretrained_vision'}.get(k,k)])for k,v in buckets.items()]

def lr_scales(cfg,groups,*,joint=False):
    il=cfg['il'];warm=int(il['max_updates']*.03)
    def scale(step,pretrained):
        if pretrained and not joint and step<il['interface_updates']:return 0.
        return (step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1.,(step-warm)/(il['max_updates']-warm))))
    return [lambda step,pre=g['name'].startswith('qwen_'):scale(step,pre)for g in groups]

class ProfileDDPEngine:
    """Disposable DDP profiling only; formal native recovery stays ZeRO2."""
    def __init__(self,model,optimizer,scheduler,accum):
        self.module=model;self.wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[torch.cuda.current_device()],find_unused_parameters=True,static_graph=False,bucket_cap_mb=25)
        self.optimizer,self.scheduler,self.accum=optimizer,scheduler,accum
        self.global_steps=self.skipped_steps=self.micro=0;self._norm=None
    def __call__(self,*a,**kw):return self.wrapped(*a,**kw)
    def train(self):self.module.train()
    def zero_grad(self):self.optimizer.zero_grad(set_to_none=True)
    def backward(self,loss,**_):loss.backward()
    def step(self):
        self.micro+=1
        if self.micro%self.accum:return
        self._norm=float(torch.nn.utils.clip_grad_norm_(self.module.parameters(),1.))
        if not math.isfinite(self._norm):raise FloatingPointError('Nonfinite DDP gradient')
        self.optimizer.step();self.scheduler.step();self.global_steps+=1
    def get_global_grad_norm(self):return self._norm

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--qualification',action='store_true');p.add_argument('--joint',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--evidence-override');p.add_argument('--profile-no-save',action='store_true');p.add_argument('--backend',choices=['zero2','ddp']);p.add_argument('--micro-batch',type=int)
    p.add_argument('--qualification-save-at',type=int,nargs='*',default=[])
    p.add_argument('--profile-dir');p.add_argument('--profile-steps',type=int,default=0)
    p.add_argument('--retain-short-graphs',action='store_true')
    p.add_argument('--checkpoint-policy',choices=('source','short_off','all_off'),default='source')
    p.add_argument('--cpu-prefetch-depth',type=int,choices=(0,1,2),default=0)
    p.add_argument('--execution-receipt')
    p.add_argument('--qualification-parent')
    a=p.parse_args();cfg=load_config(a.config);host=require_dispatch(admit_long_job=not a.qualification)
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if not a.qualification and (world!=8 or a.stop_after or a.evidence_override or a.profile_no_save or a.joint):raise ValueError('Formal IL preserves all100k/scenes/tasks/stages')
    if a.qualification and not(a.stop_after and 1<=a.stop_after<=20):raise ValueError('Qualification requires a finite1..20update window')
    if a.retain_short_graphs:
        if a.checkpoint_policy!='source':raise ValueError('Specify one checkpoint execution option')
        a.checkpoint_policy='short_off'
    tuning=a.checkpoint_policy!='source'or bool(a.cpu_prefetch_depth)
    if tuning and not a.qualification and not a.execution_receipt:
        raise ValueError('Formal execution tuning requires an explicit qualified native migration receipt')
    if a.execution_receipt and(a.qualification or not a.resume):raise ValueError('Formal migration needs exact native resume')
    if a.qualification_parent and(not a.qualification or a.resume or a.joint or world!=8):
        raise ValueError('Native continuation qualification requires eight ranks, original stage and separate output')
    if a.profile_no_save and (not a.qualification or a.resume):raise ValueError('Disposable profile cannot recover/formally train')
    if a.qualification_save_at and(not a.qualification or a.profile_no_save):raise ValueError('Extra checkpoint boundaries are qualification-only')
    backend=a.backend or cfg['execution']['backend']
    if backend=='ddp'and not a.profile_no_save:raise ValueError('DDP is first a disposable measured profile; native resume not yet qualified')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise ValueError('Qualified deterministic CUBLAS workspace changed')
    configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);lock=(out/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in(signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    qwen_identity=cp._rank_zero_call(lambda:verify_generic_qwen_identity(cfg['qwen_identity'],cfg['qwen_path']))
    data=WorldSceneData(Path(cfg['data_root'])/'TRAIN');teacher=TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN')
    bank=CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',));tracks=FutureTrackStore(Path(cfg['target_root'])/'TRAIN')
    evidence=EvidenceBank(a.evidence_override or cfg['evidence_root']);processor=QwenObservationProcessor(cfg['qwen_path'])
    if not a.qualification and (set(data.tokens)!=set(bank.manifest['scene_ids'])or set(data.tokens)!=set(teacher.records)or set(data.tokens)!=set(evidence.manifest['scenes'])):raise ValueError('Formal common-target population incomplete')
    builder=BatchBuilder(data,processor,teacher,bank,tracks,evidence)
    micro=a.micro_batch or cfg['execution']['micro_batch'];accum=32//(world*micro)
    if world*micro*accum!=32:raise ValueError('Microbatch topology changes global32')
    native_parent=a.resume or a.qualification_parent
    parent_config=(json.loads((Path(native_parent)/'identity.json').read_text())['qwen_config']if native_parent else None)
    # Exact native restoration loads every model tensor below. Avoid first
    # loading unused generic weights; construction never replaces native data.
    model=build_model(cfg,checkpoint_config=copy.deepcopy(parent_config)).to(device)
    if cfg['architecture_id']!='D'and cfg['execution']['activation_checkpointing']:model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if a.checkpoint_policy!='source':
        from starVLA.model.modules.qwen_world_compare.execution import checkpoint_flags,apply_checkpoint_policy
        apply_checkpoint_policy(model,a.checkpoint_policy,checkpoint_flags(model))
    groups=parameter_groups(model,cfg);opt=cfg['il']['optimizer']
    # ZeRO2 replaces param_groups entries with flattened master tensors. Bind
    # logical model names/shapes before handing those same groups to the engine.
    group_contract=optimizer_group_contract(model,groups)
    optimizer=torch.optim.AdamW(groups,betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'],fused=opt['fused'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scales(cfg,groups,joint=a.joint))
    ds=dict(train_batch_size=32,train_micro_batch_size_per_gpu=micro,gradient_accumulation_steps=accum,gradient_clipping=1.,bf16=dict(enabled=False),fp16=dict(enabled=False),zero_optimization=dict(stage=2,overlap_comm=True,contiguous_gradients=True,reduce_bucket_size=50000000,allgather_bucket_size=50000000,ignore_unused_parameters=True),steps_per_print=1000000)
    if backend=='zero2':
        import deepspeed
        engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=ds,dist_init_required=False)
    else:engine=ProfileDDPEngine(model,optimizer,scheduler,accum)
    qcfg=model.decision.qwen_model.config.to_dict()if cfg['architecture_id']=='D'else model.qwen.model.config.to_dict()
    identity=cp.canonical_identity(dict(schema='qwen_world_compare_IL_v1',architecture_id=cfg['architecture_id'],config=cfg,model_config=cfg['model'],qwen_config=qcfg,world=world,qualification=a.qualification,uses_rl=False,uses_metric_supervision=True,train_heads=True,source_files=source_files(),initialization=qwen_identity,parameter_roles=model.parameter_roles(),optimizer_ordered_parameter_groups=group_contract,data_sha256=data.sha,teacher_identity_sha256=file_sha(Path(cfg['teacher_cache'])/'IDENTITY.json'),candidate_bank_hash=bank.bank_hash,evidence_identity=evidence.manifest['identity'],evidence_manifest_sha256=file_sha(evidence.root/'MANIFEST.json'),target_manifest_sha256=file_sha(tracks.root/'manifest.json'),execution=dict(backend=backend,micro=micro,accum=accum,ds=ds),joint_override=a.joint))
    if tuning:
        identity['execution']['execution_tuning']=dict(checkpoint_policy=a.checkpoint_policy,cpu_prefetch_depth=a.cpu_prefetch_depth)
    migration=None
    if a.execution_receipt:
        from starVLA.rl.qwen_world_compare.execution_migration import validate_receipt
        migration=cp._rank_zero_call(lambda:validate_receipt(a.execution_receipt,identity,resume=a.resume))
    def publish():
        p=out/'IDENTITY.json'
        if p.exists()and json.loads(p.read_text())!=identity:
            previous=json.loads(p.read_text())
            if migration is None or previous!=migration['old_identity']:raise ValueError('Existing run identity mismatch')
            archive=out/'execution_lineage'/migration['sha256'];archive.mkdir(parents=True,exist_ok=True)
            cp.atomic_json(archive/'PREVIOUS_IDENTITY.json',previous)
            cp.atomic_json(archive/'MIGRATION.json',migration)
        cp.atomic_json(p,identity)
    state=dict(updates=0,cursor=0,phase='joint'if a.joint else'interface',qwen_optimizer_initialized=a.joint);last=None
    if a.resume:
        if a.execution_receipt:
            from starVLA.rl.qwen_world_compare.execution_migration import restore
            state=restore(a.resume,engine,identity,builder,a.execution_receipt)
        else:state=cp.restore(a.resume,engine,identity,builder)
        last=a.resume
    if a.qualification_parent:
        from starVLA.rl.qwen_world_compare.execution_migration import validate_patch
        parent=json.loads((Path(a.qualification_parent)/'identity.json').read_text())
        validate_patch(parent,identity,qualification=True)
        state=cp.restore(a.qualification_parent,engine,parent,builder);last=a.qualification_parent
    cp._rank_zero_call(publish)
    start=state['updates'];limit=(start+a.stop_after if a.qualification_parent else a.stop_after)if a.qualification else 100000
    first_handoff=bool(migration and json.loads((Path(a.resume)/'identity.json').read_text())==migration['old_identity'])
    if first_handoff:
        cp.atomic_json(out/f'EXECUTION_RESTORE_rank{rank}.json',dict(status='RESTORED',updates=start,cursor=state['cursor'],
            parent=a.resume,migration_sha256=migration['sha256'],optimizer_global_steps=int(engine.global_steps),
            builder=builder.state_dict(),lr=[g['lr']for g in optimizer.param_groups],native_full_state_loader=True))
    qualified_tokens=evidence.manifest['scenes']
    prefetch=None
    if a.cpu_prefetch_depth:
        from starVLA.rl.qwen_world_compare.prefetch import OrderedUpdatePrefetch
        def worker_builder():
            # Ownership of CPU observation/teacher/track readers transfers to
            # this sole worker. Only SQLite readers need worker-local creation;
            # duplicating all immutable manifests added ~40s startup per rank.
            return BatchBuilder(data,processor,teacher,CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',)),
                tracks,EvidenceBank(a.evidence_override or cfg['evidence_root']))
        def requests(update,cursor):
            result=[]
            for m in range(accum):
                positions=[rank*(32//world)+m*micro+j for j in range(micro)]
                tokens=[qualified_tokens[(cursor+i)%len(qualified_tokens)]if a.qualification and not a.qualification_parent else data.tokens[data.index(cursor+i)]for i in positions]
                result.append((tokens,positions))
            return result
        prefetch=OrderedUpdatePrefetch(builder,requests,start_update=start,start_cursor=state['cursor'],
            depth=a.cpu_prefetch_depth,builder_factory=worker_builder)
    gradient_audit=GradientRolesAudit(model)if a.qualification and not a.profile_no_save else None
    profiler=None
    if a.profile_dir:
        from starVLA.rl.qwen_world_compare.profiling import PhaseProfiler
        profiler=PhaseProfiler(Path(a.profile_dir),wait=1,warmup=1,active=a.profile_steps or 2,enabled=True,
            metadata=dict(architecture_id=cfg['architecture_id'],config_sha256=fingerprint(cfg),backend=backend,micro=micro,accum=accum,world=world,global_batch=32))
        profiler.__enter__()
    try:
        with(out/f'metrics_rank{rank}.jsonl').open('a',buffering=1)as log:
            while state['updates']<limit:
                tick=time.monotonic();interface=state['updates']<1000 and not a.joint;model.configure_stage(interface);engine.train()
                model.policy_version=f'IL:{cfg["architecture_id"]}:{state["updates"]}'
                if not interface and not state['qwen_optimizer_initialized']:initialize_joint_optimizer(engine,state)
                batches=[];err=None
                try:
                    if prefetch is not None:batches=prefetch.next(state['updates'],state['cursor'])
                    else:
                        for m in range(accum):
                            positions=[rank*(32//world)+m*micro+j for j in range(micro)]
                            tokens=[qualified_tokens[(state['cursor']+i)%len(qualified_tokens)]if a.qualification and not a.qualification_parent else data.tokens[data.index(state['cursor']+i)]for i in positions]
                            batches.append(builder.build(tokens,positions,state['updates']))
                except Exception as exc:err=repr(exc)
                consistent_cpu_data_error(err,device);data_seconds=time.monotonic()-tick
                den=update_denominators(batches,device);weights=dict(cfg['il']['loss_weights']);ramp=min(1.,(state['updates']+1)/1000)
                for key in weights:
                    if key!='x':weights[key]*=ramp
                engine.zero_grad();before=int(engine.global_steps);skipped=int(engine.skipped_steps);metrics={};torch.cuda.reset_peak_memory_stats();lrs=[g['lr']for g in optimizer.param_groups]
                for batch in batches:
                    batch=to_device(batch,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=engine(batch,denominators=den,weights=weights)
                    if result['loss'].dtype!=torch.float32:raise TypeError('Supervised loss must reduce FP32')
                    with torch.profiler.record_function('compare/backward'):engine.backward(result['loss'],scale_wrt_gas=False)
                    with torch.profiler.record_function('compare/optimizer_communication'):engine.step()
                    for k,v in result['losses'].items():metrics[k]=metrics.get(k,0.)+float(v)
                health=completed_update_guard(engine,before,skipped,device);state.update(updates=state['updates']+1,cursor=state['cursor']+32,phase='interface'if interface else'joint')
                if gradient_audit is not None:gradient_audit.finish_update()
                torch.cuda.synchronize();elapsed=time.monotonic()-tick;slow=torch.tensor(elapsed,device=device);dist.all_reduce(slow,op=dist.ReduceOp.MAX)
                row=dict(updates=state['updates'],cursor=state['cursor'],phase=state['phase'],sample_ids=[b['tokens']for b in batches],post_positions=[b['positions'][i]for b in batches for i in b['aux_scene_indices'].tolist()],losses=metrics,seconds=elapsed,slowest_rank_seconds=float(slow),data_seconds=data_seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),lr_used=lrs,lr=[g['lr']for g in optimizer.param_groups],**health)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if profiler is not None:profiler.step(update=state['updates'],scenes=32,wall_seconds=elapsed,data_wait_seconds=data_seconds,optimizer_seconds=None)
                if rank==0 and(state['updates']<=5 or state['updates']%20==0):print(json.dumps(row),flush=True)
                requested=torch.tensor(int(STOP or boundary_stop_due(host)),device=device);dist.all_reduce(requested,op=dist.ReduceOp.MAX);terminal=state['updates']==limit or bool(requested)
                handoff_boundary=first_handoff and state['updates']==start+2
                if not a.profile_no_save and(terminal or handoff_boundary or state['updates']%1000==0 or state['updates']in a.qualification_save_at):
                    st=time.monotonic();last=cp.save(out,engine,state,identity,builder)
                    if rank==0:
                        cp.atomic_json(out/'progress.json',dict(state,status='COMPLETE'if state['updates']==100000 else'STOPPED'if terminal else'RUNNING',checkpoint=last,save_seconds=time.monotonic()-st));cp.retain_rolling(out,2)
                        if handoff_boundary:cp.atomic_json(out/'EXECUTION_HANDOFF_COMPLETE.json',dict(status='PASS',from_updates=start,updates=state['updates'],checkpoint=last,migration_sha256=migration['sha256']))
                if terminal:break
        if rank==0:cp.atomic_json(out/'INVOCATION_COMPLETE.json',dict(updates=state['updates'],invocation_start_update=start,qualification=a.qualification,checkpoint=last,profile_no_save=a.profile_no_save))
        if gradient_audit is not None:
            cp.atomic_json(out/f'GRADIENT_AUDIT_start{start:06d}_rank{rank}.json',gradient_audit.result())
    except BaseException:
        cp.atomic_json(out/f'FAILURE_rank{rank}.json',dict(error=traceback.format_exc(),state=state,last_complete_checkpoint=last));raise
    finally:
        if prefetch is not None:
            prefetch.close()
            cp.atomic_json(out/f'CPU_PREFETCH_start{start:06d}_pid{os.getpid()}_rank{rank}.json',prefetch.statistics())
        if profiler is not None:profiler.__exit__(*sys.exc_info())
        bank.close();teacher.close();dist.destroy_process_group()

if __name__=='__main__':main()
