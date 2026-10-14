"""Two registered constrained group-relative policies on the audited ZeRO-2 engine."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import torch.distributed as dist
from accelerate.utils import set_seed

from starVLA.rl.flow_grpo import distributed as control
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import optimizer_groups, parameter_manifest
from starVLA.rl.flow_grpo.monitor import GradientMonitor, CheckedDeepSpeedBackward, ParameterProbe, optimizer_norm_metrics
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.stability import ReferenceKLController
from starVLA.rl.flow_grpo.trainer import make_accelerator
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions
from starVLA.rl.constrained_flow_noise.checkpoint import save_checkpoint, resume_checkpoint, export_policy
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, fingerprint, file_sha, validate_parameters, SFT_SHA256
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.effective_group_collector import EffectiveGroupCollector
from starVLA.rl.constrained_flow_noise.lagrange import CostMultipliers
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy, ReplayLabels, DEFAULT_CONFIG, DEFAULT_SOURCE
from starVLA.rl.constrained_flow_noise.runtime import OrderedScenes, collect_batch, gather_rows, update_dual, micro_inputs, archive_collection, exposure_lr, cpu_tree
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService
from starVLA.rl.constrained_flow_noise.rollout_graph import configure as configure_rollout_graph
from starVLA.rl.constrained_flow_noise.diagnostics import summarize_rows, saturation_streak
from starVLA.rl.constrained_flow_noise.monitor import run_monitor
from scripts.analysis.constrained_flow_noise.evaluate_policy import run_evaluation

STOP=False
def signal_stop(*_):
    global STOP
    STOP=True


def stop_requested(device):
    flag=torch.tensor(int(STOP),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag)


def probe_gradients(engine,output):
    """Original public ZeRO gradient API, extended to the explicit noise closure."""
    from deepspeed.utils import safe_get_full_grad
    entries=[]
    for name,parameter in engine.module.named_parameters():
        if not parameter.requires_grad:continue
        value=safe_get_full_grad(parameter)
        entries.append(dict(name=name,present=value is not None,dtype=str(value.dtype) if value is not None else None,
                            finite=bool(torch.isfinite(value).all()) if value is not None else False,
                            nonzero=int(torch.count_nonzero(value)) if value is not None else 0,
                            l2=float(value.norm(dtype=torch.float64)) if value is not None else None))
    control.rank0_call(lambda:atomic_json(Path(output)/f'gradients_update{engine.global_steps+1:06d}.json',entries),engine.device)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--variant',choices=['C_SDE','N_FLOW_NOISE'],required=True)
    p.add_argument('--population',required=True);p.add_argument('--condition-cache',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--registration',required=True)
    p.add_argument('--metric-cache',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--metric-manifest',help='Complete frozen TRAIN asset manifest, required for formal training')
    p.add_argument('--budget-registration',help='Required before formal training; both variants share it')
    p.add_argument('--kernel-qualification',required=True)
    p.add_argument('--qualification',action='store_true');p.add_argument('--resume')
    p.add_argument('--qualification-detail',choices=['full','profile'],default='full')
    p.add_argument('--qualification-raw-scenes',type=int,default=32)
    p.add_argument('--stop-after-updates',type=int,help='Cooperative saved boundary, never changes registered budget')
    p.add_argument('--checkpoint-after-first-batch',action='store_true',help='Qualification of collector interruption')
    p.add_argument('--save-every',type=int,default=256);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--export-initial',action='store_true')
    a=p.parse_args();host=require_dispatch();configure_numerics()
    detailed=a.qualification and a.qualification_detail=='full'
    if not a.qualification and a.qualification_detail!='full':raise ValueError('Profile mode is qualification-only')
    if (a.stop_after_updates or a.checkpoint_after_first_batch) and not a.qualification:
        raise ValueError('Artificial stop controls are qualification-only; formal safety/host stops use cooperative signals')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    cfg,_=resolve_config(DEFAULT_CONFIG,overrides=['checkpoint_contract.source_config='+DEFAULT_SOURCE])
    cfg['runtime']['process_group_timeout']=1800
    accelerator=make_accelerator(cfg);control.initialize_control_group(1800)
    if accelerator.num_processes!=8:raise ValueError('Registered single-host eight-GPU ZeRO-2 only')
    rank,world=accelerator.process_index,accelerator.num_processes
    set_seed(42,device_specific=True)
    pop,calibration,registration=[json.loads(Path(path).read_text()) for path in (a.population,a.calibration,a.registration)]
    kernel_qualification=json.loads(Path(a.kernel_qualification).read_text())
    if kernel_qualification['status']!='PASS':raise ValueError('Kernel/graph acceptance must pass first')
    if calibration['status']!='PASS' or calibration['population_identity']!=pop['identity_sha256']:
        raise ValueError('Missing valid TRAIN-only calibration')
    budget=json.loads(Path(a.budget_registration).read_text()) if a.budget_registration else None
    if not a.qualification and (not budget or budget['population_identity']!=pop['identity_sha256'] or budget['variants']!=['C_SDE','N_FLOW_NOISE']):
        raise ValueError('Common profiled budget must be locked before formal outcomes')
    if budget:
        if budget['sha256']!=fingerprint({k:v for k,v in budget.items() if k!='sha256'}):raise ValueError('Budget seal changed')
        if (budget['training_seed'],budget['evaluation_seed'],budget['G'],budget['K'],budget['inner_epochs'])!=(42,2101,16,10,2):
            raise ValueError('Formal seed/sampling/update contract changed')
        if a.save_every!=budget['save_every_optimizer_updates']:raise ValueError('Recovery frequency differs from locked storage/cost budget')
    if not a.qualification:
        if not a.metric_manifest:raise ValueError('Complete TRAIN scoring assets must be sealed before formal training')
        assets=json.loads(Path(a.metric_manifest).read_text())
        if assets['status']!='COMPLETE' or assets['population_identity']!=pop['identity_sha256'] or assets['count']!=len(pop['train_tokens']):
            raise ValueError('Metric cache population is incomplete or mismatched')
    total=a.qualification_raw_scenes if a.qualification else int(budget['raw_exposures_per_variant'])
    if total<16:raise ValueError('At least one unfiltered first batch required')
    tokens=registration['calibration_tokens'] if a.qualification else pop['train_tokens']
    if not set(tokens)<=set(pop['train_tokens']):raise ValueError('DEV/Navtest entered update population')
    identity_cache=json.loads((Path(a.condition_cache)/'identity.json').read_text())
    store=ConditionStore(a.condition_cache,identity_cache,readonly=not a.qualification)
    if store.identity['population_identity']!=pop['identity_sha256']:raise ValueError('Cache population differs')
    actor,_,sft=make_noise_policy(a.variant,calibration['noise_std_bounds'])
    if actor.noise_head is not None:
        if file_sha(calibration['noise_initial_path'])!=calibration['noise_initial_sha256']:raise ValueError('Reference noise initialization changed')
        actor.noise_head.load_state_dict(torch.load(calibration['noise_initial_path'],map_location='cpu',weights_only=True),strict=True)
    closure=validate_parameters(actor.policy,actor.noise_head)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    behavior=copy.deepcopy(actor).eval().requires_grad_(False)
    dual=CostMultipliers(calibration['budgets'])
    kl=ReferenceKLController(.04,cfg['algorithm']['adaptive_reference_kl'])
    output=Path(a.output)
    sources=[Path(__file__).resolve(),*sorted((Path.cwd()/'starVLA/rl/constrained_flow_noise').rglob('*.py')),
             Path.cwd()/'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
             *[Path.cwd()/'scripts/analysis/constrained_flow_noise'/n for n in ('evaluate_policy.py','analyze_ode.py')],
             *[Path.cwd()/'scripts/analysis/flow_safe_improvement'/n for n in ('sample_original_export.py','score_samples.py')]]
    identity=dict(schema='cfn_training_v1',SFT_sha256=SFT_SHA256,variant=a.variant,population_identity=pop['identity_sha256'],
                  inherited_repository_commit='6edb05d576b40a9718421e86c06d560f184a11c6',
                  SFT_source_config_sha256=file_sha(DEFAULT_SOURCE),base_numerical_config_sha256=file_sha(DEFAULT_CONFIG),
                  upstream_lock_sha256=file_sha('configs/flow_rl/upstream_lock.json'),
                  condition_identity=store.identity_sha,calibration_sha256=file_sha(a.calibration),
                  metric_manifest_sha256=file_sha(a.metric_manifest) if a.metric_manifest else None,
                  registration_sha256=file_sha(a.registration),budget_sha256=file_sha(a.budget_registration) if budget else None,
                  kernel_qualification_sha256=file_sha(a.kernel_qualification),rollout_graph=True,
                  total_raw_exposures=total,training_seed=42,evaluation_seed=2101,qualification_only=a.qualification,qualification_detail=a.qualification_detail,
                  scene_population_sha256=fingerprint(tokens),kernel=vars(actor.spec),closure=closure,
                  source_files={str(path.relative_to(Path.cwd())):file_sha(path) for path in sources},
                  numerical_profile='bf16_zero2_fp32_partition_v2; noise_FP32_functional',
                  logprob_reduction='dimension_mean_per_transition',inner_epochs=2,clip=.02,
                  PG_mode='eval_with_autograd',KL_population='unfiltered_first16',replay_population='uniform_RL_TRAIN' if not a.qualification else 'qualification_TRAIN_subset')
    identity['SFT_replay_mode']='original_action_head_train; configured_dropout; isolated_FM_and_dropout_RNG; PG_eval_restored'
    identity['sha256']=fingerprint(identity)
    def prepare():
        if output.exists() and not a.resume:raise FileExistsError('Do not overwrite a run')
        output.mkdir(parents=True,exist_ok=True)
        (output/'collection_summaries').mkdir(exist_ok=True)
        if a.resume and json.loads((output/'run_identity.json').read_text())!=identity:raise ValueError('Exact resume changed scientific or source identity')
        atomic_json(output/'run_identity.json',identity)
        snapshot=output/'source_snapshot'
        for source in sources:
            dest=snapshot/source.relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and dest.read_bytes()!=source.read_bytes():raise ValueError('Source snapshot changed')
            if not dest.exists():dest.write_bytes(source.read_bytes())
    control.rank0_call(prepare,accelerator.device)
    groups=optimizer_groups(actor.policy,sft,.1)
    if actor.noise_head is not None:groups.append(dict(params=list(actor.noise_head.parameters()),lr=1e-6,weight_decay=0.,name='noise_head'))
    if any(abs(g['lr']-1e-6)>1e-12 for g in groups):raise ValueError('Unexpected actor LR group')
    manifest=parameter_manifest(actor,groups)
    control.rank0_call(lambda:atomic_json(output/'parameter_manifest.json',manifest),accelerator.device)
    optimizer=torch.optim.AdamW(groups,betas=tuple(sft.trainer.optimizer.betas),eps=sft.trainer.optimizer.eps,weight_decay=sft.trainer.optimizer.weight_decay)
    monitor=GradientMonitor(actor)
    engine,optimizer=accelerator.prepare(actor,optimizer)
    precision=install_fp32_partitions(engine)
    control.rank0_call(lambda:atomic_json(output/'zero2_precision.json',precision),accelerator.device)
    accelerator.deepspeed_engine_wrapped=CheckedDeepSpeedBackward(engine,monitor,accelerator.device,
          boundary_probe=(lambda eng:probe_gradients(eng,output)) if detailed else None)
    actor=accelerator.unwrap_model(engine);engine.eval()
    configure_rollout_graph(behavior,True)
    # Capture frozen models before ZeRO's in-place transformation, then verify
    # current/ref initial equality after prepare rather than assuming it.
    if not a.resume:
        for name,value in actor.state_dict().items():
            if not torch.equal(value,reference.state_dict()[name]):raise ValueError('Engine changed reference initialization')
    replay_labels=ReplayLabels(sft)
    stream=OrderedScenes(tokens,42);replay_stream=OrderedScenes(tokens,42+104729)
    state=dict(phase='ready',updates=0,collections=0,raw_cursor=0,accepted_exposures=0,replay_cursor=0,
               RL_updates=0,replay_reference_only_updates=0,next_inner=0,collector=None,
               dual=dual.state_dict(),KL=kl.state_dict(),collection_LR=None,seconds_rollout=0.,seconds_scoring=0.,
               seconds_reference=0.,seconds_updates=0.,seconds_scoring_cpu=0.,completed_milestones=[],saved_behavior_path=None,
               pending_evaluation=None,completed_evaluations=[],saturation_streak=0,safety_pause=None)
    if a.resume:
        state=resume_checkpoint(accelerator,engine,a.resume,identity,behavior)
        dual.load_state_dict(state['dual']);kl.load_state_dict(state['KL'])
        if state['phase']=='collecting':behavior.load_state_dict(actor.state_dict(),strict=True)
    probe=ParameterProbe(actor)
    if state['safety_pause']:raise ValueError('Saved safety pause requires a separately authorized correction; no automatic continuation')
    if a.export_initial and not a.resume:
        control.rank0_call(lambda:export_policy(actor,cfg['sft_checkpoint'],output/'export_initial',identity),accelerator.device)
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,signal_stop)
    reward_spec=dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',cache_roots=a.metric_cache)
    if a.metric_manifest:reward_spec.update(metric_manifest=a.metric_manifest,metric_manifest_sha256=file_sha(a.metric_manifest))
    service=ConstrainedRewardService(reward_spec,workers=a.workers)
    milestones=sorted(set([total] if a.qualification else [int(x*len(pop['train_tokens'])) for x in (.25,.5,1.,2.) if int(x*len(pop['train_tokens']))<=total]+[total]))
    started=time.monotonic(); stopped=False; last_checkpoint=a.resume
    def save(label):
        nonlocal last_checkpoint
        state['dual']=dual.state_dict();state['KL']=kl.state_dict()
        last_checkpoint=str(save_checkpoint(accelerator,engine,output,state,identity,label=label,behavior=behavior))
    def fixed_monitor(raw):
        behavior.load_state_dict(actor.state_dict(),strict=True)
        version=dict(run_identity=identity['sha256'],collection='monitor',optimizer_update=state['updates'],
                     raw_exposure=raw,condition=store.identity_sha,kernel=vars(actor.spec),mode='eval')
        run_monitor(behavior,reference,store,service,pop,registration,calibration,dual.snapshot(),version,
                    output/'monitor'/f'raw{raw:06d}',rank,world)
    def evaluate_pending():
        raw=state['pending_evaluation']
        if raw is None or a.qualification:return False
        export=output/f'export_raw{raw:06d}'
        control.rank0_call(lambda:None if (export/'COMPLETE').exists() else export_policy(actor,cfg['sft_checkpoint'],export,identity),accelerator.device)
        fixed_monitor(raw)
        evaluation=budget['DEV_evaluation']
        previous=(output/'DEV'/f'raw{state["completed_evaluations"][-1]:06d}'/'report.json') if state['completed_evaluations'] else None
        report=control.rank0_result(lambda:run_evaluation(export,a.population,output/'DEV'/f'raw{raw:06d}',
                   evaluation['baseline_scores'],evaluation['override'],evaluation['cache_root'],previous,evaluation['CPU_workers']),accelerator.device)
        state['completed_evaluations'].append(raw);state['pending_evaluation']=None
        if report['severe_safety_pause_components']:
            state['safety_pause']=dict(reason='predeclared_DEV_severe_safety_degradation',raw_exposure=raw,
                                      components=report['severe_safety_pause_components'],report_sha256=report['sha256'])
        save(f'evaluated_update{state["updates"]:06d}_raw{raw:06d}')
        return bool(state['safety_pause'])
    try:
        if not a.qualification and not a.resume:
            fixed_monitor(0)
            save('initial_raw000000')
        if evaluate_pending():stopped=True
        while state['raw_cursor']<total or state['phase'] in ('collecting','updating'):
            if stopped:break
            if state['phase']=='ready':
                behavior.load_state_dict(actor.state_dict(),strict=True)
                version=dict(run_identity=identity['sha256'],collection=state['collections'],optimizer_update=state['updates'],
                             condition=store.identity_sha,kernel=vars(actor.spec),mode='eval')
                collector=EffectiveGroupCollector(version,dual.snapshot())
                state.update(phase='collecting',collector=collector.state_dict(),next_inner=0,saved_behavior_path=None)
            else:collector=EffectiveGroupCollector.restore(state['collector'])
            boundary=next((x for x in milestones if x>state['raw_cursor']),total)
            while state['phase']=='collecting':
                count=collector.next_count(boundary-state['raw_cursor'])
                if count:
                    local=control.synchronized_call(lambda:collect_batch(behavior,reference,store,service,stream,pop,collector,
                                 state['raw_cursor'],count,calibration,rank,world,failure_dir=output/'failures'),accelerator.device)
                    rows=gather_rows(local)
                    if len(rows)!=count:raise ValueError('Missing raw scene groups cannot disappear from denominator')
                    collector.add_returned(rows,collector.behavior_identity,collector.lambda_snapshot)
                    state['raw_cursor']+=count;state['collector']=collector.state_dict()
                    for _,row in rows:
                        for key,source in [('seconds_rollout','rollout_seconds'),('seconds_scoring','scoring_seconds'),('seconds_reference','reference_seconds'),('seconds_scoring_cpu','scoring_cpu_seconds')]:state[key]+=row[source]
                    if stop_requested(accelerator.device) or (a.checkpoint_after_first_batch and not a.resume):
                        save(f'collect{state["collections"]:06d}_raw{state["raw_cursor"]:06d}');stopped=True;break
                if collector.done or not count or state['raw_cursor']==boundary:
                    summary=control.rank0_result(lambda:summarize_rows(collector.rows,actor.spec,collector.accepted),accelerator.device)
                    state['saturation_streak']=saturation_streak(state['saturation_streak'],summary)
                    control.rank0_call(lambda:atomic_json(output/'collection_summaries'/f'{state["collections"]:06d}.json',summary),accelerator.device)
                    if state['saturation_streak']>=20:
                        state['safety_pause']=dict(reason='95_percent_at_one_sigma_bound_for_20_collections',collection=state['collections'])
                        save(f'sigma_pause_collect{state["collections"]:06d}');stopped=True;break
                    dual_report=update_dual(dual,collector,state['collections'],accelerator.device)
                    state['dual_report']=dual_report;state['accepted_exposures']+=len(collector.accepted)
                    state['collection_LR']=exposure_lr(state['raw_cursor'],total)
                    state['saved_behavior_path']=control.rank0_result(lambda:archive_collection(output,collector,state['collections']),accelerator.device)
                    state.update(phase='updating',collector=collector.state_dict(),next_inner=0)
            if stopped:break
            for inner in range(state['next_inner'],2):
                t0=time.monotonic();logs=[]
                if detailed:
                    before_parameters={name:value.detach().cpu().clone() for name,value in actor.named_parameters()}
                    before_masters=[value.detach().clone() for value in engine.optimizer.single_partition_of_fp32_groups]
                for group in optimizer.param_groups:group['lr']=state['collection_LR']
                used_KL=kl.value
                for micro in range(2):
                    slot=micro*world+rank
                    arguments=control.synchronized_call(lambda:micro_inputs(collector,slot,store,replay_labels,replay_stream,
                                        state['replay_cursor']+slot,accelerator.device),accelerator.device)
                    with accelerator.accumulate(engine):
                        with accelerator.autocast():
                            loss,metrics=engine(**arguments,kl_coefficient=used_KL)
                        if inner==0 and float(metrics['ratio_max_abs_from_one'])>1e-4:
                            raise ValueError('Fresh behavior/current ratio exceeded qualified layout tolerance')
                        accelerator.backward(loss)
                    logs.append({key:float(value) for key,value in metrics.items()})
                state['updates']+=1;state['next_inner']=inner+1;state['replay_cursor']+=16
                state['RL_updates' if collector.accepted else 'replay_reference_only_updates']+=1
                measured=torch.tensor(sum(x['reference_KL_weighted'] for x in logs)/2,device=accelerator.device,dtype=torch.float64)
                dist.all_reduce(measured);measured/=world
                if float(measured)<-1e-6:raise FloatingPointError('Negative reference KL beyond numerical tolerance')
                kl.advance(state['updates'],max(0.,float(measured)),min(16,len(collector.rows)))
                torch.cuda.synchronize();elapsed=time.monotonic()-t0;state['seconds_updates']+=elapsed
                log={key:sum(x[key] for x in logs)/2 for key in logs[0]}
                log.update(monitor.consume());log.update(probe.consume());log.update(optimizer_norm_metrics(engine,2,clip=1.))
                log.update(update=state['updates'],collection=state['collections'],inner=inner+1,rank=rank,
                           raw_exposure=state['raw_cursor'],accepted_exposure=state['accepted_exposures'],
                           raw_groups=len(collector.rows),accepted_groups=len(collector.accepted),
                           LR=[g['lr'] for g in optimizer.param_groups],KL_coefficient=used_KL,
                           lambda_snapshot=collector.lambda_snapshot.tolist(),dual=state['dual_report'],seconds=elapsed,
                           peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                if detailed:
                    parameter_changes={name:int(torch.count_nonzero(value.detach().cpu()!=before_parameters[name]))
                                       for name,value in actor.named_parameters()}
                    master_changes=[int(torch.count_nonzero(value.detach()!=previous))
                                    for value,previous in zip(engine.optimizer.single_partition_of_fp32_groups,before_masters)]
                    if any(count==0 for count in master_changes):
                        raise ValueError('A registered optimizer group did not change its FP32 master state')
                    log['local_FP32_master_changed_elements_by_group']=master_changes
                    control.rank0_call(lambda:atomic_json(output/f'parameter_changes_update{state["updates"]:06d}.json',parameter_changes),accelerator.device)
                    del before_parameters,before_masters
                with (output/f'training_rank{rank}.jsonl').open('a') as stream_log:stream_log.write(json.dumps(log,allow_nan=False)+'\n')
                if rank==0:print(json.dumps(log,allow_nan=False),flush=True)
                if state['next_inner']<2 and (stop_requested(accelerator.device) or (a.stop_after_updates and state['updates']>=a.stop_after_updates)):
                    save(f'inner_update{state["updates"]:06d}');stopped=True;break
            if stopped:break
            state['collections']+=1;state.update(phase='ready',collector=None,next_inner=0)
            at_milestone=state['raw_cursor'] in milestones and state['raw_cursor'] not in state['completed_milestones']
            if at_milestone:
                state['completed_milestones'].append(state['raw_cursor'])
                if not a.qualification:state['pending_evaluation']=state['raw_cursor']
            should_stop=stop_requested(accelerator.device) or bool(a.stop_after_updates and state['updates']>=a.stop_after_updates)
            if at_milestone or state['updates']%a.save_every==0 or should_stop:
                save(f'update{state["updates"]:06d}_raw{state["raw_cursor"]:06d}')
                if at_milestone:
                    export=output/f'export_raw{state["raw_cursor"]:06d}'
                    control.rank0_call(lambda:export_policy(actor,cfg['sft_checkpoint'],export,identity),accelerator.device)
            if at_milestone and not a.qualification and evaluate_pending():should_stop=True
            if should_stop:stopped=True;break
        if not stopped:
            state['phase']='complete'
        if any(p.requires_grad or p.grad is not None for model in (reference,behavior) for p in model.parameters()):
            raise ValueError('Frozen reference or behavior received an update')
        report={key:value for key,value in state.items() if key not in ('collector','dual','KL')}
        report.update(status='STOPPED_AT_SAVED_BOUNDARY' if stopped else 'COMPLETE_FIXED_RAW_BUDGET',rank=rank,host=host,
                      checkpoint=last_checkpoint,identity_sha256=identity['sha256'],wall_seconds_this_invocation=time.monotonic()-started,
                      qualification_only=a.qualification,ODE_evaluation_required=True)
        atomic_json(output/f'result_rank{rank}.json',report)
    except Exception:
        # Do not enter a new collective after a CUDA/communication failure. This
        # is failure evidence, NOT a resumable partially applied Adam boundary.
        failure=output/'failures';failure.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state=cpu_tree(state),last_valid_checkpoint=last_checkpoint,identity=identity,
                        traceback=traceback.format_exc(),resumable=False),failure/f'trainer_rank{rank}.pt')
        raise
    finally:
        service.close();monitor.close()


if __name__=='__main__':main()
