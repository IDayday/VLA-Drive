"""Independent H CPU campaign: early consolidation and real multi-scene batches.

Check/schema/self-check never dispatch. Only ``run`` may start the registered H
native/evaluation helpers. F is read-only comparison evidence, never a worker.
Native Adam updates, actual collected material and F-reference clocks stay
separate, including the Batch32 update1250 pending-inner boundary.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

from .revival_campaign import read, sha, alive, decision as long_decision


ROOT_PATH = Path('/mnt/project/qwen-world-safety-exploration-20261031/improvement_training_20261101')
CONTROL_ROOT = ROOT_PATH.parent/'standard_training_20261101'
SCHEMA = 'world_improvement_campaign_registry_v1'
PROTECTED = {'training-vlawm-tsinghua-zt', 'training-vla-zt'}
ROLES = {'IMPROVEMENT_EARLY_DECAY_G8': 'early_decay',
         'IMPROVEMENT_MULTISCENE32_G8': 'multiscene32',
         'IMPROVEMENT_VERIFIED_RETENTION_G8': 'verified_retention',
         'IMPROVEMENT_FP32_OUTPUT_G8': 'fp32_output',
         'IMPROVEMENT_MIXED_NOISE_G8': 'mixed_noise'}
FACTOR = {'early_decay': 1, 'multiscene32': 4, 'verified_retention': 1, 'fp32_output': 1, 'mixed_noise': 1}
CAPS = {'early_decay': (20000, 30000, 50000), 'multiscene32': (5000, 7500, 12500),
        'verified_retention': (20000, 30000, 50000), 'fp32_output': (20000, 30000, 50000),
        'mixed_noise': (20000, 30000, 50000)}
IMMUTABLE = ('id', 'root', 'source', 'config', 'config_sha256', 'parent_checkpoint',
             'group_size', 'trainer_module', 'improvement_role', 'owner_state')
P0_PROTOCOL = dict(schema='world_fixed50k_warmup1000_cosine_end20k_floor01_v1',
    horizon_updates=50000, warmup_updates=1000, cosine_decay_end_updates=20000, min_lr_ratio=.1)
BATCH_PROTOCOL = dict(schema='world_multiscene32_G8_sourcePPO_query_clock_v1',
    scenes_per_rank=4, group_size=8, world_size=8, inner_updates=4,
    normalization_candidates=256, optimizer_horizon_updates=12500,
    optimizer_warmup_updates=250, initial_lr_ratio=.001, min_lr_ratio=.1,
    source_update_clock_multiplier=4)
RETENTION_PROTOCOL = dict(schema='world_verified_RL_nominal_mean_retention_v1',
    capacity=256, admission_start_update=2000, history_scenes_per_rank=1,
    history_share=.25, min_score=.9, regression_margin=.01, upgrade_margin=1e-6,
    ODE_seed=2101, bank_seed=42, reservoir='uniform_over_verified_admissions')


def gates(role, cap):
    if role not in CAPS or cap not in CAPS[role]:
        raise ValueError('Explicit per-role Adam cap required')
    if role == 'fp32_output':
        return [0, 2000, 5000, 10000]+list(range(12000, cap+1, 2000))
    if role in ('early_decay','verified_retention','mixed_noise'):
        return [2000, 5000, 10000]+list(range(12000, cap+1, 2000))
    return [500, 1250, 2500]+list(range(3000, cap+1, 500))


def budget(role, updates):
    """Collected material is ceil(inner windows), not always multiplier*step."""
    factor = FACTOR[role]
    scenes = ((updates+3)//4)*8*factor
    nominal = factor*updates
    return dict(optimizer_updates=updates, F_reference_updates=nominal,
        source_update_clock_multiplier=factor, next_inner=updates % 4,
        scene_exposures=scenes, candidate_queries=scenes*8,
        F_reference_scene_exposures=nominal*2,
        F_reference_candidate_queries=nominal*16,
        scene_surplus_at_pending_boundary=scenes-nominal*2,
        candidate_surplus_at_pending_boundary=scenes*8-nominal*16,
        candidate_query_scope='fresh on-policy source-PG candidates only',
        additional_official_query_accounting_required=(role=='verified_retention' and updates>2000),
        strict_equal_source_candidate_budget=(updates % 4 == 0),
        strict_equal_candidate_budget=(updates % 4 == 0 and not(role=='verified_retention' and updates>2000)))


def configure(root):
    if Path(root).resolve() != ROOT_PATH:
        raise ValueError('Only explicit independent H root')
    loaded = sys.modules.get('starVLA.rl.qwen_world_safety_exploration.contracts')
    if loaded is not None and loaded.ROOT.resolve() != ROOT_PATH:
        raise RuntimeError('Use a fresh CPU interpreter; another study was imported')
    os.environ.update(WORLD_COMPARE_ROOT=str(ROOT_PATH),
        WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031', CUDA_VISIBLE_DEVICES='')


def validate_registry(value, root=ROOT_PATH, *, pending_source=False):
    import yaml
    if (Path(root).resolve() != ROOT_PATH or Path(value['root']).resolve() != ROOT_PATH
            or value.get('task') != 'qwen_world_safety_exploration_20261031'
            or value.get('schema', SCHEMA) != SCHEMA or value.get('training_population') != 103288):
        raise ValueError('Exact H task/schema/full103288 required')
    runs = value['runs']
    if not 1 <= len(runs) <= len(ROLES) or len({r['id'] for r in runs}) != len(runs):
        raise ValueError('Unique explicitly registered H roles; P0 may launch alone')
    for run in runs:
        role = ROLES.get(run['id'])
        if (role is None or run.get('improvement_role') != role or run.get('group_size') != 8
                or run.get('trainer_module') != 'tools.qwen_world_safety_exploration.train'
                or Path(run['owner_state']).resolve() != ROOT_PATH/'owners/IMPROVEMENT_NATIVE/STATE.json'
                or Path(run['root']).resolve() != ROOT_PATH/'runs'/run['id']
                or run['gates'] != gates(role, run['target_updates'])
                or run.get('continuation_max_updates', CAPS[role][-1]) not in CAPS[role]
                or run['target_updates'] > run.get('continuation_max_updates', CAPS[role][-1])):
            raise ValueError('Wrong H role/native lineage/Adam cadence/cap')
        if sha(run['config']) != run['config_sha256']:
            raise ValueError('Registered scientific config changed')
        if not pending_source and not (Path(run['source'])/'tools/qwen_world_safety_exploration/train.py').is_file():
            raise ValueError('Frozen per-run scientific source is not ready')
        cfg = yaml.safe_load(Path(run['config']).read_text()); research = cfg['research']
        if research.get('arm') != run['id'] or Path(research.get('campaign_root', '')).resolve() != ROOT_PATH:
            raise ValueError('Scientific arm/campaign does not match H registration')
        if role == 'early_decay':
            if research.get('family') != 'consolidation_training' or research.get('consolidation_training') != P0_PROTOCOL:
                raise ValueError('Only frozen early-decay science is allowed')
        elif role=='multiscene32' and (research.get('standard_role') != 'multiscene32'
                or research.get('batch32') != BATCH_PROTOCOL
                or research.get('structured_advantage') != 'world_SafetyEP_multiscene32_group_credit_v1'):
            raise ValueError('Only actual multi-scene32 protocol is allowed')
        elif role=='verified_retention':
            if (research.get('standard_role')!='verified_retention' or research.get('verified_retention')!=RETENTION_PROTOCOL
                    or research.get('family') is not None
                    or any(research.get(k) is not None for k in ['consolidation_training','batch32','standardized_training','revival_training'])
                    or research.get('structured_advantage')!='world_SafetyEP_group_credit_v1'):
                raise ValueError('Only registered verified-retention protocol; no other family')
        elif role == 'fp32_output':
            from starVLA.rl.qwen_world_safety_exploration.precision_recipe import validate_precision_recipe
            validate_precision_recipe(cfg)
        elif role == 'mixed_noise':
            from starVLA.rl.qwen_world_safety_exploration.mixed_exploration_recipe import validate_mixed_recipe
            validate_mixed_recipe(cfg)
        if not run.get('deployment_candidates'):
            raise ValueError('Explicit deployment routes required')
        if any(r['host'] in PROTECTED for r in run['deployment_candidates']):
            raise ValueError('Protected IL hosts cannot be H routes')
    external = value.get('external_control', {})
    if external != dict(registry=str(CONTROL_ROOT/'RUNS.json'), arm='SAFETY_G8_CONTROL', root=str(CONTROL_ROOT), read_only=True):
        raise ValueError('Exact read-only external F control required')
    control_registry = read(CONTROL_ROOT/'RUNS.json')
    control = next(r for r in control_registry['runs'] if r['id'] == 'SAFETY_G8_CONTROL')
    if (control_registry['manifest'] != value['manifest'] or control_registry['observation_data'] != value['observation_data']
            or any(r['parent_checkpoint'] != control['parent_checkpoint'] for r in runs)):
        raise ValueError('H must use the same data and fresh pinned IL70 as F')
    return value


def valid_checkpoint(run, updates, checkpoint, *, check_rank_results=False):
    from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha, fingerprint
    import yaml
    path = Path(checkpoint).resolve(); role = run['improvement_role']
    if not path.is_relative_to(Path(run['root']).resolve()/'checkpoints'):
        raise ValueError('Foreign H native checkpoint')
    seal = read(path/'COMPLETE'); identity = read(path/'identity.json')
    if (seal.get('updates') != updates or seal.get('world') != 8
            or seal.get('next_inner') != updates % 4
            or seal.get('files_sha256') != file_sha(path/'files.json')
            or identity.get('purpose') != 'formal' or identity.get('world') != 8
            or identity.get('recipe', {}).get('adaptation') != 'ATTN_LORA_MEMORY'):
        raise ValueError('Exact sealed eight-rank H boundary required')
    owned_identity=read(Path(run['root'])/'identity.json')
    if fingerprint(identity)!=fingerprint(owned_identity):
        migration=read(run['execution_migration_receipt']) if run.get('execution_migration_receipt') else {}
        if migration.get('old_identity')!=identity or migration.get('new_identity')!=owned_identity:
            raise ValueError('Checkpoint identity differs from own native lineage')
    source=owned_identity.get('execution_identity',{}).get('fields',{}).get('module_root')
    cfg=yaml.safe_load(Path(run['config']).read_text())
    if (not source or Path(source).resolve()!=Path(run['source']).resolve()
            or owned_identity.get('recipe',{}).get('research')!=cfg['research']):
        raise ValueError('Native source/scientific recipe differs from H binding')
    for rank in range(8):
        if not (path/f'rank{rank}.pt').is_file():
            raise ValueError('Missing native rank/RNG inventory')
        if check_rank_results:
            result = read(Path(run['root'])/f'result_rank{rank}.json')
            if result['updates'] != updates or result['checkpoint'] != str(path):
                raise ValueError('Missing actual eight-rank boundary completion')
    return dict(path=str(path), updates=updates, next_inner=seal['next_inner'],
        files_sha256=seal['files_sha256'], identity_sha256=file_sha(path/'identity.json'),
        actual_collection_budget=budget(role, updates))


def checkpoint_at_gate(run, updates):
    state = read(run['owner_state']) if Path(run['owner_state']).exists() else {}
    native = state.get('runs', {}).get(run['id'], {})
    if run['improvement_role'] == 'fp32_output' and updates == 0:
        checkpoint = Path(run['root'])/'checkpoints/update000000_inner0'
        if (checkpoint/'COMPLETE').exists():return valid_checkpoint(run, 0, checkpoint), native
        return None, native
    stage = native.get('stages', {}).get(str(updates), {})
    if not stage.get('complete'):return None, native
    return valid_checkpoint(run, updates, stage['checkpoint']), native


def owner_class():
    from .owner import Owner, StopBeforeDispatch
    from .dispatch import launch, job_state
    from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts, require_dispatch
    from starVLA.rl.qwen_world_safety_exploration.resources import MODEL_PYTHON
    from starVLA.rl.flow_grpo.transactions import atomic_json
    class ImprovementOwner(Owner):
        def __init__(self):
            super().__init__(selected_ids=[r['id'] for r in read(ROOT_PATH/'RUNS.json')['runs']], owner_name='IMPROVEMENT_NATIVE')
            validate_registry(self.registry); self.bindings={}; self.campaign_done=threading.Event()
            for r in self.runs:self.bindings[r['id']]={k:r[k] for k in IMMUTABLE}
            atomic_json(self.directory/'OWNER.json', process_receipt())
        def current(self, arm):
            self.registry = validate_registry(read(ROOT_PATH/'RUNS.json'))
            run = next(r for r in self.registry['runs'] if r['id'] == arm)
            binding = {k:run[k] for k in IMMUTABLE}
            if arm not in self.bindings:self.bindings[arm]=binding
            if binding != self.bindings[arm]:raise ValueError('Frozen H science/native binding changed')
            return run
        def candidate(self, run):
            legal = set(allowed_hosts(admit_long_job=True))
            return super().candidate(dict(run, deployment_candidates=[r for r in run['deployment_candidates'] if r['host'] in legal and r['host'] not in PROTECTED]))
        def run_stage(self, initial, target, resume):
            arm=initial['id']; node=self.stage(arm,target)
            if node.get('complete'):return node['checkpoint']
            while True:
                run=self.current(arm)
                if not node.get('job'):
                    if self.campaign_done.is_set() or (ROOT_PATH/'STOP_REQUESTED').exists():raise StopBeforeDispatch('H coordinator stopped before a new dispatch')
                    adopted=run.get('initial_job',run.get('job'))
                    if adopted and str(adopted) not in node.get('refused_jobs',[]) and str(adopted) not in [x['job'] for x in node.get('cooperative_returns',[])]:
                        dispatch=read(Path(adopted)/'DISPATCH.json'); command=dispatch['command']
                        if int(command[command.index('--stop-after-update')+1]) == target:
                            if command[command.index('--output')+1] != run['root'] or command[command.index('--config')+1] != run['config']:raise ValueError('Foreign initial H job')
                            attempt=int(Path(adopted).name.rsplit('_attempt',1)[1])
                            node.update(job=str(adopted),host=dispatch['host'],adopted=True,attempt=max(attempt,int(node.get('attempt',0))))
                    if not node.get('job'):
                        self.update(arm,status='WAITING_AUTHORIZED_RESOURCE',stage_target_updates=target)
                        chosen=self.candidate(run)
                        if chosen is None:time.sleep(30);continue
                        route,sample=chosen
                        try:require_dispatch(host=route['host'],admit_long_job=True)
                        except PermissionError:
                            if route['host'] not in allowed_hosts(admit_long_job=True):continue
                            raise
                        latest=Path(run['root'])/'latest.json'
                        if latest.exists():
                            saved=read(latest);resume=saved['checkpoint']
                            if saved['updates']>=target:raise ValueError('Unadopted native progress at/past gate; explicit review required')
                        attempt=int(node.get('attempt',0))+1; name=f'{arm}/formal_to{target:06}_attempt{attempt:03}'
                        command=[MODEL_PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','-m',run['trainer_module'],
                            '--config',run['config'],'--manifest',self.registry['manifest'],'--il-checkpoint',run['parent_checkpoint'],
                            '--data',self.registry['observation_data'],'--parent-selection',self.registry['parent_selection'],
                            '--output',run['root'],'--stop-after-update',str(target),'--save-every',str(run.get('save_every',500))]
                        if resume:
                            command+=['--resume',resume]
                            if run.get('execution_migration_receipt'):command+=['--migration-receipt',run['execution_migration_receipt']]
                        try:job=launch(route['host'],run.get('execution_source',run['source']),command,name,gpus='0,1,2,3,4,5,6,7',expected_uuids=[g['uuid'] for g in sample['gpus']],share_minimum=route['minimum_free_MiB'],lease_slot=route['lease_slot'])
                        except Exception:
                            node.update(attempt=attempt,job=str(ROOT_PATH/'jobs'/name),dispatch_error=traceback.format_exc());self.update(arm,status='DISPATCH_TRANSPORT_REVIEW',job=node['job']);raise
                        node.update(attempt=attempt,job=str(job),host=route['host'],resume=resume)
                receipt=job_state(node['job']);latest_path=Path(run['root'])/'latest.json'
                updates=read(latest_path)['updates'] if latest_path.exists() else 0
                self.update(arm,status='TRAINING' if receipt.get('status')=='RUNNING' else receipt.get('status','STARTING'),updates=updates,job=node['job'],host=node.get('host'),stage_target_updates=target,nominal_F_reference_updates=FACTOR[run['improvement_role']]*updates)
                if receipt.get('status')=='REFUSED_RESOURCE':
                    if receipt.get('returncode')!=75 or receipt.get('child_not_started') is not True:raise RuntimeError('Ambiguous refusal: no scientific retry')
                    node.setdefault('refused_jobs',[]).append(node.pop('job'));time.sleep(30);continue
                if 'returncode' not in receipt:time.sleep(10);continue
                if receipt['returncode']!=0:raise RuntimeError('Scientific/transport failure retained; no automatic retry: '+node['job'])
                latest=read(latest_path);cp=valid_checkpoint(run,latest['updates'],latest['checkpoint'],check_rank_results=True)
                if latest['updates']!=target:
                    if (ROOT_PATH/'STOP_REQUESTED').exists():raise StopBeforeDispatch('H cooperative checkpoint retained')
                    own_stop=Path(node['job'])/'STOP_REQUESTED'
                    migration_due=node.get('host') not in allowed_hosts(admit_long_job=True) or receipt.get('migration_required') is True
                    if latest['updates']<target and own_stop.exists() and migration_due:
                        node.setdefault('cooperative_returns',[]).append(dict(interrupted_job=node['job'],job=node.pop('job'),checkpoint=cp,stop_file_sha256=sha(own_stop),reason='HOST_AUTHORIZATION_BOUNDARY',migration_required=receipt.get('migration_required')))
                        resume=cp['path'];self.update(arm,status='WAITING_AUTHORIZED_MIGRATION',checkpoint=resume);continue
                    raise ValueError('Native completion outside registered H gate')
                node.update(complete=True,checkpoint=cp['path'],budget=cp['actual_collection_budget'],end_unix=time.time())
                self.update(arm,status='STAGE_COMPLETE',updates=target,checkpoint=cp['path'],actual_collection_budget=node['budget']);return cp['path']
        def run_arm(self, initial):
            arm=initial['id'];resume=None
            try:
                while True:
                    run=self.current(arm)
                    if run.get('improvement_disposition'):self.update(arm,status='COMPLETED_AT_REGISTERED_CAP',disposition=run['improvement_disposition']);return
                    for target in run['gates']:
                        # Native0 is saved by the first normal invocation. It
                        # is an evaluation gate, never a zero-budget trainer.
                        if target != 0:resume=self.run_stage(run,target,resume)
                    if run['target_updates']>=run.get('continuation_max_updates',CAPS[run['improvement_role']][-1]):self.update(arm,status='COMPLETED_AT_REGISTERED_CAP');return
                    self.update(arm,status='WAITING_COMPLETE_LONG_GATE_REVIEW')
                    if self.campaign_done.wait(15):return
            except StopBeforeDispatch:self.update(arm,status='STOPPED_AT_COOPERATIVE_BOUNDARY')
            except Exception:self.update(arm,status='FAILED_REVIEW_REQUIRED',error=traceback.format_exc())
    return ImprovementOwner


def process_receipt(**extra):
    return dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=__file__,source_sha256=sha(__file__),started_unix=time.time(),**extra)


def evaluation_class():
    from . import eval_owner
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from starVLA.rl.qwen_world_safety_exploration.contracts import file_sha
    eval_owner.checkpoint_at_gate=checkpoint_at_gate  # This H-only CPU process; old sources/processes untouched.
    class ImprovementEvaluation(eval_owner.EvaluationOwner):
        def __init__(self):
            self.registry_path=ROOT_PATH/'RUNS.json';self.directory=ROOT_PATH/'owners/IMPROVEMENT_EVAL';self.directory.mkdir(parents=True,exist_ok=True)
            self.owner_lock=(self.directory/'owner.lock').open('a');fcntl.flock(self.owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.state_path=self.directory/'STATE.json';self.state=read(self.state_path) if self.state_path.exists() else dict(schema=eval_owner.SCHEMA,runs={},evaluations={})
            self.max_shards=32;self.max_workers=64;self.minimum=32768
            self.claims_path=ROOT_PATH/'resources/EVALUATION_CLAIMS.json';self.claims_path.parent.mkdir(parents=True,exist_ok=True)
            self.registry=self.registry_now();self.initial_bindings={r['id']:self.binding(r) for r in self.registry['runs']}
            self.state.update(schema=eval_owner.SCHEMA,status='RUNNING',registry=str(self.registry_path),max_shards=32,max_workers=64,minimum_free_MiB=32768,unknown_or_partial_results_have_no_score=True)
            atomic_json(self.directory/'OWNER.json',process_receipt(registry=str(self.registry_path)));self.flush()
        def registry_now(self):return validate_registry(super().registry_now())
        def samples(self):return [(host,s) for host,s in super().samples() if host not in PROTECTED]
    return ImprovementEvaluation


def run_review(args):
    from .review import Review,BASELINES
    from .notify import refresh
    from tools.qwen_world_research.evaluate import read_sealed
    from starVLA.rl.qwen_world_safety_exploration.contracts import fingerprint
    from starVLA.rl.flow_grpo.transactions import atomic_json
    review=Review()
    while True:
        registry=validate_registry(read(ROOT_PATH/'RUNS.json'));pending=[];published=[];ready=0
        for run in registry['runs']:
            for u in run['gates']:
                ev=ROOT_PATH/f"evaluations/{run['id']}_u{u:06}/NAVTEST";summary=ev/'SUMMARY.json'
                if not summary.exists():pending.append(str(ev));continue
                value=read_sealed(summary)
                if value.get('status')!='COMPLETE' or value.get('scenes')!=12146 or value.get('invalid_scenes')!=0:raise ValueError('Only complete H NAVTEST')
                ready+=1;point=review.out/f"{run['id']}_u{u:06}.json";cu=FACTOR[run['improvement_role']]*u
                if not point.exists():
                    payload=dict(schema='world_improvement_full_population_review_v1',status='COMPLETE',arm=run['id'],updates=u,PDMS=value['PDMS'],evaluation=str(ev),summary_sha256=value['sha256'],actual_collection_budget=budget(run['improvement_role'],u),comparisons={name:review.compare(ev,path,label=name) for name,path in BASELINES.items()},NAVTEST_never_enters_training_selection=True)
                    payload['sha256']=fingerprint(payload);atomic_json(point,payload);published.append(str(point))
                if u == 0:continue  # Initial execution precision vs IL70, no fictitious F-Control0.
                base=CONTROL_ROOT/f'evaluations/SAFETY_G8_CONTROL_u{cu:06}/NAVTEST';paired_path=review.out/f"{run['id']}_{u}_vs_SAFETY_G8_CONTROL_{cu}.json"
                if paired_path.exists():continue
                if not (base/'SUMMARY.json').exists():pending.append(f'READ_ONLY_F_CONTROL_u{cu:06}');continue
                control=read_sealed(base/'SUMMARY.json')
                if control.get('status')!='COMPLETE' or control.get('scenes')!=12146 or control.get('invalid_scenes')!=0:raise ValueError('External F control is partial')
                comparison=review.compare(ev,base,label='READ_ONLY_F_CONTROL');material=budget(run['improvement_role'],u)
                targetIL=review.compare(ev,BASELINES['IL70'],label='IL70')['result'];controlIL=review.compare(base,BASELINES['IL70'],label='IL70')['result']
                high=targetIL['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']-controlIL['fixed_0p9_cohorts']['fixed_cohorts']['high']['mean_delta_PDMS']
                contract=('same_source_PG_candidates_and_scenes_additional_ODE_history_queries_separate' if material['additional_official_query_accounting_required'] else 'same_candidate_queries_and_scene_exposures_different_Adam_count' if material['strict_equal_candidate_budget'] else 'nominal_4x_clock_with_explicit_pending_material_surplus')
                payload=dict(schema='world_improvement_external_control_comparison_v1',status='COMPLETE',arm=run['id'],updates=u,reference_arm='SAFETY_G8_CONTROL',reference_updates=cu,external_control_root=str(CONTROL_ROOT),comparison=comparison,fixed_IL70_high_advantage_PDMS=high,actual_collection_budget=material,budget_contract=contract,F_evaluation_or_training_launched_here=False)
                payload['sha256']=fingerprint(payload);atomic_json(paired_path,payload);published.append(str(paired_path))
        state=dict(status='WAITING_COMPLETE_NAVTEST' if pending else 'COMPLETE_AT_REGISTERED_GATES',ready=ready,pending=pending,published=published,heartbeat_unix=time.time(),external_control_read_only=True,no_training_mutations=True)
        atomic_json(review.out/'STATE.json',state)
        if published:refresh(review.out/'STATE.json','H full paired evidence published')
        if not args.watch or not pending:return state
        if (review.out/'STOP_OWNER').exists():atomic_json(review.out/'OWNER_STOPPED.json',process_receipt());return
        time.sleep(max(10,args.interval))


def run_campaign(args):
    from starVLA.rl.flow_grpo.transactions import atomic_json
    from .notify import refresh
    from .review import Review
    folder=ROOT_PATH/'owners/IMPROVEMENT_CAMPAIGN';folder.mkdir(parents=True,exist_ok=True)
    guard=(folder/'owner.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=owner_class()();state=read(folder/'STATE.json') if (folder/'STATE.json').exists() else dict(decisions={},helpers={})
    atomic_json(folder/'OWNER.json',process_receipt(allow_continuation=args.allow_continuation));source=Path(__file__).resolve().parents[2]
    def helper(name,directory,command):
        directory.mkdir(parents=True,exist_ok=True);receipt=directory/'HELPER_LAUNCH.json'
        if (directory/'STOP_OWNER').exists():return
        if receipt.exists():
            previous=read(receipt)
            if alive(previous):return
            atomic_json(directory/f"HELPER_LAUNCH_{previous['pid']}_{previous['start_ticks']}.json",previous)
        with (directory/'cpu.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u','-m','tools.qwen_world_safety_exploration.improvement_campaign',*command],cwd=source,env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,start_ticks=int(Path('/proc',str(process.pid),'stat').read_text().split()[21]),host=socket.gethostname(),source=str(source),command=command,at_unix=time.time());atomic_json(receipt,value);state['helpers'][name]=value
    def point(run,u):
        path=ROOT_PATH/f"analysis/PILOT_COMPARISONS/{run['id']}_{u}_vs_SAFETY_G8_CONTROL_{FACTOR[run['improvement_role']]*u}.json"
        if not path.exists():return None
        value=read(path)
        if value.get('status')!='COMPLETE':return None
        return dict(value['comparison']['result'],fixed_IL70_high_advantage_PDMS=value.get('fixed_IL70_high_advantage_PDMS',0))
    def own_trend(run,current,previous):
        path=ROOT_PATH/f"analysis/CONTINUATION/{run['id']}_{current}_vs_own_{previous}.json"
        if path.exists():return read(path)['comparison']['result']
        target=ROOT_PATH/f"evaluations/{run['id']}_u{current:06}/NAVTEST";base=ROOT_PATH/f"evaluations/{run['id']}_u{previous:06}/NAVTEST"
        if not (target/'SUMMARY.json').exists() or not (base/'SUMMARY.json').exists():return None
        review=Review.__new__(Review);review.cache={};comparison=review.compare(target,base,label='own_previous_long_gate')
        atomic_json(path,dict(status='COMPLETE',comparison=comparison,at_unix=time.time()));return comparison['result']
    def update_budget(run,choice):
        cap=run['target_updates'];key=f"{run['id']}:{cap}";path=folder/f"DECISION_{run['id']}_u{cap:06}.json"
        atomic_json(path,dict(schema='world_improvement_conditional_continuation_v1',arm=run['id'],from_cap=cap,physical_reference_updates=FACTOR[run['improvement_role']]*cap,decision=choice,at_unix=time.time()))
        evdir=ROOT_PATH/'owners/IMPROVEMENT_EVAL';stop=evdir/'STOP_OWNER';pending=state.setdefault('pending_budget_updates',{})
        if key not in pending:
            if stop.exists():raise RuntimeError('External H evaluator STOP_OWNER requires review')
            pending[key]=dict(reason='H_budget_binding_CPU_adoption',decision_key=key,nonce=str(time.time_ns()),host=socket.gethostname())
            atomic_json(folder/'STATE.json',state)
        marker=pending[key]
        if marker['host']!=socket.gethostname():raise RuntimeError('Foreign-host adoption requires explicit review')
        if stop.exists() and read(stop)!=marker:raise RuntimeError('External H evaluator stop marker differs')
        if not stop.exists():atomic_json(stop,marker)
        launch=evdir/'HELPER_LAUNCH.json'
        if launch.exists() and alive(read(launch)):return
        with (ROOT_PATH/'RUNS.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);value=validate_registry(read(ROOT_PATH/'RUNS.json'));live=next(r for r in value['runs'] if r['id']==run['id'])
            if live['target_updates']==cap:
                if choice['next_cap'] is None:live['improvement_disposition']=choice['action']
                else:live.update(target_updates=choice['next_cap'],gates=gates(live['improvement_role'],choice['next_cap']))
                live['continuation_receipt']=str(path);atomic_json(ROOT_PATH/'RUNS.json',value)
        if read(stop)!=marker:raise RuntimeError('H evaluator stop changed during budget adoption')
        stop.unlink();pending.pop(key);state['decisions'][key]=dict(path=str(path),**choice);refresh(path,'H conditional native continuation')
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={}
        try:
            while True:
                registry=validate_registry(read(ROOT_PATH/'RUNS.json'))
                for run in registry['runs']:
                    if run['id'] not in futures:futures[run['id']]=pool.submit(owner.run_arm,run)
                helper('evaluation',ROOT_PATH/'owners/IMPROVEMENT_EVAL',['evaluate','--interval','30'])
                helper('review',ROOT_PATH/'analysis/PILOT_COMPARISONS',['review','--watch','--interval','60'])
                for run in registry['runs']:
                    cap=run['target_updates'];role=run['improvement_role'];key=f"{run['id']}:{cap}";node=owner.state['runs'].get(run['id'],{})
                    if not node.get('stages',{}).get(str(cap),{}).get('complete') or key in state['decisions'] or run.get('improvement_disposition'):continue
                    if not args.allow_continuation or cap>=run.get('continuation_max_updates',CAPS[role][-1]):continue
                    previous=10000//FACTOR[role] if cap==CAPS[role][0] else CAPS[role][CAPS[role].index(cap)-1]
                    current,prior,trend=point(run,cap),point(run,previous),own_trend(run,cap,previous)
                    if not current or not prior or not trend:continue
                    choice=long_decision(FACTOR[role]*cap,role,current,prior,trend)
                    if choice.get('next_cap') is not None:choice['next_cap']//=FACTOR[role]
                    update_budget(run,choice)
                done=all(f.done() for f in futures.values())
                evpath=ROOT_PATH/'owners/IMPROVEMENT_EVAL/STATE.json';evstate=read(evpath) if evpath.exists() else {}
                reviewpath=ROOT_PATH/'analysis/PILOT_COMPARISONS/STATE.json';reviewstate=read(reviewpath) if reviewpath.exists() else {}
                state.update(status='TRAINING_COMPLETE_EVALUATIONS_PENDING' if done else 'RUNNING',runs=owner.state['runs'],heartbeat_unix=time.time(),host=socket.gethostname(),external_control_read_only=str(CONTROL_ROOT),no_early_score_elimination=True)
                atomic_json(folder/'STATE.json',state)
                if (ROOT_PATH/'STOP_REQUESTED').exists():
                    for n in owner.state['runs'].values():
                        if n.get('job') and not n.get('status','').startswith(('FAILED','STOPPED','COMPLETED')):(Path(n['job'])/'STOP_REQUESTED').touch(exist_ok=True)
                if not args.allow_continuation:
                    if all(n.get('stages',{}).get(str(next(r['target_updates'] for r in registry['runs'] if r['id']==arm)),{}).get('complete') or n.get('status','').startswith('FAILED') for arm,n in owner.state['runs'].items()) and len(owner.state['runs'])==len(registry['runs']):owner.campaign_done.set()
                if done and evstate.get('status') in ('COMPLETE_AT_REGISTERED_GATES','FINISHED_WITH_REVIEW_REQUIRED') and (reviewstate.get('status')=='COMPLETE_AT_REGISTERED_GATES' or evstate.get('status')=='FINISHED_WITH_REVIEW_REQUIRED'):break
                time.sleep(30)
        finally:owner.campaign_done.set()
    state.update(status='COMPLETE_AT_APPROVED_BUDGETS',finished_unix=time.time());atomic_json(folder/'STATE.json',state);refresh(folder/'STATE.json','H current approved budgets finished')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=('schema','self-check','check','run','evaluate','review'));parser.add_argument('--root',default=str(ROOT_PATH));parser.add_argument('--registry');parser.add_argument('--allow-continuation',action='store_true');parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=float,default=30);parser.add_argument('--max-shards',type=int,default=32);parser.add_argument('--max-workers',type=int,default=64);args=parser.parse_args()
    if args.mode=='schema':
        print(json.dumps(dict(schema=SCHEMA,root=str(ROOT_PATH),roles=ROLES,per_role_caps=CAPS,native_owner=str(ROOT_PATH/'owners/IMPROVEMENT_NATIVE/STATE.json'),external_control=dict(registry=str(CONTROL_ROOT/'RUNS.json'),arm='SAFETY_G8_CONTROL',root=str(CONTROL_ROOT),read_only=True),required_registry_fields=['task','root','training_population','runs','manifest','observation_data','parent_selection','evaluation','external_control'],required_run_fields=list(IMMUTABLE)+['architecture_id','target_updates','gates','deployment_candidates'],gate_examples={r:gates(r,CAPS[r][0]) for r in CAPS},batch1250_material=budget('multiscene32',1250)),indent=2));return
    if args.mode=='self-check':
        assert gates('early_decay',20000)==[2000,5000,10000,12000,14000,16000,18000,20000]
        assert gates('multiscene32',5000)==[500,1250,2500,3000,3500,4000,4500,5000]
        assert budget('multiscene32',1250)['scene_exposures']==10016 and budget('multiscene32',1250)['candidate_queries']==80128
        assert budget('multiscene32',1250)['next_inner']==2 and not budget('multiscene32',1250)['strict_equal_candidate_budget']
        assert budget('multiscene32',5000)['candidate_queries']==budget('early_decay',20000)['candidate_queries']==320000
        assert gates('verified_retention',20000)==gates('early_decay',20000)
        assert budget('verified_retention',5000)['additional_official_query_accounting_required'] and not budget('verified_retention',5000)['strict_equal_candidate_budget']
        print('PASS: per-role gates, native pending1250 and explicit query/scheduler clocks; no dispatch');return
    configure(args.root)
    if args.registry and Path(args.registry).resolve()!=ROOT_PATH/'RUNS.json':raise ValueError('Only H registry')
    registry=validate_registry(read(ROOT_PATH/'RUNS.json'),pending_source=args.mode=='check')
    if args.mode=='check':
        owner_class();evaluation_class()
        print(json.dumps(dict(status='CPU_REGISTRY_AND_H_OWNERS_IMPORT_COMPATIBLE_NO_DISPATCH',root=str(ROOT_PATH),arms=[r['id'] for r in registry['runs']],initial_caps=[r['target_updates'] for r in registry['runs']])));return
    if args.mode=='evaluate':
        if args.max_shards!=32 or args.max_workers!=64 or args.interval<10:raise ValueError('Qualified up-to32 shards/64 workers and bounded polling required')
        return evaluation_class()().run(interval=args.interval)
    if args.mode=='review':return run_review(args)
    run_campaign(args)


if __name__=='__main__':main()
