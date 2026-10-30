"""Adopt the five existing RL jobs and advance their bounded research campaign.

Only administrative state and new, sealed adaptive stage censuses are written.
The scientific runtime/snapshots are never edited. A first-stage job must already
be registered; this owner cannot accidentally launch a duplicate first stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

DEFAULT_ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
FIRST_ARMS = ('PPO_HARD', 'PPO_FULL', 'PPO_ADAPTIVE', 'CISPO_FULL', 'NFT_FULL', 'PPO_BATCHSTD', 'PPO_SAFE_COMFORT', 'PPO_ANCHOR40')
GATES = (1000, 5000, 10000, 20000, 35000, 50000)
FIRST_PRIORITY_GATE = 5000
SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')
SCHEMA = 'world_rl_optimization_campaign_owner_v1'


class ArmStoppedByResearchDecision(Exception):
    """A separately recorded allocation decision, never a scientific failure."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}.{threading.get_ident()}')
    with temp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False, default=str)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def job_path(value):
    if isinstance(value, (str, Path)):
        return str(value)
    for key in ('receipt', 'job', 'path', 'receipt_root', 'job_root'):
        if value.get(key):
            return str(value[key])
    raise ValueError('Job descriptor has no receipt path: '+str(value))


def result_code(result):
    value = result.get('returncode', result.get('exit_code'))
    if value is None:
        raise ValueError('Job result has no explicit return code')
    return int(value)


def native_checkpoint(run_root, expected_update=None):
    """Read the native atomic pointer/seal, leaving tensor verification to resume."""
    run_root = Path(run_root)
    candidates = []
    for name in ('progress.json', 'latest.json', 'LATEST.json'):
        record = read_json(run_root/name, {})
        if record.get('checkpoint'):
            candidates.append(Path(record['checkpoint']))
    if expected_update is not None:
        candidates.insert(0, run_root/'checkpoints'/f'update{expected_update:06}_inner0')
    available = []
    for path in dict.fromkeys(candidates):
        seal = read_json(path/'COMPLETE', {})
        if 'updates' not in seal:
            continue
        if expected_update is not None and int(seal['updates']) != expected_update:
            continue
        if expected_update is not None and int(seal.get('next_inner', 0)) != 0:
            continue
        available.append((int(seal['updates']), path, seal))
    if not available:
        raise FileNotFoundError('No complete native checkpoint at requested boundary: '+str(expected_update))
    update, path, seal = max(available, key=lambda value:value[0])
    return dict(path=str(path.resolve()), updates=update, phase=seal.get('phase'),
                next_inner=seal.get('next_inner'), files_sha256=seal.get('files_sha256'))


def metric(summary, name):
    means = summary.get('means', {})
    if name in means:
        return 100*float(means[name])
    aliases = {'no_at_fault_collisions':'NC', 'drivable_area_compliance':'DAC',
               'time_to_collision_within_bound':'TTC'}
    if aliases.get(name) in summary:
        return float(summary[aliases[name]])
    raise ValueError('Completed summary lacks safety metric '+name)


def select_survivors(summaries, *, score_tolerance=.25, safety_tolerance=.25, max_challengers=2):
    """Declared point-estimate compute allocation, explicitly using observed Test.

    This is not a statistical claim of algorithmic non-inferiority. One training
    seed is being screened. The PPO_FULL control is always retained.
    """
    if 'PPO_FULL' not in summaries:
        raise ValueError('A complete matched PPO_FULL control is required before selection')
    control = summaries['PPO_FULL']; control_score = float(control['PDMS'])
    decisions = {}; eligible = []
    for arm, summary in summaries.items():
        score_delta = float(summary['PDMS'])-control_score
        safety_delta = {k:metric(summary,k)-metric(control,k) for k in SAFETY}
        acceptable = score_delta >= -score_tolerance and min(safety_delta.values()) >= -safety_tolerance
        decisions[arm] = dict(PDMS=float(summary['PDMS']), delta_vs_PPO_FULL_PDMS=score_delta,
                              safety_delta_points=safety_delta,
                              point_noninferior_for_compute_allocation=acceptable,
                              clearly_below_control_gt1point=score_delta < -1.)
        if arm != 'PPO_FULL' and acceptable:
            eligible.append(arm)
    eligible.sort(key=lambda arm:(float(summaries[arm]['PDMS']),
                                 min(decisions[arm]['safety_delta_points'].values())), reverse=True)
    selected = ['PPO_FULL']+eligible[:max_challengers]
    for arm, decision in decisions.items():
        decision['selected_for_next_gate'] = arm in selected
        decision['reason'] = ('mandatory matched control' if arm=='PPO_FULL' else
                              'best eligible challenger' if arm in selected else
                              'point PDMS more than1 below control' if decision['clearly_below_control_gt1point'] else
                              'held by predeclared score/safety screen or parallel compute limit')
    return selected, dict(selected=selected, decisions=decisions,
                          Test_selection_used=True, training_scene_selection_uses_NAVTEST=False,
                          score_tolerance_PDMS_points=score_tolerance,
                          safety_tolerance_percentage_points=safety_tolerance,
                          maximum_noncontrol_survivors=max_challengers,
                          interpretation='point-estimate budget allocation; not a statistical non-inferiority claim')


def select_at_gate(gate, summaries, *, score_tolerance=.25, safety_tolerance=.25):
    """Scores set priority; only separate research receipts end an arm.

    The later continuation policy supersedes the original max2/.25-point cap.
    No score, including a small or unconfirmed deficit, ends an experiment.
    At1k even priority is independent of scores.
    """
    selected=[arm for arm in FIRST_ARMS if arm in summaries]
    score_priority=gate>=FIRST_PRIORITY_GATE
    if score_priority:
        selected=([arm for arm in selected if arm=='PPO_FULL']+
                  sorted((arm for arm in selected if arm!='PPO_FULL'),
                         key=lambda arm:float(summaries[arm]['PDMS']),reverse=True))
    return selected,dict(selected=selected,priority_order=selected,
        diagnostic_only=True,automatic_pruning=False,first_pruning_gate=None,
        first_budget_priority_gate=FIRST_PRIORITY_GATE,Test_selection_used=score_priority,
        training_scene_selection_uses_NAVTEST=False,
        continuation_policy_supersedes_max2_budget_cap=True,
        decisions={arm:dict(selected_for_next_gate=True,resource_priority=index,
            reason='healthy experiment retains its planned budget; score informs resource priority only')
            for index,arm in enumerate(selected)},
        interpretation='default budget retained unless a separate evidence-based research stop receipt supersedes it')


class CampaignOwner:
    def __init__(self, root, *, score_tolerance=.25, safety_tolerance=.25,
                 resource_retries=2, dispatcher=None):
        self.root = Path(root).resolve()
        self.owner_root = self.root/'owners/CAMPAIGN'; self.owner_root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.owner_root/'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.capacity_changed = threading.Condition(self.mutex)
        self.capacity_epoch = 0
        self.dispatcher = dispatcher or importlib.import_module('tools.qwen_world_research.dispatch')
        self.score_tolerance = score_tolerance; self.safety_tolerance = safety_tolerance
        self.resource_retries = resource_retries
        self.runs = {row['id']:row for row in read_json(self.root/'RUNS.json')['runs']}
        for arm in FIRST_ARMS:
            if arm not in self.runs or not self.runs[arm].get('job'):
                raise ValueError('First-stage job must already be dispatched; owner will not duplicate '+arm)
        self.parent = str(self.root/'IL_PARENT')
        self.data = str(self.root/'data/NAVTRAIN')
        self.panel = str(self.root/'data/panels/TRAIN_DIAGNOSTIC_001024.json')
        self.navtest_data = '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST'
        self.navtest_manifest = str(self.root/'data/panels/NAVTEST_012146.json')
        if not Path(self.navtest_manifest).exists():
            population=read_json('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
            body=dict(schema='world_rl_research_evaluation_census_v1',cohort='NAVTEST',
                      count=12146,tokens=population['navtest_tokens'],
                      data_manifest_sha256=sha_file(Path(self.navtest_data)/'manifest.json'),
                      train_feedback_allowed=False,used_for_actor_update=False,
                      selection_rule='entire canonical fixed12146 NAVTEST; observed-Test disclosure retained')
            atomic_json(self.navtest_manifest,dict(body,sha256=fingerprint(body)))
        self.state = read_json(self.owner_root/'STATE.json', {})
        if self.state and self.state.get('schema') != SCHEMA:
            raise ValueError('Existing campaign owner state has a different schema')
        self.state.setdefault('schema',SCHEMA);self.state.setdefault('runs',{})
        self.state.setdefault('results',[]);self.state.setdefault('events',[])
        self.state.setdefault('selections',{});self.state.setdefault('started_at',now())
        for arm in FIRST_ARMS:
            arm_state = self.state['runs'].setdefault(arm,dict(id=arm,status='ADOPTING_EXISTING_FIRST_STAGE',stages={}))
            arm_state.setdefault('stages',{})
            arm_state['run_root'] = self.runs[arm]['root']
            arm_state['source'] = self.runs[arm].get('source')
        owner = dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                     source_file=str(Path(__file__).resolve()),source_file_sha256=sha_file(__file__),
                     started_at=now(),scope='adopt existing jobs; independent arm budgets; never edit training snapshots')
        atomic_json(self.owner_root/'OWNER.json',owner)
        self.state['owner'] = owner
        self.state['status'] = 'RUNNING'
        self.state['selection_policy'] = dict(score_tolerance_PDMS_points=score_tolerance,
            safety_tolerance_percentage_points=safety_tolerance,Test_selection_used=True,
            max_challengers=None,control='PPO_FULL',gates=list(GATES),
            automatic_pruning=False,continuation_policy_supersedes_max2_budget_cap=True,
            first_pruning_gate=None,first_budget_priority_gate=FIRST_PRIORITY_GATE,
            first_gate_diagnostic_only=True,
            first_gate_Test_selection_used=False,default_budget_updates=50000,
            evidence_based_arm_stops=True,arm_decisions=str(self.owner_root/'arm_decisions'),
            later_user_futility_instruction_supersedes_unconditional_continuation=True,
            first_gate_global_barrier=False,first_gate_waits_for_NAVTEST=False,
            first_gate_results_required_before_first_priority=False,
            independent_arm_pipelines=True,global_training_barriers=False,
            periodic_NAVTEST_async=True,nonadaptive_TRAIN_diagnostics_async=True,
            adaptive_waits_for_own_TRAIN_only=True,
            correction_registered_before_first_complete_new_1k_NAVTEST=True)
        self.eval_pool = ThreadPoolExecutor(max_workers=12,thread_name_prefix='research_eval')
        self.feedback_pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix='adaptive_feedback')
        self.evaluation_futures = {}
        self.baseline_future = None
        self.flush()

    def flush(self):
        with self.mutex:
            self.state['updated_at'] = now()
            atomic_json(self.owner_root/'STATE.json',self.state)
            for arm, value in self.state['runs'].items():
                atomic_json(self.root/'owners'/arm/'STATE.json',value)

    def event(self, event, **details):
        with self.mutex:
            self.state['events'].append(dict(at=now(),event=event,**details))
            self.state['events'] = self.state['events'][-300:]
            self.flush()

    def update_arm(self, arm, **values):
        with self.mutex:
            self.state['runs'][arm].update(values)
            self.flush()

    def arm_stop_decision(self, arm):
        path=self.owner_root/'arm_decisions'/f'{arm}.json'
        decision=read_json(path,{})
        if not decision:return None
        if (decision.get('arm')!=arm or decision.get('disposition')!='STOP_FUTILITY'
                or not decision.get('reason') or not decision.get('evidence')
                or not all(Path(p).is_file() for p in decision['evidence'])):
            raise ValueError('Incomplete evidence-based arm decision: '+str(path))
        return dict(decision,path=str(path))

    def stop_arm(self, arm, decision):
        """Cooperatively seal this arm, release its exact job and retain results."""
        current=next(r for r in read_json(self.root/'RUNS.json')['runs'] if r['id']==arm)
        job=Path(current['job'])
        self.update_arm(arm,status='STOPPING_BY_RESEARCH_DECISION',research_decision=decision)
        if not (job/'RESULT.json').exists():
            (job/'STOP_REQUESTED').touch()
            self.event('research_arm_cooperative_stop_requested',arm=arm,job=str(job),decision=decision['path'])
        result=self.dispatcher.wait_job(str(job))
        self.dispatcher.wait_transport_exit(str(job))
        if result_code(result)!=0 and not (result_code(result)==75 and result.get('child_not_started') is True):
            raise RuntimeError('Research stop did not exit cleanly; retain failure: '+json.dumps(result)[:1500])
        checkpoint=native_checkpoint(current['root'])
        receipt=dict(arm=arm,status='STOPPED_FUTILITY',decision=decision['path'],job=str(job),
            result=result,checkpoint=checkpoint,at=now(),native_job_exited=True,
            optimizer_state_retained=True,training_seed_unchanged=True)
        atomic_json(self.owner_root/'arm_decisions'/f'{arm}.STOPPED.json',receipt)
        self.update_arm(arm,status='STOPPED_FUTILITY',updates=checkpoint['updates'],
                        checkpoint=checkpoint['path'],research_stop_receipt=receipt)
        self.update_run_registry(arm,status='STOPPED_FUTILITY',completed_updates=checkpoint['updates'],
                                 last_checkpoint=checkpoint['path'],research_decision=decision['path'])
        self.capacity_released()
        self.event('arm_stopped_futility',arm=arm,updates=checkpoint['updates'],decision=decision['path'])

    def capacity_released(self):
        """A real adopted worker exited; wake dispatchers without journal polls."""
        with self.capacity_changed:
            self.capacity_epoch += 1
            self.capacity_changed.notify_all()

    def wait_capacity(self, node, *, status, label, arm=None, since=None):
        """Resource refusal is a wait state, independent of scientific failure."""
        with self.capacity_changed:
            epoch=self.capacity_epoch if since is None else since
            node['status']=status
            if arm:self.state['runs'][arm]['status']=status
            self.event('resource_capacity_wait',label=label,status=status)
            released=self.capacity_changed.wait_for(lambda:self.capacity_epoch!=epoch,timeout=1800)
        if not released:self.event('capacity_wait_heartbeat',label=label,status=status)

    def update_run_registry(self, arm, **values):
        # dispatch.py uses the same cross-thread/process lock; preserve every
        # unrelated row/field rather than rewriting the registration snapshot.
        with (self.root/'RUNS.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            registry = read_json(self.root/'RUNS.json')
            row = next(r for r in registry['runs'] if r['id']==arm)
            row.update(values)
            atomic_json(self.root/'RUNS.json',registry)
            self.runs[arm] = dict(row)

    def stage(self, arm, target):
        with self.mutex:
            return self.state['runs'][arm]['stages'].setdefault(str(target),{})

    def run_evaluation(self, label, checkpoint, *, cohort, node, baseline=None):
        if node.get('summary'):
            return node['summary']
        arguments = dict(root=self.root,label=label,parent_checkpoint=self.parent,
                         rl_checkpoint=checkpoint,cohort=cohort,
                         manifest=self.panel if cohort=='TRAIN_DIAGNOSTIC' else self.navtest_manifest,
                         data=self.data if cohort=='TRAIN_DIAGNOSTIC' else self.navtest_data,
                         baseline=baseline)
        attempts=int(node.get('resource75_requeues',0))
        resource_epoch=None
        while True:
            spec = node.get('spec')
            while spec is None:
                if node.get('resource_capacity_wait'):
                    if resource_epoch is not None:
                        self.wait_capacity(node,status='WAITING_EVAL_CAPACITY',label=label,since=resource_epoch)
                    with self.mutex:node.pop('resource_capacity_wait',None);self.flush()
                with self.capacity_changed:dispatch_epoch=self.capacity_epoch
                try:
                    spec = self.dispatcher.dispatch_evaluation(**arguments)
                except BlockingIOError as error:
                    with self.mutex:
                        node.update(status='WAITING_EVAL_CAPACITY',capacity_reason=str(error))
                        self.flush()
                    self.event('evaluation_waiting_capacity',label=label,reason=str(error))
                    self.wait_capacity(node,status='WAITING_EVAL_CAPACITY',label=label,since=dispatch_epoch)
                    continue
                with self.mutex:
                    node['spec']=spec;node['status']='DISPATCHED';self.flush()
            results=[]
            with self.capacity_changed:resource_epoch=self.capacity_epoch
            for job in spec['jobs']:
                result=self.dispatcher.wait_job(job_path(job));results.append(result)
                if not (result_code(result)==75 and result.get('child_not_started') is True):
                    with self.mutex:
                        reported=node.setdefault('capacity_release_reported_jobs',[])
                        if job_path(job) not in reported:
                            reported.append(job_path(job));self.flush()
                            self.capacity_released()
            bad = [r for r in results if result_code(r)!=0]
            if bad:
                resource_only = all(result_code(r)==75 and r.get('child_not_started') is True for r in bad)
                with self.mutex:
                    node.setdefault('failed_attempts',[]).append(dict(at=now(),spec=spec,results=bad))
                    self.flush()
                if resource_only:
                    # Same semantic label/output. Dispatcher must reuse good
                    # shards and create new receipts only for refused children.
                    with self.mutex:
                        attempts+=1;node['resource75_requeues']=attempts
                        node.pop('spec',None)
                        node['resource_capacity_wait']=attempts>self.resource_retries
                        self.flush()
                    self.event('evaluation_resource75_requeue',label=label,attempt=attempts)
                    continue
                raise RuntimeError('Evaluation child failed; no scientific retry: '+json.dumps(bad,default=str)[:2500])
            summary = self.dispatcher.finish_evaluation(spec)
            if summary.get('status')!='COMPLETE' or summary.get('invalid_scenes',0)!=0:
                raise RuntimeError('Evaluation did not produce its complete declared population: '+label)
            if cohort=='NAVTEST' and int(summary.get('scenes',0))!=12146:
                raise RuntimeError('Partial Navtest cannot become a selection result')
            if cohort=='TRAIN_DIAGNOSTIC' and int(summary.get('scenes',0))!=1024:
                raise RuntimeError('Partial TRAIN panel cannot become a stage feedback bank')
            with self.mutex:
                node['summary']=summary;node['completed_at']=now()
                self.state['results'].append(dict(label=label,cohort=cohort,summary=summary))
                self.flush()
            self.event('evaluation_complete',label=label,cohort=cohort,PDMS=summary.get('PDMS'))
            return summary

    def baseline(self):
        with self.mutex:
            node = self.state.setdefault('IL70_TRAIN_BASELINE',{})
            adopted=read_json(self.root/'IL_BASELINE_EVALUATION.json',{})
            if adopted and not node.get('spec'):
                node['spec']=adopted
        self.run_evaluation('IL70_TRAIN_001024',None,cohort='TRAIN_DIAGNOSTIC',node=node)
        spec=node['spec']
        if not spec.get('training_bank') or not Path(spec['training_bank']).exists():
            raise RuntimeError('Shared IL TRAIN baseline has no complete training-feedback bank')
        return dict(output=spec['output'],training_bank=spec['training_bank'])

    def next_adaptive_stage(self, arm, update, current_training_bank, baseline_bank):
        from starVLA.rl.qwen_world_research_data.adaptive import prepare_stage
        from starVLA.rl.qwen_world_research_data import adaptive as generator
        from starVLA.rl.qwen_world_research_data.manifest import read_sealed
        manifest=self.runs[arm]['manifest'];stream=read_sealed(manifest)
        exposures=stream['stage_exposures'];world=8
        period=exposures*4//world
        if period!=1000 or update%period:
            raise ValueError('Adaptive continuation must follow the registered1000-update boundaries')
        index=update//period
        path=Path(stream['stage_registry'])/f'stage{index:06}.json'
        receipt_path=path.with_name(f'stage{index:06}_GENERATION.json')
        if path.exists():
            # An already published future stage is immutable and adopted by
            # identity, never regenerated after new scores arrive.
            body=read_sealed(path)
            if body['stream_manifest_sha256']!=sha_file(manifest) or body['stage_index']!=index:
                raise ValueError('Existing adaptive stage does not bind this stable stream')
            self.event('adaptive_stage_adopted',arm=arm,stage=index,path=str(path))
            return
        body=prepare_stage(manifest,index,current_bank=current_training_bank,
                           baseline_bank=baseline_bank,world_size=world)
        atomic_json(receipt_path,dict(status='PUBLISHED',at=now(),stage_index=index,
            stage_path=str(path),stage_file_sha256=sha_file(path),stage_sha256=body['sha256'],
            generator_source=str(Path(generator.__file__).resolve()),generator_source_sha256=sha_file(generator.__file__),
            generator_recipe_commit='f2f5575d5b6353c05f4b7be51a089ebd9ffd7f35',
            current_bank=current_training_bank,current_bank_sha256=sha_file(current_training_bank),
            baseline_bank=baseline_bank,baseline_bank_sha256=sha_file(baseline_bank),
            stage0_unchanged=True,NAVTEST_used_for_scene_selection=False,
            selection='35%uniform,40%low-score,25%IL-regression; repeat across batches; distinct within global batch'))
        self.event('adaptive_stage_published',arm=arm,stage=index,path=str(path))

    def train_to(self, arm, target, *, resume=None, adopt_first=False):
        node=self.stage(arm,target)
        if node.get('checkpoint') and node.get('train_completed'):
            return node['checkpoint']
        if adopt_first and 'train_job' not in node:
            with self.mutex:
                node['train_job']=self.runs[arm]['job']
                node['adopted_initial_dispatch']=True
                self.flush()
        attempts=int(node.get('resource75_requeues',0))
        resource_epoch=None
        while True:
            if self.arm_stop_decision(arm):raise ArmStoppedByResearchDecision()
            if not node.get('train_job'):
                if (node.get('resource_requeue_pending') and attempts>self.resource_retries
                        and resource_epoch is not None):
                    self.wait_capacity(node,status='WAITING_TRAIN_CAPACITY',label=arm,arm=arm,since=resource_epoch)
                current=next(r for r in read_json(self.root/'RUNS.json')['runs'] if r['id']==arm)
                if (current.get('stage_budget_updates')==target and current.get('job')
                        and not node.get('resource_requeue_pending')):
                    new_job=current['job']
                    with self.mutex:node['adopted_published_dispatch']=True
                else:
                    new_job=str(self.dispatcher.dispatch_training(self.root,arm,target,
                        resume=resume,source=self.runs[arm].get('source')))
                with self.mutex:
                    node['train_job']=new_job
                    node.pop('resource_requeue_pending',None)
                    self.flush()
            job=str(node['train_job'])
            self.update_arm(arm,status='TRAINING',stage_target_updates=target,job=job)
            with self.capacity_changed:resource_epoch=self.capacity_epoch
            result=self.dispatcher.wait_job(job)
            if not (result_code(result)==75 and result.get('child_not_started') is True):
                self.capacity_released()
            with self.mutex:
                node['train_result']=result;self.flush()
            code=result_code(result)
            if code==75 and result.get('child_not_started') is True:
                with self.mutex:
                    node.setdefault('previous_train_jobs',[]).append(dict(job=job,result=result))
                    attempts+=1;node['resource75_requeues']=attempts;node.pop('train_job',None)
                    node['resource_requeue_pending']=True;self.flush()
                self.event('training_resource75_requeue',arm=arm,target=target,attempt=attempts)
                continue
            if code!=0:
                raise RuntimeError('Formal training failed; preserve evidence and do not retry: '+json.dumps(result,default=str)[:2500])
            if self.arm_stop_decision(arm):raise ArmStoppedByResearchDecision()
            checkpoint=native_checkpoint(self.runs[arm]['root'],expected_update=target)
            with self.mutex:
                node.update(checkpoint=checkpoint['path'],train_completed=True,updates=target,train_completed_at=now())
            self.update_arm(arm,status='TRAINING_BOUNDARY_COMPLETE',updates=target,checkpoint=checkpoint['path'])
            self.update_run_registry(arm,status='STAGE_TRAIN_COMPLETE',completed_updates=target,job=job,
                                     last_checkpoint=checkpoint['path'])
            self.event('training_stage_complete',arm=arm,updates=target,checkpoint=checkpoint['path'])
            return checkpoint['path']

    def evaluation_future(self, arm, update, checkpoint, cohort):
        """Adopt each immutable checkpoint/cohort once, independently of training."""
        key=(arm,update,cohort)
        with self.mutex:
            node=self.stage(arm,update).setdefault(cohort,{})
            if key not in self.evaluation_futures:
                def evaluate():
                    if node.get('evaluation_error'):
                        raise RuntimeError('Prior evaluation failure requires review: '+node['evaluation_error'])
                    try:
                        baseline=self.baseline_future.result() if cohort=='TRAIN_DIAGNOSTIC' else None
                        suffix='TRAIN1024' if baseline else 'NAVTEST'
                        summary=self.run_evaluation(f'{arm}_u{update:06}_{suffix}',checkpoint,
                            cohort=cohort,node=node,baseline=baseline['output'] if baseline else None)
                    except Exception:
                        error=traceback.format_exc()
                        with self.mutex:
                            node.update(status='FAILED_REQUIRES_REVIEW',evaluation_error=error)
                            self.state['runs'][arm]['evaluation_alert']=dict(updates=update,cohort=cohort,error=error)
                            self.flush()
                        self.event('asynchronous_evaluation_failed',arm=arm,updates=update,cohort=cohort,error=error[-2500:])
                        raise
                    with self.mutex:node['status']='COMPLETE';self.flush()
                    # Aggregation is administrative: its failure must not turn a
                    # valid scientific result into a failed training experiment.
                    try:self.aggregate_priorities()
                    except Exception:self.event('priority_aggregation_warning',error=traceback.format_exc()[-2500:])
                    return summary
                feedback=cohort=='TRAIN_DIAGNOSTIC' and self.runs[arm]['data_policy']=='adaptive'
                pool=self.feedback_pool if feedback else self.eval_pool
                self.evaluation_futures[key]=pool.submit(evaluate)
            return self.evaluation_futures[key]

    def check_arm_evaluations(self, arm, *, wait_all=False):
        with self.mutex:
            futures=[future for key,future in self.evaluation_futures.items() if key[0]==arm
                and (key[2]=='TRAIN_DIAGNOSTIC' or key[1] in GATES)]
        for future in futures:
            if wait_all or future.done():future.result()

    def queue_boundary(self, arm, update, checkpoint):
        adaptive=self.runs[arm]['data_policy']=='adaptive'
        train=self.evaluation_future(arm,update,checkpoint,'TRAIN_DIAGNOSTIC')
        if update in GATES:self.evaluation_future(arm,update,checkpoint,'NAVTEST')
        if adaptive and update<50000:
            if update>=int(self.state['runs'][arm].get('updates',0)):
                self.update_arm(arm,status='WAITING_OWN_TRAIN_FEEDBACK')
            train.result()
            baseline=self.baseline_future.result()
            bank=self.stage(arm,update)['TRAIN_DIAGNOSTIC']['spec'].get('training_bank')
            if not bank or not Path(bank).exists():raise RuntimeError('Current TRAIN diagnostic bank not complete')
            self.next_adaptive_stage(arm,update,bank,baseline['training_bank'])
        self.event('boundary_evaluations_queued',arm=arm,updates=update,
            training_waits_for='own TRAIN bank only' if adaptive and update<50000 else 'none',
            other_arms_waited_for=False,NAVTEST_waited_for=False)

    def aggregate_priorities(self):
        """Publish available matched-budget evidence without a training barrier."""
        with self.mutex:
            for gate in GATES:
                summaries={arm:self.stage(arm,gate)['NAVTEST']['summary'] for arm in FIRST_ARMS
                    if self.stage(arm,gate).get('NAVTEST',{}).get('summary')}
                if not summaries:continue
                if 'PPO_FULL' in summaries:
                    _,decision=select_at_gate(gate,summaries,score_tolerance=self.score_tolerance,
                                              safety_tolerance=self.safety_tolerance)
                else:
                    decision=dict(selected=list(summaries),Test_selection_used=False,
                        control_comparison_pending=True,
                        reason='Matched control unavailable; no paired screen or training hold')
                decision.update(available_result_arms=list(summaries),
                    pending_result_arms=[arm for arm in FIRST_ARMS if arm not in summaries],
                    training_barrier=False,automatic_pruning=False,
                    default_budget_updates=50000,evidence_based_arm_stops=True)
                for arm in summaries:
                    stop=self.arm_stop_decision(arm)
                    if stop:
                        decision.setdefault('decisions',{}).setdefault(arm,{}).update(
                            selected_for_next_gate=False,reason=stop['reason'],decision_receipt=stop['path'])
                        decision['selected']=[a for a in decision['selected'] if a!=arm]
                self.state['selections'][str(gate)]=decision
            if (self.state.get('status') in ('RUNNING','RUNNING_CONTROL_REVIEW_REQUIRED')
                    and self.state['runs']['PPO_FULL'].get('status')=='FAILED_REQUIRES_REVIEW'):
                self.state['status']='RUNNING_CONTROL_REVIEW_REQUIRED'
            self.flush()

    def run_arm(self, arm):
        """One serial native-training chain; other arms and scores never gate it."""
        target=None
        try:
            if self.state['runs'][arm].get('status')=='FAILED_REQUIRES_REVIEW':
                self.event('failed_arm_preserved_for_review',arm=arm)
                return 'FAILED_REQUIRES_REVIEW'
            stop=self.arm_stop_decision(arm)
            if stop:
                if self.state['runs'][arm].get('status')!='STOPPED_FUTILITY':self.stop_arm(arm,stop)
                return 'STOPPED_FUTILITY'
            adaptive=self.runs[arm]['data_policy']=='adaptive'
            boundaries=range(1000,50001,1000) if adaptive else GATES
            checkpoint=self.state['runs'][arm].get('checkpoint')
            for target in boundaries:
                if self.arm_stop_decision(arm):raise ArmStoppedByResearchDecision()
                self.check_arm_evaluations(arm)
                if (self.owner_root/'STOP_AFTER_CURRENT_STAGE').exists():
                    self.update_arm(arm,status='HELD_BY_STOP_REQUEST');return 'HELD_BY_STOP_REQUEST'
                node=self.stage(arm,target)
                if not node.get('train_completed'):
                    committed=int(self.state['runs'][arm].get('updates',0))
                    dispatched=int(self.runs[arm].get('stage_budget_updates',0))
                    if target<=committed or target<dispatched:
                        # Recover only this historical immutable checkpoint; do
                        # not adopt a newer RUNS.job as an old first-stage job.
                        old=native_checkpoint(self.runs[arm]['root'],expected_update=target)
                        with self.mutex:
                            node.update(checkpoint=old['path'],train_completed=True,updates=target,
                                        adopted_historical_checkpoint=True)
                            self.flush()
                    else:
                        checkpoint=self.train_to(arm,target,resume=self.state['runs'][arm].get('checkpoint') or checkpoint,adopt_first=target==1000)
                        self.check_arm_evaluations(arm)
                checkpoint=node['checkpoint']
                if target>int(self.state['runs'][arm].get('updates',0)):
                    self.update_arm(arm,updates=target,checkpoint=checkpoint)
                self.queue_boundary(arm,target,checkpoint)
            self.update_arm(arm,status='TRAINING_COMPLETE_EVALUATIONS_PENDING')
            self.check_arm_evaluations(arm,wait_all=True)
            self.update_arm(arm,status='COMPLETE',updates=50000)
            self.update_run_registry(arm,status='COMPLETE',completed_updates=50000)
            self.event('arm_complete',arm=arm,updates=50000)
            return 'COMPLETE'
        except ArmStoppedByResearchDecision:
            try:
                self.stop_arm(arm,self.arm_stop_decision(arm))
                return 'STOPPED_FUTILITY'
            except Exception:
                error=traceback.format_exc()
                self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error)
                self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
                self.event('research_stop_requires_review',arm=arm,error=error[-2500:])
                return 'FAILED_REQUIRES_REVIEW'
        except Exception:
            error=traceback.format_exc()
            self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error)
            self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
            self.event('arm_stopped_after_failure',arm=arm,target=target,error=error[-2500:])
            return 'FAILED_REQUIRES_REVIEW'

    def run(self):
        self.event('owner_adopted_independent_pipelines',arms=list(FIRST_ARMS),global_training_barriers=False)
        self.baseline_future=self.feedback_pool.submit(self.baseline)
        # Extra matched safety probes reuse already sealed native checkpoints;
        # they never change a learner's target or become a training barrier.
        for probe in read_json(self.root/'ADDITIONAL_NAVTEST_PROBES.json',{}).get('evaluations',[]):
            arm=probe['arm'];update=int(probe['updates'])
            if self.arm_stop_decision(arm):continue
            checkpoint=native_checkpoint(self.runs[arm]['root'],update)['path']
            if (checkpoint!=probe['checkpoint'] or
                    sha_file(Path(checkpoint)/'COMPLETE')!=probe['checkpoint_seal_sha256']):
                raise ValueError('Additional matched probe checkpoint binding changed')
            self.evaluation_future(arm,update,checkpoint,'NAVTEST')
        with self.mutex:
            self.state.pop('current_gate',None)
            self.state['scheduling']='independent per-arm native budgets; asynchronous periodic evaluations'
            self.flush()
        pool=ThreadPoolExecutor(max_workers=len(FIRST_ARMS),thread_name_prefix='research_arm')
        pending={pool.submit(self.run_arm,arm):arm for arm in FIRST_ARMS}
        while pending:
            done,_=wait(pending,timeout=1800,return_when=FIRST_COMPLETED)
            if not done:
                self.event('owner_heartbeat',waiting_arms=list(pending.values()))
                continue
            for future in done:
                arm=pending.pop(future)
                self.event('arm_pipeline_finished',arm=arm,status=future.result())
        pool.shutdown(wait=True)
        self.aggregate_priorities()
        statuses={arm:self.state['runs'][arm]['status'] for arm in FIRST_ARMS}
        self.state['status']=('COMPLETE_WITH_RESEARCH_STOPS' if 'STOPPED_FUTILITY' in statuses.values()
                              and all(s in ('COMPLETE','STOPPED_FUTILITY') for s in statuses.values()) else
                              'COMPLETE' if all(s=='COMPLETE' for s in statuses.values()) else
                              'HELD_BY_STOP_REQUEST' if 'HELD_BY_STOP_REQUEST' in statuses.values() else
                              'COMPLETE_WITH_FAILED_ARMS_REVIEW_REQUIRED')
        self.event('campaign_pipelines_finished',statuses=statuses)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--score-tolerance',type=float,default=.25)
    parser.add_argument('--safety-tolerance',type=float,default=.25)
    parser.add_argument('--resource-retries',type=int,default=2)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if not 0<=args.score_tolerance<=1 or not 0<=args.safety_tolerance<=1 or not 0<=args.resource_retries<=3:
        parser.error('Bounded tolerances/retries required')
    if args.plan_only:
        runs=read_json(args.root/'RUNS.json')['runs']
        print(json.dumps(dict(root=str(args.root),adopt=[{k:r.get(k) for k in ('id','job','source','root')} for r in runs if r['id'] in FIRST_ARMS],
                              gates=GATES,always_continue='control; other arms unless separate evidence-based stop',max_challengers=None,
                              automatic_pruning=False,continuation_policy_supersedes_max2_budget_cap=True,
                              adaptive_intermediate_updates=1000,heartbeat_seconds=1800,
                              first_pruning_gate=None,first_budget_priority_gate=FIRST_PRIORITY_GATE,
                              first_gate_diagnostic_only=True,
                              first_gate_Test_selection_used=False,default_budget_updates=50000,
                              evidence_based_arm_stops=True,
                              first_gate_global_barrier=False,first_gate_waits_for_NAVTEST=False,
                              first_gate_results_required_before_first_priority=False,
                              independent_arm_pipelines=True,global_training_barriers=False,
                              periodic_NAVTEST_async=True,nonadaptive_TRAIN_diagnostics_async=True,
                              adaptive_waits_for_own_TRAIN_only=True,
                              Test_selection_used=True,no_new_jobs_launched=True),indent=2))
        return
    owner=CampaignOwner(args.root,score_tolerance=args.score_tolerance,
                        safety_tolerance=args.safety_tolerance,resource_retries=args.resource_retries)
    try:
        owner.run()
    finally:
        owner.flush()
        owner.eval_pool.shutdown(wait=True)
        owner.feedback_pool.shutdown(wait=True)


if __name__=='__main__':
    main()
