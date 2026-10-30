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
FIRST_ARMS = ('PPO_HARD', 'PPO_FULL', 'PPO_ADAPTIVE', 'CISPO_FULL', 'NFT_FULL')
GATES = (1000, 5000, 10000, 20000, 35000, 50000)
FIRST_PRIORITY_GATE = 5000
SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')
SCHEMA = 'world_rl_optimization_campaign_owner_v1'


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
    """Scores set resource priority only; every healthy arm keeps its budget.

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
        interpretation='all healthy arms continue; any future futility/reserve decision requires a separate evidence-based receipt')


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
                     started_at=now(),scope='adopt five existing jobs; continue bounded gates; never edit training snapshots')
        atomic_json(self.owner_root/'OWNER.json',owner)
        self.state['owner'] = owner
        self.state['status'] = 'RUNNING'
        self.state['selection_policy'] = dict(score_tolerance_PDMS_points=score_tolerance,
            safety_tolerance_percentage_points=safety_tolerance,Test_selection_used=True,
            max_challengers=None,control='PPO_FULL',gates=list(GATES),
            automatic_pruning=False,continuation_policy_supersedes_max2_budget_cap=True,
            first_pruning_gate=None,first_budget_priority_gate=FIRST_PRIORITY_GATE,
            first_gate_diagnostic_only=True,
            first_gate_Test_selection_used=False,all_healthy_arms_continue_to=50000,
            first_gate_global_barrier=False,first_gate_waits_for_NAVTEST=False,
            first_gate_results_required_before_first_priority=True,
            correction_registered_before_first_complete_new_1k_NAVTEST=True)
        self.eval_pool = ThreadPoolExecutor(max_workers=12,thread_name_prefix='research_eval')
        self.nav_futures = {}
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

    def capacity_released(self):
        """A real adopted worker exited; wake dispatchers without journal polls."""
        with self.capacity_changed:
            self.capacity_epoch += 1
            self.capacity_changed.notify_all()

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
        for attempt in range(self.resource_retries+1):
            spec = node.get('spec')
            while spec is None:
                with self.capacity_changed:epoch=self.capacity_epoch
                try:
                    spec = self.dispatcher.dispatch_evaluation(**arguments)
                except BlockingIOError as error:
                    with self.mutex:
                        node.update(status='WAITING_CAPACITY',capacity_reason=str(error))
                        self.flush()
                    self.event('evaluation_waiting_capacity',label=label,reason=str(error))
                    # Capacity waiting is independent of scientific/resource75
                    # retry accounting. Actual worker exits wake it immediately;
                    # unrelated jobs can also release capacity between the
                    # sparse1800-second administrative heartbeats.
                    with self.capacity_changed:
                        released=self.capacity_changed.wait_for(lambda:self.capacity_epoch!=epoch,timeout=1800)
                    if not released:self.event('capacity_wait_heartbeat',label=label)
                    continue
                with self.mutex:
                    node['spec']=spec;node['status']='DISPATCHED';self.flush()
            results=[]
            for job in spec['jobs']:
                results.append(self.dispatcher.wait_job(job_path(job)))
                self.capacity_released()
            bad = [r for r in results if result_code(r)!=0]
            if bad:
                resource_only = all(result_code(r)==75 and r.get('child_not_started') is True for r in bad)
                with self.mutex:
                    node.setdefault('failed_attempts',[]).append(dict(at=now(),spec=spec,results=bad))
                    self.flush()
                if resource_only and attempt<self.resource_retries:
                    # Same semantic label/output. Dispatcher must reuse good
                    # shards and create new receipts only for refused children.
                    with self.mutex:node.pop('spec',None)
                    self.event('evaluation_resource75_requeue',label=label,attempt=attempt+1)
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
        raise RuntimeError('Unreachable evaluation retry state')

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
        while True:
            if not node.get('train_job'):
                current=next(r for r in read_json(self.root/'RUNS.json')['runs'] if r['id']==arm)
                if current.get('stage_budget_updates')==target and current.get('job'):
                    new_job=current['job']
                    with self.mutex:node['adopted_published_dispatch']=True
                else:
                    new_job=str(self.dispatcher.dispatch_training(self.root,arm,target,
                        resume=resume,source=self.runs[arm].get('source')))
                with self.mutex:
                    node['train_job']=new_job
                    self.flush()
            job=str(node['train_job'])
            self.update_arm(arm,status='TRAINING',stage_target_updates=target,job=job)
            result=self.dispatcher.wait_job(job)
            self.capacity_released()
            with self.mutex:
                node['train_result']=result;self.flush()
            code=result_code(result)
            if code==75 and result.get('child_not_started') is True and attempts<self.resource_retries:
                with self.mutex:
                    node.setdefault('previous_train_jobs',[]).append(dict(job=job,result=result))
                    attempts+=1;node['resource75_requeues']=attempts;node.pop('train_job',None)
                # Do not adopt the same refused receipt when selecting a new
                # bounded attempt at the top of the loop.
                fresh_job=str(self.dispatcher.dispatch_training(self.root,arm,target,
                    resume=resume,source=self.runs[arm].get('source')))
                with self.mutex:node['train_job']=fresh_job;self.flush()
                self.event('training_resource75_requeue',arm=arm,target=target,attempt=attempts)
                continue
            if code!=0:
                raise RuntimeError('Formal training failed; preserve evidence and do not retry: '+json.dumps(result,default=str)[:2500])
            checkpoint=native_checkpoint(self.runs[arm]['root'],expected_update=target)
            with self.mutex:
                node.update(checkpoint=checkpoint['path'],train_completed=True,updates=target,train_completed_at=now())
            self.update_arm(arm,status='TRAINING_BOUNDARY_COMPLETE',updates=target,checkpoint=checkpoint['path'])
            self.update_run_registry(arm,status='STAGE_TRAIN_COMPLETE',completed_updates=target,job=job,
                                     last_checkpoint=checkpoint['path'])
            self.event('training_stage_complete',arm=arm,updates=target,checkpoint=checkpoint['path'])
            return checkpoint['path']

    def nav_evaluation(self, arm, update, checkpoint, node):
        """Adopt one durable evaluation; retain failures across CPU-owner restarts."""
        key=(arm,update)
        with self.mutex:
            if key not in self.nav_futures:
                def evaluate():
                    if node.get('evaluation_error'):
                        raise RuntimeError('Prior evaluation failure requires review: '+node['evaluation_error'])
                    try:
                        return self.run_evaluation(f'{arm}_u{update:06}_NAVTEST',checkpoint,
                                                   cohort='NAVTEST',node=node)
                    except Exception:
                        error=traceback.format_exc()
                        with self.mutex:
                            node.update(status='FAILED_REQUIRES_REVIEW',evaluation_error=error)
                            self.flush()
                        self.event('asynchronous_evaluation_failed',arm=arm,updates=update,error=error[-2500:])
                        raise
                self.nav_futures[key]=self.eval_pool.submit(evaluate)
            return self.nav_futures[key]

    def check_first_diagnostic(self, arm):
        future=self.nav_futures.get((arm,1000))
        if future is not None and future.done():
            future.result()  # A real scoring failure forbids further dispatches.

    def evaluate_boundary(self, arm, update, checkpoint, *, navtest, wait_navtest=True):
        baseline=self.baseline_future.result()
        node=self.stage(arm,update)
        with self.mutex:
            train_node=node.setdefault('TRAIN_DIAGNOSTIC',{})
            nav_node=node.setdefault('NAVTEST',{})
            historical=update<int(self.state['runs'][arm].get('updates',0))
        if not historical:self.update_arm(arm,status='EVALUATING',updates=update)
        train=self.eval_pool.submit(self.run_evaluation,f'{arm}_u{update:06}_TRAIN1024',checkpoint,
                                    cohort='TRAIN_DIAGNOSTIC',node=train_node,baseline=baseline['output'])
        nav=self.nav_evaluation(arm,update,checkpoint,nav_node) if navtest else None
        train_summary=train.result()
        if self.runs[arm]['data_policy']=='adaptive' and update<50000:
            bank=train_node['spec'].get('training_bank')
            if not bank or not Path(bank).exists():raise RuntimeError('Current TRAIN diagnostic bank not complete')
            self.next_adaptive_stage(arm,update,bank,baseline['training_bank'])
        if not historical:
            self.update_arm(arm,status='TRAIN_PANEL_COMPLETE',updates=update)
        if nav is not None and wait_navtest:
            summary=nav.result()
            if not historical:self.update_arm(arm,status='GATE_EVALUATED',updates=update)
            return summary
        return train_summary

    def advance_arm(self, arm, target, first=False):
        try:
            if first:
                checkpoint=self.train_to(arm,1000,adopt_first=True)
                self.evaluate_boundary(arm,1000,checkpoint,navtest=True,wait_navtest=False)
                self.event('first_diagnostic_released_training',arm=arm,next_gate=target,
                           NAVTEST_waited_for=False,other_arms_waited_for=False)
            self.check_first_diagnostic(arm)
            prior=int(self.state['runs'][arm].get('updates',0))
            checkpoint=self.state['runs'][arm].get('checkpoint')
            if prior>=target:
                node=self.stage(arm,target)
                if node.get('NAVTEST',{}).get('summary'):return node['NAVTEST']['summary']
                checkpoint=node.get('checkpoint') or (checkpoint if prior==target else None)
                if not checkpoint:raise RuntimeError('Historical gate has no preserved native checkpoint')
                return self.evaluate_boundary(arm,target,checkpoint,navtest=True)
            adaptive=self.runs[arm]['data_policy']=='adaptive'
            boundaries=list(range(prior+1000,target+1,1000)) if adaptive else [target]
            for boundary in boundaries:
                self.check_first_diagnostic(arm)
                if (self.owner_root/'STOP_AFTER_CURRENT_STAGE').exists():
                    self.update_arm(arm,status='HELD_BY_STOP_REQUEST')
                    return None
                checkpoint=self.train_to(arm,boundary,resume=checkpoint)
                self.check_first_diagnostic(arm)
                summary=self.evaluate_boundary(arm,boundary,checkpoint,navtest=boundary==target)
            return summary
        except Exception:
            error=traceback.format_exc()
            self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error)
            self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
            self.event('arm_stopped_after_failure',arm=arm,target=target,error=error[-2500:])
            return None

    def collect_first_diagnostics(self):
        """First-1k scores may overlap training, but precede the first selection."""
        summaries={}
        for arm in FIRST_ARMS:
            future=self.nav_futures.get((arm,1000))
            if future is None:
                if self.state['runs'][arm].get('status')!='FAILED_REQUIRES_REVIEW':
                    raise RuntimeError('Healthy arm has no adopted first NAVTEST: '+arm)
                continue
            try:
                summaries[arm]=future.result()
            except Exception:
                error=traceback.format_exc()
                self.update_arm(arm,status='FAILED_REQUIRES_REVIEW',error=error)
                self.update_run_registry(arm,status='FAILED_REQUIRES_REVIEW',controller_error=error)
                self.event('arm_stopped_after_first_diagnostic_failure',arm=arm,error=error[-2500:])
        _,decision=select_at_gate(1000,summaries)
        with self.mutex:self.state['selections']['1000']=decision;self.flush()
        self.event('first_diagnostics_collected_before_pruning',complete_arms=list(summaries),
                   first_pruning_gate=None,first_budget_priority_gate=FIRST_PRIORITY_GATE,
                   Test_selection_used=False)
        return summaries

    def parallel_gate(self, arms, target, *, first=False):
        results={}
        pool=ThreadPoolExecutor(max_workers=len(arms),thread_name_prefix='research_arm')
        pending={pool.submit(self.advance_arm,arm,target,first):arm for arm in arms}
        while pending:
            done,_=wait(pending,timeout=1800,return_when=FIRST_COMPLETED)
            if not done:
                self.event('owner_heartbeat',target=target,waiting_arms=list(pending.values()))
                continue
            for task in done:
                arm=pending.pop(task);summary=task.result()
                if summary is not None:results[arm]=summary
        pool.shutdown(wait=True)
        return results

    def run(self):
        self.event('owner_adopted_existing_jobs',arms=list(FIRST_ARMS))
        self.baseline_future=self.eval_pool.submit(self.baseline)
        selected=list(FIRST_ARMS)
        # The first wave advances each arm independently through its own TRAIN
        # feedback to5k. No first-1k score or other arm gates this continuation.
        for index,target in enumerate(GATES[1:]):
            if (self.owner_root/'STOP_AFTER_CURRENT_STAGE').exists():
                self.state['status']='HELD_BY_STOP_REQUEST';self.flush();return
            with self.mutex:self.state['current_gate']=target;self.flush()
            summaries=self.parallel_gate(selected,target,first=index==0)
            if index==0:
                first_summaries=self.collect_first_diagnostics()
                summaries={arm:value for arm,value in summaries.items() if arm in first_summaries}
            if 'PPO_FULL' not in summaries:
                self.state['status']='RUNNING_CONTROL_REVIEW_REQUIRED'
                selected=list(summaries)
                self.state['selections'][str(target)]=dict(selected=selected,
                    reason='No complete matched control; retain every healthy arm without paired screening',
                    Test_selection_used=False,control_review_required=True)
                self.event('matched_control_missing_healthy_arms_continue',gate=target,selected=selected)
                if not selected:
                    self.state['status']='NO_HEALTHY_ARMS_REVIEW_REQUIRED';self.flush();return
                if target!=50000:continue
            if target==50000:
                for arm in summaries:
                    self.update_arm(arm,status='COMPLETE',updates=50000)
                    self.update_run_registry(arm,status='COMPLETE',completed_updates=50000)
                self.state['status']='COMPLETE' if 'PPO_FULL' in summaries else 'COMPLETE_CONTROL_REVIEW_REQUIRED'
                self.event('campaign_complete',arms=list(summaries));return
            selected,decision=select_at_gate(target,summaries,score_tolerance=self.score_tolerance,
                                            safety_tolerance=self.safety_tolerance)
            self.state['selections'][str(target)]=decision
            for arm in summaries:
                self.update_arm(arm,selection_decision=decision['decisions'][arm])
            self.event('next_gate_selected',gate=target,selected=selected,
                       automatic_pruning=False,first_pruning_gate=None,
                       Test_selection_used=decision['Test_selection_used'])


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
                              gates=GATES,always_continue='all healthy arms',max_challengers=None,
                              automatic_pruning=False,continuation_policy_supersedes_max2_budget_cap=True,
                              adaptive_intermediate_updates=1000,heartbeat_seconds=1800,
                              first_pruning_gate=None,first_budget_priority_gate=FIRST_PRIORITY_GATE,
                              first_gate_diagnostic_only=True,
                              first_gate_Test_selection_used=False,all_healthy_arms_continue_to=50000,
                              first_gate_global_barrier=False,first_gate_waits_for_NAVTEST=False,
                              first_gate_results_required_before_first_priority=True,
                              Test_selection_used=True,no_new_jobs_launched=True),indent=2))
        return
    owner=CampaignOwner(args.root,score_tolerance=args.score_tolerance,
                        safety_tolerance=args.safety_tolerance,resource_retries=args.resource_retries)
    try:
        owner.run()
    finally:
        owner.flush()
        owner.eval_pool.shutdown(wait=True)


if __name__=='__main__':
    main()
