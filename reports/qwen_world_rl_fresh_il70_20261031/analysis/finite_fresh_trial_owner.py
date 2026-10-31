"""Finite, sealed-registration fresh-IL trial owner; no implicit first dispatch.

CLI: --registration REG [--owner-root O] --prepare|--launch|--run|--self-check.
Reuses CampaignOwner native/evaluation transactions and CadenceDispatch. This
file does not modify their source or take the main campaign/cadence owner lock.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import yaml

A = Path(__file__).resolve().parent
SOURCE = Path('/mnt/project/VLA-Drive-qwen-world-rl-research-20261030')
DEFAULT_ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
sys.path[:0] = [str(A), str(SOURCE)]
os.environ.setdefault('WORLD_POSTCOT_TASK', 'qwen_world_rl_optimization_20261030')
from tools.qwen_world_research import dispatch as D
from tools.qwen_world_research.campaign_owner import (
    CampaignOwner, ArmHeldForResourceLease, atomic_json, fingerprint,
    job_path, native_checkpoint, now, read_json, result_code,
)
from navtest_cadence_owner_v2 import CadenceDispatch, Events, NoLongerAuthorized

SCHEMA = 'world_finite_sealed_fresh_trial_owner_v1'
# An already existing recipe must not be revived by relabeling its registration.
LEGACY_ARMS = frozenset('PPO_HARD PPO_FULL PPO_ADAPTIVE CISPO_FULL NFT_FULL '
    'PPO_BATCHSTD PPO_SAFE_COMFORT PPO_ANCHOR40 PPO_MEMORYLR20 PPO_SAFETY_EP '
    'PPO_PREF_REPLAY PPO_PREF_CFM PPO_NOISE03 PPO_LASTDET PPO_RESCUE_REPLAY '
    'PPO_RESCUE_CFM PPO_SAFETY_LRLOW PPO_SAFETY_LRCOS PPO_SAFETY_DATA '
    'PPO_SAFETY_LRDATA PPO_SAFETY_SUCCESS_MEMORY PPO_SAFETY_FPO'.split())
BINDING_KEYS = ('module', 'config', 'config_sha256', 'source', 'root', 'manifest')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal(row):
    return any(word in str(row.get('status', '')).upper()
               for word in ('STOP', 'FAIL', 'CANCEL', 'HELD', 'PAUSED', 'DEFER'))


def alive(record):
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text().split()
        return int(stat[21]) == int(record['start_ticks']) and stat[2] not in ('Z', 'X')
    except (OSError, KeyError, ValueError):
        return False


class Contract:
    """Small immutable scientific binding; RUNS supplies mutable dispatch only."""
    def __init__(self, registration, root=DEFAULT_ROOT, owner_root=None):
        self.path = Path(registration).resolve()
        self.root = Path(root).resolve()
        self.file_sha256 = sha(self.path)
        self.registration = read_json(self.path)
        declared = self.registration.get('arms', [])
        if not isinstance(declared, list) or not declared or any(not isinstance(x, dict) for x in declared):
            raise ValueError('Registration requires explicit arms[] objects')
        self.arms = {row['id']: dict(row) for row in declared}
        if len(self.arms) != len(declared) or set(self.arms) & LEGACY_ARMS:
            raise ValueError('Duplicate or legacy arm cannot be adopted as a new fresh trial')
        seal = self.registration.get('sha256')
        if seal and seal != fingerprint({k:v for k,v in self.registration.items() if k != 'sha256'}):
            raise ValueError('Registration canonical body seal mismatch')
        self.targets = tuple(self.registration.get('gates', []))
        self.cap = self.registration.get('optimizer_update_cap_per_arm')
        if (not self.targets or any(type(x) is not int or x <= 0 for x in self.targets)
                or tuple(sorted(set(self.targets))) != self.targets or self.targets[-1] != self.cap
                or self.targets[0] != 1000 or type(self.cap) is not int or self.cap > 10000):
            raise ValueError('Finite fresh gates must start1k, increase uniquely, end at explicit cap<=10k')
        name = self.registration.get('owner_name')
        configured_owner = self.registration.get('owner_root')
        self.owner_root = Path(owner_root or configured_owner or self.root/'owners'/(name or 'FINITE_FRESH_TRIAL')).resolve()
        if not self.owner_root.is_relative_to(self.root/'owners') or self.owner_root.name in ('CAMPAIGN','NAVTEST_CADENCE','SAFETY_FRESH_PAIR','LR_STUDY'):
            raise ValueError('Owner root must be a separate dedicated owners/ directory')
        self.plan = self.owner_root/'LAUNCH_PLAN.json'
        self.baselines = self.registration.get('comparisons', {'primary':'PPO_SAFETY_FPO','secondary':['PPO_SAFETY_EP']})
        if not isinstance(self.baselines.get('primary'), str) or not isinstance(self.baselines.get('secondary', []), list):
            raise ValueError('comparisons requires primary arm string and secondary[]')
        for arm, row in self.arms.items():
            if any(not isinstance(row.get(k), str) or not row[k] for k in BINDING_KEYS):
                raise ValueError('Missing sealed arm binding: '+arm)
            if not row['module'].startswith('starVLA.rl.qwen_world_research_algorithms.'):
                raise ValueError('Unregistered module namespace')
            self.validate_config(arm, row)

    def validate_config(self, arm, row):
        if sha(row['config']) != row['config_sha256']:
            raise ValueError('Scientific config bytes changed: '+arm)
        config = yaml.safe_load(Path(row['config']).read_text())
        research = config.get('research', {})
        if (research.get('arm') != arm or config.get('seed') != 42 or config.get('world_size') != 8
                or int(config.get('max_steps', 0)) < self.cap
                or any(research.get(k) is not None for k in ('native_fork','warmstart'))):
            raise ValueError('Only freshIL70 seed42/world8, no RL warmstart/nativefork: '+arm)

    def rows(self, *, require_job=True, only_arm=None, evaluation=False):
        if sha(self.path) != self.file_sha256:
            raise ValueError('Reviewed registration bytes changed; new prepare/review required')
        if terminal(self.registration):
            raise NoLongerAuthorized('Registration stopped or held')
        registry = {row['id']:row for row in read_json(self.root/'RUNS.json')['runs']}
        output = {}
        for arm, proof in self.arms.items():
            if only_arm is not None and arm != only_arm:continue
            row = registry[arm]
            if not evaluation and (terminal(row) or row.get('new_training_authorized') is False):
                raise NoLongerAuthorized('Trial no longer authorized: '+arm)
            if evaluation and row.get('new_evaluation_authorized') is False:
                raise NoLongerAuthorized('New evaluation explicitly held: '+arm)
            if any(row.get(k) != proof[k] for k in BINDING_KEYS):
                raise ValueError('RUNS scientific identity changed: '+arm)
            if (int(row.get('research_budget_updates', 0)) != self.cap
                    or row.get('initial_updates') != 0 or row.get('initial_scene_cursor') != 0
                    or row.get('training_seed') != 42 or row.get('fresh_Adam') is not True
                    or row.get('scientific_registration') != str(self.path)
                    or not Path(row['source']).is_dir()):
                raise ValueError('Fresh identity/cap/registration binding changed: '+arm)
            if require_job and (not row.get('job') or row.get('stage_budget_updates') not in self.targets):
                raise ValueError('Root must register and dispatch first1k before owner starts')
            if require_job and not (Path(row['job'])/'LAUNCH.json').is_file():
                raise ValueError('Root first job has no actual published LAUNCH receipt')
            if row.get('job') and not Path(row['job']).resolve().is_relative_to(self.root/'jobs'/arm):
                raise ValueError('First job belongs to a different arm')
            self.validate_config(arm, proof)
            output[arm] = dict(row)
        return output


class TrialDispatch:
    def __init__(self, owner):
        self.owner = owner
        self.nav = CadenceDispatch(owner)

    def __getattr__(self, name):
        return getattr(D, name)

    def wait_job(self, job):
        result = D.wait_job(job)
        if result_code(result) == 75 and result.get('child_not_started') is True:
            D.wait_transport_exit(job)  # Refusal receipt alone is not transport exit.
        return result

    def dispatch_training(self, root, arm, target, **kwargs):
        while True:
            self.owner.authorized(arm, target)
            epoch = self.owner.capacity_epoch
            try:
                return D.dispatch_training(root, arm, target, **kwargs)
            except (BlockingIOError, PermissionError) as error:
                node = self.owner.stage(arm, target)
                node['capacity_reason'] = str(error)
                self.owner.wait_capacity(node, status='WAITING_TRAIN_CAPACITY', label=arm, arm=arm, since=epoch)

    def dispatch_evaluation(self, **arguments):
        arm, target, suffix = self.owner.parse_label(arguments['label'])
        self.owner.authorized(arm, target, evaluation=True)
        return self.nav.dispatch_evaluation(**arguments) if suffix == 'NAVTEST' else D.dispatch_evaluation(**arguments)


class FiniteFreshOwner(CampaignOwner):
    def __init__(self, contract):
        self.contract = contract
        self.root, self.owner_root = contract.root, contract.owner_root
        self.owner_root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.owner_root/'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.capacity_changed = threading.Condition(self.mutex)
        self.capacity_epoch = 0
        self.resource_retries = 1  # One short exact-prechild retry, then event/1800s wait.
        self.runs = contract.rows()
        self.arm_locks = []
        for arm in sorted(self.runs):
            folder = self.root/'owners'/arm; folder.mkdir(parents=True, exist_ok=True)
            lock = (folder/'finite_fresh_owner.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.arm_locks.append(lock)  # Also prevents a second owner-root from adopting this arm.
        self.dispatcher = TrialDispatch(self)
        self.parent = str(self.root/'IL_PARENT')
        self.data = str(self.root/'data/NAVTRAIN')
        self.panel = str(self.root/'data/panels/TRAIN_DIAGNOSTIC_001024.json')
        self.navtest_data = '/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST'
        self.navtest_manifest = str(self.root/'data/panels/NAVTEST_012146.json')
        self.state = read_json(self.owner_root/'STATE.json', {})
        if self.state and (self.state.get('schema') != SCHEMA or self.state.get('registration_sha256') != contract.file_sha256):
            raise ValueError('Existing state belongs to another sealed trial')
        self.state.update(schema=SCHEMA, status='RUNNING_FINITE_FRESH_TRIAL', registration=str(contract.path),
            registration_sha256=contract.file_sha256, targets=list(contract.targets), global_budget_cap=contract.cap,
            initial_global_updates=0, main_campaign_lock_used=False, training_waits_for_intermediate_evaluation=False,
            budget_wait=False, final_native_status='COMPLETED_AT_REGISTERED_CAP', automatic_scientific_retry=False)
        for key in ('runs','events','results'):
            self.state.setdefault(key, {} if key == 'runs' else [])
        for arm, row in self.runs.items():
            self.state['runs'].setdefault(arm, dict(id=arm, updates=0, stages={}, status='ADOPTING_ROOT_FIRST_JOB'))
            self.state['runs'][arm].update(run_root=row['root'], source=row['source'], research_budget_updates=contract.cap)
        self.eval_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='finite_trial_eval')
        self.review_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='finite_trial_CPUreview')
        self.evaluation_futures = {}
        atomic_json(self.owner_root/'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            started_at=now(), source_file=str(Path(__file__).resolve()), source_sha256=sha(__file__),
            registration=str(contract.path), registration_sha256=contract.file_sha256))
        self.flush()

    def authorized(self, arm, target, evaluation=False):
        if arm not in self.contract.arms or target not in self.contract.targets:
            raise NoLongerAuthorized('Unregistered arm or gate')
        if (self.owner_root/'STOP_AFTER_CURRENT_STAGE').exists():
            raise NoLongerAuthorized('Owner held; preserve existing children')
        row = self.contract.rows(only_arm=arm, evaluation=evaluation)[arm]
        if evaluation and row.get('new_evaluation_authorized') is False:
            raise NoLongerAuthorized('No new evaluations authorized')
        return row

    def arm_stop_decision(self, arm):
        self.authorized(arm, self.active_targets[arm])
        row = next(r for r in read_json(self.root/'RUNS.json')['runs'] if r['id'] == arm)
        if row.get('job') and (Path(row['job'])/'STOP_REQUESTED').exists():
            raise NoLongerAuthorized('Cooperative stop requested; preserve native seal')
        return None

    def parse_label(self, label):
        try:
            arm, rest = label.rsplit('_u', 1)
            target, suffix = rest.split('_', 1)
            target = int(target)
        except (ValueError, TypeError):
            raise NoLongerAuthorized('Invalid trial evaluation label')
        if (arm not in self.contract.arms or target not in self.contract.targets
                or suffix not in ('NAVTEST','TRAIN1024') or label != f'{arm}_u{target:06}_{suffix}'):
            raise NoLongerAuthorized('Unregistered arm/gate/cohort label')
        return arm, target, suffix

    def require_allowed(self, label):
        arm, target, suffix = self.parse_label(label)
        if suffix != 'NAVTEST':
            raise NoLongerAuthorized('NAV adapter only')
        self.authorized(arm, target, evaluation=True)

    def wait_capacity(self, node, *, status, label, arm=None, since=None):
        if since is not None and since != self.capacity_epoch:
            return
        node['status'] = status
        self.event('waiting_actual_capacity', label=label, status=status)
        events, descriptors = Events(), []
        try:
            for path, names in [(self.root, {'RUNS.json'}), (self.root/'resources', {'EVALUATION_CLAIMS.json'}),
                    (self.owner_root, {'STOP_AFTER_CURRENT_STAGE'}), (Path('/mnt/project'), {'server_dispatch_policy.json'})]:
                events.add(path, names)
            jobs = {r['job'] for r in read_json(self.root/'RUNS.json')['runs'] if r.get('job')}
            jobs.update(r['receipt'] for r in read_json(self.root/'resources/EVALUATION_CLAIMS.json', {}).values() if r.get('receipt'))
            for job in jobs:
                path = Path(job); events.add(path, {'RESULT.json'})
                launch = read_json(path/'LAUNCH.json', {})
                try:
                    pid = launch['transport_pid']; stat = Path(f'/proc/{pid}/stat').read_text().split()
                    if int(stat[21]) != launch['transport_start_ticks'] or stat[2] in ('Z','X'):continue
                    fd = os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else int(ctypes.CDLL(None).syscall(434, pid, 0))
                    if fd >= 0:descriptors.append(fd)
                except (OSError, KeyError, ValueError):pass
            if events.wait(1800, descriptors):self.capacity_released()
            else:self.event('capacity_1800s_heartbeat', label=label)
        finally:
            events.close()
            for fd in descriptors:os.close(fd)
        if arm:self.authorized(arm, self.active_targets[arm])

    def train_to(self, arm, target):
        row = self.authorized(arm, target)
        self.active_targets[arm] = target
        first = target == self.contract.targets[0]
        node = self.stage(arm, target)
        if first and not node.get('train_job') and not node.get('previous_train_jobs') and row.get('stage_budget_updates') != target:
            raise ValueError('First gate must be root-dispatched or already recorded in this owner state')
        return super().train_to(arm, target, resume=self.state['runs'][arm].get('checkpoint'), adopt_first=first and not node.get('previous_train_jobs'))

    def queue_boundary(self, arm, target, checkpoint):
        for cohort, suffix in [('NAVTEST','NAVTEST'), ('TRAIN_DIAGNOSTIC','TRAIN1024')]:
            key = (arm, target, cohort)
            if key in self.evaluation_futures:continue
            node = self.stage(arm, target).setdefault(cohort, {})
            def evaluate(arm=arm, target=target, cohort=cohort, suffix=suffix, node=node, checkpoint=checkpoint):
                try:
                    if node.get('evaluation_error'):return None  # Never retry scientific failure.
                    primary = self.contract.baselines['primary']
                    baseline = read_json(self.root/'evaluation_specs'/f'{primary}_u{target:06}_{suffix}.json', {}).get('output')
                    result = self.run_evaluation(f'{arm}_u{target:06}_{suffix}', checkpoint, cohort=cohort, node=node, baseline=baseline)
                    node['status'] = 'COMPLETE'; self.flush()
                    self.review_pool.submit(self.review_ready)
                    return result
                except NoLongerAuthorized as error:
                    node.update(status='HELD_BY_STOP_REQUEST', reason=str(error)); self.flush()
                except Exception:
                    node.update(status='FAILED_REQUIRES_REVIEW', evaluation_error=traceback.format_exc()); self.flush()
            self.evaluation_futures[key] = self.eval_pool.submit(evaluate)

    def review_ready(self):
        """Existing pure CPU value/contrast API; old arms are read-only baselines."""
        try:
            reviewer = self.contract.registration.get('reviewer')
            if reviewer:
                script = Path(reviewer['script'])
                if sha(script) != reviewer['sha256']:
                    raise ValueError('Registered CPU reviewer changed')
                with (self.owner_root/'reviewer.log').open('ab', buffering=0) as log:
                    subprocess.run([D.MODEL, str(script), *reviewer.get('argv', ['--registration',str(self.contract.path)])], cwd=A,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                        stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
                return
            if self.root != DEFAULT_ROOT:
                self.event('CPU_review_requires_attention', reason='Existing metric-cache helper root differs; no implicit path rebinding')
                return
            from safety_fresh_pair_review import value, il_value, annotate, contrast
            output = read_json(self.owner_root/'MATCHED_REVIEW.json', {})
            output.setdefault('schema', 'world_finite_fresh_trial_matched_review_v1')
            for key in ('contrasts','TRAIN_diagnostics'):output.setdefault(key, {})
            for cohort, key in [('NAVTEST','contrasts'), ('TRAIN_DIAGNOSTIC','TRAIN_diagnostics')]:
                il = il_value(cohort); scores = il['values'][:,0]
                for target in self.contract.targets:
                    for arm in self.runs:
                        current = value(arm, target, cohort)
                        if current is None:continue
                        baselines = [(self.contract.baselines['primary'], True)] + [(b,False) for b in self.contract.baselines.get('secondary', [])]
                        for baseline_arm, primary in baselines:
                            label = f'{arm}_u{target:06}_minus_{baseline_arm}'
                            if label in output[key]:continue
                            baseline = value(baseline_arm, target, cohort)
                            if baseline is None:continue
                            pair = annotate(current, baseline, scores >= .9)
                            pair['primary_comparison'] = primary
                            pair['fixed_IL70_low_lt09'] = contrast(current, baseline, scores < .9)
                            pair['fixed_IL70_zero'] = contrast(current, baseline, scores == 0)
                            output[key][label] = pair
            expected = len(self.runs)*len(self.contract.targets)*(1+len(self.contract.baselines.get('secondary', [])))
            output.update(status='COMPLETE' if all(len(output[k]) == expected for k in ('contrasts','TRAIN_diagnostics')) else 'PARTIAL_ACTUAL_COMPLETE_PAIRS_ONLY',
                updated_at=now(), registration=str(self.contract.path), registration_sha256=self.contract.file_sha256,
                initial_updates=0, registered_cap=self.contract.cap, expected_each=expected, no_automatic_promotion=True)
            atomic_json(self.owner_root/'MATCHED_REVIEW.json', output)
            self.state['matched_review'] = dict(path=str(self.owner_root/'MATCHED_REVIEW.json'), status=output['status']); self.flush()
        except Exception:
            self.event('CPU_review_requires_attention', error=traceback.format_exc()[-2500:])

    def run_one(self, arm):
        if terminal(self.state['runs'][arm]):return
        try:
            for target in self.contract.targets:
                self.authorized(arm, target)
                checkpoint = self.train_to(arm, target)
                self.queue_boundary(arm, target, checkpoint)  # Enqueue only; never wait here.
            self.update_arm(arm, status='COMPLETED_AT_REGISTERED_CAP', training_status='COMPLETED_AT_REGISTERED_CAP',
                updates=self.contract.cap, evaluation_status='PENDING_REGISTERED_GATES', budget_wait=False)
            self.update_run_registry(arm, status='COMPLETED_AT_REGISTERED_CAP', completed_updates=self.contract.cap)
            for (name, _, _), future in list(self.evaluation_futures.items()):
                if name == arm:future.result()
            complete = all(self.stage(arm,u).get(c,{}).get('summary') for u in self.contract.targets for c in ('NAVTEST','TRAIN_DIAGNOSTIC'))
            self.update_arm(arm, evaluation_status='ALL_REGISTERED_EVALUATIONS_COMPLETE' if complete else 'REQUIRES_EVALUATION_REVIEW')
        except (NoLongerAuthorized, ArmHeldForResourceLease) as error:
            self.update_arm(arm, status='HELD_BY_STOP_REQUEST', reason=str(error))
        except Exception:
            error = traceback.format_exc()
            self.update_arm(arm, status='FAILED_REQUIRES_REVIEW', error=error)
            self.update_run_registry(arm, status='FAILED_REQUIRES_REVIEW', controller_error=error)

    def run_trial(self):
        self.active_targets = {}
        try:
            with ThreadPoolExecutor(max_workers=len(self.runs), thread_name_prefix='finite_fresh_native') as pool:
                list(pool.map(self.run_one, self.runs))
            self.eval_pool.shutdown(wait=True)
            self.review_pool.submit(self.review_ready)
            self.review_pool.shutdown(wait=True)
            complete = all(r.get('training_status') == 'COMPLETED_AT_REGISTERED_CAP' for r in self.state['runs'].values())
            self.state.update(status='COMPLETED_AT_REGISTERED_CAP' if complete else 'FINISHED_WITH_FAILURE_OR_HOLD', budget_wait=False)
            self.flush(); return self.state
        finally:
            self.eval_pool.shutdown(wait=True); self.review_pool.shutdown(wait=True)


def bindings():
    return {str(p):sha(p) for p in [Path(__file__).resolve(), A/'navtest_cadence_owner_v2.py',
        A/'safety_fresh_pair_review.py', A/'rescue_pair_analyzer.py',
        SOURCE/'tools/qwen_world_research/campaign_owner.py', SOURCE/'tools/qwen_world_research/dispatch.py']}


def prepare(contract):
    try:rows = contract.rows(); ready = True; reason = None
    except (ValueError, KeyError, FileNotFoundError, NoLongerAuthorized) as error:rows = {}; ready = False; reason = str(error)
    plan = dict(schema=SCHEMA+'_launch_plan', status='READY_FOR_ROOT_REVIEW' if ready else 'WAITING_ROOT_FIRST_DISPATCH',
        script=str(Path(__file__).resolve()), started=False, registration=str(contract.path), registration_sha256=contract.file_sha256,
        owner_root=str(contract.owner_root), bindings=bindings(), arms=list(contract.arms), targets=list(contract.targets), cap=contract.cap,
        root_first_jobs={a:r['job'] for a,r in rows.items()}, ready_for_owner=ready, waiting_reason=reason,
        adopt_only_first_job=True, training_waits_for_intermediate_evaluation=False, max_active_evaluations=2,
        NAVTEST='full12146, existing CadenceDispatch up-to32 lawful shards; frozen layouts retained', TRAIN='fixed1024 at each gate',
        finite_end='COMPLETED_AT_REGISTERED_CAP', budget_wait=False, automatic_scientific_retry=False,
        argv=[D.MODEL, str(Path(__file__).resolve()), '--registration', str(contract.path), '--root', str(contract.root), '--owner-root', str(contract.owner_root), '--launch'])
    atomic_json(contract.plan, plan); return plan


def verify_plan(contract):
    plan = read_json(contract.plan)
    if plan.get('registration_sha256') != contract.file_sha256 or plan.get('bindings') != bindings():
        raise ValueError('Prepared registration/dependency seal changed; review again')
    if not plan.get('ready_for_owner'):raise ValueError('Prepare is not ready for owner')
    contract.rows(); return plan


def launch(contract):
    plan = verify_plan(contract)
    with (contract.owner_root/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(alive(read_json(contract.owner_root/name, {})) for name in ('OWNER.json','LAUNCH.json')):
            raise RuntimeError('Exact trial owner already alive')
        command = [D.MODEL, '-u', str(Path(__file__).resolve()), '--registration', str(contract.path), '--root', str(contract.root), '--owner-root', str(contract.owner_root), '--run']
        with (contract.owner_root/'stdout.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=A, env=dict(os.environ, PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=process.pid, start_ticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21]),
            argv=command, started_at=now(), source_sha256=sha(__file__), registration=str(contract.path),
            registration_sha256=contract.file_sha256, bindings=plan['bindings'])
        atomic_json(contract.owner_root/'LAUNCH.json', record); return record


def self_check():
    """Two narrow CPU mocks: no duplicate first job; native precedes eval waits."""
    from types import SimpleNamespace
    from unittest.mock import patch
    arm = 'MOCK_NEW_FRESH'
    owner = object.__new__(FiniteFreshOwner)
    owner.contract = SimpleNamespace(targets=(1000,5000), cap=5000, arms={arm:{}})
    owner.runs = {arm:dict(root='run', source='frozen', job='first')}
    owner.state = dict(runs={arm:dict(updates=0, stages={}, status='ADOPTING_ROOT_FIRST_JOB')})
    owner.active_targets = {}; owner.mutex = threading.RLock(); owner.capacity_changed = threading.Condition(owner.mutex)
    owner.capacity_epoch = owner.resource_retries = 0
    owner.authorized = lambda a,t,**kw:dict(stage_budget_updates=1000, job='first')
    owner.arm_stop_decision = lambda a:None
    owner.flush = lambda:None; owner.event = lambda *a,**kw:None
    owner.update_run_registry = lambda *a,**kw:None
    owner.dispatcher = SimpleNamespace(wait_job=lambda job:dict(returncode=0), dispatch_training=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Duplicate initial dispatch')))
    with patch('tools.qwen_world_research.campaign_owner.native_checkpoint', return_value=dict(path='cp1000',updates=1000)):
        assert owner.train_to(arm,1000) == 'cp1000'
    assert owner.stage(arm,1000)['adopted_initial_dispatch'] is True
    seen = []
    def train(a,t):
        seen.append(t); owner.stage(a,t).update(checkpoint='cp'+str(t),train_completed=True); return 'cp'+str(t)
    class Pending:
        def result(self):assert seen == [1000,5000]
    owner.train_to = train; owner.evaluation_futures = {}
    def queue(a,t,cp):
        for c in ('NAVTEST','TRAIN_DIAGNOSTIC'):
            owner.stage(a,t).setdefault(c,{})['summary'] = dict(status='COMPLETE')
            owner.evaluation_futures[(a,t,c)] = Pending()
    owner.queue_boundary = queue
    owner.run_one(arm)
    assert seen == [1000,5000] and owner.state['runs'][arm]['status'] == 'COMPLETED_AT_REGISTERED_CAP'
    return dict(status='PASS', source_sha256=sha(__file__), checks=['root_first_job_adopted_no_redispatch','native5k_before_any_evaluation_wait'], GPU=0, dispatches=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--owner-root', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    for action in ('prepare','launch','run','self-check'):mode.add_argument('--'+action, action='store_true')
    args = parser.parse_args()
    if args.self_check:print(json.dumps(self_check())); return
    if not args.registration:parser.error('--registration is required')
    contract = Contract(args.registration, args.root, args.owner_root)
    if args.prepare:result = prepare(contract)
    elif args.launch:result = launch(contract)
    else:verify_plan(contract); result = FiniteFreshOwner(contract).run_trial()
    print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
