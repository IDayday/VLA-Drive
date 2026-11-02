"""Independent finite asynchronous Safety NAVTEST owner; never launches training.

Run with explicit WORLD_COMPARE_ROOT/TASK. Each registered frozen run.source is
used for every export/evaluate/summarize command. Pre-child resource75 is requeued; one confirmed CUDA OOM shard may be
retried unchanged on an idle GPU while retaining all four leases. Other failures
preserve receipts for review, never zero scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from starVLA.rl.qwen_world_safety_exploration.contracts import (ROOT, TASK, allowed_hosts,
    file_sha, fingerprint)
from starVLA.rl.qwen_world_safety_exploration.resources import probe, MODEL_PYTHON, gpu_activity_issue
from starVLA.rl.flow_grpo.transactions import atomic_json
from .dispatch import launch, job_state
from .notify import refresh

SCHEMA = 'world_Safety_exploration_async_evaluation_owner_v1'
COUNTS = {'NAVTEST': 12146}
LEASES = {name: Path('/mnt/project')/folder for name, folder in (
    ('primary', 'runtime_gpu_leases'), ('shared', 'runtime_gpu_shared_workers'),
    ('research', 'runtime_gpu_research_slots'), ('evaluation', 'runtime_gpu_evaluation_slots'))}
KNOWN_ROOTS = ('/mnt/project/qwen-sourceflow-', '/mnt/project/qwen-world-postcot-',
               '/mnt/project/qwen-world-four-frameworks-', '/mnt/project/qwen-world-abcd-rl-best-20261031/','/mnt/project/qwen-world-safety-exploration-20261031/','/mnt/project/qwen-world-rl-optimization-20261030/','/mnt/project/qwen-world-task-suites-20261031/')
TERMINAL_NATIVE = ('FAILED', 'STOPPED', 'COMPLETED_AT_REGISTERED_CAP')


def completion_priority(entry):
    current = (entry.get('arm') == 'REVIVAL_G16' or
               entry.get('arm') == 'IMPROVEMENT_VERIFIED_RETENTION_G8')
    completed = sum(node.get('status') == 'COMPLETE' for node in entry.get('ranks', {}).values())
    if entry.get('arm') == 'IMPROVEMENT_VERIFIED_RETENTION_G8':
        return (0, int(entry.get('updates', 0)), -completed, entry.get('label', ''))
    return (not current, -completed, int(entry.get('updates', 0)), entry.get('label', ''))


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def retryable(value):
    return (value.get('status') == 'REFUSED_RESOURCE' and value.get('returncode') == 75
            and value.get('child_not_started') is True)


def lease_free(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def usable_gpu(gpu, sample, *, minimum, reserved):
    if gpu['uuid'] in reserved or gpu['free_MiB'] < minimum:
        return False
    if gpu_activity_issue(gpu, sample['owners']):
        return False
    owners = [row for row in sample['owners'] if row['uuid'] == gpu['uuid']]
    for row in owners:
        if (not row.get('visible') or not isinstance(row.get('output'), str)
                or not row['output'].startswith(KNOWN_ROOTS)):
            return False
        if 'evaluat' in row.get('entry', '').lower():
            return False  # No second evaluation owner on this physical device.
    uuid = gpu['uuid']+'.lock'
    if not lease_free(LEASES['evaluation']/uuid):
        return False
    # A guardian may reserve all old namespaces without allocating CUDA memory.
    if all(not lease_free(LEASES[name]/uuid) for name in ('primary', 'shared', 'research')):
        return False
    return True


def checkpoint_at_gate(run, updates):
    state = read(run['owner_state'], {})
    native = state.get('runs', {}).get(run['id'], {})
    stage = native.get('stages', {}).get(str(updates), {})
    if not stage.get('complete'):
        return None, native
    path = Path(stage['checkpoint']).resolve()
    if not path.is_relative_to(Path(run['root']).resolve()/'checkpoints'):
        raise ValueError('Foreign checkpoint outside registered native run')
    seal = read(path/'COMPLETE', {})
    identity = read(path/'identity.json', {})
    if (seal.get('updates') != updates or seal.get('world') != 8
            or seal.get('next_inner') != 0 or seal.get('files_sha256') != file_sha(path/'files.json')
            or identity.get('recipe',{}).get('adaptation') != 'ATTN_LORA_MEMORY'
            or identity.get('purpose') != 'formal' or identity.get('world') != 8):
        raise ValueError('Evaluation requires the exact sealed formal eight-rank gate')
    for rank in range(8):
        if not (path/f'rank{rank}.pt').is_file():
            raise ValueError('Incomplete native checkpoint rank inventory')
    return dict(path=str(path), updates=updates, files_sha256=seal['files_sha256'],
                identity_sha256=file_sha(path/'identity.json')), native


def complete_summary(path, *, architecture, cohort):
    value = read(path)
    if value is None:
        return None
    body = dict(value); saved = body.pop('sha256', None)
    if (fingerprint(body) != saved or value.get('status') != 'COMPLETE'
            or value.get('IL_updates') != 70000 or value.get('cohort') != cohort
            or value.get('schema') != 'world_rl_research_evaluation_summary_v1' or value.get('scenes') != COUNTS[cohort]
            or value.get('expected_scenes') != COUNTS[cohort] or value.get('invalid_scenes') != 0):
        raise ValueError('Only sealed full-population canonical summaries are publishable')
    return value


class EvaluationOwner:
    def __init__(self, registry, *, max_shards=32, max_workers=32, minimum=32768):
        if TASK != 'qwen_world_safety_exploration_20261031' or not os.environ.get('WORLD_COMPARE_ROOT'):
            raise PermissionError('Explicit independent Safety ROOT/TASK required before import')
        self.registry_path = Path(registry).resolve()
        if self.registry_path != (ROOT/'RUNS.json').resolve():
            raise ValueError('Only this task RUNS registry may authorize evaluations')
        self.directory = ROOT/'owners/RESEARCH_EVALUATION'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner_lock = (self.directory/'owner.lock').open('a')
        fcntl.flock(self.owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.directory/'STATE.json'
        self.state = read(self.state_path, dict(schema=SCHEMA, runs={}, evaluations={}))
        self.max_shards, self.max_workers, self.minimum = max_shards, max_workers, minimum
        self.claims_path = ROOT/'resources/EVALUATION_CLAIMS.json'
        self.claims_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self.registry_now()
        self.initial_bindings = {r['id']: self.binding(r) for r in self.registry['runs']}
        self.state.update(schema=SCHEMA, status='RUNNING', registry=str(self.registry_path),
            max_shards=max_shards, max_workers=max_workers, minimum_free_MiB=minimum,
            unknown_or_partial_results_have_no_score=True)
        atomic_json(self.directory/'OWNER.json', dict(pid=os.getpid(),
            start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=str(Path(__file__).resolve()), source_sha256=file_sha(__file__), started_unix=time.time(),
            registry=str(self.registry_path), task=TASK, root=str(ROOT)))
        self.flush()

    @staticmethod
    def binding(run):
        value = {key: run[key] for key in ('id', 'architecture_id', 'root', 'source', 'config',
                                          'config_sha256', 'parent_checkpoint', 'gates', 'target_updates')}
        value.update(evaluation_source=run.get('evaluation_source'),
                     evaluation_source_receipt=run.get('evaluation_source_receipt'))
        return value

    def registry_now(self):
        value = read(self.registry_path)
        if value.get('task') != TASK or not 1 <= len(value.get('runs', [])) <= 16:
            raise ValueError('The independent finite registered research arms are required')
        if len({row['id']for row in value['runs']}) != len(value['runs']):
            raise ValueError('Unique registered research ids required')
        for cohort,location in value['evaluation']['data_paths'].items():
            m=read(Path(location)/'manifest.json',{})
            if m.get('cohort')!=cohort or len(m.get('tokens',[]))!=COUNTS[cohort]:
                raise ValueError('Missing or changed canonical evaluation population: '+str(location))
        for run in value['runs']:
            from .evaluation_execution import evaluation_source
            if file_sha(run['config']) != run['config_sha256']:
                raise ValueError('Registered scientific config bytes changed')
            if not Path(run['root']).resolve().is_relative_to(ROOT.resolve()/'runs'):
                raise ValueError('Foreign run root')
            if not (Path(evaluation_source(run))/'tools/qwen_world_safety_exploration/evaluate.py').is_file():
                raise ValueError('Frozen run source lacks the registered evaluator')
        return value

    def flush(self):
        self.state['heartbeat_unix'] = time.time()
        atomic_json(self.state_path, self.state)
        signature = fingerprint({k:{'status':v.get('status'),'summary':v.get('summary')}
            for k,v in self.state.get('evaluations',{}).items()
            if v.get('status') in ('COMPLETE','FAILED_REVIEW_REQUIRED')})
        if signature != getattr(self,'_report_signature',None):
            self._report_signature = signature
            refresh(self.state_path,'Safety exploration evaluation milestone or failure')

    def receipt(self, node):
        if not node.get('job'):
            return None
        try:
            return job_state(node['job'])
        except (FileNotFoundError, ValueError):
            node.update(status='DISPATCH_UNCERTAIN_REVIEW_REQUIRED')
            return None

    def finished(self, node):
        value = self.receipt(node)
        if value is None:
            return False
        if retryable(value):
            node.setdefault('refused_jobs', []).append(node.pop('job'))
            node.update(status='RESOURCE_REQUEUE')
            self.flush()
            return False
        if 'returncode' not in value:
            node['status'] = 'RUNNING'
            return False
        if value['returncode'] != 0:
            command=node.get('command',[])
            log=Path(node['job'])/'stdout.log'
            text=log.read_text(errors='replace')if log.exists()else ''
            oom=(value.get('returncode')==1 and value.get('status')=='FAILED' and not value.get('forced_expiry')
                and 'tools.qwen_world_safety_exploration.evaluate'in command and '--rank'in command
                and '--world'in command and '--summarize'not in command
                and 'torch.OutOfMemoryError: CUDA out of memory.'in text)
            if oom and not node.get('evaluation_exclusive_recovery'):
                previous=dict(job=node['job'],result=value,stdout_sha256=file_sha(log),reason='CONFIRMED_CUDA_OOM_RETRY_SAME_SPEC_ON_IDLE_EXCLUSIVE_GPU')
                node.setdefault('operator_failures',[]).append(previous);node.pop('job');node.pop('result',None)
                node.update(status='RESOURCE_REQUEUE',evaluation_exclusive_recovery=True)
                self.flush();return False
            node.update(status='FAILED_REVIEW_REQUIRED', result=value)
            return False
        node['status'] = 'EXITED'
        return True

    def requeue_resource_boundary(self, entry, rank, node, rank_receipt):
        """Resume only the SAME partial shard after a proven resource return.

        Existing scene receipts, source, export, seed and world stay immutable.
        A nonresource stop or failed/forced child retains the review path below.
        """
        from starVLA.rl.qwen_world_safety_exploration.contracts import authorization
        value = self.receipt(node)
        host = node.get('host')
        if (not value or value.get('returncode') != 0 or value.get('forced_expiry')
                or value.get('status') == 'FORCED_EXIT_REVIEW_REQUIRED'
                or rank_receipt.get('status') != 'STOPPED_AT_SCENE_BOUNDARY'):
            return False
        job = Path(node['job']).resolve()
        if not job.is_relative_to((ROOT/'jobs/evaluation').resolve()):
            return False
        grant, _ = authorization()
        guard = (host in grant.get('temporary_hosts', [])
                 and time.time() >= grant['expires_unix']-grant['checkpoint_guard_seconds'])
        authorized = host in allowed_hosts(admit_long_job=True)
        priority_yield = None
        if authorized:
            from .native_priority import verified_yield
            priority_yield = verified_yield(entry, rank, node, value, rank_receipt)
        resource_return = not authorized and ((job/'STOP_REQUESTED').exists() or guard)
        if not resource_return and priority_yield is None:
            return False
        output = Path(entry['evaluation']).resolve()
        identity = read(output/'identity.json', {})
        spec = read(output/'SPEC.json', {})
        identity_body = dict(identity);identity_sha = identity_body.pop('sha256', None)
        spec_body = dict(spec);spec_sha = spec_body.pop('sha256', None)
        if (not identity_sha or fingerprint(identity_body) != identity_sha
                or spec_sha != entry.get('spec_sha256') or fingerprint(spec_body) != spec_sha
                or spec.get('world') != entry['world'] or spec.get('seed') != 2101
                or spec.get('cohort') != entry['cohort'] or spec.get('arm') != entry['arm']
                or node.get('source') != spec.get('source')
                or identity.get('seed') != 2101 or identity.get('cohort') != entry['cohort']
                or rank_receipt.get('identity') != identity_sha
                or rank_receipt.get('rank') != rank or rank_receipt.get('world') != entry['world']
                or rank_receipt.get('host') != host or value.get('host') != host):
            return False
        command = node.get('command', [])
        def argument(flag):
            return command[command.index(flag)+1] if command.count(flag) == 1 and command.index(flag)+1 < len(command) else None
        if (argument('-m') != 'tools.qwen_world_safety_exploration.evaluate'
                or argument('--rank') != str(rank) or argument('--world') != str(entry['world'])
                or argument('--seed') != '2101' or argument('--cohort') != entry['cohort']
                or argument('--output') is None or Path(argument('--output')).resolve() != output
                or value.get('command') != command
                or Path(value.get('cwd', '')).resolve() != Path(node['source']).resolve()):
            return False
        node.setdefault('resource_returns', []).append(dict(job=str(job), result=value,
            rank_receipt=rank_receipt, source=node['source'], command=list(command),
            spec_sha256=spec_sha, identity_sha256=identity_sha,
            reason='OWNED_NATIVE_PRIORITY_YIELD' if priority_yield is not None else 'EXIT0_OWNED_SCENE_BOUNDARY_RESOURCE_RETURN',
            priority_yield_receipt=priority_yield, at_unix=time.time(),
            same_source_world_rank_seed=True, completed_scene_cache_preserved=True))
        node.pop('job');node.pop('result', None)
        node.update(status='RESOURCE_REQUEUE')
        self.flush()
        return True

    def launch_job(self, run, node, command, family, route=None):
        from .evaluation_execution import evaluation_source
        source = evaluation_source(run)
        attempt = int(node.get('attempt', 0))+1
        name = f'evaluation/{family}_attempt{attempt:03}'
        directory = ROOT/'jobs'/name
        # Persist intent before transport: an ambiguous Popen is never retried.
        node.update(attempt=attempt, job=str(directory), command=command,
                    source=source, status='DISPATCHING', dispatched_unix=time.time())
        self.flush()
        kwargs = {}
        if route:
            kwargs = dict(gpus=str(route['gpu']), expected_uuids=[route['uuid']],
                          share_minimum=self.minimum, lease_slot='evaluation')
            if route.get('exclusive_recovery'):
                kwargs=dict(gpus=str(route['gpu']),expected_uuids=[route['uuid']],minimum=61440,
                    lease_slot='evaluation',exclusive_all_slots=True)
            node.update(host=route['host'], gpu_uuid=route['uuid'])
        try:
            launch(node['host'], source, command, name,
                   worker_source=Path(__file__).resolve().parents[2], **kwargs)
            node['status'] = 'STARTING'
        except Exception:
            node.update(status='DISPATCH_UNCERTAIN_REVIEW_REQUIRED', dispatch_error=traceback.format_exc())
        self.flush()

    def samples(self):
        def sample_host(host):
            try:
                sample = probe(host)
                if (sample['worker_python_exists'] and sample['MemAvailable_bytes'] >= 64*2**30
                        and sample['load1'] < sample['CPU_affinity']*1.25):
                    return host, sample
            except Exception:
                pass  # Unreachable authorized hosts remain eligible next pass.
            return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            return [row for row in pool.map(sample_host, allowed_hosts(admit_long_job=True)) if row]

    def discover(self):
        self.registry = self.registry_now()
        current_ids={r['id']for r in self.registry['runs']}
        if not set(self.initial_bindings)<=current_ids:raise ValueError('Registered scientific arms cannot disappear')
        for run in self.registry['runs']:
            if run['id'] not in self.initial_bindings:
                self.initial_bindings[run['id']]=self.binding(run)
                self.state.setdefault('new_run_admissions',{})[run['id']]=dict(binding=self.binding(run),
                    registry_sha256=file_sha(self.registry_path),admitted_unix=time.time())
            if self.binding(run) != self.initial_bindings[run['id']]:
                raise ValueError('Run scientific/source binding changed; explicit owner review required')
            record = self.state['runs'].setdefault(run['id'], dict(gates={}))
            for updates in run['gates']:
                gate = record['gates'].setdefault(str(updates), {})
                if gate.get('status') == 'FAILED_REVIEW_REQUIRED':
                    continue
                cp, native = checkpoint_at_gate(run, updates)
                if cp is None:
                    status=native.get('status','')
                    terminal=(status.startswith(('FAILED','STOPPED')) or
                        (status=='COMPLETED_AT_REGISTERED_CAP' and native.get('updates',-1)>=run['target_updates']))
                    if terminal:
                        gate.update(status='SKIPPED_NATIVE_TERMINAL',native_status=status)
                    elif gate.get('status')=='SKIPPED_NATIVE_TERMINAL':
                        gate.update(status='WAITING_NATIVE_CHECKPOINT',native_status=status)
                    continue
                label = f"{run['id']}_u{updates:06}"
                gate.update(status='CHECKPOINT_READY', checkpoint=cp, label=label)
                for cohort in COUNTS:
                    key = label+'_'+cohort
                    out = ROOT/'evaluations'/label/cohort
                    from .evaluation_execution import evaluation_source
                    spec = dict(schema=SCHEMA, label=label, arm=run['id'], architecture_id=run['architecture_id'],
                        updates=updates, checkpoint=cp, source=evaluation_source(run), config_sha256=run['config_sha256'],
                        parent_checkpoint=run['parent_checkpoint'], parent_selection=self.registry['parent_selection'],
                        cohort=cohort, count=COUNTS[cohort], world=self.max_shards,
                        inference='single_candidate_cleanx_K10_shift5', seed=2101)
                    spec['sha256'] = fingerprint(spec)
                    out.mkdir(parents=True, exist_ok=True)
                    saved = read(out/'SPEC.json')
                    if saved is not None and saved != spec:
                        raise ValueError('Existing evaluation world/checkpoint/source spec must never change')
                    if saved is None:
                        if (out/'identity.json').exists() or list((out/'scenes').glob('*.json')):
                            raise ValueError('Existing unowned evaluation requires explicit adoption review')
                        atomic_json(out/'SPEC.json', spec)
                    entry = self.state['evaluations'].setdefault(key, dict(label=label, arm=run['id'],
                        architecture_id=run['architecture_id'], updates=updates, cohort=cohort,
                        evaluation=str(out), spec_sha256=spec['sha256'], world=self.max_shards, ranks={}))
                    result = complete_summary(out/'SUMMARY.json', architecture=run['architecture_id'], cohort=cohort)
                    if result is not None:
                        entry.update(status='COMPLETE', summary=str(out/'SUMMARY.json'), result=result)

    def helper_active(self):
        for record in self.state['runs'].values():
            for gate in record['gates'].values():
                for node in [gate.get('export', {})]:
                    if node.get('job') and node.get('status') not in ('COMPLETE', 'FAILED_REVIEW_REQUIRED'):
                        if not self.finished(node) and node.get('job'):
                            return True
        for entry in self.state['evaluations'].values():
            node = entry.get('summarizer', {})
            if node.get('job') and node.get('status') not in ('COMPLETE', 'FAILED_REVIEW_REQUIRED'):
                if not self.finished(node) and node.get('job'):
                    return True
        return False

    def advance(self, samples):
        helper_busy = self.helper_active()
        runs = {run['id']:run for run in self.registry['runs']}
        for arm, record in self.state['runs'].items():
            run = runs[arm]
            for updates, gate in record['gates'].items():
                if not gate.get('checkpoint') or gate.get('status') == 'FAILED_REVIEW_REQUIRED':
                    continue
                export = ROOT/'exports'/gate['label'];node = gate.setdefault('export', {})
                if (export/'COMPLETE').exists():
                    meta, seal = read(export/'export.json'), read(export/'COMPLETE')
                    if (seal['export_sha256'] != file_sha(export/'export.json')
                            or meta.get('IL_updates') != 70000 or meta['qualification']
                            or meta.get('RL_updates') != int(updates)
                            or meta.get('RL_checkpoint_files_sha256') != gate['checkpoint']['files_sha256']):
                        raise ValueError('Changed or foreign sealed export')
                    node.update(status='COMPLETE', path=str(export), model_sha256=meta['model_sha256'])
                    continue
                if node.get('job'):
                    if self.finished(node):
                        node.update(status='FAILED_REVIEW_REQUIRED', error='Exporter exited without sealed export')
                    if node.get('status') == 'FAILED_REVIEW_REQUIRED':
                        gate['status'] = 'FAILED_REVIEW_REQUIRED'
                    continue
                if not helper_busy and samples and self.allow_new(run):
                    code = ('import sys;from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint as export_policy;'
                            'export_policy(sys.argv[1],sys.argv[2],sys.argv[3])')
                    command = [MODEL_PYTHON, '-u', '-c', code, run['parent_checkpoint'], str(export), gate['checkpoint']['path']]
                    node['host'] = min(samples, key=lambda pair:pair[1]['load1']/pair[1]['CPU_affinity'])[0]
                    self.launch_job(run, node, command, gate['label']+'/export')
                    helper_busy = True
        for entry in self.state['evaluations'].values():
            if entry.get('status') == 'COMPLETE':
                continue
            for rank in range(entry['world']):
                node = entry['ranks'].setdefault(str(rank), {})
                if node.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED') or not node.get('job'):
                    continue
                if self.finished(node):
                    rank_receipt = read(Path(entry['evaluation'])/f'rank{rank}.json', {})
                    if rank_receipt.get('status') != 'COMPLETE' or rank_receipt.get('world') != entry['world']:
                        if self.requeue_resource_boundary(entry, rank, node, rank_receipt):
                            continue
                        node.update(status='FAILED_REVIEW_REQUIRED', error='Worker exited without full fixed shard; no implicit retry')
                    else:
                        node['status'] = 'COMPLETE'
                if node.get('status') == 'FAILED_REVIEW_REQUIRED':
                    entry.update(status='FAILED_REVIEW_REQUIRED', failure_rank=rank)
            # A failed shard blocks publication, but other detached workers can
            # still finish. Harvest their receipts so completed workers do not
            # permanently consume the concurrency budget. Failed shards retain
            # their review state; this does not authorize a scientific retry.
            if entry.get('status') == 'FAILED_REVIEW_REQUIRED':
                continue
            if all(entry['ranks'][str(rank)].get('status') == 'COMPLETE' for rank in range(entry['world'])):
                node = entry.setdefault('summarizer', {})
                if node.get('job'):
                    if self.finished(node):
                        node['status'] = 'COMPLETE'
                        result = complete_summary(Path(entry['evaluation'])/'SUMMARY.json',
                            architecture=entry['architecture_id'], cohort=entry['cohort'])
                        if result is None:
                            entry.update(status='FAILED_REVIEW_REQUIRED', error='Summarizer exited without full summary')
                        else:
                            entry.update(status='COMPLETE', result=result, summary=str(Path(entry['evaluation'])/'SUMMARY.json'))
                    elif node.get('status') == 'FAILED_REVIEW_REQUIRED':
                        entry.update(status='FAILED_REVIEW_REQUIRED', error='Summarizer failed')
                elif not helper_busy and samples and self.allow_new(runs[entry['arm']]):
                    node['host'] = min(samples, key=lambda pair:pair[1]['load1']/pair[1]['CPU_affinity'])[0]
                    command = [MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_safety_exploration.evaluate',
                        '--summarize', '--output', entry['evaluation'], '--summary-output', str(Path(entry['evaluation'])/'SUMMARY.json')]
                    self.launch_job(runs[entry['arm']], node, command, entry['label']+'/'+entry['cohort']+'/summarize')
                    helper_busy = True

    def stop_new(self):
        return (ROOT/'STOP_REQUESTED').exists() or (self.directory/'STOP_NEW_EVALUATIONS').exists()

    def allow_new(self, run):
        return not self.stop_new() and run.get('new_evaluation_authorized', True) is not False

    def dispatch_ranks(self, samples):
        if self.stop_new():
            return
        live = sum(bool(node.get('job')) and node.get('status') not in ('COMPLETE', 'FAILED_REVIEW_REQUIRED')
            for entry in self.state['evaluations'].values() for node in entry['ranks'].values())
        if live >= self.max_workers:
            return
        with (ROOT/'resources/EVALUATION_DISPATCH.lock').open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            claims = read(self.claims_path, {})
            for uuid, claim in list(claims.items()):
                try:
                    if 'returncode' in job_state(claim['receipt']):
                        del claims[uuid]
                except (KeyError, FileNotFoundError, ValueError):
                    pass  # Unknown claims remain reserved, never stolen.
            routes = [dict(host=host,gpu=gpu['index'],uuid=gpu['uuid'],free_MiB=gpu['free_MiB'],
                           has_owners=any(o['uuid']==gpu['uuid']for o in sample['owners']))
                for host, sample in samples for gpu in sample['gpus']
                if usable_gpu(gpu, sample, minimum=self.minimum, reserved=claims)]
            from .native_priority import filter_eval_routes
            routes = filter_eval_routes(routes, samples)
            runs = {run['id']:run for run in self.registry['runs']}
            # Finish a nearly-complete checkpoint before opening more low ranks.
            # Existing worlds/specs/rank commands are immutable; only order changes.
            entries = sorted(self.state['evaluations'].values(), key=completion_priority)
            for entry in entries:
                for rank in range(entry['world']):
                    if not routes or live >= self.max_workers:
                        break
                    gate = self.state['runs'][entry['arm']]['gates'][str(entry['updates'])]
                    if gate.get('export', {}).get('status') != 'COMPLETE' or entry.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED'):
                        continue
                    node = entry['ranks'].setdefault(str(rank), {})
                    if node.get('job') or node.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED', 'DISPATCH_UNCERTAIN_REVIEW_REQUIRED'):
                        continue
                    index=0
                    retry_hosts=node.get('operator_retry_hosts')
                    if retry_hosts:
                        index=next((i for i,r in enumerate(routes)if r['host']in retry_hosts),None)
                        if index is None:continue
                    if node.get('evaluation_exclusive_recovery'):
                        index=next((i for i,r in enumerate(routes)if(not retry_hosts or r['host']in retry_hosts)and not r['has_owners']and r['free_MiB']>=61440 and
                            all(lease_free(LEASES[name]/(r['uuid']+'.lock'))for name in ('primary','shared','research'))),None)
                        if index is None:continue
                    route=routes.pop(index);run=runs[entry['arm']]
                    if node.get('evaluation_exclusive_recovery'):route['exclusive_recovery']=True
                    if not self.allow_new(run):
                        routes.insert(0, route)
                        continue
                    data = self.registry['evaluation']['data_paths'][entry['cohort']]
                    command = [MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_safety_exploration.evaluate',
                        '--parent-checkpoint',run['parent_checkpoint'],'--rl-checkpoint',gate['checkpoint']['path'],
                        '--export', gate['export']['path'], '--output', entry['evaluation'],
                        '--cohort', entry['cohort'], '--data', data, '--manifest',self.registry['evaluation']['manifests'][entry['cohort']],
                        '--rank', str(rank), '--world', str(entry['world']), '--seed', '2101']
                    if rank == 0:
                        command += ['--native-parity-output', str(Path(entry['evaluation'])/'native_export_parity.json')]
                    self.launch_job(run, node, command, entry['label']+'/'+entry['cohort']+f'/rank{rank:02}', route)
                    # Every successful/ambiguous launch claims immediately, even
                    # if another launch later fails, preserving partial dispatch.
                    claims[route['uuid']] = dict(receipt=node['job'], owner='RESEARCH_EVALUATION',
                        label=entry['label'], cohort=entry['cohort'], rank=rank,
                        host=route['host'], updated_unix=time.time())
                    atomic_json(self.claims_path, claims)
                    entry['status'] = 'RUNNING';live += 1
            atomic_json(self.claims_path, claims)

    def done(self):
        for run in self.registry['runs']:
            for update in run['gates']:
                gate = self.state['runs'].get(run['id'], {}).get('gates', {}).get(str(update), {})
                if gate.get('status') in ('SKIPPED_NATIVE_TERMINAL', 'FAILED_REVIEW_REQUIRED'):
                    continue
                if not gate.get('checkpoint'):
                    return False
                if any(self.state['evaluations'].get(gate['label']+'_'+cohort, {}).get('status')
                       not in ('COMPLETE', 'FAILED_REVIEW_REQUIRED') for cohort in COUNTS):
                    return False
        return True

    def tick(self):
        self.discover()
        needs_resources = any(gate.get('checkpoint') and gate.get('status') != 'FAILED_REVIEW_REQUIRED'
            for record in self.state['runs'].values() for gate in record['gates'].values()) and not self.done()
        samples = self.samples() if needs_resources and not self.stop_new() else []
        self.advance(samples)
        self.dispatch_ranks(samples)
        self.flush()

    def run(self, *, interval=30, once=False):
        while True:
            if (self.directory/'STOP_OWNER').exists():
                self.state['status'] = 'STOPPED_OWNER_ONLY_EXISTING_WORKERS_UNTOUCHED';break
            try:
                self.tick()
                if self.done():
                    failures = (any(entry.get('status') == 'FAILED_REVIEW_REQUIRED' for entry in self.state['evaluations'].values())
                        or any(gate.get('status') in ('FAILED_REVIEW_REQUIRED','SKIPPED_NATIVE_TERMINAL') for record in self.state['runs'].values() for gate in record['gates'].values()))
                    self.state['status'] = 'FINISHED_WITH_REVIEW_REQUIRED' if failures else 'COMPLETE_AT_REGISTERED_GATES'
                    break
            except Exception:
                self.state.update(status='OWNER_REVIEW_REQUIRED', error=traceback.format_exc())
                self.flush()
                raise
            if once:
                break
            time.sleep(interval)
        self.flush()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registry', default=str(ROOT/'RUNS.json'))
    parser.add_argument('--max-shards', type=int, default=32)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--minimum-free-mib', type=int, default=32768)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true', help='One real authorized scheduling pass; not a dry run')
    args = parser.parse_args()
    if not 1 <= args.max_shards <= 32 or not 1 <= args.max_workers <= 64 or args.interval < 5 or args.minimum_free_mib < 8192:
        parser.error('Require shards/workers1..32, interval>=5s and valid measured sharing headroom')
    EvaluationOwner(args.registry, max_shards=args.max_shards, max_workers=args.max_workers,
                    minimum=args.minimum_free_mib).run(interval=args.interval, once=args.once)


if __name__ == '__main__':
    main()
