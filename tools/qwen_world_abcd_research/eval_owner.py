"""Independent finite asynchronous ABCD DEV/NAVTEST owner; never launches training.

Run with explicit WORLD_COMPARE_ROOT/TASK. Each registered frozen run.source is
used for every export/evaluate/summarize command. Only proven pre-child resource
75 is requeued. Other failures preserve receipts for review, never zero scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from starVLA.rl.qwen_world_compare.contracts import (ROOT, TASK, allowed_hosts,
    file_sha, fingerprint)
from starVLA.rl.qwen_world_compare.resources import probe, MODEL_PYTHON, gpu_activity_issue
from starVLA.rl.flow_grpo.transactions import atomic_json
from .dispatch import launch, job_state

SCHEMA = 'ABCD_best_parent_async_evaluation_owner_v1'
COUNTS = {'DEV': 1696, 'NAVTEST': 12146}
LEASES = {name: Path('/mnt/project')/folder for name, folder in (
    ('primary', 'runtime_gpu_leases'), ('shared', 'runtime_gpu_shared_workers'),
    ('research', 'runtime_gpu_research_slots'), ('evaluation', 'runtime_gpu_evaluation_slots'))}
KNOWN_ROOTS = ('/mnt/project/qwen-sourceflow-', '/mnt/project/qwen-world-postcot-',
               '/mnt/project/qwen-world-four-frameworks-', '/mnt/project/qwen-world-abcd-rl-best-20261031/')
TERMINAL_NATIVE = ('FAILED', 'STOPPED', 'COMPLETED_AT_REGISTERED_CAP')


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
            or identity.get('architecture_id') != run['architecture_id']
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
            or value.get('architecture_id') != architecture or value.get('cohort') != cohort
            or value.get('qualification') is not False or value.get('scenes') != COUNTS[cohort]
            or value.get('scene_denominator') != COUNTS[cohort] or value.get('invalid_scenes') != 0):
        raise ValueError('Only sealed full-population canonical summaries are publishable')
    return value


class EvaluationOwner:
    def __init__(self, registry, *, max_shards=32, max_workers=32, minimum=32768):
        if TASK != 'qwen_world_abcd_rl_best_20261031' or not os.environ.get('WORLD_COMPARE_ROOT'):
            raise PermissionError('Explicit independent ABCD ROOT/TASK required before import')
        self.registry_path = Path(registry).resolve()
        if self.registry_path != (ROOT/'RUNS.json').resolve():
            raise ValueError('Only this task RUNS registry may authorize evaluations')
        self.directory = ROOT/'owners/ABCD_EVALUATION'
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
        return {key: run[key] for key in ('id', 'architecture_id', 'root', 'source', 'config',
                                          'config_sha256', 'parent_export', 'gates', 'target_updates')}

    def registry_now(self):
        value = read(self.registry_path)
        if value.get('task') != TASK or len(value.get('runs', [])) != 4:
            raise ValueError('Exactly the independent registered four architecture arms required')
        if {row['architecture_id'] for row in value['runs']} != set('ABCD'):
            raise ValueError('Duplicate or missing architecture registration')
        for run in value['runs']:
            if file_sha(run['config']) != run['config_sha256']:
                raise ValueError('Registered scientific config bytes changed')
            if not Path(run['root']).resolve().is_relative_to(ROOT.resolve()/'runs'):
                raise ValueError('Foreign run root')
            if not (Path(run['source'])/'tools/qwen_world_abcd_research/evaluate.py').is_file():
                raise ValueError('Frozen run source lacks the registered evaluator')
        return value

    def flush(self):
        self.state['heartbeat_unix'] = time.time()
        atomic_json(self.state_path, self.state)

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
            node.update(status='FAILED_REVIEW_REQUIRED', result=value)
            return False
        node['status'] = 'EXITED'
        return True

    def launch_job(self, run, node, command, family, route=None):
        attempt = int(node.get('attempt', 0))+1
        name = f'evaluation/{family}_attempt{attempt:03}'
        directory = ROOT/'jobs'/name
        # Persist intent before transport: an ambiguous Popen is never retried.
        node.update(attempt=attempt, job=str(directory), command=command,
                    source=run['source'], status='DISPATCHING', dispatched_unix=time.time())
        self.flush()
        kwargs = {}
        if route:
            kwargs = dict(gpus=str(route['gpu']), expected_uuids=[route['uuid']],
                          share_minimum=self.minimum, lease_slot='evaluation')
            node.update(host=route['host'], gpu_uuid=route['uuid'])
        try:
            launch(node['host'], run['source'], command, name,
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
        for run in self.registry['runs']:
            if self.binding(run) != self.initial_bindings[run['id']]:
                raise ValueError('Run scientific/source binding changed; explicit owner review required')
            record = self.state['runs'].setdefault(run['id'], dict(gates={}))
            for updates in run['gates']:
                gate = record['gates'].setdefault(str(updates), {})
                if gate.get('status') in ('FAILED_REVIEW_REQUIRED', 'SKIPPED_NATIVE_TERMINAL'):
                    continue
                cp, native = checkpoint_at_gate(run, updates)
                if cp is None:
                    if any(native.get('status', '').startswith(prefix) for prefix in TERMINAL_NATIVE):
                        gate.update(status='SKIPPED_NATIVE_TERMINAL', native_status=native.get('status'))
                    continue
                label = f"{run['id']}_u{updates:06}"
                gate.update(status='CHECKPOINT_READY', checkpoint=cp, label=label)
                for cohort in COUNTS:
                    key = label+'_'+cohort
                    out = ROOT/'evaluations'/label/cohort
                    spec = dict(schema=SCHEMA, label=label, arm=run['id'], architecture_id=run['architecture_id'],
                        updates=updates, checkpoint=cp, source=run['source'], config_sha256=run['config_sha256'],
                        parent_export=run['parent_export'], parent_selection=self.registry['parent_selection'],
                        cohort=cohort, count=COUNTS[cohort], world=self.max_shards,
                        inference='single_candidate_cleanx_K10_shift5', seed=2101)
                    if cohort == 'NAVTEST':
                        selected = read(self.registry['parent_selection'])['selected'][run['architecture_id']]
                        spec['paired_IL_baseline'] = str(Path(selected['evaluation_identity']).parent)
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
                            or meta['architecture_id'] != run['architecture_id'] or meta['qualification']
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
                    code = ('import sys;from starVLA.rl.qwen_world_abcd_research.deployment import export_policy;'
                            'export_policy(sys.argv[1],sys.argv[2],sys.argv[3])')
                    command = [MODEL_PYTHON, '-u', '-c', code, run['parent_export'], str(export), gate['checkpoint']['path']]
                    node['host'] = min(samples, key=lambda pair:pair[1]['load1']/pair[1]['CPU_affinity'])[0]
                    self.launch_job(run, node, command, gate['label']+'/export')
                    helper_busy = True
        for entry in self.state['evaluations'].values():
            if entry.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED'):
                continue
            for rank in range(entry['world']):
                node = entry['ranks'].setdefault(str(rank), {})
                if node.get('status') == 'COMPLETE' or not node.get('job'):
                    continue
                if self.finished(node):
                    rank_receipt = read(Path(entry['evaluation'])/f'rank{rank}.json', {})
                    if rank_receipt.get('status') != 'COMPLETE' or rank_receipt.get('world') != entry['world']:
                        node.update(status='FAILED_REVIEW_REQUIRED', error='Worker exited without full fixed shard; no implicit retry')
                    else:
                        node['status'] = 'COMPLETE'
                if node.get('status') == 'FAILED_REVIEW_REQUIRED':
                    entry.update(status='FAILED_REVIEW_REQUIRED', failure_rank=rank)
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
                    command = [MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_abcd_research.evaluate',
                        '--summarize', '--evaluation', entry['evaluation'], '--output', str(Path(entry['evaluation'])/'SUMMARY.json')]
                    if entry['cohort'] == 'NAVTEST':
                        selected = read(self.registry['parent_selection'])['selected'][entry['architecture_id']]
                        command += ['--baseline', str(Path(selected['evaluation_identity']).parent)]
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
            routes = [dict(host=host, gpu=gpu['index'], uuid=gpu['uuid'])
                for host, sample in samples for gpu in sample['gpus']
                if usable_gpu(gpu, sample, minimum=self.minimum, reserved=claims)]
            runs = {run['id']:run for run in self.registry['runs']}
            # Round-robin ranks across cohorts so DEV does not block NAVTEST.
            entries = list(self.state['evaluations'].values())
            for rank in range(self.max_shards):
                for entry in entries:
                    if not routes or live >= self.max_workers:
                        break
                    gate = self.state['runs'][entry['arm']]['gates'][str(entry['updates'])]
                    if gate.get('export', {}).get('status') != 'COMPLETE' or entry.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED'):
                        continue
                    node = entry['ranks'].setdefault(str(rank), {})
                    if node.get('job') or node.get('status') in ('COMPLETE', 'FAILED_REVIEW_REQUIRED', 'DISPATCH_UNCERTAIN_REVIEW_REQUIRED'):
                        continue
                    route = routes.pop(0);run = runs[entry['arm']]
                    if not self.allow_new(run):
                        routes.insert(0, route)
                        continue
                    parent = read(Path(run['parent_export'])/'export.json')['IL_parent']
                    data = str(Path(parent['identity']['config']['data_root'])/entry['cohort'])
                    command = [MODEL_PYTHON, '-u', '-m', 'tools.qwen_world_abcd_research.evaluate',
                        '--export', gate['export']['path'], '--output', entry['evaluation'],
                        '--cohort', entry['cohort'], '--data', data, '--rank', str(rank), '--world', str(entry['world']), '--seed', '2101']
                    if entry['cohort'] == 'NAVTEST':
                        command += ['--selection', self.registry['parent_selection']]
                    if rank == 0 and entry['cohort'] == 'DEV':
                        command += ['--native-parity-output', str(Path(entry['evaluation'])/'native_export_parity.json')]
                    self.launch_job(run, node, command, entry['label']+'/'+entry['cohort']+f'/rank{rank:02}', route)
                    # Every successful/ambiguous launch claims immediately, even
                    # if another launch later fails, preserving partial dispatch.
                    claims[route['uuid']] = dict(receipt=node['job'], owner='ABCD_EVALUATION',
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
                        or any(gate.get('status') == 'FAILED_REVIEW_REQUIRED' for record in self.state['runs'].values() for gate in record['gates'].values()))
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
    parser.add_argument('--max-workers', type=int, default=32)
    parser.add_argument('--minimum-free-mib', type=int, default=32768)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true', help='One real authorized scheduling pass; not a dry run')
    args = parser.parse_args()
    if not 1 <= args.max_shards <= 32 or not 1 <= args.max_workers <= 32 or args.interval < 5 or args.minimum_free_mib < 8192:
        parser.error('Require shards/workers1..32, interval>=5s and valid measured sharing headroom')
    EvaluationOwner(args.registry, max_shards=args.max_shards, max_workers=args.max_workers,
                    minimum=args.minimum_free_mib).run(interval=args.interval, once=args.once)


if __name__ == '__main__':
    main()
