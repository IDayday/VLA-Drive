"""CPU handoff for the four fixed endpoints; reuse qualified final entry points.

No trainer, early Navtest, checkpoint selection or automatic worker retry.
Ambiguous/failed stages retain their intent and stop for explicit inspection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import traceback

from scripts.analysis.fm_transfer import dispatch_final_evaluation as dispatch
from scripts.analysis.fm_transfer.final_lock import readiness, authenticated, EVALUATION_SOURCE
from scripts.analysis.fm_transfer.milestone_controller import remote, worker_allowed
from starVLA.rl.fm_transfer.contracts import ROOT, require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
SOURCE = ROOT / 'source/d58485b08078a6319b2835793d8c79ffbbd3e9d5'


def read(path):
    return json.loads(Path(path).read_text())


def verify_source_snapshot(path):
    """Accept the two existing publisher schemas, never a conflicting commit."""
    path = Path(path)
    seal = read(path / 'SOURCE_COMPLETE.json')
    identities = [seal[k] for k in ('commit', 'git_sha') if k in seal]
    if (len(path.name) != 40 or any(c not in '0123456789abcdef' for c in path.name)
            or not identities or any(value != path.name for value in identities)
            or seal.get('git_archive_extraction_complete') is not True):
        raise ValueError('Controller and worker sources must be immutable published snapshots')
    return seal


def configuration(root, control_source, hosts):
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError('Distinct candidate hosts required')
    value = dict(schema='fm_transfer_final_handoff_v1', root=str(Path(root).resolve()),
        controller_source=str(Path(control_source).resolve()), worker_source=str(SOURCE),
        evaluation_source=str(EVALUATION_SOURCE), hosts=list(hosts), models=list(dispatch.MODELS),
        training_seed=42, inference_seed=2101, modes=['E_NATIVE', 'E_SHIFT5'],
        fixed_RL_endpoint=20000, no_automatic_retry=True, no_training=True)
    value['sha256'] = fingerprint(value)
    return value


def stage(control, name, command, *, host=None, source=None):
    """Durable one-attempt commands; a lost response never authorizes replay."""
    source=SOURCE if source is None else Path(source)
    directory = Path(control) / 'stages' / name
    intent = dict(command=list(command), host=host, source=str(source))
    if directory.exists():
        old = read(directory / 'intent.json')
        result = read(directory / 'result.json') if (directory / 'result.json').exists() else None
        if (old != intent or not result or result['status'] != 'COMPLETE' or result.get('exit_code') != 0
                or result.get('command') != list(command) or result.get('host') != host):
            raise RuntimeError('Unresolved/changed stage; inspect without automatic retry: ' + name)
        return result
    require_dispatch()
    if host:
        worker_allowed(host)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / 'intent.json', intent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    invocation = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=' + str(source),
                  'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', *command]
    local = host is None or host == socket.gethostname().removesuffix('-worker-0')
    actual = command if local else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                   host, shlex.join(invocation)]
    result = dict(status='RUNNING', start_unix=time.time(), command=command, host=host)
    atomic_json(directory / 'result.json', result)
    try:
        with (directory / 'command.log').open('xb') as log:
            child = subprocess.Popen(actual, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            result['child'] = dict(pid=child.pid,
                start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
            atomic_json(directory / 'result.json', result)
            code = child.wait()
        result.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
        if code:
            raise RuntimeError('Existing entry point failed; preserve workers and inspect: ' + name)
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        result['end_unix'] = time.time()
        atomic_json(directory / 'result.json', result)
    return result


def probe(host):
    program = ('import json,os,subprocess,time\nfrom pathlib import Path\n' +
               inspect.getsource(dispatch.gpu_snapshot) + '''
value=gpu_snapshot();capacity=float(value['CPU_affinity']);limits=[]
p=Path('/sys/fs/cgroup/cpu.max')
if p.exists():
 text=p.read_text().split();limits.append(dict(path=str(p),value=text))
 if text[0]!='max':capacity=min(capacity,int(text[0])/int(text[1]))
for root in ['/sys/fs/cgroup/cpu','/sys/fs/cgroup/cpu,cpuacct']:
 q=Path(root)/'cpu.cfs_quota_us';p=Path(root)/'cpu.cfs_period_us'
 if q.exists() and p.exists():
  quota,period=int(q.read_text()),int(p.read_text());limits.append(dict(path=root,quota=quota,period=period))
  if quota>0:capacity=min(capacity,quota/period)
value.update(CPU_capacity=capacity,CPU_quota_observations=limits)
print(json.dumps(value))
''')
    value = json.loads(remote(host, [PYTHON, '-c', program]))
    value['gpus'] = {int(k): v for k, v in value['gpus'].items()}
    return value


def available_devices(samples, reserved=()):
    devices, reasons, seen = [], {}, set(reserved)
    for host, pair in samples.items():
        first, second = pair
        if set(first['gpus']) != set(second['gpus']):
            raise ValueError('Physical inventory changed between observations')
        capacity = min(float(s.get('CPU_capacity', s['CPU_affinity'])) for s in pair)
        spare = max(0, math.floor(capacity - max(float(s['load'][0]) for s in pair)))
        admitted = 0
        for gpu in sorted(second['gpus']):
            reason = None
            if first['gpus'][gpu]['uuid'] != second['gpus'][gpu]['uuid']:
                raise ValueError('Physical GPU identity changed between observations')
            uuid = second['gpus'][gpu]['uuid']
            if uuid in seen:
                reason = 'aliased_or_reserved_UUID'
            elif admitted >= spare:
                reason = 'CPU_headroom'
            else:
                try:
                    dispatch.admit_devices(first, second, [(0, gpu)], seen)
                except RuntimeError as exc:
                    reason = str(exc)
            if reason:
                reasons[f'{host}:{gpu}'] = reason
            else:
                devices.append(dict(host=host, gpu=gpu, GPU_uuid=uuid))
                seen.add(uuid)
                admitted += 1
    return devices, reasons


def allocate(models, devices):
    """Use all admitted unique GPUs, assigning contiguous shards per model."""
    if len({d['GPU_uuid'] for d in devices}) != len(devices):
        raise ValueError('A physical device cannot be allocated twice')
    if not devices:
        return {}
    count = min(len(models), len(devices))
    if not count:
        return {}
    plans, offset = {}, 0
    for index, model in enumerate(models[:count]):
        size = len(devices) // count + int(index < len(devices) % count)
        if not 1 <= size <= 64:
            raise ValueError('Unsupported final shard count')
        plans[model] = dict(shards=size, assignments=[dict(rank=rank, **device)
            for rank, device in enumerate(devices[offset:offset + size])])
        offset += size
    return plans


def model_state(root, model, plan, is_live=dispatch.live, *, evaluation_source=None):
    evaluation_source=EVALUATION_SOURCE if evaluation_source is None else Path(evaluation_source)
    output = Path(root) / 'evaluations' / (model + '_final_NAVTEST')
    receipts = [authenticated(p) for p in sorted(output.glob('dispatches/*.json'))]
    if not receipts:
        raise RuntimeError('Reserved route has no worker receipt; inspect before launching again')
    by_rank = {r['rank']: r for r in receipts}
    if len(receipts) != plan['shards'] or set(by_rank) != set(range(plan['shards'])):
        raise RuntimeError('Partial/duplicate final dispatch requires explicit inspection')
    manifest = authenticated(output / 'manifest.json') if (output / 'manifest.json').exists() else None
    done, reserved = [], []
    for item in plan['assignments']:
        row = by_rank[item['rank']]
        if (any(row[k] != item[k] for k in ('host', 'gpu', 'GPU_uuid', 'rank')) or
                row['shards'] != plan['shards'] or row['source'] != str(evaluation_source) or
                row['output'] != str(output) or row['model'] != model):
            raise ValueError('Final worker differs from its reserved route')
        alive = is_live(row)
        path = output / f"rank{row['rank']}.json"
        report = read(path) if path.exists() else None
        complete = bool(report and report['status'] == 'ASSIGNED_COMPLETE')
        if complete and (not manifest or report['identity_sha256'] != manifest['sha256']):
            raise ValueError('Foreign final worker completion')
        if not alive and not complete:
            raise RuntimeError('Final worker failed; no automatic retry')
        if alive:
            reserved.append(row['GPU_uuid'])
        done.append(complete)
    return dict(status='WORKERS_EXITED_COMPLETE' if all(done) and not reserved else 'RUNNING',
                reserved=reserved, completed_shards=sum(done))


def validate_summary(result, manifest, selection, model):
    chosen = selection['models'][model]
    lock_body = {k: v for k, v in selection.items() if k != 'sha256'}
    if (result['status'] != 'COMPLETE_POPULATION' or result['scenes'] != 12146 or result['logs'] != 136
            or result['identity'] != manifest or manifest['cohort'] != 'NAVTEST'
            or manifest['seed'] != 2101 or manifest['modes'] != ['E_NATIVE', 'E_SHIFT5']
            or manifest['qualification_only'] or manifest['final_lock'] != lock_body
            or manifest['population_identity'] != selection['population_identity']
            or manifest['source_sha256'] != selection['sampler_sha256']
            or manifest['transfer_evaluation_sha256'] != selection['transfer_evaluation_sha256']
            or any(manifest[k] != chosen[k] for k in ('weight_sha256', 'stage', 'new_SFT'))
            or manifest['export_receipt_sha256'] != chosen['export_receipt_sha256']):
        raise ValueError('Final summary has not proved its complete locked population')


def tick(value, control):
    root, control = Path(value['root']), Path(control)
    ready = readiness(root)
    if ready['missing']:
        return dict(status=ready['status'], missing=ready['missing'], Navtest_opened=False)
    lock = root / 'final_navtest/LOCK.json'
    stage(control, 'lock', [PYTHON, '-m', 'scripts.analysis.fm_transfer.final_lock',
          'lock', '--root', str(root), '--source', str(EVALUATION_SOURCE), '--output', str(lock)])
    selection = authenticated(lock)
    if selection['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST' or set(selection['models']) != set(dispatch.MODELS):
        raise ValueError('Wrong final selection')
    routes_path = control / 'routes.json'
    routes = read(routes_path) if routes_path.exists() else {}
    states = {model: model_state(root, model, plan) for model, plan in routes.items()}
    reserved = {u for state in states.values() for u in state['reserved']}
    missing_models = [m for m in dispatch.MODELS if m not in routes]
    if missing_models:
        hosts, excluded = [], {}
        for host in value['hosts']:
            try:
                worker_allowed(host)
                hosts.append(host)
            except PermissionError as exc:
                excluded[host] = str(exc)
        def two(host):
            try:
                first = probe(host); time.sleep(2); return first, probe(host)
            except (OSError, subprocess.SubprocessError, PermissionError) as exc:
                return dict(observation_failed=repr(exc), capacity_not_inferred=True)
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
            observations = dict(zip(hosts, pool.map(two, hosts)))
        samples = {h: pair for h, pair in observations.items() if not isinstance(pair, dict)}
        excluded.update({h: pair for h, pair in observations.items() if isinstance(pair, dict)})
        devices, reasons = available_devices(samples, reserved)
        admission = dict(time=time.time(), observations=samples, excluded_hosts=excluded,
                         excluded_devices=reasons, selected_devices=devices)
        admission_path = control / ('inventory_' + str(time.time_ns()) + '.json')
        atomic_json(admission_path, admission)
        planned = allocate(missing_models, devices)
        routes.update(planned)
        atomic_json(routes_path, routes)
        for model, plan in planned.items():
            for host in dict.fromkeys(d['host'] for d in plan['assignments']):
                rows = [d for d in plan['assignments'] if d['host'] == host]
                stage(control, model + '_' + host, [PYTHON, '-m',
                    'scripts.analysis.fm_transfer.dispatch_final_evaluation', '--lock', str(lock),
                    '--source', str(EVALUATION_SOURCE), '--model', model, '--shards', str(plan['shards']),
                    '--ranks', ','.join(str(d['rank']) for d in rows),
                    '--gpus', ','.join(str(d['gpu']) for d in rows)], host=host)
        if planned:
            return dict(status='FINAL_PARTITIONS_STARTED', planned=planned, Navtest_opened=True)
    analyses = {}
    for model, state in states.items():
        if state['status'] != 'WORKERS_EXITED_COMPLETE':
            continue
        output = root / 'evaluations' / (model + '_final_NAVTEST')
        summary = root / 'analysis' / (output.name + '.json')
        stage(control, 'analyze_' + model, [PYTHON, '-m', 'scripts.analysis.fm_transfer.analyze_evaluation',
              '--evaluation', str(output), '--output', str(summary)])
        result = read(summary)
        validate_summary(result, authenticated(output / 'manifest.json'), selection, model)
        analyses[model] = dict(path=str(summary), sha256=file_sha(summary))
    paired = {}
    for variant in ('A_DDP_TIME', 'B_SOURCE_ACTION_DIT'):
        if not all(variant + '_' + stage in analyses for stage in ('SFT', 'RL')):
            continue
        output = root / 'evaluations' / (variant + '_RL_final_NAVTEST')
        reference = root / 'evaluations' / (variant + '_SFT_final_NAVTEST')
        summary = root / 'analysis' / (output.name + '_paired_own_SFT.json')
        stage(control, 'paired_' + variant, [PYTHON, '-m', 'scripts.analysis.fm_transfer.analyze_evaluation',
              '--evaluation', str(output), '--reference', str(reference), '--output', str(summary)])
        result = read(summary)
        current = read(analyses[variant + '_RL']['path'])
        baseline = read(analyses[variant + '_SFT']['path'])
        if (result['status'] != 'COMPLETE_POPULATION' or result['reference'] != str(reference)
                or result['identity'] != current['identity'] or result['scenes'] != 12146 or result['logs'] != 136):
            raise ValueError('Incomplete own-SFT final comparison')
        for mode in value['modes']:
            comparison = result['modes'][mode]['paired_comparison']
            delta = current['modes'][mode]['mean']['score'] - baseline['modes'][mode]['mean']['score']
            if (abs(delta - comparison['PDMS_log_cluster_bootstrap']['mean']) > 1e-12 or
                    comparison['reference_safe_population'] != baseline['modes'][mode]['fully_safe_scenes']):
                raise ValueError('Final paired gain or fixed reference-safe population disagrees')
        paired[variant] = dict(path=str(summary), sha256=file_sha(summary))
    complete = len(analyses) == 4 and len(paired) == 2
    return dict(status='ALL_FINAL_POPULATIONS_AND_PAIRS_COMPLETE' if complete else
                'WAITING_FOR_FINAL_WORKERS_OR_CAPACITY', states=states, analyses=analyses,
                paired=paired, final_lock_sha256=file_sha(lock), Navtest_opened=bool(routes))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--control', type=Path, default=ROOT / 'final_handoff_v1')
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require_dispatch()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only controller required')
    source = Path(__file__).resolve().parents[3]
    value = configuration(ROOT, source, args.hosts)
    for path in (SOURCE, EVALUATION_SOURCE, source):
        verify_source_snapshot(path)
    args.control.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'final_handoff.owner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.control / 'plan.json'
        if path.exists() and read(path) != value:
            raise ValueError('Explicit handoff required for changed controller or routing')
        if not path.exists() and ((ROOT / 'final_navtest/LOCK.json').exists() or
                                 list((ROOT / 'final_navtest/workers').glob('*/*.json'))):
            raise ValueError('Existing final work requires explicit adoption')
        atomic_json(path, value)
        owner = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                     start_unix=time.time(), plan_sha256=value['sha256'], source_sha256=file_sha(__file__))
        atomic_json(args.control / 'owner.json', owner)
        try:
            while True:
                require_dispatch()
                result = tick(value, args.control)
                result.update(owner=owner, observed_unix=time.time())
                atomic_json(args.control / 'status.json', result)
                if result['status'] == 'ALL_FINAL_POPULATIONS_AND_PAIRS_COMPLETE':
                    atomic_json(args.control / 'COMPLETE.json', result)
                    break
                if args.once or (args.control / 'STOP_AFTER_TICK').exists():
                    break
                time.sleep(30)
        except BaseException:
            atomic_json(args.control / 'failure.json', dict(owner=owner, error=traceback.format_exc(),
                        end_unix=time.time(), workers_preserved=True, automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
