"""Independent, bounded official NAVTEST queue for the two formal AUG runs.

No training state or old CAMPAIGN is edited. Import/discovery/CLI inspection do
not dispatch. Root explicitly starts --run --execute, or calls tick(...).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import allowed_hosts, file_sha, fingerprint
from tools.d44_native_rl.launch import launch, DispatchPreflightFailed

ROOT = Path('/mnt/project/d46-priority-rl-20261103')
SOURCE = Path(__file__).resolve().parents[2]
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASK = 'd46_priority_rl_20261103'
SCHEMA = 'd46_augustus_native_v1'
PARENT_SHA = '386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
WORLD, POPULATION = 32, 12146
SPARE_GPUS = {'training-vlawm-zt2': (0, 1, 2, 3, 4, 5, 7),
              'training-vlawm-tsinghua-zt4': (4, 7),
              'training-rl-zt2': (1, 7)}
EARLY = {'AUG1_FULL': (100, 200), 'AUG2_LORA': (200, 400)}
EARLY.update(AUG3_FULL_ALIGNED=(100,200,400),AUG4_LORA_ALIGNED=(200,400,800))
VERSIONS={name:('_v2'if 'ALIGNED'in name else'_v1')for name in EARLY}
SCIENCES={name:('augustus_aligned_science_v2'if 'ALIGNED'in name else'augustus_science_v1')for name in EARLY}
TERMINAL = ('COMPLETE', 'FAILED_REQUIRES_REVIEW', 'FAILED_SUMMARY_REQUIRES_REVIEW',
            'DISPATCH_REQUIRES_RECONCILIATION')


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def initial_state():
    return dict(schema='d46_augustus_official_evaluation_queue_v1', cases={}, at=time.time(),
                world=WORLD, max_active_cases=2, training_labels_exported=False)


def state_path(root):
    return Path(root) / 'operations/AUGUSTUS_CAMPAIGN_20261103/EVAL_STATE.json'


def inspect_checkpoint(root, route, checkpoint):
    """Small sealed metadata now; the original evaluator rehashes real weights."""
    root = Path(root).resolve()
    exp = route['experiment']
    output = Path(route['formal_output']).resolve()
    cp = Path(checkpoint).resolve()
    if exp not in EARLY or output != root / 'runs' / (exp + VERSIONS[exp]) or cp.parent != output / 'checkpoints':
        raise ValueError('Not this registered formal AUG checkpoint')
    identity, seal, files = (read(cp / name, {}) for name in ('identity.json', 'COMPLETE', 'files.json'))
    body = dict(identity); digest = body.pop('sha256', None)
    recipe = identity.get('recipe', {})
    if (not files or seal.get('files_sha256') != file_sha(cp / 'files.json')
            or identity.get('sha256') != fingerprint(body)
            or files.get('identity.json') != file_sha(cp / 'identity.json')
            or identity.get('schema') != 'd44_native_rl_v1' or identity.get('architecture_id') != 'D'
            or identity.get('algorithm_schema') != SCHEMA or identity.get('experiment_id') != exp
            or identity.get('parent_model_sha256') != PARENT_SHA or identity.get('qualification') is not False
            or recipe.get('experiment_id') != exp or recipe.get('parent') != 'D_IL46K'
            or recipe.get('training_uses_navtest') is not False
            or identity.get('augustus_protocol_sha256') != fingerprint(recipe.get('augustus', {}))
            or seal.get('world') != route['route']['world'] or identity.get('world') != seal.get('world')
            or seal.get('phase') not in ('READY', 'COMPLETE') or seal.get('next_inner') != 0
            or type(seal.get('updates')) is not int or seal['updates'] <= 0):
        raise ValueError('Formal AUG algorithm/parent/protocol/native seal mismatch')
    source = Path(route['route']['source']).resolve()
    if source != root / 'runtime'/SCIENCES[exp] or not identity.get('source_files'):
        raise ValueError('Only the frozen AUG scientific source may evaluate this checkpoint')
    for name, expected in identity['source_files'].items():
        p = (source / name).resolve()
        if not p.is_relative_to(source) or file_sha(p) != expected:
            raise ValueError('Frozen scientific source changed: ' + name)
    for name in ('optimizer.pt', 'transaction.json') + tuple(f'rank{i}.pt' for i in range(seal['world'])):
        if name not in files or not (cp / name).is_file():
            raise ValueError('Incomplete native file inventory: ' + name)
    return dict(checkpoint=str(cp), updates=seal['updates'], identity_sha256=digest,
                inventory_sha256=seal['files_sha256'], payload_sha256=files['optimizer.pt'],
                parent_model_sha256=PARENT_SHA, algorithm_schema=SCHEMA, experiment=exp,
                policy_signature=fingerprint(dict(parent=PARENT_SHA, trainable_payload=files['optimizer.pt'], scope=identity['scope'])),
                scene_budget_clock=seal['updates']*recipe['global_scenes']/recipe['inner_updates'],
                source=str(source), protocol_sha256=identity['augustus_protocol_sha256'])


def discover(root, state, routes):
    for exp, route in routes.items():
        if exp not in EARLY or route.get('experiment') != exp:
            continue
        if route.get('phase') != 'formal' and 'formal' not in route.get('stages', {}):
            continue  # Qualification weights are never evaluated as formal RL.
        output = Path(route['formal_output'])
        candidates = [(output / 'checkpoints' / f'update{u:06d}_inner0', 'early_exact_saved_gate')
                      for u in EARLY[exp]]
        for p in sorted((output / 'evaluation_requests').glob('epoch*.json')):
            request = read(p)
            if request.get('seed') != 2101 or not request.get('checkpoint'):
                raise ValueError('Changed native evaluation request: ' + str(p))
            candidates.append((Path(request['checkpoint']), str(p)))
        for cp, origin in candidates:
            if not (cp / 'COMPLETE').is_file():
                continue
            binding = inspect_checkpoint(root, route, cp)
            if origin != 'early_exact_saved_gate':
                request = read(origin)
                if request.get('updates') != binding['updates'] or request.get('source_run') != binding['identity_sha256']:
                    raise ValueError('Native request does not bind this exact sealed checkpoint')
            key = exp + '_u' + format(binding['updates'], '06d')
            case = state['cases'].get(key)
            if case is not None:
                if case['binding'] != binding:
                    raise ValueError('Registered immutable checkpoint was replaced')
                if origin not in case['origins']:case['origins'].append(origin)
                continue
            state['cases'][key] = dict(id=key, binding=binding, origins=[origin], status='PENDING',
                jobs={}, next_attempt={}, created=time.time(), world=WORLD,
                output=str(Path(root) / 'evaluations' / key / 'evaluation'))
    return state


def reconcile(case):
    done = active = 0
    for rank in range(WORLD):
        jobs = case['jobs'].get(str(rank), [])
        if not jobs:continue
        job = Path(jobs[-1]['job']);result = read(job / 'RESULT.json')
        if result is None:
            active += 1;continue
        row = read(Path(case['output']) / f'rank{rank}.json', {})
        if result.get('returncode') == 0:
            if (row.get('status') == 'COMPLETE' and row.get('rank') == rank and row.get('world') == WORLD
                    and row.get('new', -1) + row.get('reused', -1) == len(range(rank, POPULATION, WORLD))):
                done += 1
            else:
                case.update(status='FAILED_REQUIRES_REVIEW', failed_rank=rank,
                            error='EXIT0 without the full exact-rank population', failed_result=str(job / 'RESULT.json'))
        elif result.get('returncode') != 75 or result.get('child_not_started') is not True:
            case.update(status='FAILED_REQUIRES_REVIEW', failed_rank=rank, failed_result=str(job / 'RESULT.json'))
    case.update(completed_shards=done, active_shards=active)
    return done == WORLD


def il_baseline(root):
    # Same contract lookup as tools.d44_native_rl.campaign.il_baseline when its
    # ROOT is D46; no mutation of that module's process-global campaign root.
    return read(Path(root) / 'PARENT_CONTRACT.json')['complete_navtest']['original_evaluation']


def summarize_case(root, case):
    from tools.qwen_world_compare.evaluate import summarize
    from tools.d44_native_rl.analyze import analyze
    baseline = il_baseline(root);output = Path(case['output'])
    identity=read(output/'identity.json',{});body=dict(identity);digest=body.pop('sha256',None)
    binding=case['binding']
    if (fingerprint(body)!=digest or identity.get('checkpoint')!=binding['checkpoint']
            or identity.get('D44_training_identity')!=binding['identity_sha256']
            or identity.get('model_sha256')!=binding['policy_signature']
            or identity.get('updates')!=binding['updates'] or identity.get('cohort')!='NAVTEST'
            or identity.get('qualification')is not False or identity.get('seed')!=2101
            or identity.get('count')!=POPULATION or identity.get('inference')!='single_candidate_cleanx_K10_shift5'):
        raise ValueError('Full evaluation does not bind this registered AUG checkpoint/seed/protocol')
    summary = summarize(output, baseline)
    if (summary.get('status') != 'COMPLETE' or summary.get('scenes') != POPULATION
            or summary.get('physical_logs') != 44 or summary.get('invalid_scenes') != 0
            or summary.get('qualification') is not False or summary.get('seed') != 2101
            or summary.get('evaluation_identity_sha256')!=digest):
        raise ValueError('Full official12146/44/invalid0 result required')
    for name, value in [('SUMMARY.json', summary)]:
        path = output.parent / name
        if path.exists() and read(path) != value:raise ValueError('Preserve previous official summary')
        atomic_json(path, value)
    case['summary'] = str(output.parent / 'SUMMARY.json')
    paired = analyze(output, baseline)
    paired['analysis_schema'] = 'D46_complete_B1_paired_gain_loss_v1'
    gain = paired['gain_loss']
    gain['fixed_reference_cohorts'] = {k.replace('IL44', 'IL46'): v for k, v in gain['fixed_reference_cohorts'].items()}
    gain['limits'] = gain['limits'].replace('IL44', 'IL46')
    paired.pop('sha256', None);paired['sha256'] = fingerprint(paired)
    path = output.parent / 'PAIRED_ANALYSIS.json'
    if path.exists() and read(path) != paired:raise ValueError('Preserve previous paired analysis')
    atomic_json(path, paired)
    case.update(status='COMPLETE', paired_analysis=str(path), completed_at=time.time())


def inventory(host, ops_source):
    """Only the three explicit spare hosts; no probes of six learner nodes."""
    if host not in SPARE_GPUS:raise ValueError('Host is not an authorized evaluation spare')
    code = '''import json,pathlib,subprocess,sys
sys.path.insert(0, OPS_SOURCE)
from tools.qwen_world_compare.job_worker import known_owner_args
apps={}
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 p,u,m=line.split(',');apps.setdefault(u.strip(),[]).append((int(p),int(m)))
rows=[]
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
 i,u,free,used=line.split(',');owners=[]
 for pid,mem in apps.get(u.strip(),[]):
  try:
   proc=pathlib.Path('/proc')/str(pid);args=proc.joinpath('cmdline').read_bytes().decode().split('\\0');start=int(proc.joinpath('stat').read_text().split()[21])
   entry=args[args.index('-m')+1]if '-m'in args else None
   known=entry in ('tools.d46_augustus.evaluate','tools.d44_native_rl.evaluate','tools.qwen_world_compare.evaluate','tools.qwen_world_safety_exploration.evaluate') and known_owner_args(args)
  except(OSError,ValueError,IndexError):known=False;start=None
  owners.append(dict(pid=pid,start_ticks=start,used_MiB=mem,known=known))
 rows.append(dict(index=int(i),uuid=u.strip(),free_MiB=int(free),used_MiB=int(used),owners=owners))
print(json.dumps(rows))
'''.replace('OPS_SOURCE', repr(str(Path(ops_source).resolve())))
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, '/usr/bin/python3', '-'],
                       input=code, text=True, capture_output=True, timeout=20)
    if r.returncode:raise DispatchPreflightFailed(r.stderr)
    return json.loads(r.stdout)


def available_slots(host, rows, reserved):
    slots = []
    for row in rows:
        if row['index'] not in SPARE_GPUS[host] or row['free_MiB'] < 24000:continue
        owners = row['owners']
        if any(o.get('known') is not True for o in owners) or len(owners) >= 2:continue
        if not owners and row['free_MiB'] < 79000:continue  # Hidden/unattributed memory.
        if row['used_MiB'] - sum(o['used_MiB'] for o in owners) > 1024:continue
        existing = max(len(owners), reserved.get((host, row['index']), 0))
        slots.extend((host, row['index']) for _ in range(max(0, 2 - existing)))
    return slots


def tick(root, eval_state, routes, *, ops_source=SOURCE, execute=True):
    """Caller owns the independent EVAL lock; persist before/after every launch."""
    root = Path(root);state = eval_state
    if state.get('schema') != initial_state()['schema']:raise ValueError('Foreign evaluation state')
    discover(root, state, routes)
    reserved = {}
    for case in state['cases'].values():
        if case['status'] != 'COMPLETE' and reconcile(case) and execute:
            try:summarize_case(root, case)
            except Exception as error:case.update(status='FAILED_SUMMARY_REQUIRES_REVIEW', error=repr(error))
        for jobs in case['jobs'].values():
            if jobs and not (Path(jobs[-1]['job']) / 'RESULT.json').exists():
                for gpu in jobs[-1]['gpus']:
                    key = (jobs[-1]['host'], gpu);reserved[key] = reserved.get(key, 0) + 1
    state['at'] = time.time()
    if not execute:return state
    state_path(root).parent.mkdir(parents=True,exist_ok=True)
    active = [c for c in state['cases'].values() if c.get('active_shards')]
    selected = list(active)
    for case in sorted(state['cases'].values(), key=lambda c: (-c.get('completed_shards', 0), c['binding']['scene_budget_clock'], c['created'])):
        if len(selected) >= 2:break
        if case not in selected and case['status'] not in TERMINAL:selected.append(case)
    if len(active) > 2:
        state['error'] = 'More than two active checkpoint cases; no new dispatch';return state
    pool = []
    for host in SPARE_GPUS:
        if host not in allowed_hosts(admit_long_job=True):continue
        try:pool.extend(available_slots(host, inventory(host, ops_source), reserved))
        except (OSError, ValueError, subprocess.SubprocessError, DispatchPreflightFailed) as error:
            state.setdefault('inventory_errors', {})[host] = dict(at=time.time(), error=repr(error))
    for rank in range(WORLD):
        for case in selected:
            if case['status'] in TERMINAL:continue
            jobs = case['jobs'].setdefault(str(rank), [])
            if jobs:
                result = read(Path(jobs[-1]['job']) / 'RESULT.json')
                if result is None or result.get('returncode') == 0:continue
                if (result.get('returncode') != 75 or result.get('child_not_started') is not True
                        or time.time() < result.get('end_unix', time.time()) + 300):continue
            if not pool:continue
            host, gpu = pool.pop(0)
            if host not in allowed_hosts(admit_long_job=True):continue
            number = case['next_attempt'].get(str(rank), 1);case['next_attempt'][str(rank)] = number + 1
            job = root / 'jobs/AUGUSTUS_EVALUATE' / case['id'] / f'rank{rank:02d}_attempt{number:03d}'
            command = [PYTHON, '-m', 'tools.d46_augustus.evaluate', '--checkpoint', case['binding']['checkpoint'],
                       '--output', case['output'], '--rank', str(rank), '--world', str(WORLD)]
            atomic_json(state_path(root), state)
            try:
                receipt = launch(host, case['binding']['source'], job, command, [gpu], share_min_free_mib=24000,
                                 worker_source=ops_source, task_root=root, task=TASK)
            except DispatchPreflightFailed as error:
                case.setdefault('preflight_failures', []).append(dict(job=str(job), at=time.time(), error=repr(error)))
                continue
            except Exception as error:
                case.update(status='DISPATCH_REQUIRES_RECONCILIATION', ambiguous_job=str(job), error=repr(error))
                atomic_json(state_path(root), state);continue
            jobs.append(receipt);case['status'] = 'RUNNING'
            atomic_json(state_path(root), state)
    for case in state['cases'].values():reconcile(case)
    state.update(at=time.time(), complete=bool(state['cases']) and all(c['status'] == 'COMPLETE' for c in state['cases'].values()))
    atomic_json(state_path(root), state)
    from .complete_events import notify
    changed=False
    for case in state['cases'].values():
        if case['status']!='COMPLETE':continue
        try:
            summary=read(case['summary']);sha=summary['sha256']
            if case.get('report_notified_sha256')==sha:continue
            receipt=notify(state_path(root),case['id'],sha,root=root)
            case['report_notified_sha256']=receipt['notified_sha']
            case['report_notification']=receipt
        except Exception as error:
            case['report_notification_error']=repr(error)
        changed=True
    if changed:atomic_json(state_path(root),state)
    return state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=ROOT);p.add_argument('--routes-state', type=Path, required=True)
    p.add_argument('--once', action='store_true');p.add_argument('--run', action='store_true')
    p.add_argument('--execute', action='store_true');p.add_argument('--interval', type=int, default=1800)
    a = p.parse_args()
    if a.run and not a.execute:p.error('--run requires explicit --execute')
    if not a.run and not a.once:p.error('Select --once (inspection) or --run --execute')
    path = state_path(a.root);path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            from .event_wait import wait_for_events
            if not a.routes_state.is_file():
                wait_for_events([a.routes_state.parent],a.interval)
                continue
            state = read(path, initial_state());routes = read(a.routes_state)['routes']
            state = tick(a.root, state, routes, ops_source=SOURCE, execute=a.execute)
            print(json.dumps(dict(cases=len(state['cases']), execute=a.execute, at=state['at'])), flush=True)
            if a.once:break
            paths=[a.routes_state.parent]
            for row in routes.values():
                output=Path(row['formal_output'])
                paths += [output,output/'checkpoints',output/'evaluation_requests']
            for case in state['cases'].values():
                paths.append(Path(case['output']))
                paths += [Path(jobs[-1]['job'])for jobs in case['jobs'].values()if jobs]
            pending=any(case['status']not in TERMINAL for case in state['cases'].values())
            # Only a queued resource admission gets a bounded five-minute
            # retry; idle/active jobs are driven by native/result events.
            timeout=min(a.interval,300)if pending and not any(c.get('active_shards')for c in state['cases'].values())else a.interval
            wait_for_events(paths,timeout,ignore=('EVAL_STATE.json','EVAL_OWNER.json'))


if __name__ == '__main__':main()
