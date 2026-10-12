"""Navtest request queue and independent GPU shards using frozen FP32 evaluation.

Only routing, progress and memory guards are new. Model restoration, reader,
scene noise, planning and canonical scoring come from the immutable 0c54 source.
No training process, checkpoint, historical asset or existing controller is changed.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid

FROZEN = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
DEFAULT_OUTPUT = ART / 'navsim_navtest_dispatch_20261011T0225'
INPUTS = ART / 'navsim_navtest_current_v1_0c54'
INDEX = Path('/mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json')
DEVKIT = Path('/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1')
SCRIPT = Path(__file__).resolve()
EXECUTED_TOOL_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
METRICS = ('score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
    'time_to_collision_within_bound','comfort','driving_direction_compliance')


class InfrastructureError(RuntimeError):
    pass


class ResourcesUnavailable(RuntimeError):
    pass


def now(): return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def first_failure(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    try: os.link(temp, path)
    except FileExistsError: pass
    finally: temp.unlink(missing_ok=True)


def attempt_root(destination, attempt):
    return destination / 'attempts' / attempt if attempt else destination


def new_attempt(destination, routes):
    """Never overwrite a previous attempt's process reports or cost evidence."""
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:12]
    root = attempt_root(destination, attempt)
    atomic(root / 'ATTEMPT.json', dict(attempt=attempt, routes=routes, started_utc=now(),
        routing_tool_sha256=digest(SCRIPT), status='STARTING'))
    return attempt


def validate_record_union(destination, expected, identity):
    """Coverage comes from global immutable records, independent of worker count."""
    records = list((destination / 'records').glob('*.json'))
    if len(records) != len(expected) or {p.stem for p in records} != set(expected):
        raise ValueError('Incomplete prediction population')
    for path in records:
        saved = json.loads(path.read_text())
        if saved.get('token') != path.stem or saved['identity'] != identity or digest(destination / 'predictions' / (path.stem + '.npz')) != saved['sha256']:
            raise ValueError('Prediction union token/identity/hash mismatch')


def valid_score_completion(destination, req):
    """A filename is not a success certificate; diagnostic failed banks stay invalid."""
    score_path = destination / 'scores/COMPLETE.json'
    if not score_path.exists(): return False
    prediction = json.loads((destination / 'identity.json').read_text())
    full = json.loads((destination / 'COMPLETE.json').read_text())
    scores = json.loads(score_path.read_text())
    score_identity = json.loads((destination / 'scores/identity.json').read_text())
    if (prediction['training_identity'] != req['training_identity'] or prediction['checkpoint'] != req['tag'] or
        prediction['checkpoint_files_sha256'] != req['checkpoint_files_sha256'] or prediction['updates'] != req['updates'] or
        prediction['evaluation_source_sha'] != SHA or prediction['sampling_seed'] != 42 or
        full['identity'] != prediction['identity'] or full['scenes'] != 12146 or not full['full_population'] or
        score_identity['prediction_identity'] != prediction['identity'] or scores['identity'] != score_identity['identity']):
        raise ValueError('Completed bank differs from requested prediction/scoring identity')
    index=json.loads(INDEX.read_text());expected={r['token']:r['log'] for r in index}
    if len(index)!=12146 or len(expected)!=12146 or len(set(expected.values()))!=136:
        raise ValueError('Expected complete canonical Navtest index')
    validate_record_union(destination,expected,prediction['identity'])
    for variant in ('q0', 'q_final'):
        value = scores[variant]
        saved = json.loads((destination / 'scores' / variant / 'summary.json').read_text())
        if value != saved or value['identity'] != score_identity['identity'] or value['scenes'] != 12146 or value['valid'] is not True or value['failed'] != 0:
            raise ValueError('Invalid completed score bank: ' + variant)
        if value.get('scene_csv_sha256') != digest(destination / 'scores' / variant / 'scenes.csv'):
            raise ValueError('Completed score CSV changed: ' + variant)
        validate_score_csv(destination/'scores'/variant/'scenes.csv',value,expected,variant)
    if (destination / 'INFERENCE_FAILED.json').exists(): raise ValueError('Unresolved inference failure')
    return True


def validate_score_csv(path, summary, expected, variant):
    """Check independently measured population and means, not declared counts."""
    with Path(path).open() as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=12146 or len({r.get('token') for r in rows})!=12146 or {r.get('token') for r in rows}!=set(expected):
        raise ValueError('Completed score CSV population/uniqueness differs: '+variant)
    values={k:[] for k in METRICS}
    for row in rows:
        if row.get('log')!=expected[row['token']] or row.get('status')!='ok' or row.get('variant')!=variant:
            raise ValueError('Completed score CSV token/log/status/variant differs: '+variant)
        for key in METRICS:
            value=float(row[key])
            if not math.isfinite(value) or value < -1e-9 or value > 1+1e-9:
                raise ValueError('Completed score CSV metric nonfinite/outside [0,1]: '+key)
            values[key].append(value)
    for key,bank in values.items():
        declared=float(summary['metrics'][key])
        if not math.isfinite(declared) or abs(math.fsum(bank)/len(bank)-declared)>1e-12:
            raise ValueError('Completed score metric mean differs: '+key)
    if abs(sum(v==0 for v in values['score'])/len(rows)-summary['zero_fraction'])>1e-12:
        raise ValueError('Completed score zero fraction differs')


def reconcile_infrastructure_leases(destination, index):
    """Called only after dead-worker evidence; validate successes before fencing."""
    expected = {r['token'] for r in index}
    identity = json.loads((destination / 'identity.json').read_text())['identity'] if (destination/'identity.json').exists() else None
    verified = set()
    for path in (destination / 'records').glob('*.json'):
        saved = json.loads(path.read_text())
        if path.stem not in expected or saved.get('token') != path.stem or saved['identity'] != identity or digest(destination/'predictions'/(path.stem+'.npz')) != saved['sha256']:
            raise ValueError('Prior prediction invalid; cannot infrastructure-resume scientific corruption')
        verified.add(path.stem)
    claims = destination / 'work_leases'; claims.mkdir(parents=True, exist_ok=True)
    expired = []
    with (claims/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in claims.glob('*.json'):
            old = json.loads(path.read_text())
            tokens = {index[i]['token'] for i in range(old['first'], old['end'])}
            if old['status'] == 'COMPLETE':
                if not tokens <= verified: raise ValueError('Completed lease lacks valid scene records')
                continue
            value = dict(old, status='EXPIRED', owner=uuid.uuid4().hex, heartbeat_unix=0,
                expired_after_verified_worker_exit_utc=now(), previous_owner=old['owner'])
            atomic(path, value); expired.append(path.name)
    return dict(verified_prior_scenes=len(verified), fenced_leases=expired)


def review_load_oom(destination, failure, evidence):
    """Review only a hash-bound CUDA allocation failure before any scene ran.

    This is an explicit operator recovery, never an automatic scientific-error
    retry. Preserve the original exception/classification and every attempt log.
    """
    if evidence.get('reviewed_failure_kind') != 'CUDA_load_oom_after_qualification':
        raise ValueError('Unsupported load OOM review')
    stack = failure.get('error_traceback', '')
    if not failure.get('error', '').startswith('OutOfMemoryError(') or "model.to('cuda').eval()" not in stack:
        raise ValueError('Review requires the original model-loading CUDA OOM stack')
    if 'plan_from_current' in stack or 'scene_noise(' in stack:
        raise ValueError('Inference/model-computation errors cannot use load OOM review')
    free = re.search(r'of which ([0-9.]+) MiB is free', stack)
    if not free or float(free.group(1)) >= 512:
        raise ValueError('No evidence of exhausted physical GPU memory')
    root = destination / 'attempts' / failure['attempt']
    reports = sorted((root / 'shards').glob('*.json'))
    bound = evidence.get('attempt_files_sha256', {})
    if not reports or len(reports) != json.loads(reports[0].read_text())['shards']:
        raise ValueError('Load OOM review needs every original shard report')
    for path in reports:
        report = json.loads(path.read_text()); log = root / (path.stem + '.log')
        if any(bound.get(str(p.relative_to(destination))) != digest(p) for p in (path, log)):
            raise ValueError('Load OOM attempt evidence hash differs')
        error = report.get('error', '')
        peer = "RuntimeError('Peer inference failure; stop owned worker at scene boundary')"
        if report.get('scenes_done') != 0 or report.get('status') != 'FAILED' or not (
                error.startswith('OutOfMemoryError(') or error == peer):
            raise ValueError('Load OOM review cannot clear an executed scene or another error')
    if any((destination / 'records').glob('*.json')) or any((destination / 'predictions').glob('*.npz')):
        raise ValueError('Load-only recovery must not contain scene outputs')


def recover_infrastructure(args):
    """Explicit recovery; no process signals and no clearing scientific failures."""
    authorized()
    destination = args.output/'jobs'/args.request_key
    failure_path = destination/'INFERENCE_FAILED.json'
    failure = json.loads(failure_path.read_text())
    evidence = json.loads(args.recovery_evidence.read_text())
    if evidence.get('classification') != 'infrastructure' or evidence.get('failure_sha256') != digest(failure_path) or evidence.get('request_key') != args.request_key or not evidence.get('reason'):
        raise ValueError('Recovery evidence must bind exact failure hash/request and reason')
    if (failure.get('classification') != 'infrastructure'
            and evidence.get('reviewed_failure_kind') == 'CUDA_load_oom_after_qualification'):
        review_load_oom(destination, failure, evidence)
    elif failure.get('classification') != 'infrastructure':
        # Old wrappers did not record exception stacks. A narrow, explicit
        # transport review may recover this known SSH/stdout failure; preserve
        # the original marker verbatim, including its original classification.
        if failure.get('error') != "BrokenPipeError(32, 'Broken pipe')" or evidence.get('reviewed_failure_kind') != 'SSH_stdout_broken_pipe':
            raise ValueError('Scientific/unclassified failure has no supported transport review')
        report=destination/'attempts'/failure['attempt']/'shards'/('shard_'+str(failure['shard']).zfill(3)+'.json')
        log=destination/'attempts'/failure['attempt']/('shard_'+str(failure['shard']).zfill(3)+'.log')
        if evidence.get('worker_log_sha256')!=digest(log) or evidence.get('worker_report_sha256')!=digest(report) or 'Timeout, server ' not in log.read_text():
            raise ValueError('Transport review lacks exact SSH timeout/progress evidence')
        progress=json.loads(report.read_text())
        if progress['error']!=failure['error'] or progress['scenes_done']%16 or not evidence.get('original_traceback_unavailable'):
            raise ValueError('Transport review must disclose missing original stack and bind progress-boundary failure')
    proofs = []
    reports = [json.loads(p.read_text()) for p in (destination/'attempts').glob('*/shards/*.json')]
    reports.append(failure)
    seen = set()
    for report in reports:
        host = report.get('host', '').removesuffix('-worker-0'); pid = report.get('pid')
        start = report.get('proc_start_ticks')
        if not host or not pid or (host, pid) in seen: continue
        seen.add((host,pid)); authorized(host)
        code = "from pathlib import Path;import json;p=Path('/proc/'+str("+str(pid)+")+'/stat');print(json.dumps(dict(exists=p.exists(),start_ticks=p.read_text().rsplit(')',1)[1].split()[19] if p.exists() else None)))"
        command = ['python','-c',code]
        if host != socket.gethostname().removesuffix('-worker-0'):
            command = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(command)]
        proof = json.loads(subprocess.check_output(command,text=True,timeout=20))
        if proof['exists'] and (start is None or proof['start_ticks'] == start):
            raise ValueError('Prior owned worker still alive or unverifiable; cannot fence leases')
        proofs.append(dict(host=host,pid=pid,expected_start_ticks=start,observed=proof))
    if (destination/'scores/INVALID.json').exists(): raise ValueError('Scientific scoring failure cannot be infrastructure-cleared')
    index=json.loads((args.inputs/'index.json').read_text())
    metric_index=json.loads(INDEX.read_text())
    input_identity=json.loads((args.inputs/'identity.json').read_text())['identity']
    if len(index)!=12146 or {(r['token'],r['log']) for r in index}!={(r['token'],r['log']) for r in metric_index} or json.loads((destination/'identity.json').read_text())['input_identity']!=input_identity:
        raise ValueError('Recovery lease order/input identity/population differs')
    reconciled = reconcile_infrastructure_leases(destination,index)
    recovery = destination/'recoveries'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12])
    recovery.mkdir(parents=True)
    atomic(recovery/'RECOVERY.json',dict(evidence=evidence,original_failure_sha256=digest(failure_path),
        original_classification=failure['classification'],recovery_classification='explicit reviewed infrastructure',
        worker_exit_proofs=proofs,reconciled=reconciled,observed_utc=now()))
    failure_path.replace(recovery/'INFERENCE_FAILED.json')
    print(json.dumps(dict(recovered=args.request_key,evidence=str(recovery/'RECOVERY.json'),**reconciled)),flush=True)


def authorized(host=None):
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    hosts = policy['task_authorizations']['structured_world_fgtr_round1']['latest_explicit_resource_hosts']
    evaluation = ART / 'NAVSIM_EVALUATION_POLICY_20261011.json'
    if evaluation.exists(): hosts = json.loads(evaluation.read_text()).get('inference_allowed_hosts', hosts)
    local = socket.gethostname().removesuffix('-worker-0')
    if (host or local) not in hosts: raise ValueError('Host outside latest five-host authorization')
    selected = host or local
    if selected in policy.get('global_dispatch_prohibitions', {}):
        raise ValueError('Host has a current global deployment prohibition')
    reservation = policy.get('host_reservations', {}).get(selected, {})
    if reservation.get('exclude_new_other_workloads') and reservation.get('task') != 'structured_world_fgtr_round1':
        raise ValueError('Host reserved for another active task')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FROZEN, text=True).strip() != SHA:
        raise ValueError('Wrong frozen evaluation source')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=FROZEN, text=True).strip():
        raise ValueError('Frozen evaluation source is dirty')


def gpu_rows():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    result = []
    for row in text.strip().splitlines():
        index, uuid, total, used, free, util = [v.strip() for v in row.split(',')]
        result.append({'index': int(index), 'uuid': uuid, 'total_MiB': int(total),
            'used_MiB': int(used), 'free_MiB': int(free), 'utilization_percent': int(util)})
    return result


def train_snapshot():
    p = ART / 'formal_navsim_local_G3_G2_v3_io/G3_EVENT_LOCAL/metrics.jsonl'
    if not p.exists(): return {}
    with p.open('rb') as stream:
        stream.seek(max(0, p.stat().st_size - 300000)); lines = stream.read().splitlines()
    values = []
    for line in lines:
        try: values.append(json.loads(line))
        except (ValueError, UnicodeDecodeError): pass
    values = values[-32:]
    return {'observed_utc': now(), 'last_update': values[-1]['updates'] if values else None,
        'count': len(values), 'median_update_seconds': statistics.median(r['seconds'] for r in values) if values else None,
        'latest_metric_mtime': p.stat().st_mtime}


def inventory():
    """Read only current process metadata; publish GPU binding fields, no secrets."""
    candidates = ['/tmp/structured-world-round1/envs/qwen/bin/python',
        '/var/tmp/structured-world-fgtr-round1-20261008/qwen/bin/python',
        '/var/tmp/navsim-navtest-eval-20261011/qwen/bin/python']
    samples = []
    for i in range(3):
        samples.append(gpu_rows())
        if i < 2: time.sleep(.5)
    devices = samples[-1]; owned = {}; process_evidence = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            argv = [v.decode() for v in (path/'cmdline').read_bytes().split(b'\0') if v]
            if str(FROZEN/'tools/structured_world/train_vla.py') not in argv or '--output' not in argv: continue
            env = dict(v.split('=',1) for v in (path/'environ').read_bytes().decode().split('\0') if '=' in v)
            if 'LOCAL_RANK' not in env: continue
            run = Path(argv[argv.index('--output')+1])
            source = json.loads((run/'identity.json').read_text())
            if source['scope'] != 'formal' or source['dataset'] != 'navsim' or source['training_source_sha'] != SHA:
                continue
            if source['config']['structured_world']['group'] not in ('G0_NO_FUTURE_LABEL','G1_FULL_UNIFORM','G2_EVENT_UNIFORM','G3_EVENT_LOCAL'): continue
            local = int(env['LOCAL_RANK']); mask = env.get('CUDA_VISIBLE_DEVICES')
            visible = mask.split(',') if mask else [str(r['index']) for r in devices]
            binding = visible[local]
            row = next(r for r in devices if binding in (str(r['index']),r['uuid']))
            evidence = dict(pid=int(path.name),proc_start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19],
                GPU_uuid=row['uuid'],local_rank=local,visible_GPU_mask=mask,run=str(run),training_identity=source['identity'],
                group=source['config']['structured_world']['group'],source=SHA,
                binding_basis='exact registered frozen train_vla.py --output and LOCAL_RANK; source set_device(local)')
            # NVML may report the host PID while /proc is a container namespace.
            namespace = next((line.split()[1:] for line in (path/'status').read_text().splitlines()
                              if line.startswith('NSpid:')), [])
            evidence['pid_namespace_aliases'] = [int(value) for value in namespace]
            owned.setdefault(row['uuid'],[]).append(evidence); process_evidence.append(evidence)
        except (OSError,ValueError,KeyError,IndexError,StopIteration,UnicodeError): continue
    apps = []; query_error = None
    try:
        text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.STDOUT)
        for line in text.strip().splitlines():
            pid,gpu,memory = [v.strip() for v in line.split(',')]
            apps.append(dict(pid=int(pid),GPU_uuid=gpu,used_MiB=int(memory)))
    except (subprocess.SubprocessError,ValueError) as error: query_error=repr(error)
    return dict(host=socket.gethostname(),python=next((p for p in candidates if Path(p).is_file()),None),samples=samples,
        registered_training_by_GPU=owned,process_evidence=process_evidence,compute_apps=apps,compute_apps_error=query_error,
        namespace_visibility='visible registered rank bindings; additional hidden occupants cannot be enumerated when compute-apps is empty',
        cpu_count=os.cpu_count())


def cpu_budget(requested):
    affinity = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    quotas = []
    for root in [Path('/sys/fs/cgroup'),*Path('/sys/fs/cgroup').parents]:
        path = root/'cpu.max'
        if path.exists():
            quota,period = path.read_text().split()
            if quota != 'max': quotas.append(int(quota)/int(period))
    v1=Path('/sys/fs/cgroup/cpu')
    if (v1/'cpu.cfs_quota_us').exists():
        q=int((v1/'cpu.cfs_quota_us').read_text());p=int((v1/'cpu.cfs_period_us').read_text())
        if q>0:quotas.append(q/p)
    capacity=min([affinity,*quotas]);load=os.getloadavg()[0]
    workers=min(requested,max(1,int(min(capacity, max(1,(affinity-load-16)/2)))))
    return dict(workers=workers,affinity_CPUs=affinity,cgroup_quota_CPUs=quotas,
        effective_capacity_CPUs=capacity,host_load_1min=load,reserved_host_CPUs=16,native_threads_per_score_worker=1,
        scoring_hosts=1)


def request(args):
    authorized(); run = args.run.resolve()
    source = json.loads((run / 'identity.json').read_text())
    complete = json.loads((run / 'checkpoints' / args.tag / 'COMPLETE.json').read_text())
    if source['scope'] != 'formal' or source['dataset'] != 'navsim' or source['training_source_sha'] != SHA:
        raise ValueError('Request must identify a formal frozen-source NAVSIM checkpoint')
    if complete['identity'] != source['identity']: raise ValueError('Foreign checkpoint')
    group = source['config']['structured_world']['group']
    key = group + '_' + str(complete['completed']).zfill(7)
    value = {'key': key, 'group': group, 'run': str(run), 'tag': args.tag, 'updates': complete['completed'],
        'training_identity': source['identity'], 'population': 'navtest', 'scenes': 12146, 'logs': 136,
        'seed': 42, 'precision': 'FP32 / TF32 disabled', 'variants': ['q0', 'q_final'],
        'checkpoint_files_sha256': complete['file_sha256'], 'priority': args.priority}
    path = args.output / 'requests' / (key + '.json')
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != 'priority'} != {k: v for k, v in value.items() if k != 'priority'}:
            raise ValueError('Conflicting queue request')
        print(json.dumps({'already_enqueued': key, 'request': str(path)}), flush=True); return
    atomic(path, value); print(json.dumps({'enqueued': key, 'request': str(path)}), flush=True)


def prediction_contract(source, checkpoint, data):
    # Same scientific contract as frozen infer_checkpoint.py; host/shard routing
    # is recorded separately, never represented as a different scene population.
    return {'schema': 'structured_world_FP32_predictions_v1', 'training_identity': source['identity'],
        'training_source_sha': source['training_source_sha'], 'checkpoint': checkpoint['tag'],
        'updates': checkpoint['completed'], 'checkpoint_files_sha256': checkpoint['file_sha256'],
        'evaluation_source_sha': SHA, 'evaluation_source_files': {p: digest(FROZEN / p) for p in
            ('tools/structured_world/infer_checkpoint.py', 'starVLA/dataloader/structured_world/current_inputs.py')},
        'input_identity': data.identity['identity'], 'sampling_seed': 42,
        'sampling': 'independent CPU scene-bound Gaussian, original 10-step FM, one residual and one final candidate',
        'precision': 'FP32 native ZeRO master restore, strict=True; FP32 inference; TF32 disabled',
        'supervision_heads_removed': False, 'scene_diagnostics': False}


def leased_indices(output, data, report):
    """Small GPU work leases let faster hosts consume more scenes.

    Claims change routing only. An expired lease receives a fresh generation;
    old workers must check ownership before publishing any prediction.
    """
    claims = output / 'work_leases'; claims.mkdir(parents=True, exist_ok=True)
    report['work_routing'] = '32-scene dynamic leases, 900s expiry, generation checked before publish'
    report['scenes_total'] = None; report['shared_population_total'] = len(data)
    for attempt in range(len(data) + 1):
        acquired = None
        with (claims / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for first in range(0, len(data), 32):
                path = claims / (str(first).zfill(6) + '.json')
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous['status'] == 'COMPLETE' or time.time() - previous['heartbeat_unix'] < 900: continue
                claim = {'first': first, 'end': min(first + 32, len(data)), 'owner': uuid.uuid4().hex,
                    'status': 'ACTIVE', 'heartbeat_unix': time.time(), 'pid': os.getpid(),
                    'host': socket.gethostname(), 'shard': report['shard'], 'attempt': report.get('attempt')}
                atomic(path, claim); acquired = path, claim; break
        if acquired is None: return
        path, claim = acquired
        for i in range(claim['first'], claim['end']):
            yield i, path, claim['owner']
            with (claims / 'queue.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                fresh = json.loads(path.read_text())
                if fresh['owner'] != claim['owner']: raise RuntimeError('Work lease generation changed')
                fresh['heartbeat_unix'] = time.time(); atomic(path, fresh)
        with (claims / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fresh = json.loads(path.read_text())
            if fresh['owner'] != claim['owner']: raise RuntimeError('Completed lease generation changed')
            fresh['status'] = 'COMPLETE'; fresh['heartbeat_unix'] = time.time(); atomic(path, fresh)


def publish_owned(path, temp, record, value, lease, owner):
    if lease is None:
        temp.replace(path); atomic(record, value); return
    with (lease.parent / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if json.loads(lease.read_text())['owner'] != owner: raise RuntimeError('Lost lease before prediction publication')
        temp.replace(path); atomic(record, value)


def inference_idle_seconds(scene_seconds, duty_cycle):
    """Yield only between completed scenes; never change tensors, RNG or sampling."""
    if not math.isfinite(duty_cycle) or not .05 <= duty_cycle <= 1:
        raise ValueError('Inference duty cycle must be finite and in [0.05, 1]')
    if not math.isfinite(scene_seconds) or scene_seconds < 0:
        raise ValueError('Invalid completed-scene wall time')
    return scene_seconds * (1. / duty_cycle - 1.)


def scheduling_duty_cycle(args):
    duty = getattr(args, 'cohost_duty_cycle', 1.)
    policy_path = getattr(args, 'scheduling_policy', None)
    if policy_path:
        policy = json.loads(policy_path.read_text())
        host = socket.gethostname().removesuffix('-worker-0')
        duty = policy.get('owned_training_duty_by_host', {}).get(
            host, policy.get('owned_training_inference_duty_cycle', duty))
    inference_idle_seconds(0., duty)  # Validate even when a scene is already cached.
    return duty


def worker(args):
    authorized()
    import numpy as np
    import torch
    sys.path.insert(0, str(FROZEN))
    from tools.structured_world.infer_checkpoint import load_model, scene_noise
    from starVLA.dataloader.structured_world.current_inputs import CurrentInputs
    torch.cuda.set_device(0); torch.set_num_threads(2)
    physical_uuid = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != physical_uuid:
        raise InfrastructureError('Actual CUDA device UUID differs from the routed physical GPU')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * (1 << 30) / total, 0)
    data = CurrentInputs(args.inputs)
    state_root = attempt_root(args.output, args.attempt) / 'shards'; state_root.mkdir(parents=True, exist_ok=True)
    report_path = state_root / ('shard_' + str(args.shard).zfill(3) + '.json')
    report = {'status': 'LOADING', 'pid': os.getpid(), 'host': socket.gethostname(), 'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'actual_CUDA_UUID': physical_uuid,
        'shard': args.shard, 'shards': args.shards, 'attempt': args.attempt, 'started_utc': now(), 'memory_cap_GiB': args.memory_cap_gib,
        'proc_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'routing_tool_sha256': EXECUTED_TOOL_SHA256, 'scenes_done': 0, 'scenes_total': len(range(args.shard, len(data), args.shards)),
        'GPU_before': gpu_rows()}
    if len(data) == 12146:
        profile = args.output.parents[1] / 'qualification' / args.output.name / socket.gethostname().removesuffix('-worker-0') / 'QUALIFICATION_PASSED.json'
        qualified = json.loads(profile.read_text())
        device = next(r for r in report['GPU_before'] if r['uuid'] == report['visible_gpu'])
        if device['free_MiB'] * (1 << 20) < qualified['peak_reserved_bytes'] + 6 * (1 << 30):
            raise InfrastructureError('Resource changed: measured peak plus CUDA overhead cannot preserve five GiB')
    atomic(report_path, report)
    if args.fp32_export and (args.fp32_export.parent / 'FP32_EXPORT_COMPLETE.json').exists():
        from omegaconf import OmegaConf
        from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
        export = json.loads((args.fp32_export.parent / 'FP32_EXPORT_COMPLETE.json').read_text())
        source = json.loads((args.run / 'identity.json').read_text())
        checkpoint = json.loads((args.run / 'checkpoints' / args.tag / 'COMPLETE.json').read_text())
        if export['training_identity'] != source['identity'] or export['checkpoint'] != args.tag or export['checkpoint_files_sha256'] != checkpoint['file_sha256']:
            raise ValueError('FP32 export not bound to the requested native checkpoint')
        if digest(args.fp32_export) != export['sha256']: raise ValueError('Immutable FP32 export hash differs')
        model = VLAStructuredFGTR(OmegaConf.create(source['config'])).float()
        state = torch.load(args.fp32_export, map_location='cpu', weights_only=False)
        model.load_state_dict(state, strict=True); del state
        # CPU construction/hash verification can take long enough for another
        # task to acquire the GPU. Recheck immediately before CUDA allocation.
        before_cuda = next(r for r in gpu_rows() if r['uuid'] == report['visible_gpu'])
        needed = qualified['peak_reserved_bytes'] if len(data) == 12146 else args.memory_cap_gib * (1 << 30)
        report['GPU_before_model_to_cuda'] = before_cuda
        atomic(report_path, report)
        if before_cuda['free_MiB'] * (1 << 20) < needed + 5 * (1 << 30):
            raise InfrastructureError('Resource changed during CPU model loading; no CUDA model allocation attempted')
        model.to('cuda').eval(); model.inference_fp32 = True
        model.shared_geometry_identity = source['shared_geometry_identity']
        report['FP32_export'] = str(args.fp32_export); report['FP32_export_sha256'] = export['sha256']
    else:
        model, source, checkpoint = load_model(args.run, args.tag, 'cuda')
        if args.fp32_export:
            args.fp32_export.parent.mkdir(parents=True, exist_ok=True)
            temp = args.fp32_export.with_suffix('.tmp')
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, temp); del state
            temp.replace(args.fp32_export); args.fp32_export.chmod(0o444)
            atomic(args.fp32_export.parent / 'FP32_EXPORT_COMPLETE.json', {'training_identity': source['identity'],
                'checkpoint': args.tag, 'checkpoint_files_sha256': checkpoint['file_sha256'],
                'sha256': digest(args.fp32_export), 'frozen_evaluation_source': SHA,
                'origin': 'unmodified frozen load_model: verified native ZeRO FP32 masters and strict loaded model state',
                'exported_utc': now(), 'dtype': 'unchanged native FP32 tensors/buffers'})
    checkpoint = dict(checkpoint, tag=args.tag)
    if source['dataset'] != data.identity['dataset']: raise ValueError('Input dataset differs from checkpoint')
    contract = prediction_contract(source, checkpoint, data)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    target = args.output / 'identity.json'
    if target.exists() and json.loads(target.read_text())['identity'] != identity: raise ValueError('Foreign predictions')
    atomic(target, dict(contract, identity=identity))
    for name in ('predictions', 'records'): (args.output / name).mkdir(parents=True, exist_ok=True)
    report['status'] = 'INFERENCE'; report['prediction_identity'] = identity
    report['loaded_utc'] = now(); report['load_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['scheduling_idle_seconds'] = 0.
    report['scheduling_policy'] = str(args.scheduling_policy) if args.scheduling_policy else None
    report['cohost_duty_cycle'] = scheduling_duty_cycle(args)
    atomic(report_path, report); start = time.monotonic(); costs = []
    work = leased_indices(args.output, data, report) if len(data) == 12146 else (
        (i, None, None) for i in range(args.shard, len(data), args.shards))
    for i, lease, owner in work:
        if (args.output / 'INFERENCE_FAILED.json').exists(): raise RuntimeError('Peer inference failure; stop owned worker at scene boundary')
        scene_seconds = 0.
        token = data.index[i]['token']; record = args.output / 'records' / (token + '.json')
        path = args.output / 'predictions' / (token + '.npz')
        if record.exists():
            saved = json.loads(record.read_text())
            if saved['identity'] != identity or digest(path) != saved['sha256']: raise ValueError('Prediction identity/hash changed')
        else:
            before = time.monotonic(); observation = data[i]
            noise = scene_noise(token, 42, model.steps, 'cuda')
            with torch.no_grad():
                planned = model.plan_from_current([observation], initial_noise=noise)
                values = {key: planned[key][0].cpu().numpy() for key in ('q0', 'q_final', 'out_of_range')}
            if any(not np.isfinite(values[k]).all() or values[k].shape != (8, 3) for k in ('q0', 'q_final')):
                raise FloatingPointError('Invalid trajectory; never exclude the scene')
            temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            with temp.open('wb') as stream: np.savez_compressed(stream, **values)
            # The checksum binds temporary payload bytes before atomic publish.
            publish_owned(path, temp, record, {'token': token, 'identity': identity, 'sha256': digest(temp)}, lease, owner)
            scene_seconds = time.monotonic() - before
            costs.append(scene_seconds)
        duty = scheduling_duty_cycle(args)
        idle_seconds = inference_idle_seconds(scene_seconds, duty)
        if idle_seconds:
            # q0/q_final/out_of_range have already been copied to CPU and published.
            # No CUDA work is pending here; other useful work may use the device.
            time.sleep(idle_seconds)
            report['scheduling_idle_seconds'] += idle_seconds
        report['cohost_duty_cycle'] = duty
        report['scenes_done'] += 1
        if report['scenes_done'] <= 4 or report['scenes_done'] % 16 == 0:
            report.update(observed_utc=now(), inference_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                median_scene_seconds=statistics.median(costs[-32:]) if costs else None)
            atomic(report_path, report)
            try:
                print(json.dumps({k: report[k] for k in ('shard', 'scenes_done', 'peak_reserved_bytes', 'observed_utc')}), flush=True)
            except BrokenPipeError:
                # Optional stdout progress is never a scientific/model failure.
                report['stdout_transport_detached']=True;atomic(report_path,report)
    torch.cuda.synchronize()
    report.update(status='COMPLETE', completed_utc=now(), inference_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        median_scene_seconds=statistics.median(costs) if costs else None, GPU_after=gpu_rows())
    atomic(report_path, report)


def routed(host, command, environment, stdout_path=None):
    authorized(host)
    env = dict(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', **environment)
    argv = ['env'] + [k + '=' + str(v) for k, v in env.items()] + command
    if host == socket.gethostname().removesuffix('-worker-0'): return argv
    remote='cd '+shlex.quote(str(FROZEN))+' && exec '+shlex.join(argv)
    if stdout_path:remote+=' >> '+shlex.quote(str(stdout_path))+' 2>&1'
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8',host,remote]


def inspect_route(route):
    command = [route['python'], '-B', str(SCRIPT), '--inspect']
    result = subprocess.check_output(routed(route['host'], command, {}), text=True)
    return json.loads(result.strip().splitlines()[-1])


def refresh_routes(args, request):
    """Choose today's safe unique GPUs; a prior route list is never a fixed quota."""
    policy = json.loads((ART / 'NAVSIM_EVALUATION_POLICY_20261011.json').read_text())
    fallback = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())['task_authorizations']['structured_world_fgtr_round1']['latest_explicit_resource_hosts']
    hosts = policy.get('inference_allowed_hosts', fallback)
    old = json.loads(args.routes.read_text()).get('routes', []) if args.routes.exists() else []
    prior = {r['host']: r for r in old}
    cost_path=args.output/'COHOST_COST_POLICY.json'
    cost_policy=json.loads(cost_path.read_text()) if cost_path.exists() else {}
    def scan(host):
        try: authorized(host)
        except ValueError as error: return host, {'error': str(error)}
        argv = ['python','-B',str(SCRIPT),'--inventory']
        if host != socket.gethostname().removesuffix('-worker-0'):
            argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', host, shlex.join(argv)]
        try:
            result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if result.returncode: return host, {'error': result.stderr.strip()[-600:]}
            return host, json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as error: return host, {'error': repr(error)}
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool: inspected = dict(pool.map(scan, hosts))
    routes = []; seen = set(); skipped = {}
    for host in hosts:
        info = inspected[host]
        if host in cost_policy.get('quarantined_hosts',{}):
            skipped[host]=cost_policy['quarantined_hosts'][host];continue
        if 'error' in info: skipped[host] = info['error']; continue
        candidates = []
        for row in info['samples'][-1]:
            samples = [next(r for r in s if r['uuid'] == row['uuid']) for s in info['samples']]
            free = min(r['free_MiB'] for r in samples)
            if row['uuid'] in seen: continue
            if free < 20 * 1024: continue
            owned = info['registered_training_by_GPU'].get(row['uuid'],[])
            aliases = {pid for v in owned for pid in [v['pid'], *v.get('pid_namespace_aliases', [])]}
            device_apps = [r for r in info['compute_apps'] if r['GPU_uuid']==row['uuid']]
            extra = [r for r in device_apps if r['pid'] not in aliases]
            if max(r['used_MiB'] for r in samples) > 128 and (not owned or extra): continue
            # Hidden namespace occupants cannot be inferred to be our training.
            if owned and row['used_MiB'] - sum(r.get('used_MiB',0) for r in device_apps) > 512:
                continue
            if owned and cost_policy.get('exclude_high_utilization_training') and max(r['utilization_percent'] for r in samples)>20:
                continue
            if not owned and max(r['utilization_percent'] for r in samples)>20:continue
            candidates.append((row,free,owned))
        if not candidates: skipped[host] = 'duplicate UUIDs, insufficient headroom, or unrelated high utilization'; continue
        python = info['python'] or prior.get(host, {}).get('python')
        readiness = ART/'navsim_navtest_runtime_staging_20261011/20261011T023958Z/stage_v1'/host/'READINESS.json'
        if readiness.exists():
            certified=json.loads(readiness.read_text())
            if not certified['passed'] or not certified['CPU_imports_passed'] or certified['source_sha']!=SHA or not certified['private_overlay_all_bytes_and_links_verified']['passed']:
                skipped[host]='Mencius runtime readiness invalid';continue
            if any(digest(Path(p))!=checksum for p,checksum in certified['evidence_sha256'].items()):
                skipped[host]='Mencius readiness evidence hash differs';continue
            python=certified['python'];info['certified_runtime_readiness']=str(readiness)
            info['certified_runtime_readiness_sha256']=digest(readiness)
        if not python:
            skipped[host]='Missing certified isolated Qwen runtime; staging owned by Mencius';continue
        probe = [python, '-B', '-c', "import torch,transformers,deepspeed,flash_attn; print(torch.__version__,transformers.__version__,deepspeed.__version__,flash_attn.__version__,torch.cuda.is_available())"]
        try:
            runtime = subprocess.check_output(routed(host, probe, {}), text=True, stderr=subprocess.STDOUT, timeout=40)
            info['runtime_probe'] = runtime.strip().splitlines()[-1]
            if info['runtime_probe'] != '2.5.1+cu124 4.57.0 0.16.9 2.7.4.post1 True':
                skipped[host] = 'Qwen/CUDA runtime differs: ' + info['runtime_probe']; continue
        except Exception as error: skipped[host] = 'Qwen runtime probe failed: ' + repr(error); continue
        for row, free, owned in candidates:
            seen.add(row['uuid'])
            routes.append({'host': host, 'uuid': row['uuid'], 'index': row['index'], 'python': python,
                'memory_cap_gib': min(26, int(free / 1024) - 6), 'cohosting_verified_own_training_or_idle': True,
                'ownership': 'registered owned structured-world training' if owned else 'repeatedly physically empty GPU',
                'owner_evidence':owned,
                'cohost_duty_cycle': cost_policy.get('owned_training_duty_by_host', {}).get(host,
                    cost_policy.get('owned_training_inference_duty_cycle', 1.)) if owned else 1.,
                'scheduling_policy': str(cost_path) if owned else None,
                'minimum_reserve_GiB': 5})
    atomic(args.output / 'resource_snapshots' / (request['key'] + '.json'), {'observed_utc': now(), 'inspected': inspected,
        'selected_routes': routes, 'excluded': skipped,'cohost_cost_policy':cost_policy,
        'selection': 'current safe unique GPU subset, no fixed host/GPU/shard count'})
    if not routes: raise ResourcesUnavailable('No currently qualified candidate GPUs')
    return routes


def launch_worker(args, request, destination, route, shard, count, inputs, attempt=None):
    command = [route['python'], '-B', '-u', str(SCRIPT), '--worker', '--run', request['run'],
        '--tag', request['tag'], '--inputs', str(inputs), '--output', str(destination),
        '--shard', str(shard), '--shards', str(count), '--memory-cap-gib', str(route['memory_cap_gib'])]
    if route.get('fp32_export'): command += ['--fp32-export', route['fp32_export']]
    duty = route.get('cohost_duty_cycle', 1.)
    inference_idle_seconds(0., duty)
    command += ['--cohost-duty-cycle', str(duty)]
    if route.get('scheduling_policy'):
        command += ['--scheduling-policy', str(route['scheduling_policy'])]
    if attempt: command += ['--attempt', attempt]
    root = attempt_root(destination, attempt)
    log = root / ('shard_' + str(shard).zfill(3) + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
    argv = routed(route['host'], command, {'CUDA_VISIBLE_DEVICES': route['uuid']},stdout_path=log)
    with log.open('a') as stream:
        proc = subprocess.Popen(argv, cwd=FROZEN, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(root / 'launches' / ('shard_' + str(shard).zfill(3) + '.json'),
        {'pid': proc.pid, 'host': route['host'], 'GPU_uuid': route['uuid'], 'shard': shard, 'shards': count,
         'command': argv, 'launched_utc': now(), 'controller_pid': os.getpid()})
    return proc


def qualify(args, request, routes):
    qualified = []; hosts = sorted({r['host'] for r in routes})
    export_path = args.output / 'FP32_exports' / request['key'] / 'model_fp32.pt'
    first_host = 'training-vlawm-zt' if 'training-vlawm-zt' in hosts else hosts[0]
    for route in routes: route['fp32_export'] = str(export_path)
    def qualify_host(host):
        subset = [r for r in routes if r['host'] == host]; route = subset[0]
        inspected = inspect_route(route); devices = {r['uuid']: r for r in inspected['GPUs']}
        if route['uuid'] not in devices: raise ValueError('GPU UUID absent from routed host')
        if min(devices[r['uuid']]['free_MiB'] for r in subset) < route['memory_cap_gib'] * 1024 + 5120:
            raise ValueError('Physical GPU headroom below cap plus five GiB')
        destination = args.output / 'qualification' / request['key'] / host
        before = train_snapshot(); atomic(destination / 'LOAD_BEFORE.json', {'training': before, 'inspection': inspected})
        saved = destination / 'QUALIFICATION_PASSED.json'
        reused = saved.exists() and (destination / 'shards/shard_000.json').exists()
        if reused:
            previous = json.loads(saved.read_text())
            launch=json.loads((destination/'launches/shard_000.json').read_text())
            command=launch['command']
            reused = previous['request']['checkpoint_files_sha256'] == request['checkpoint_files_sha256'] and previous['request']['training_identity'] == request['training_identity'] and route['python'] in str(command)
            if not reused:
                archived=destination.parent/'profile_archive'/(host+'_'+uuid.uuid4().hex[:12])
                archived.parent.mkdir(parents=True,exist_ok=True);destination.replace(archived)
                atomic(destination/'LOAD_BEFORE.json',{'training':before,'inspection':inspected,'previous_profile_preserved':str(archived)})
        if not reused:
            process = launch_worker(args, request, destination, route, 0, 1, args.inputs.parent / (args.inputs.name + '_memory4'))
            code = process.wait()
            if code: raise RuntimeError('Memory qualification failed on ' + host + '; exit ' + str(code))
        result = json.loads((destination / 'shards/shard_000.json').read_text())
        peak = result['peak_reserved_bytes']; after = inspect_route(route)
        minimum_free = min(r['free_MiB'] for r in after['GPUs'] if r['uuid'] in {v['uuid'] for v in subset})
        if peak > route['memory_cap_gib'] * (1 << 30) or minimum_free * (1 << 20) < peak + 5 * (1 << 30):
            raise ValueError('Measured peak cannot preserve five GiB physical headroom')
        atomic(destination / 'QUALIFICATION_PASSED.json', {'passed': True, 'host': host,
            'request': request, 'peak_reserved_bytes': peak, 'peak_allocated_bytes': result['peak_allocated_bytes'],
            'available_minimum_MiB': minimum_free, 'minimum_reserve_GiB': 5, 'scenes': 4,
            'training_before': before, 'training_after': train_snapshot(), 'after_inspection': after,
            'qualification_predictions_are_not_formal_outputs': True, 'reused_same_checkpoint_profile': reused})
        for r in subset: r['profile_median_scene_seconds'] = result['median_scene_seconds']
        return subset
    # Restore the native checkpoint once; subsequent profiles/workers use the
    # immutable, strictly bound FP32 export to avoid 32 native restoration storms.
    qualified += qualify_host(first_host)
    remaining = [host for host in hosts if host != first_host]
    if remaining:
        with ThreadPoolExecutor(max_workers=len(remaining)) as pool:
            for subset in pool.map(qualify_host, remaining): qualified += subset
    first_route = next(r for r in qualified if r['host'] == first_host)
    reference = args.output / 'qualification' / request['key'] / 'FROZEN_REFERENCE'
    debug = args.inputs.parent / (args.inputs.name + '_memory4')
    command = [first_route['python'], '-B', '-u', str(FROZEN / 'tools/structured_world/infer_checkpoint.py'),
        '--run', request['run'], '--tag', request['tag'], '--inputs', str(debug), '--output', str(reference), '--sampling-seed', '42']
    reference.mkdir(parents=True, exist_ok=True)
    if not (reference / 'COMPLETE.json').exists():
        with (reference / 'inference.log').open('a') as stream:
            result = subprocess.run(routed(first_host, command, {'CUDA_VISIBLE_DEVICES': first_route['uuid']}),
                cwd=FROZEN, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('Unmodified frozen inference parity reference failed')
    import numpy as np
    for host in hosts:
        candidate = args.output / 'qualification' / request['key'] / host
        if json.loads((candidate / 'identity.json').read_text()) != json.loads((reference / 'identity.json').read_text()):
            raise ValueError('New worker changed frozen prediction identity')
        for row in json.loads((debug / 'index.json').read_text()):
            with np.load(candidate / 'predictions' / (row['token'] + '.npz')) as new, np.load(reference / 'predictions' / (row['token'] + '.npz')) as original:
                for key in ('q0', 'q_final', 'out_of_range'):
                    if not np.array_equal(new[key], original[key]): raise ValueError('Frozen inference parity failed: ' + host + '/' + key)
    atomic(args.output / 'qualification' / request['key'] / 'FROZEN_PARITY_PASSED.json',
        {'passed': True, 'scenes': 4, 'hosts': hosts, 'variants': ['q0', 'q_final', 'out_of_range'],
         'comparison': 'bitwise against unmodified frozen infer_checkpoint.py native FP32 master restore',
         'shared_prediction_identity': json.loads((reference / 'identity.json').read_text())['identity'],
         'reference_command': command, 'observed_utc': now()})
    return qualified


def finish_predictions(destination, shards, attempt=None):
    identity = json.loads((destination / 'identity.json').read_text())
    index = json.loads(INDEX.read_text()); expected = {r['token'] for r in index}
    if len(index) != 12146 or len(expected) != 12146 or len({r['log'] for r in index}) != 136:
        raise ValueError('Canonical Navtest index population changed')
    validate_record_union(destination, expected, identity['identity'])
    root = attempt_root(destination, attempt)
    reports = [json.loads((root / 'shards' / ('shard_' + str(i).zfill(3) + '.json')).read_text()) for i in range(shards)]
    if any(r['status'] != 'COMPLETE' or r['prediction_identity'] != identity['identity'] for r in reports):
        raise ValueError('Incomplete/foreign shard report')
    # Successful prior predictions may come from completed leases during a
    # resumed attempt; exact record membership/hashes define the population.
    historical = [json.loads(p.read_text()) for p in (destination / 'attempts').glob('*/shards/*.json')]
    if not attempt: historical.extend(reports)
    atomic(destination / 'COMPLETE.json', {'identity': identity['identity'], 'scenes': 12146, 'full_population': True,
        'seconds': max(r['inference_seconds'] for r in reports), 'GPU_hours': sum(r.get('inference_seconds', 0) for r in historical)/3600,
        'shards': shards, 'final_attempt': attempt, 'attempt_reports': len(historical),
        'coverage_basis': 'exact global unique token membership and identity/payload hashes, including successful prior attempts',
        'routing': 'independent GPU processes with small work leases, exact full-input identity and original scene-bound noise',
        'completed_utc': now()})


def execute(args, req):
    destination = args.output / 'jobs' / req['key']
    if valid_score_completion(destination, req):
        atomic(args.output / 'completed' / (req['key'] + '.json'), {'request': req,
            'score_complete': str(destination / 'scores/COMPLETE.json'), 'valid': True, 'completed_utc': now()})
        return
    if (destination / 'INFERENCE_FAILED.json').exists():
        raise RuntimeError('Unresolved failure; explicit infrastructure recovery evidence required; scientific failures cannot be cleared')
    routes = refresh_routes(args, req)
    if not routes or len({r['uuid'] for r in routes}) != len(routes): raise ValueError('Empty/duplicate physical GPU routing')
    if any(not r.get('cohosting_verified_own_training_or_idle') for r in routes): raise ValueError('Unverified GPU ownership/cohosting route')
    policy = json.loads((ART / 'NAVSIM_EVALUATION_POLICY_20261011.json').read_text())
    if not policy['dispatch_navtest_evaluation'] or policy['dispatch_development_evaluation']:
        raise ValueError('Current user evaluation policy differs')
    debug_ready = args.inputs.parent / (args.inputs.name + '_memory4') / 'COMPLETE.json'
    while not debug_ready.exists():
        atomic(args.output / 'STATE.json', {'status': 'WAITING_FOR_PROFILE_ASSETS', 'request': req, 'observed_utc': now()})
        time.sleep(10)
    atomic(args.output / 'STATE.json', {'status': 'QUALIFYING_MEMORY', 'request': req, 'observed_utc': now()})
    routes = qualify(args, req, routes)
    ready = args.inputs / 'ASSETS_VERIFIED.json'
    heading = args.inputs / 'HEADING_VERIFIED.json'
    while not ready.exists() or not heading.exists():
        atomic(args.output / 'STATE.json', {'status': 'WAITING_FOR_FULL_ASSETS', 'request': req,
            'reason': 'full pixels/calibration/legal state audit and direct canonical quaternion heading audit both required', 'observed_utc': now()})
        time.sleep(10)
    assets = json.loads(ready.read_text())
    head = json.loads(heading.read_text())
    if not assets['passed'] or assets['scenes'] != 12146 or not head['passed'] or head['input_identity'] != assets['input_identity'] or head['populations']['navtest_full']['bitwise_raw_direct_state_verified'] != 12146:
        raise ValueError('Input assets not qualified')
    # Asset preparation/profile waits must not turn an old inventory into a
    # launch authorization. Recheck now and qualify any newly admitted host.
    routes=qualify(args,req,refresh_routes(args,req))
    atomic(destination / 'REGISTRATION.json', {'request': req, 'frozen_evaluation_source': SHA, 'routes': routes,
        'inputs': str(args.inputs), 'input_identity': assets['input_identity'], 'metric_index': str(INDEX),
        'metric_index_sha256': digest(INDEX), 'training_unchanged': True, 'sampling_seed': 42,
        'variants': ['q0', 'q_final'], 'inference': 'FP32, strict native master restore, TF32 disabled, one original 10-step proposal',
        'routing_tool_sha256': digest(SCRIPT), 'registered_utc': now()})
    budget=cpu_budget(args.score_workers);score_workers=budget['workers']
    atomic(destination/'CPU_BUDGET.json',budget)
    score_command = [args.canonical_python, '-B', '-u', str(SCRIPT.with_name('stream_navsim_navtest_pdms.py')),
        '--predictions', str(destination), '--index', str(INDEX), '--devkit', str(DEVKIT),
        '--output', str(destination / 'scores'), '--workers', str(score_workers)]
    authorized()
    with (destination / 'canonical_score.log').open('a') as stream:
        scorer = subprocess.Popen(score_command, cwd=FROZEN, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(destination / 'SCORER_LAUNCH.json', {'pid': scorer.pid, 'command': score_command, 'launched_utc': now(),
        'mode': 'CPU process pool starts before GPU shards; atomic-ready predictions streamed immediately'})
    processes = []
    try:
      if not (destination / 'COMPLETE.json').exists():
        attempt = new_attempt(destination, routes)
        for i, route in enumerate(routes):
            processes.append(launch_worker(args, req, destination, route, i, len(routes), args.inputs, attempt))
        atomic(args.output / 'STATE.json', {'status': 'INFERENCE_RUNNING', 'request': req, 'shards': len(routes),
            'launch_pids': [p.pid for p in processes], 'attempt': attempt, 'observed_utc': now(), 'training': train_snapshot()})
        codes = [p.wait() for p in processes]
        if any(codes):
            first_failure(destination / 'INFERENCE_FAILED.json', {'classification': 'unclassified', 'exit_codes': codes, 'observed_utc': now()})
            raise RuntimeError('Inference shard failed; outputs retained; exit codes ' + repr(codes))
        finish_predictions(destination, len(routes), attempt)
      atomic(args.output / 'STATE.json', {'status': 'CANONICAL_SCORING', 'request': req, 'pid': scorer.pid,
        'command': score_command, 'observed_utc': now(), 'training': train_snapshot()})
      if scorer.wait(): raise RuntimeError('Canonical Navtest scoring failed; failed scenes retained')
      if not valid_score_completion(destination, req): raise ValueError('Missing valid complete score bank')
    except BaseException as error:
      first_failure(destination / 'INFERENCE_FAILED.json', {'classification': 'unclassified', 'error': repr(error), 'observed_utc': now()})
      # Peers observe the atomic failure at their next scene boundary. Wait only
      # for owned GPU launchers; never signal training or an unrelated process.
      for process in processes: process.wait()
      if scorer.poll() is None:
          try: scorer.wait(timeout=30)
          except subprocess.TimeoutExpired: scorer.terminate(); scorer.wait(timeout=30)
      raise
    if processes and all(process.returncode == 0 for process in processes) and scorer.returncode == 0:
        atomic(destination / 'OWNED_WORKERS_EXITED.json', {
            'key': req['key'], 'training_identity': req['training_identity'],
            'checkpoint_files_sha256': req['checkpoint_files_sha256'],
            'all_gpu_launchers_exited': True, 'scorer_exited': True,
            'launchers': [{'pid': process.pid, 'returncode': process.returncode} for process in processes],
            'scorer_pid': scorer.pid, 'scorer_returncode': scorer.returncode,
            'evidence': 'Synchronous local/SSH launchers and canonical scorer all waited to successful exit',
            'observed_utc': now(), 'routing_tool_sha256': EXECUTED_TOOL_SHA256})
    atomic(args.output / 'completed' / (req['key'] + '.json'), {'request': req,
        'score_complete': str(destination / 'scores/COMPLETE.json'), 'valid': True, 'completed_utc': now(), 'training': train_snapshot()})


def serve(args):
    authorized(); args.output.mkdir(parents=True, exist_ok=True)
    if args.cadence_policy and args.exit_when_idle:
        raise ValueError('Cadence capture requires a persistent dispatcher, not --exit-when-idle')
    cadence = None
    if args.cadence_policy:
        import navtest_cadence
        cadence = navtest_cadence.load_policy(args.cadence_policy)
    with (args.output / 'controller.lock').open('a') as lock, ExitStack() as cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(args.output / 'CONTROLLER.json', {'pid': os.getpid(), 'host': socket.gethostname(),
            'started_utc': now(), 'routes_file': str(args.routes), 'tool': str(SCRIPT), 'tool_sha256': EXECUTED_TOOL_SHA256,
            'cadence_policy': str(args.cadence_policy) if cadence else None,
            'cadence_policy_sha256': digest(args.cadence_policy) if cadence else None})
        observer = navtest_cadence.Observer(cadence, args.output, request, valid_score_completion) if cadence else None
        if observer:
            cleanup.callback(observer.close)
        while True:
            authorized()
            if observer and observer.error:
                raise RuntimeError('Cadence capture failed; evidence retained') from observer.error
            paths = list((args.output / 'requests').glob('*.json'))
            reqs = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: (r['priority'], r['key']))
            todo = []
            for r in reqs:
                marker = args.output / 'completed' / (r['key'] + '.json')
                if marker.exists():
                    if json.loads(marker.read_text()).get('valid') is not True or not valid_score_completion(args.output/'jobs'/r['key'], r):
                        raise ValueError('Queue success marker does not certify a valid full bank')
                elif not cadence or navtest_cadence.eligible(cadence, r): todo.append(r)
            if not todo:
                atomic(args.output / 'STATE.json', {'status': 'IDLE_AWAITING_REQUESTS', 'observed_utc': now()})
                if args.exit_when_idle: return
                time.sleep(10); continue
            req = todo[0]
            try: execute(args, req)
            except ResourcesUnavailable as error:
                atomic(args.output / 'STATE.json', {'status': 'WAITING_FOR_AVAILABLE_RESOURCES',
                    'request': req, 'reason': str(error), 'observed_utc': now()})
                time.sleep(30)
            except Exception as error:
                atomic(args.output / 'STATE.json', {'status': 'FAILED', 'request': req, 'error': repr(error), 'observed_utc': now()})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('enqueue', 'serve', 'worker', 'inspect', 'inventory', 'recover-infrastructure', 'verify-complete'): modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--routes', type=Path, default=DEFAULT_OUTPUT / 'ROUTES.json')
    parser.add_argument('--run', type=Path); parser.add_argument('--tag')
    parser.add_argument('--priority', type=int, default=100)
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--memory-cap-gib', type=float, default=26)
    parser.add_argument('--cohost-duty-cycle', type=float, default=1.,
                        help='Wall-time pacing between complete scenes only; no numerical change')
    parser.add_argument('--scheduling-policy', type=Path,
                        help='Optional live routing-only pacing policy for authorized training cohosts')
    parser.add_argument('--fp32-export', type=Path)
    parser.add_argument('--attempt')
    parser.add_argument('--request-key')
    parser.add_argument('--recovery-evidence', type=Path)
    parser.add_argument('--canonical-python', default='/root/miniconda3/envs/navsim/bin/python')
    parser.add_argument('--score-workers', type=int, default=12)
    parser.add_argument('--exit-when-idle', action='store_true')
    parser.add_argument('--cadence-policy', type=Path,
                        help='Explicit evaluation-only cadence override; frozen training is unchanged')
    args = parser.parse_args()
    if args.enqueue: request(args)
    elif args.inventory: print(json.dumps(inventory()))
    elif args.recover_infrastructure: recover_infrastructure(args)
    elif args.verify_complete:
        authorized();req=json.loads((args.output/'requests'/(args.request_key+'.json')).read_text())
        if not valid_score_completion(args.output/'jobs'/args.request_key,req):raise ValueError('Valid completion unavailable')
        print(json.dumps(dict(valid=True,key=args.request_key,scenes=12146,variants=['q0','q_final'],tool_sha256=digest(SCRIPT))),flush=True)
    elif args.worker:
        try: worker(args)
        except BaseException as error:
            first_failure(args.output / 'INFERENCE_FAILED.json', {'host': socket.gethostname(), 'pid': os.getpid(),
                'shard': args.shard, 'attempt': args.attempt, 'classification': 'infrastructure' if isinstance(error, InfrastructureError) else 'scientific_or_unclassified',
                'proc_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                'error_traceback':traceback.format_exc(),'executed_tool_sha256':EXECUTED_TOOL_SHA256,
                'error': repr(error), 'observed_utc': now()})
            report = attempt_root(args.output, args.attempt) / 'shards' / ('shard_' + str(args.shard).zfill(3) + '.json')
            if report.exists():
                value = json.loads(report.read_text()); value.update(status='FAILED', error=repr(error), failed_utc=now()); atomic(report, value)
            raise
    elif args.inspect:
        authorized(); print(json.dumps({'host': socket.gethostname(), 'GPUs': gpu_rows(),
            'training': train_snapshot(), 'observed_utc': now(), 'tool_sha256': digest(SCRIPT)}))
    else: serve(args)


if __name__ == '__main__': main()
