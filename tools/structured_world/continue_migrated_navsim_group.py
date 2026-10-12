"""Continue the migrated G2 on zt4 after its existing 25k launcher exits.

Run on zt4 with --prepare --output NEW_CONTROL, then --execute --output
NEW_CONTROL. Preparation uses CPU/proc metadata only. No signals or retries.
ENV_PRIVATE.json contains only guard.ENV_KEYS training variables, never tokens.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adopt_navsim_evaluation_policy as guard

MIG = Path('/mnt/project/iqe-runtime-audit-20261009/migrations/20261012_vlawm_zt_offline')
ORIGINAL = guard.ART/'navsim_G2_vlawm_preflight_20261011/20261011T032024Z/preparation_v3/G2_MANIFEST.json'
HOST, GROUP = 'training-vlawm-zt4', 'G2_EVENT_UNIFORM'
CADENCE = guard.ART/'NAVSIM_NAVTEST_CADENCE_20261012.json'
read, require, digest = guard.read, guard.require, guard.digest


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
    tmp.replace(path)


def same(reference):
    try: current = guard.process(reference['pid'])
    except (FileNotFoundError, ProcessLookupError): return None
    require(current['start_ticks'] == int(reference['start_ticks']), 'PID reused; inspect manually')
    require(current['state'] == 'Z' or current['argv'] == reference['argv'], 'Process argv changed')
    return current


def workers(run):
    found = set()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: argv = [os.fsdecode(v) for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError, ProcessLookupError): continue
        if str(guard.ROOT/'tools/structured_world/train_vla.py') in argv and str(run) in argv:
            found.add(int(p.name))
    return found


def stage_command(argv, target):
    command = list(argv); at = command.index('--stop-after')
    require(command.count('--stop-after') == 1 and command.count('--resume') == 1, 'Original argv differs')
    if target == 100000: del command[at:at+2]
    else: command[at+1] = str(target)
    return command


def bank(manifest, target=None):
    value = guard.native(manifest['run'], target)
    require(value and value['identity'] == manifest['identity'], 'Native identity changed or checkpoint absent')
    receipt = read(value['complete_file'])
    require(all((Path(value['complete_file']).parent/n).is_file() for n in receipt['file_sha256']), 'Native payload missing')
    require(receipt['status'] == ('COMPLETE' if value['completed'] == 100000 else 'PAUSED'),
            'Not a PAUSED stage boundary; never automatically restart an interrupted stage')
    return value


def boundary(manifest, before):
    report = read(Path(manifest['run'])/'RESUME_BOUNDARY_VERIFIED.json')
    require(report['passed'] is True and report['identity'] == manifest['identity']
            and report['completed'] == before, 'Native resume boundary differs')
    return report


def checks(manifest):
    require(socket.gethostname().removesuffix('-worker-0') == HOST, 'Execute/prepare locally on zt4 only')
    guard.authorize(HOST); guard.clean_source(); guard.evaluation_policy([GROUP])
    for path, sha in manifest['references'].items(): require(digest(path) == sha, 'Bound source/evidence changed: '+path)
    require(read(Path(manifest['run'])/'identity.json')['identity'] == manifest['identity'], 'Run identity changed')
    require(not (Path(manifest['run'])/'STOP_AFTER_CHECKPOINT').exists(), 'Explicit stop marker exists')
    task = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']
    require(task['shared_GPU_deployment_authorized'], 'Sharing authorization revoked')


def gpu_ready(manifest):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    for line in raw.splitlines():
        if not line.strip(): continue
        pid, uuid = [s.strip() for s in line.split(',')]
        require(pid in manifest['cache'] and uuid in manifest['uuids'], 'Unregistered genuine GPU task; no eviction')
        proc = same(manifest['cache'][pid]); require(proc and proc['state'] != 'Z', 'Original cache worker changed')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    rows = [list(map(str.strip, line.split(','))) for line in raw.splitlines() if line.strip()]
    require(sorted(r[0] for r in rows) == manifest['uuids'], 'zt4 physical UUIDs changed')
    return all(int(r[1]) >= 50*1024 for r in rows)


def prepare(output, migration, original):
    launch = read(migration/'TARGET_LAUNCH.json'); request = read(migration/'SOURCE_STOP_REQUEST.json')
    old = read(original); run = Path(request['source_argv'][request['source_argv'].index('--output')+1])
    exited = read(migration/'SOURCE_EXIT_CONFIRMED.json'); parent = read(migration/'PARENT_CHECKPOINT_VERIFIED.json')
    require(exited['status'] == 'SOURCE_EXIT_CONFIRMED' and parent['status'] == 'PASS'
            and exited['latest_tag'] == parent['tag'] == launch['parent_tag']
            and parent['receipt_sha256'] == launch['parent_receipt_sha256'], 'Original pause/parent verification differs')
    require(launch['host'] == HOST and launch['argv'] == request['source_argv']+['--resume']
            and launch['cwd'] == request['source_cwd'] == str(guard.ROOT)
            and launch['identity'] == old['expected_native_identity'] and old['groups'] == [GROUP]
            and old['training_plan'][0]['command'] == request['source_argv'], 'Original migration/ancestry differs')
    require([p['target'] for p in old['training_plan']] == list(guard.TARGETS)
            and all(stage_command(launch['argv'], p['target']) == p['command'] for p in old['training_plan'][1:]),
            'Original 25/50/75/100k stage commands differ')
    authorization = read(migration/'SHARED_RESOURCE_AUTHORIZATION.json')
    require(authorization['host'] == HOST and authorization['shared_GPU_deployment_authorized'], 'Sharing evidence differs')
    uuids = sorted(g['uuid'] for g in read(migration/'SHARED_GPU_ADMISSION.json')['gpus'])
    require(len(set(uuids)) == 8 and len(launch['cache_workers_preserved']) == 8, 'Expected original eight-GPU/cache inventory')
    current = same(launch); require(current and current['state'] != 'Z', 'Prepare while the exact original launcher is live')
    require(Path('/proc', str(current['pid'])).stat().st_uid == os.getuid(), 'Launch environment belongs to another UID')
    env = {os.fsdecode(k): os.fsdecode(v) for field in Path('/proc', str(current['pid']), 'environ').read_bytes().split(b'\0')
           for k, _, v in [field.partition(b'=')] if os.fsdecode(k) in guard.ENV_KEYS}
    require(same(launch) and workers(run) == {current['pid'], *current['children']}, 'Unexpected same-run workers')
    cache = {str(pid): guard.process(pid) for pid in launch['cache_workers_preserved']}
    require(all('iqe.vision_cache.cli' in p['argv'] and 'cache-build' in p['argv'] for p in cache.values()), 'Original IQE workers differ')
    refs = dict(old['preserved_files_sha256'])
    paths = [original, Path(__file__), Path(guard.__file__), guard.NAVTEST_TOOL, CADENCE,
             run/'identity.json', *[migration/n for n in ('TARGET_LAUNCH.json', 'SOURCE_STOP_REQUEST.json',
             'SOURCE_EXIT_CONFIRMED.json', 'PARENT_CHECKPOINT_VERIFIED.json', 'SHARED_RESOURCE_AUTHORIZATION.json', 'SHARED_GPU_ADMISSION.json')]]
    refs.update({str(p.resolve()): digest(p) for p in paths})
    cadence = read(CADENCE)
    require(cadence['observation_updates'] == [25000, 50000, 60000, 70000, 80000, 90000, 100000]
            and any(r['run'] == str(run) and r['training_identity'] == launch['identity'] for r in cadence['runs']), 'G2 cadence registration differs')
    queue = guard.ART/'physical_gpu_locks'/('queue_'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()+'.lock')
    physical = queue.parent/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')
    require(str(queue) == old['queue_lock'] and launch['lineage_lock'] == str(run/'HOST_MIGRATION_RUNTIME.lock'), 'Original lock differs')
    manifest = {'run': str(run), 'identity': launch['identity'], 'current': launch, 'cache': cache,
                'uuids': uuids, 'locks': [str(queue), str(physical), launch['lineage_lock']], 'references': refs,
                'training_targets': list(guard.TARGETS), 'navtest_targets': [25000, 50000, 100000],
                'initial_boundary_sha256': digest(run/'RESUME_BOUNDARY_VERIFIED.json'),
                'routing_policy_sha256_at_prepare': digest(guard.POLICY),
                'original_manifest': str(original), 'created_unix': time.time()}
    checks(manifest); boundary(manifest, launch['parent_completed'])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write(output/'ENV_PRIVATE.json', env); manifest['environment_sha256'] = digest(output/'ENV_PRIVATE.json')
    write(output/'REGISTRATION.json', manifest); write(output/'STATE.json', {'status': 'PREPARED', 'GPU_jobs_started': 0})


def execute(output):
    m = read(output/'REGISTRATION.json'); env_file = output/'ENV_PRIVATE.json'
    require(env_file.stat().st_mode & 0o777 == 0o600 and env_file.stat().st_uid == os.getuid()
            and digest(env_file) == m['environment_sha256'], 'Private original environment changed/unsafe')
    saved = read(env_file); require(set(saved) <= set(guard.ENV_KEYS), 'Unexpected saved environment key')
    require(read(output/'STATE.json')['status'] == 'PREPARED', 'Already attempted; inspect receipts, no automatic retry')
    with (output/'EXECUTION_STARTED.json').open('x') as f: json.dump({'pid': os.getpid()}, f)
    state = lambda status, **fields: write(output/'STATE.json', {'status': status, 'unix': time.time(), **fields})
    def log(**fields):
        with (output/'executed_commands.jsonl').open('a') as f: f.write(json.dumps({'unix': time.time(), **fields})+'\n')
    try:
        checks(m)
        with ExitStack() as held:
            fds = [held.enter_context(Path(p).open('a')) for p in m['locks']]
            for f in fds[:2]: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                checks(m); current = same(m['current'])
                if current and current['state'] != 'Z':
                    require(workers(m['run']) <= {current['pid'], *current['children']}, 'Other same-run training process')
                    state('WAITING_CURRENT_25K', pid=current['pid']); time.sleep(5); continue
                if current: require(os.waitstatus_to_exitcode(current['exit_status']) == 0, 'Original launcher failed')
                require(not workers(m['run']), 'Original descendants or another same-run training process remain')
                try: fcntl.flock(fds[2], fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError: state('WAITING_LINEAGE_LOCK'); time.sleep(5)
            log(stage='original_launcher_finished', observed_exit_code=0 if current else None)
            require(digest(Path(m['run'])/'RESUME_BOUNDARY_VERIFIED.json') == m['initial_boundary_sha256'], 'Initial resume report changed')
            boundary(m, m['current']['parent_completed']); bank(m, 25000)
            for target in guard.TARGETS:
                before = bank(m)
                require(before['completed'] == (25000 if target == 25000 else guard.TARGETS[guard.TARGETS.index(target)-1]), 'Unexpected stage boundary')
                if target != 25000:
                    while True:
                        checks(m); require(not workers(m['run']), 'Another same-run training process')
                        if gpu_ready(m): break
                        state('WAITING_GPU_HEADROOM', target=target); time.sleep(10)
                    command = stage_command(m['current']['argv'], target)
                    record = {'target': target, 'command': command, 'native_before': before}
                    state('RUNNING', **record); log(stage='launch', **record)
                    with (output/f'train_to_{target}.log').open('ab') as stream:
                        child = subprocess.Popen(command, cwd=guard.ROOT, env={**os.environ, **saved}, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(f.fileno() for f in fds))
                        log(stage='process', pid=child.pid, start_ticks=guard.process(child.pid)['start_ticks'], target=target)
                        rc = child.wait()
                    log(stage='exit', target=target, returncode=rc); require(rc == 0, 'Real training failed; no retry')
                    require(not workers(m['run']), 'Training descendants remain'); checks(m)
                    log(stage='resume_boundary', target=target, report=boundary(m, before['completed']))
                milestone = bank(m, target)
                if target in m['navtest_targets']:
                    log(stage='queue_navtest', checkpoint=milestone, receipt=guard.queue_navtest({'group': GROUP, 'run': m['run'], 'target': target}))
            done = read(Path(m['run'])/'TRAINING_COMPLETE.json')
            require(done['identity'] == m['identity'] and done['updates'] == 100000, '100k completion absent')
            state('TRAINING_100K_COMPLETE_NAVTEST_REQUESTS_QUEUED', evaluation_complete=False)
    except BaseException as error:
        state('FAILED', error_type=type(error).__name__, error=str(error), workers_signaled=False, automatic_retry=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--migration', type=Path, default=MIG); parser.add_argument('--original-manifest', type=Path, default=ORIGINAL)
    args = parser.parse_args()
    os.umask(0o077)
    if args.prepare: prepare(args.output, args.migration, args.original_manifest)
    else: execute(args.output)
    print(json.dumps({'control': str(args.output), 'status': read(args.output/'STATE.json')['status']}))


if __name__ == '__main__': main()
