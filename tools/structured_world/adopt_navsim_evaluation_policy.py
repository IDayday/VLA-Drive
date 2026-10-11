"""Adopt an existing NAV training child and queue asynchronous Navtest requests.

The old supervisory process is stopped, retaining its locks until its child
exits naturally. Only that supervisor is then retired. Frozen worker commands,
native checkpoints and historical registrations are preserved.
"""
import argparse
import ast
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
TREE = '11915ed4b770a1a9060784f31661d15125a256e7f41cf8474fc3fa9ca8d72173'
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
POLICY = ART/'NAVSIM_EVALUATION_POLICY_20261011.json'
NAVTEST_TOOL = Path(__file__).resolve().with_name('run_navsim_navtest.py')
NAVTEST_OUTPUT = ART/'navsim_navtest_dispatch_20261011T0225'
TARGETS = (25000, 50000, 75000, 100000)
ENV_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'MAX_JOBS', 'TOKENIZERS_PARALLELISM', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES',
            'PYTHONHASHSEED', 'PYTORCH_CUDA_ALLOC_CONF', 'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'LD_LIBRARY_PATH',
            'PATH', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CUDA_HOME', 'TORCH_EXTENSIONS_DIR',
            'TRITON_CACHE_DIR', 'HF_HOME', 'TRANSFORMERS_CACHE', 'HF_HUB_OFFLINE')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def process(pid):
    p = Path('/proc')/str(pid)
    stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': stat[0], 'ppid': int(stat[1]), 'start_ticks': int(stat[19]),
            'exit_status': int(stat[49]),
            'argv': [os.fsdecode(v) for v in (p/'cmdline').read_bytes().split(b'\0') if v],
            'children': [int(v) for v in (p/'task'/str(pid)/'children').read_text().split()]}


def verify_process(reference, stopped=False):
    current = process(reference['pid'])
    require(current['start_ticks'] == reference['start_ticks'], 'PID was reused')
    if current['state'] != 'Z':
        require(current['argv'] == reference['argv'], 'Process command changed')
    if stopped:
        require(current['state'] in ('T', 't'), 'Legacy supervisor is not stopped')
    return current


def authorize(host):
    task = read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']
    require(host in task['allowed_hosts'] and host in task.get('latest_explicit_resource_hosts', task['allowed_hosts']),
            'Task host authorization changed')


def evaluation_policy(groups):
    p = read(POLICY)
    require(p['population'] == 'navtest' and p['expected_scenes'] == 12146
            and p['dispatch_development_evaluation'] is False and p['dispatch_navtest_evaluation'] is True
            and p['observation_updates'] == list(TARGETS) and set(groups) <= set(p['groups'])
            and p['training_recipe_change'] is False and p['sampling_seed'] == 42,
            'Navtest-only policy is missing or changed')
    return p


def clean_source():
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
            'Frozen source differs')


def pair_plan(argv, groups):
    runner = ROOT/'tools/structured_world/run_navsim_pair.py'
    body = next(n.body for n in ast.parse(runner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    at = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    ns = {'argparse': argparse, 'Path': Path, 'ROOT': ROOT}
    exec(compile(ast.Module(body=body[:at]+ast.parse('args=parser.parse_args('+repr(argv)+')').body,
                           type_ignores=[]), str(runner), 'exec'), ns)
    args = ns['args']
    training = next(n for n in ast.walk(ast.Module(body=body, type_ignores=[]))
                    if isinstance(n, ast.If) and ast.unparse(n.test) == 'completed < target')
    plans = []
    for group in groups:
        run = args.output/group
        for target in TARGETS:
            exists = (run/'checkpoints/latest').exists() or target > 25000
            values = {**ns, 'group': group, 'target': target, 'run_root': run,
                      'config': ROOT/f'configs/structured_world/navsim_{group}_seed42.yaml',
                      'latest': SimpleNamespace(exists=lambda: exists)}
            exec(compile(ast.Module(body=training.body[:-1], type_ignores=[]), str(runner), 'exec'), values)
            plans.append({'group': group, 'target': target, 'run': str(run), 'command': values['command']})
    return plans


def capture(pid):
    parent = process(pid)
    require(len(parent['children']) == 1, 'Supervisor must have exactly one direct stage child')
    child = process(parent['children'][0])
    require(child['state'] != 'Z' and child['ppid'] == pid, 'Training child is not active')
    argv = parent['argv']
    script = next(Path(a) for a in argv if a.endswith('.py'))
    kind = script.name
    value = lambda flag: argv[argv.index(flag)+1]
    references = {str(script): digest(script)}
    if kind == 'resume_navsim_group.py':
        control = Path(value('--output')); old = read(control/'REGISTRATION.json')
        state = read(control/'STATE.json'); groups = ['G1_FULL_UNIFORM']
        original = Path(old['original_controller'])
        original_argv = read(original)['argv']
        runner_index = original_argv.index(str(ROOT/'tools/structured_world/run_navsim_pair.py'))
        plans = pair_plan(original_argv[runner_index+1:], groups)
        host = old['destination_host']; uuids = old['expected']['physical_gpu_UUIDs']
        sharing = old['allow_low_util_sharing']; reserve = old['required_free_MiB']
        references.update(old['preserved_files_sha256'])
        routed = shlex.split(child['argv'][-1])
        require(child['argv'][0] == 'ssh' and routed[-len(state['command']):] == state['command'], 'SSH child differs')
        route_prefix = routed[:-len(state['command'])]
    elif kind == 'run_navsim_pair.py':
        control = Path(value('--output')); old = read(control/'REGISTRATION.json'); state = read(control/'STATE.json')
        groups = old['groups']; plans = pair_plan(argv[argv.index(str(script))+1:], groups)
        require(value('--host') == 'local', 'Capture this controller on its local training host')
        host = socket.gethostname().removesuffix('-worker-0'); uuids = old['physical_gpu_UUIDs']
        sharing = False; reserve = 0; route_prefix = []
        require(child['argv'] == state['command'], 'Local pair training child differs')
        references[str(control/'REGISTRATION.json')] = digest(control/'REGISTRATION.json')
    elif kind == 'deploy_navsim_g0.py':
        # Extract the reviewed original routing command function without running its main.
        ns = {'Path': Path, 'ROOT': ROOT, 'ART': ART, 'require': require}
        tree = ast.parse(script.read_text())
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    ('CONTROL', 'LOCAL', 'PYTHON', 'GEOMETRY', 'DINO') for t in n.targets):
                exec(compile(ast.Module(body=[n], type_ignores=[]), str(script), 'exec'), ns)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_command')
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), 'exec'), ns)
        control = ns['CONTROL']; old = read(control/'REGISTRATION.json'); state = read(control/'STATE.json')
        groups = ['G0_NO_FUTURE_LABEL']; run = Path(old['run_root'])
        plans = [{'group': groups[0], 'target': t, 'run': str(run),
                  'command': ns['native_command'](run, target=t, resume=t > 25000)} for t in TARGETS]
        host = 'training-rl-zt2'; uuids = old['physical_gpu_UUIDs']; sharing = False; reserve = 0; route_prefix = []
        require(child['argv'] == state['command'], 'G0 training child differs')
        references[str(control/'REGISTRATION.json')] = digest(control/'REGISTRATION.json')
        references[str(control/'QUALIFICATION_COMPLETE.json')] = digest(control/'QUALIFICATION_COMPLETE.json')
    else:
        raise RuntimeError('Unsupported supervisory process: '+kind)
    require(state['status'] == 'RUNNING' and '_train_to_' in state['stage'], 'Only an active training stage may be adopted')
    current = next(p for p in plans if p['group']+'_train_to_'+str(p['target']) == state['stage'])
    require(current['command'] == state['command'], 'Original frozen training plan differs from active stage')
    locks = []
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        name = os.readlink(fd)
        if name.endswith('.lock') and str(ART/'physical_gpu_locks') in name:
            require('FLOCK' in (fd.parent.parent/'fdinfo'/fd.name).read_text(), 'Registered lock is not held')
            locks.append(name)
    require(locks, 'Supervisor has no physical GPU lease')
    physical = ART/'physical_gpu_locks'/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')
    require(str(physical) in locks, 'Supervisor physical lock differs from its registered UUIDs')
    for p in plans:
        config = Path(p['command'][p['command'].index('--config')+1])
        references[str(config)] = digest(config)
        require('--debug-updates' not in p['command'] and '--debug-scenes' not in p['command'], 'Debug training is forbidden')
    env = dict(v.split(b'=', 1) for v in (Path('/proc')/str(child['pid'])/'environ').read_bytes().split(b'\0') if b'=' in v)
    authorize(host); evaluation_policy(groups); clean_source()
    resume_reference = None
    if '--resume' in current['command']:
        report = Path(current['run'])/'RESUME_BOUNDARY_VERIFIED.json'
        resume_reference = {'file': str(report), 'sha256': digest(report), 'report': read(report)}
        require(resume_reference['report']['passed'], 'Running native resume was not verified')
    return {'kind': 'NAVSIM_Navtest_policy_supervisor_adoption_v1', 'created_unix': time.time(),
            'supervisor_host': socket.gethostname().removesuffix('-worker-0'), 'training_host': host,
            'old_supervisor': parent, 'adopted_child': child, 'current_stage': current,
            'groups': groups, 'training_plan': plans, 'original_locks': sorted(locks),
            'physical_gpu_UUIDs': uuids, 'allow_low_util_sharing': sharing, 'required_free_MiB': reserve,
            'route_prefix': route_prefix, 'training_environment': {k: os.fsdecode(env[k.encode()]) for k in ENV_KEYS if k.encode() in env},
            'preserved_files_sha256': references, 'adopted_resume_reference': resume_reference,
            'training_source_sha': SHA, 'execution_source_sha256': TREE}


def native(run, target=None):
    run = Path(run)
    pointer = run/'checkpoints/latest'
    if target is None and not pointer.exists():
        return None
    tag = f'milestone_{target:07d}' if target is not None else pointer.read_text().strip()
    require(Path(tag).name == tag, 'Unsafe native checkpoint tag')
    complete = run/'checkpoints'/tag/'COMPLETE.json'; c = read(complete); identity = read(run/'identity.json')
    require(c['identity'] == identity['identity'] and c['ranks'] == 8 and len(c['file_sha256']) == 17
            and identity['training_source_sha'] == SHA and identity['training_source_tree_sha256'] == TREE
            and identity['schedule_horizon'] == 100000 and identity['effective_global_batch'] == 32
            and identity['FM_repeat'] == 8 and identity['microbatch'] == 4 and identity['world_size'] == 8
            and identity['training_seed'] == 42 and identity['scenes_per_epoch'] == 101592,
            'Native training identity differs')
    if target is not None:
        require(c['completed'] == target, 'Milestone differs')
    return {'completed': c['completed'], 'identity': c['identity'], 'tag': tag,
            'complete_file': str(complete), 'complete_sha256': digest(complete)}


def queue_navtest(plan):
    policy = evaluation_policy([plan['group']])
    checkpoint = native(plan['run'], plan['target'])
    output = Path(policy.get('navtest_dispatch_output', NAVTEST_OUTPUT))
    path = output/'requests'/f"{plan['group']}_{plan['target']:07d}.json"
    priority = read(path)['priority'] if path.exists() else 100
    command = [sys.executable, '-B', str(NAVTEST_TOOL), '--enqueue', '--run', plan['run'],
               '--tag', checkpoint['tag'], '--output', str(output), '--priority', str(priority)]
    result = subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    receipt = json.loads(result.strip().splitlines()[-1]); request = read(receipt['request'])
    require(request['run'] == str(Path(plan['run']).resolve()) and request['tag'] == checkpoint['tag']
            and request['updates'] == plan['target'] and request['group'] == plan['group']
            and request['training_identity'] == checkpoint['identity'] and request['population'] == 'navtest'
            and request['scenes'] == 12146 and request['seed'] == 42
            and request['variants'] == ['q0', 'q_final'], 'Newton enqueue contract differs')
    return {'request': receipt['request'], 'request_sha256': digest(receipt['request']),
            'command': command, 'routing_tool_sha256': digest(NAVTEST_TOOL), 'policy_sha256': digest(POLICY)}


def check_references(manifest):
    clean_source()
    for name, sha in manifest['preserved_files_sha256'].items():
        require(digest(name) == sha, 'Preserved scientific/routing evidence changed: '+name)


def routed(manifest, command):
    if not manifest['route_prefix']:
        return command
    prefix = manifest['route_prefix']
    require(prefix[:4] == ['cd', str(ROOT), '&&', 'env'], 'Original SSH route prefix differs')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', manifest['training_host'],
            'cd '+shlex.quote(prefix[1])+' && env '+shlex.join(prefix[4:]+command)]


def wait_resource(manifest):
    command = ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']
    while True:
        authorize(manifest['training_host'])
        rows = [[v.strip() for v in row.split(',')] for row in subprocess.check_output(routed(manifest, command), text=True).splitlines()]
        require(sorted(r[0] for r in rows) == manifest['physical_gpu_UUIDs'], 'Physical GPU UUIDs changed')
        sharing = manifest['allow_low_util_sharing']
        if sharing:
            require(read('/mnt/project/server_dispatch_policy.json')['task_authorizations']['structured_world_fgtr_round1']['shared_GPU_deployment_authorized'], 'Sharing revoked')
        if all(int(r[3]) <= 5 and int(r[2]) >= manifest['required_free_MiB'] and (sharing or int(r[1]) <= 128) for r in rows):
            return
        time.sleep(10)


def adopt(manifest, output):
    output.mkdir(parents=True, exist_ok=False)
    state = lambda status, **values: write(output/'STATE.json', {'status': status, 'unix': time.time(), **values})
    def log(record):
        with (output/'executed_commands.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
    write(output/'REGISTRATION.json', {**manifest, 'controller_code_sha256': digest(__file__), 'evaluation_policy': read(POLICY)})
    try:
        with ExitStack() as held:
            lease = ART/'physical_gpu_locks'/f"navtest_adoption_{manifest['supervisor_host']}_{manifest['old_supervisor']['pid']}.lock"
            fd = held.enter_context(lease.open('a')); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_references(manifest); evaluation_policy(manifest['groups']); authorize(manifest['training_host'])
            parent = verify_process(manifest['old_supervisor']); child = verify_process(manifest['adopted_child'])
            require(parent['children'] == [child['pid']], 'Supervisor children changed')
            pfd = os.pidfd_open(parent['pid'])
            held.callback(os.close, pfd)
            signal.pidfd_send_signal(pfd, signal.SIGSTOP)
            for _ in range(100):
                if verify_process(manifest['old_supervisor'])['state'] in ('T', 't'): break
                time.sleep(.01)
            verify_process(manifest['old_supervisor'], stopped=True)
            state('ADOPTED_TRAINING_CHILD_RUNNING', child_pid=child['pid'], old_lockkeeper_pid=parent['pid'],
                  future_DEV_dispatch_disabled=True, stage=manifest['current_stage'], policy_sha256=digest(POLICY))
            for plan in manifest['training_plan']:
                if (Path(plan['run'])/'checkpoints'/f"milestone_{plan['target']:07d}"/'COMPLETE.json').exists():
                    log({'stage': 'queue_navtest_existing_milestone', 'request': queue_navtest(plan)})
            while True:
                verify_process(manifest['old_supervisor'], stopped=True)
                child = verify_process(manifest['adopted_child'])
                if child['state'] == 'Z': break
                time.sleep(5)
            exit_code = os.waitstatus_to_exitcode(child['exit_status'])
            log({'stage': 'adopted_child_exit', 'pid': child['pid'], 'raw_wait_status': child['exit_status'], 'exit_code': exit_code})
            # The child is now a retained zombie: no running worker/SSH is signaled.
            verify_process(manifest['old_supervisor'], stopped=True)
            require(process(parent['pid'])['children'] == [child['pid']], 'Supervisor children changed before retirement')
            signal.pidfd_send_signal(pfd, signal.SIGKILL)
            require(exit_code == 0, 'Adopted real training stage failed')
            current = manifest['current_stage']; native(current['run'], current['target'])
            reference = manifest['adopted_resume_reference']
            if reference:
                require(digest(reference['file']) == reference['sha256'] and reference['report']['passed'],
                        'Adopted resume boundary report changed')
                log({'stage': 'adopted_native_resume_verified', **reference})
            queue_locks = [ART/'physical_gpu_locks'/('queue_'+hashlib.sha256(str(Path(p['run']).resolve()).encode()).hexdigest()+'.lock')
                           for p in manifest['training_plan']]
            paths = sorted(set(map(Path, manifest['original_locks'])) | set(queue_locks))
            while True:
                with ExitStack() as attempt:
                    try:
                        for path in paths:
                            fd = attempt.enter_context(path.open('a')); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held.enter_context(attempt.pop_all()); break
                    except BlockingIOError:
                        state('WAITING_FOR_RESOURCE_LOCK', active_training_child=False)
                time.sleep(5)
            for plan in manifest['training_plan']:
                before = native(plan['run'])
                if before is None or before['completed'] < plan['target']:
                    require(before is not None if '--resume' in plan['command'] else before is None,
                            'Native resume/fresh initialization differs from original plan')
                    check_references(manifest); evaluation_policy(manifest['groups']); wait_resource(manifest)
                    require(not (Path(plan['run'])/'STOP_AFTER_CHECKPOINT').exists(), 'Explicit stop request exists')
                    command = routed(manifest, plan['command'])
                    record = {'stage': plan['group']+'_train_to_'+str(plan['target']), 'command': plan['command'],
                              'native_before': before, 'started_unix': time.time()}
                    state('RUNNING_ORIGINAL_TRAINING', **record); log(record)
                    env = {**os.environ, **manifest['training_environment']}
                    with (output/(record['stage']+'.log')).open('a') as stream:
                        result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    require(result.returncode == 0, 'Real training stage failed: '+record['stage'])
                    if before:
                        boundary = read(Path(plan['run'])/'RESUME_BOUNDARY_VERIFIED.json')
                        require(boundary['passed'] and boundary['identity'] == before['identity'] and boundary['completed'] == before['completed'],
                                'Native resume boundary differs')
                        log({'stage': 'native_resume_verified', 'report': boundary})
                milestone = native(plan['run'], plan['target'])
                log({'stage': 'queue_navtest', 'milestone': milestone, 'request': queue_navtest(plan)})
            state('ORIGINAL_100K_TRAINING_COMPLETE_NAVTEST_REQUESTS_QUEUED', groups=manifest['groups'])
    except BaseException as error:
        state('FAILED', error=repr(error), genuine_workers_signaled=False, legacy_DEV_dispatch_remains_held=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-pid', type=int)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.capture_pid:
        require(not args.execute and not args.output.exists(), 'Capture writes one fresh reviewable manifest')
        write(args.output, capture(args.capture_pid))
    else:
        require(args.manifest, '--manifest is required')
        manifest = read(args.manifest)
        require(manifest['supervisor_host'] == socket.gethostname().removesuffix('-worker-0'), 'Adopt on the supervisor host')
        if args.execute:
            authorize(manifest['supervisor_host']); adopt(manifest, args.output)
        else:
            print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
