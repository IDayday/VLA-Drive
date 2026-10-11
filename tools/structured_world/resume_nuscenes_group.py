"""Routing-only NU G1 recovery. Launch with the local qwen Python; never kill jobs.

G1 remains method-qualified on its original host. G3 supplies destination-only
environment/hardware evidence, not a new G1 qualification or scientific result.
Inspect mode performs only reads (including SSH); worker tools own run artifacts.
"""
import argparse
import ast
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shlex
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-nuscenes-formal-frozen-20261009')
SHA = '41da7c198cf0ac4f799aee2a27a0a8126de30230'
IDENTITY = '7bc94fb37bbfe65be4491a0d9666cb7d1e294cd1c8a2253a362e4bd26bea7a7a'
TARGETS = (4356, 8712, 13068, 17424)
POLICY = Path('/mnt/project/server_dispatch_policy.json')
LOCAL_PYTHON = Path('/tmp/structured-world-round1/envs/qwen/bin/python')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authorize(host):
    task = read(POLICY)['task_authorizations']['structured_world_fgtr_round1']
    require(host in task['allowed_hosts'] and host in task.get('latest_explicit_resource_hosts', task['allowed_hosts']),
            'Task dispatch authorization changed: '+host)


def original_plan(path):
    """Extract parser/training/evaluation expressions from the immutable runner."""
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
            and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA,
            'Original frozen source must be clean at exact SHA')
    controller = read(path)
    command = controller['command']
    runner = ROOT/'tools/structured_world/run_nuscenes_group.py'
    require(command[:3] == [str(LOCAL_PYTHON), '-u', str(runner)] and controller['training_source_sha'] == SHA,
            'Not the original frozen NU controller')
    ns = runpy.run_path(str(runner))
    body = next(n.body for n in ast.parse(runner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    parse = body[:end] + ast.parse('args = parser.parse_args('+repr(command[3:])+')').body
    exec(compile(ast.Module(body=parse, type_ignores=[]), str(runner), 'exec'), ns)
    args = ns['args']
    require(args.group == 'G1_FULL_UNIFORM' and args.phase == 'formal' and not args.allow_sharing, 'Wrong original queue')
    ns['config'] = ROOT/'configs/structured_world/nuscenes_G1_FULL_UNIFORM_seed42.yaml'
    training = next(n for n in ast.walk(ast.Module(body=body, type_ignores=[]))
                    if isinstance(n, ast.FunctionDef) and n.name == 'training')
    exec(compile(ast.Module(body=[training], type_ignores=[]), str(runner), 'exec'), ns)
    ns['evaluation_calls'] = {n.args[0].left.value: n.args[1] for n in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run'
        and isinstance(n.args[0], ast.BinOp) and isinstance(n.args[0].left, ast.Constant)
        and n.args[0].left.value in ('val_infer_', 'val_score_', 'val_fields_')}
    return args, ns


def commands(args, ns, target):
    """Pure command planning; no model imports, writes or worker dispatch."""
    require(target in TARGETS, 'Unregistered epoch observation boundary')
    run = args.output/args.group
    values = {**ns, 'run_root': run, 'target': target, 'tag': f'milestone_{target:07d}',
        'predictions': run/f'val_{target}_seed42', 'planning': run/f'val_{target}_planning',
        'fields': run/f'val_{target}_scene_metrics'}
    train = ns['training'](run, 'formal')
    if target < TARGETS[-1]:
        train += ['--stop-after', str(target)]
    return [('train_to_'+str(target), train+['--resume'], True, None)] + [
        (prefix+str(target), eval(compile(ast.Expression(node), '<frozen evaluation>', 'eval'), values), gpu,
         values[key]/'COMPLETE.json') for prefix, gpu, key in
        (('val_infer_', True, 'predictions'), ('val_score_', False, 'planning'), ('val_fields_', False, 'fields'))
        for node in [ns['evaluation_calls'][prefix]]]


def latest(run):
    tag = (run/'checkpoints/latest').read_text().strip()
    require(re.fullmatch(r'(latest|milestone)_\d{7}', tag), 'Unsafe native latest pointer')
    complete = read(run/'checkpoints'/tag/'COMPLETE.json')
    names = {'mp_rank_00_model_states.pt'} | {f'rng_rank{i}.pt' for i in range(8)} | {
        f'zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(8)}
    require(complete['identity'] == IDENTITY and complete['ranks'] == 8 and complete['tag'] == tag
            and complete['completed'] == int(tag.split('_')[1]) and 0 < complete['completed'] <= TARGETS[-1]
            and complete['status'] in ('RUNNING', 'PAUSED', 'COMPLETE') and set(complete['file_sha256']) == names
            and {p.name for p in (run/'checkpoints'/tag).glob('*.pt')} == names
            and all((run/'checkpoints'/tag/name).is_file() for name in names), 'Incomplete/foreign native checkpoint')
    return {'tag': tag, 'completed': complete['completed'], 'complete_file': str(run/'checkpoints'/tag/'COMPLETE.json'),
            'complete_sha256': digest(run/'checkpoints'/tag/'COMPLETE.json'), 'identity': IDENTITY}


def reviewed(path, references):
    q = read(path)
    require(q['passed'] and q['deployment_probe_passed'] and q['current_only_prefetch_passed']
            and q['real_eight_rank_recovery']['passed'] and q['real_eight_rank_recovery']['completed'] == 20
            and q['profile']['updates'] == 16, 'Incomplete reviewed qualification')
    expected = {q['raw_qualification_file']: q['raw_qualification_sha256'], **q['deployment_probe_files_sha256']}
    for filename, sha in expected.items():
        require(digest(filename) == sha, 'Qualification evidence changed: '+filename)
    references.update(expected)
    references[str(path)] = digest(path)
    return q


def route(host, command):
    authorize(host)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            'cd '+shlex.quote(str(ROOT))+' && env OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false '+shlex.join(command)]


def readiness(args, options, ns, registration, original, destination, references):
    for filename, sha in references.items():
        require(digest(filename) == sha, 'Preserved evidence changed: '+filename)
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
            and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SHA,
            'Original frozen source must be clean at exact SHA')
    require(ns['execution_source_digest']() == original['execution_source_sha256']
            and ns['scientific_config_digest'](ns['config']) == original['scientific_config_sha256'], 'Frozen scientific source changed')
    code = '''import hashlib,json,runpy,subprocess
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
root,assets,geometry,acceptance,val_inputs,val_labels=map(Path,__import__('sys').argv[1:])
n=runpy.run_path(str(root/'tools/structured_world/run_nuscenes_group.py'))
print(json.dumps({'sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'dirty':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
 'source_digest':n['execution_source_digest'](), 'config_digest':n['scientific_config_digest'](root/'configs/structured_world/nuscenes_G1_FULL_UNIFORM_seed42.yaml'),
 'destination_config_digest':n['scientific_config_digest'](root/'configs/structured_world/nuscenes_G3_EVENT_LOCAL_seed42.yaml'),
 'staging':read(assets/'STAGING_COMPLETE.json'), 'labels':read(assets/'train_labels/identity.json'),
 'labels_complete':read(assets/'train_labels/COMPLETE.json'), 'label_count':len(read(assets/'train_labels/index.json')),
 'dino':read(assets/'current_C1/identity.json'), 'dino_complete':read(assets/'current_C1/COMPLETE.json'),
 'geometry_sha256':sha(geometry),'acceptance_sha256':sha(acceptance),
 'val_count':len(read(val_labels/'index.json')), 'val_inputs':read(val_inputs/'identity.json'),
 'val_inputs_complete':read(val_inputs/'COMPLETE.json'), 'val_input_count':len(read(val_inputs/'index.json'))}))'''
    remote = json.loads(subprocess.check_output(route(options.destination_host, [args.qwen_python, '-B', '-c', code,
        *map(str, (ROOT, args.fast_assets, args.geometry, args.execution_acceptance, args.val_inputs, args.val_labels))]), text=True))
    require(remote['sha'] == SHA and not remote['dirty'] and remote['source_digest'] == original['execution_source_sha256']
            and remote['config_digest'] == original['scientific_config_sha256']
            and remote['destination_config_digest'] == destination['scientific_config_sha256'], 'Destination scientific source mismatch')
    for key in ('label_identity', 'DINO_identity'):
        require(remote['staging'][key] == registration[key], 'Destination staging identity mismatch: '+key)
    require(remote['staging']['scenes'] == remote['label_count'] == remote['labels_complete']['scenes'] == remote['dino_complete']['scenes'] == 23230
            and remote['staging']['cameras'] == 6 and remote['staging']['image_files'] == 139380
            and remote['labels']['identity'] == remote['labels_complete']['identity'] == registration['label_identity']
            and remote['labels']['population_kind'] == 'full_train_population' and not remote['labels_complete']['errors']
            and remote['dino']['identity'] == remote['dino_complete']['identity'] == registration['DINO_identity']
            and remote['dino']['structured_cache_identity'] == registration['label_identity']
            and remote['geometry_sha256'] == read(args.output/args.group/'identity.json')['initialization_assets']['shared_geometry']['sha256']
            and remote['acceptance_sha256'] == read(args.output/args.group/'identity.json')['execution_acceptance']['sha256']
            and remote['val_count'] == remote['val_input_count'] == remote['val_inputs_complete']['scenes'] == len(read(args.val_labels/'index.json')) == 4969
            and remote['val_inputs_complete']['identity'] == remote['val_inputs']['identity'] == read(args.val_inputs/'identity.json')['identity'], 'Destination assets incomplete/mismatched')
    lines = subprocess.check_output(route(options.destination_host, ['nvidia-smi',
        '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']), text=True).splitlines()
    devices = [dict(zip(('uuid', 'used_MiB', 'free_MiB', 'utilization'), [row[0], *map(int, row[1:])]))
               for row in ([v.strip() for v in line.split(',')] for line in lines)]
    require(len(devices) == 8 and sorted(r['uuid'] for r in devices) == destination['physical_gpu_UUIDs'], 'Destination physical GPUs changed')
    return remote, devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-controller', 'destination-qualification', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--destination-host', required=True)
    parser.add_argument('--wait-for-resource', action='store_true')
    parser.add_argument('--inspect-only', action='store_true')
    options = parser.parse_args()
    require(Path(sys.executable).absolute() == LOCAL_PYTHON, 'Use the original local qwen Python for CPU scoring')
    args, ns = original_plan(options.original_controller)
    run = args.output/args.group
    output = options.output.resolve()
    protected = (args.output, ROOT, args.qualification.parent, options.destination_qualification.parent)
    require(not any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected), 'Control output overlaps preserved artifacts')
    if not options.inspect_only:
        output.mkdir(parents=True, exist_ok=False)
    def state(status, **record):
        if not options.inspect_only:
            ns['atomic_json'](output/'STATE.json', {'status': status, 'unix': time.time(), **record})
    try:
        references = {str(p): digest(p) for p in (options.original_controller, args.output/'REGISTRATION.json', run/'identity.json',
            args.execution_acceptance, args.val_inputs/'identity.json', args.val_inputs/'COMPLETE.json', args.val_labels/'index.json')}
        for name in ('EXECUTED_CONTROLLER.json', 'QUEUE_FAILURE.json'):
            if (args.output/name).exists(): references[str(args.output/name)] = digest(args.output/name)
        original = reviewed(args.qualification, references)
        destination = reviewed(options.destination_qualification, references)
        registration, identity = read(args.output/'REGISTRATION.json'), read(run/'identity.json')
        require(original['group'] == args.group and original['host'] == args.host and destination['group'] == 'G3_EVENT_LOCAL'
                and destination['host'] == options.destination_host and registration['source_sha'] == identity['training_source_sha'] == SHA
                and identity['identity'] == IDENTITY and registration['host'] == args.host and registration['group'] == args.group
                and registration['phase'] == 'formal' and registration['updates'] == list(TARGETS), 'Original/destination evidence has foreign identity')
        for key in ('execution_source_sha256', 'label_identity', 'DINO_identity', 'geometry_identity'):
            require(registration[key] == original[key] == destination[key], 'Migration evidence mismatch: '+key)
        require(registration['scientific_config_sha256'] == original['scientific_config_sha256']
                and registration['physical_gpu_UUIDs'] == original['physical_gpu_UUIDs']
                and identity['training_source_tree_sha256'] == original['execution_source_sha256']
                and (identity['schedule_horizon'], identity['effective_global_batch'], identity['microbatch'], identity['FM_repeat'], identity['scenes_per_epoch']) == (17424, 32, 4, 8, 23230)
                and args.geometry_identity == original['geometry_identity'] and args.dino_identity == original['DINO_identity'], 'Original scientific contract changed')
        uuids = destination['physical_gpu_UUIDs']
        require(uuids == sorted(set(uuids)) and len(uuids) == 8, 'Eight qualified destination UUIDs required')
        required_MiB = math.ceil((original['profile']['peak_rank0_GPU_allocated_GiB']+12)*1024)
        queue_lock = args.gpu_lock_root/('queue_'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()+'.lock')
        gpu_lock = args.gpu_lock_root/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')
        checkpoint = latest(run)
        with (run/'metrics.jsonl').open('rb') as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size-65536))
            lastmetrics = json.loads(stream.read().splitlines()[-1])['updates']
        manifest = {'kind': 'routing_only_migration', 'group': args.group, 'original_host': args.host,
            'destination_host': options.destination_host, 'original_run_root': str(run), 'training_evaluation_source_sha': SHA,
            'controller_code_sha256': digest(__file__), 'immutable_runner': str(ROOT/'tools/structured_world/run_nuscenes_group.py'),
            'execution_source_sha256': original['execution_source_sha256'], 'scientific_config_sha256': original['scientific_config_sha256'],
            'preserved_files_sha256': references, 'original_method_qualification': str(args.qualification),
            'destination_environment_qualification': str(options.destination_qualification), 'original_registration': registration,
            'destination_profile': destination['profile'], 'physical_gpu_UUIDs': uuids, 'required_free_MiB': required_MiB,
            'idle_limits': {'used_MiB': 128, 'utilization': 5}, 'locks': [str(queue_lock), str(gpu_lock)],
            'native_checkpoint': checkpoint, 'lastmetrics_updates': lastmetrics, 'allow_sharing': False,
            'limits': 'G3 is destination environment/hardware/profile evidence only; original G1 qualification is unchanged. Native resume, no new initialization.',
            'plan': [{'stage': s, 'command': c, 'gpu': g, 'complete': str(p) if p else None,
                      'skip': checkpoint['completed'] >= t if p is None else p.exists()} for t in TARGETS for s, c, g, p in commands(args, ns, t)]}
        if options.inspect_only:
            manifest['readiness'], manifest['GPU_snapshot'] = readiness(args, options, ns, registration, original, destination, references)
            manifest['resources_idle'] = all(r['free_MiB'] >= required_MiB and r['used_MiB'] <= 128 and r['utilization'] <= 5 for r in manifest['GPU_snapshot'])
            print(json.dumps(manifest, indent=2)); return
        ns['atomic_json'](output/'REGISTRATION.json', manifest)
        def wait(reason, stage):
            state('WAITING_FOR_RESOURCE', reason=reason, stage=stage, native_checkpoint=latest(run))
            if not options.wait_for_resource: raise RuntimeError(reason)
            time.sleep(30)
        with ExitStack() as held:
            args.gpu_lock_root.mkdir(parents=True, exist_ok=True)
            while True:
                authorize(options.destination_host)
                with ExitStack() as attempt:
                    try:
                        for path in (queue_lock, gpu_lock):
                            lock = attempt.enter_context(path.open('a'))
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held.enter_context(attempt.pop_all()); break
                    except BlockingIOError:
                        pass
                wait('Queue or physical GPU lock is held', 'acquire_locks')
            for target in TARGETS:
                for stage, command, gpu, complete in commands(args, ns, target):
                    before = latest(run)
                    if complete is not None:
                        milestone = read(run/'checkpoints'/f'milestone_{target:07d}'/'COMPLETE.json')
                        require(milestone['completed'] == target and milestone['identity'] == IDENTITY, 'Wrong observation boundary')
                    if complete is None and before['completed'] >= target: continue
                    if complete is not None and complete.exists():
                        require(read(complete)['scenes'] == 4969, 'Existing validation marker has foreign population')
                        continue
                    while True:
                        remote, snapshot = readiness(args, options, ns, registration, original, destination, references)
                        if all(r['free_MiB'] >= required_MiB and r['used_MiB'] <= 128 and r['utilization'] <= 5 for r in snapshot): break
                        wait('Destination is occupied or lacks memory reserve', stage)
                    authorize(options.destination_host if gpu else socket.gethostname().removesuffix('-worker-0'))
                    require(latest(run) == before, 'Native checkpoint changed before dispatch')
                    require(not (run/'STOP_AFTER_CHECKPOINT').exists(), 'Run has a checkpoint stop request')
                    record = {'stage': stage, 'command': command, 'host': options.destination_host if gpu else 'local CPU',
                              'GPU_snapshot': snapshot, 'native_checkpoint': before, 'started_unix': time.time()}
                    state('RUNNING', **record)
                    with (output/'executed_commands.jsonl').open('a') as log: log.write(json.dumps(record)+'\n')
                    with (output/(stage+'.log')).open('a') as log:
                        result = subprocess.run(route(options.destination_host, command) if gpu else command, cwd=ROOT,
                            env={**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}, stdout=log, stderr=subprocess.STDOUT)
                    require(result.returncode == 0, f'Stage {stage} failed with exit {result.returncode}')
                    if complete is None:
                        boundary = read(run/'RESUME_BOUNDARY_VERIFIED.json')
                        require(boundary['passed'] and boundary['identity'] == IDENTITY and boundary['completed'] == before['completed']
                                and latest(run)['completed'] == target, 'Native recovery boundary/progress verification failed')
                        record['resume_boundary'] = {'file': str(run/'RESUME_BOUNDARY_VERIFIED.json'),
                            'sha256': digest(run/'RESUME_BOUNDARY_VERIFIED.json'), 'completed_reference': before['completed'], 'report': boundary}
                    else:
                        require(complete.is_file() and read(complete)['scenes'] == 4969, 'Stage did not complete the fixed validation population')
                    with (output/'executed_commands.jsonl').open('a') as log: log.write(json.dumps({**record, 'status': 'PASSED', 'ended_unix': time.time()})+'\n')
            state('NUSCENES_24EPOCH_TRAIN_AND_VAL_COMPLETE', native_checkpoint=latest(run), training_seeds_completed=[42])
    except BaseException as error:
        state('FAILED', error=repr(error), checkpoints_preserved=True, unrelated_processes_stopped=False)
        raise


if __name__ == '__main__':
    main()
