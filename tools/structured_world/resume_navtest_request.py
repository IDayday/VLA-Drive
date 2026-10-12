"""Resume one externally recovered request, then exec the frozen persistent serve.

No recovery, process signals, retries, model changes or historical-bank scan here.
Run only after frozen recover_infrastructure and all old launchers/scorer exit.
The original Observer runs concurrently; frozen execute supplies GPU admission.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

DISPATCHER_SHA256 = '558b6fa44ea15deeed0fec6baa2f63b1524f5926662b14de542d4e1eaa5bb5d6'
CADENCE_SHA256 = 'd3885dc109615f2c2e4a99a2d59cf05c4117fbf7735ec6fba69fed1211e97ddc'
MODEL_SOURCE = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_modules(path):
    path = path.resolve()
    modules = []
    for name, source, checksum in [('dispatcher', path, DISPATCHER_SHA256),
            ('cadence', path.with_name('navtest_cadence.py'), CADENCE_SHA256)]:
        require(sha(source) == checksum, 'Frozen ' + name + ' source hash differs')
        spec = importlib.util.spec_from_file_location('_resume_navtest_' + name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    require(modules[0].SHA == MODEL_SOURCE and modules[1].SOURCE == MODEL_SOURCE,
            'Frozen model source differs')
    return modules


def exited(record):
    """Known local PID only: absent, zombie or recorded identity replaced."""
    require(record.get('host', socket.gethostname()).removesuffix('-worker-0') ==
            socket.gethostname().removesuffix('-worker-0'), 'Cannot prove remote launcher exit locally')
    pid = int(record['pid'])
    require(pid > 0 and pid != os.getpid(), 'Invalid previous process PID')
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return dict(pid=pid, exists=False)
    start = record.get('proc_start_ticks')
    require(fields[0] == 'Z' or (start is not None and str(start) != fields[19]),
            'Previous owned process still alive or PID identity unverifiable: ' + str(pid))
    return dict(pid=pid, exists=True, state=fields[0], start_ticks=fields[19],
                expected_start_ticks=start)


def validate(args, dispatcher, cadence):
    dispatcher.authorized()
    policy = cadence.load_policy(args.cadence_policy)
    require(Path(policy['dispatch_output']).resolve() == args.output.resolve(), 'Policy/output differs')
    require(re.fullmatch(r'[A-Za-z0-9_]+', args.request_key), 'Invalid request key')
    req_path = args.output / 'requests' / (args.request_key + '.json')
    req = read(req_path)
    require(req['key'] == args.request_key and req['population'] == 'navtest' and
            req['scenes'] == 12146 and req['logs'] == 136 and req['seed'] == 42 and
            req['precision'] == 'FP32 / TF32 disabled' and req['variants'] == ['q0', 'q_final'],
            'Request scientific contract differs')
    require(any(r['training_identity'] == req['training_identity'] and r['group'] == req['group']
                for r in policy['runs']) and cadence.eligible(policy, req), 'Request not registered/eligible')
    job = args.output / 'jobs' / req['key']
    require(not (job / 'INFERENCE_FAILED.json').exists(), 'Unresolved failure; recover externally first')
    require(not (job / 'scores/INVALID.json').exists(), 'Scientific score failure cannot resume')
    registration = read(job / 'REGISTRATION.json')
    require(registration['request'] == req and registration['frozen_evaluation_source'] == MODEL_SOURCE
            and Path(registration['inputs']).resolve() == args.inputs.resolve(), 'Original registration differs')
    identity = read(job / 'identity.json')
    require(all(identity[k] == v for k, v in {
        'training_identity': req['training_identity'], 'checkpoint': req['tag'],
        'updates': req['updates'], 'checkpoint_files_sha256': req['checkpoint_files_sha256'],
        'evaluation_source_sha': MODEL_SOURCE,
        'input_identity': read(args.inputs / 'identity.json')['identity']}.items()), 'Prediction/source/input binding differs')
    recoveries = sorted((job / 'recoveries').glob('*/RECOVERY.json'))
    require(bool(recoveries), 'Missing original infrastructure recovery receipt')
    recovery_path = recoveries[-1]; recovery = read(recovery_path)
    evidence = recovery['evidence']; proofs = recovery['worker_exit_proofs']
    failure_path = recovery_path.with_name('INFERENCE_FAILED.json')
    require(evidence.get('classification') == 'infrastructure' and evidence.get('reason') and
            evidence.get('request_key') == req['key'] and evidence.get('failure_sha256') ==
            recovery['original_failure_sha256'] == sha(failure_path) and bool(proofs) and
            recovery['reconciled']['verified_prior_scenes'] >= 0, 'Recovery receipt not bound to original failure')
    require(all(p['observed']['exists'] is False or (p.get('expected_start_ticks') is not None and
                str(p['expected_start_ticks']) != str(p['observed']['start_ticks'])) for p in proofs),
            'Recovery did not prove original remote worker identities gone')
    covered = {(p['host'].removesuffix('-worker-0'), int(p['pid'])): p for p in proofs}
    for path in (job / 'attempts').glob('*/shards/*.json'):
        report = read(path)
        proof = covered.get((report['host'].removesuffix('-worker-0'), int(report['pid'])))
        require(proof is not None and proof.get('expected_start_ticks') == report.get('proc_start_ticks'),
                'Current worker report not covered by external recovery receipt')
    controller = read(args.output / 'CONTROLLER.json')
    old_processes = [exited(controller)]
    # Launch records contain local SSH launcher PIDs, despite their remote host field.
    for path in sorted((job / 'attempts').glob('*/launches/*.json')):
        launch = read(path)
        old_processes.append(exited({k: v for k, v in launch.items() if k != 'host'}))
    old_processes.append(exited(read(job / 'SCORER_LAUNCH.json')))
    bindings = {str(p): sha(p) for p in [req_path, args.cadence_policy,
                Path(policy['base_policy_path']), job / 'REGISTRATION.json',
                job / 'identity.json', args.inputs / 'identity.json', recovery_path, failure_path]}
    cost = args.output / 'COHOST_COST_POLICY.json'
    if cost.exists(): bindings[str(cost)] = sha(cost)
    return req, policy, dict(bindings=bindings, previous_controller=controller,
                            previous_local_process_exit_proofs=old_processes)


def serve_argv(args):
    return [sys.executable, '-B', '-u', str(args.dispatcher), '--serve',
            '--output', str(args.output), '--inputs', str(args.inputs), '--routes', str(args.routes),
            '--cadence-policy', str(args.cadence_policy), '--score-workers', str(args.score_workers),
            '--canonical-python', args.canonical_python]


def resume(args, dispatcher, cadence):
    # NB: frozen serve re-acquires this same lock after exec; a race loses safely.
    with (args.output / 'controller.lock').open('a') as lock, ExitStack() as cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        req, policy, proof = validate(args, dispatcher, cadence)
        stamp = dispatcher.now().replace(':', '').replace('+', '_') + '_' + str(os.getpid())
        record = args.output / 'resume_requests' / req['key'] / stamp / 'ADMISSION.json'
        proof.update(request=req, pid=os.getpid(), host=socket.gethostname(), started_utc=dispatcher.now(),
                     wrapper=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
                     dispatcher=str(args.dispatcher), dispatcher_sha256=DISPATCHER_SHA256,
                     cadence_sha256=CADENCE_SHA256, status='EXTERNAL_RECOVERY_VERIFIED',
                     gpu_admission='Frozen execute refreshes routes and qualifies before launching',
                     persistent_serve_argv=serve_argv(args))
        dispatcher.atomic(record, proof)
        dispatcher.atomic(args.output / 'CONTROLLER.json', dict(pid=os.getpid(), host=socket.gethostname(),
            proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            started_utc=dispatcher.now(), routes_file=str(args.routes), tool=str(args.dispatcher),
            tool_sha256=DISPATCHER_SHA256, cadence_policy=str(args.cadence_policy),
            cadence_policy_sha256=sha(args.cadence_policy), wrapper=str(Path(__file__).resolve()),
            targeted_request=req['key'], admission=str(record)))
        observer = cadence.Observer(policy, args.output, dispatcher.request, dispatcher.valid_score_completion)
        cleanup.callback(observer.close)
        try:
            require(not observer.error, 'Cadence observer failed before execute')
            dispatcher.execute(args, req)  # No historical completed-bank loop.
            require(not observer.error, 'Cadence observer failed during execute')
            completed = read(args.output / 'completed' / (req['key'] + '.json'))
            require(completed.get('valid') is True and completed['request'] == req,
                    'Frozen execute did not certify target completion')
            observer.close()  # Join before releasing the dispatch lock.
            cleanup.pop_all()  # Successful close needs no second join.
            require(not observer.error, 'Cadence observer failed during close')
        except BaseException as error:
            dispatcher.atomic(record.with_name('FAILED.json'), dict(error=repr(error), observed_utc=dispatcher.now()))
            raise  # No serve handoff, retry, or clearing of the original failure.
        dispatcher.atomic(record.with_name('COMPLETE.json'), dict(request_key=req['key'],
                          observed_utc=dispatcher.now(), completed_receipt=str(args.output / 'completed' / (req['key'] + '.json'))))
    argv = serve_argv(args)
    os.execv(sys.executable, argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dispatcher', 'output', 'inputs', 'cadence-policy']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--request-key', required=True)
    parser.add_argument('--score-workers', type=int, default=12)
    parser.add_argument('--routes', type=Path)
    parser.add_argument('--canonical-python', default='/root/miniconda3/envs/navsim/bin/python')
    args = parser.parse_args(argv)
    require(args.score_workers > 0, 'Positive score worker count required')
    args.dispatcher = args.dispatcher.resolve()
    args.output = args.output.resolve(); args.inputs = args.inputs.resolve()
    args.cadence_policy = args.cadence_policy.resolve()
    args.routes = args.routes.resolve() if args.routes else args.output / 'ROUTES.json'
    # Defaults from frozen main; execute/qualify/launch_worker consume these.
    for key, value in dict(run=None, tag=None, priority=100, shard=0, shards=1,
            memory_cap_gib=26, cohost_duty_cycle=1., scheduling_policy=None,
            fp32_export=None, attempt=None, recovery_evidence=None, exit_when_idle=False).items():
        setattr(args, key, value)
    return args


def main():
    args = parse_args()
    dispatcher, cadence = load_modules(args.dispatcher)
    resume(args, dispatcher, cadence)


if __name__ == '__main__':
    main()
