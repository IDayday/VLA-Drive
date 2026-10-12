"""CPU-only closure of individually registered SSH-255 attempts with a complete bank.

Never launches inference, changes the evaluator, repairs rows, or signals a PID.
Use the canonical navsim Python. Original negative evidence remains archived.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
COLD = Path('/mnt/project/VLA-Drive-structured-world-fgtr-navtest-cadence-frozen-20261012')
TOOL = COLD/'tools/structured_world/run_navsim_navtest.py'
TOOL_SHA = '91721cc10790b4890e5f9ce2aaf6016ba58df560def474745a23305698a099e5'
FAILURE_SHA = '0d3b25b1280d58d56a76bed6ccff0943f9b3b027ba042d8a4a68bbfc3a666f2f'
KEY = 'G3_EVENT_LOCAL_0070000'
ATTEMPT = '20261012T010941_4a8a639482f4'
CASES = {
    'g3_70k': dict(cold=COLD, tool_sha=TOOL_SHA, failure_sha=FAILURE_SHA,
        key=KEY, attempt=ATTEMPT, updates=70000, exit_codes=[255]*8+[0]*16),
    'g3_90k': dict(cold=Path('/mnt/project/VLA-Drive-structured-world-navtest-window-v2-frozen-20261012'),
        tool_sha='2d8bbe3a182f50da5eb09d8eabf6083d207eddcbe0e296e63dfcf4c1e52a6a44',
        failure_sha='7e3ec434dc4d286221c01618d17c8d6effda6f3a937dc4da73633903e5290828',
        key='G3_EVENT_LOCAL_0090000',
        attempt='20261012T151338_03c468953475', updates=90000,
        exit_codes=[0]*8+[255]*8, closed_by_remote_host=True),
}


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): sha.update(chunk)
    return sha.hexdigest()


def load_guard(case=None):
    case = CASES['g3_70k'] if case is None else case
    tool = case['cold']/'tools/structured_world/run_navsim_navtest.py'
    require(digest(tool) == case['tool_sha'], 'Frozen dispatcher source changed')
    spec = importlib.util.spec_from_file_location('transport_original_dispatcher', tool)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def failure_contract(value, case=None):
    case = CASES['g3_70k'] if case is None else case
    require(value['classification'] == 'unclassified' and value['exit_codes'] == case['exit_codes'],
            'Only the registered SSH-255 failure is recoverable by this operator')


def report_contract(report, launch, i, prediction_identity, case):
    require(report['shard'] == launch['shard'] == i and report['attempt'] == case['attempt']
            and report['shards'] == launch['shards'] == len(case['exit_codes'])
            and report['prediction_identity'] == prediction_identity
            and report['routing_tool_sha256'] == case['tool_sha'] and launch['command'][0] == 'ssh',
            'Foreign worker/launcher evidence')
    require(report['status'] == ('INFERENCE' if case['exit_codes'][i] == 255 else 'COMPLETE')
            and not report.get('error') and report['scenes_done'] > 0,
            'Unsupported worker/scientific failure')


def score_row(row, token, log, variant, metrics):
    require(row['token'] == token and row['log'] == log and row['variant'] == variant and row['status'] == 'ok',
            'Foreign/failed canonical score row; no scientific failure recovery')
    require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in metrics),
            'Nonfinite/out-of-range canonical score')


def absent_local(pid):
    require(not Path('/proc', str(pid)).exists(), 'Prior local PID still exists; do not signal or guess PID reuse')
    return {'host': socket.gethostname(), 'pid': pid, 'exists': False}


def worker_exit_proofs(reports):
    proofs = []
    for host in sorted({r['host'].removesuffix('-worker-0') for r in reports}):
        refs = [{'pid': r['pid'], 'start_ticks': r['proc_start_ticks']} for r in reports
                if r['host'].removesuffix('-worker-0') == host]
        code = "from pathlib import Path;import json;refs="+repr(refs)+";out=[]\n"
        code += "for r in refs:\n p=Path('/proc')/str(r['pid']);out.append(dict(r,exists=p.exists(),observed_start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19] if p.exists() else None))\nprint(json.dumps(out))"
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', host, 'python -']
        values = json.loads(subprocess.check_output(command, input=code, text=True, timeout=20))
        require(len(values) == len(refs) and {r['pid'] for r in values} == {r['pid'] for r in refs}, 'Incomplete remote PID proof')
        for value in values:
            require(not value['exists'] or value['observed_start_ticks'] != value['start_ticks'], 'Original remote GPU worker is still alive')
            proofs.append(dict(host=host, **value))
    return proofs


def validate_scene(destination, entry, prediction, metric_names):
    import numpy as np
    token = entry['token']; record_path = destination/'records'/(token+'.json')
    record = read(record_path); payload = destination/'predictions'/(token+'.npz')
    require(record['token'] == token and record['identity'] == prediction['identity'] and digest(payload) == record['sha256'],
            'Prediction token/identity/payload hash changed')
    cache_hash = digest(entry['cache_path']); hashes = [digest(record_path)]
    with np.load(payload, allow_pickle=False) as values:
        for variant in ('q0', 'q_final'):
            trajectory = values[variant]
            require(trajectory.shape == (8, 3) and np.isfinite(trajectory).all(), 'Invalid original trajectory')
            bank = destination/'scores'/variant
            score_file = bank/'records'/(token+'.json'); row = read(score_file)
            score_row(row, token, entry['log'], variant, metric_names)
            proposal = bank/'proposals'/(token+'.npz')
            require(digest(proposal) == row['proposal_sha256'] and cache_hash == row['metric_cache_sha256'], 'Original canonical score input hash changed')
            with np.load(proposal, allow_pickle=False) as saved:
                require(np.array_equal(saved['trajectory'], trajectory), 'Scored proposal differs from original prediction')
            hashes.append(digest(score_file))
    return token, hashes


def execute(output, case_name='g3_70k'):
    case = CASES[case_name]
    COLD, TOOL_SHA, FAILURE_SHA = case['cold'], case['tool_sha'], case['failure_sha']
    KEY, ATTEMPT, updates = case['key'], case['attempt'], case['updates']
    shards = len(case['exit_codes'])
    require(isinstance(FAILURE_SHA, str) and len(FAILURE_SHA) == 64,
            'Exact original failure hash must be reviewed and registered before execution')
    guard = load_guard(case); guard.authorized()
    require(socket.gethostname() == 'training-vla-zt-worker-0', 'Recover on the original controller/scorer host')
    destination = output/'jobs'/KEY; failure = destination/'INFERENCE_FAILED.json'
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(digest(failure) == FAILURE_SHA, 'Failure changed; do not reinterpret another attempt')
        failure_value = read(failure); failure_contract(failure_value, case)
        require(not (destination/'scores/INVALID.json').exists(), 'Scientific scoring failure retained')
        require(not (destination/'COMPLETE.json').exists() and not (output/'completed'/(KEY+'.json')).exists(), 'Already finalized or conflicting completion')
        request = read(output/'requests'/(KEY+'.json')); prediction = read(destination/'identity.json')
        require(request['key'] == KEY and request['population'] == 'navtest' and request['scenes'] == 12146
                and request['logs'] == 136 and request['seed'] == 42 and request['variants'] == ['q0', 'q_final'], 'Request protocol changed')
        require(prediction['training_identity'] == request['training_identity'] and prediction['checkpoint'] == request['tag']
                and prediction['updates'] == request['updates'] == updates and prediction['checkpoint_files_sha256'] == request['checkpoint_files_sha256']
                and prediction['evaluation_source_sha'] == guard.SHA and prediction['sampling_seed'] == 42, 'Prediction contract differs')
        contract = {k: v for k, v in prediction.items() if k != 'identity'}
        require(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() == prediction['identity'], 'Prediction identity hash changed')
        native = read(Path(request['run'])/'checkpoints'/request['tag']/'COMPLETE.json')
        require(native['identity'] == request['training_identity'] and native['completed'] == updates
                and native['file_sha256'] == request['checkpoint_files_sha256'], 'Pinned checkpoint metadata differs')
        registration = read(destination/'REGISTRATION.json'); inputs = Path(registration['inputs'])
        index = read(guard.INDEX); expected = {r['token'] for r in index}
        require(len(index) == len(expected) == 12146 and len({r['log'] for r in index}) == 136, 'Canonical population differs')
        require({(r['token'], r['log']) for r in read(inputs/'index.json')} == {(r['token'], r['log']) for r in index}
                and read(inputs/'identity.json')['identity'] == prediction['input_identity'], 'Current input population/identity differs')
        for folder, suffix in [('records', '.json'), ('predictions', '.npz'), ('scores/q0/records', '.json'), ('scores/q_final/records', '.json')]:
            require({p.stem for p in (destination/folder).glob('*'+suffix)} == expected, 'Missing/extra scene records: '+folder)
        root = destination/'attempts'/ATTEMPT
        require(read(root/'ATTEMPT.json')['routing_tool_sha256'] == TOOL_SHA, 'Attempt source changed')
        reports = [read(root/'shards'/f'shard_{i:03d}.json') for i in range(shards)]
        launches = [read(root/'launches'/f'shard_{i:03d}.json') for i in range(shards)]
        for i, (report, launch) in enumerate(zip(reports, launches)):
            report_contract(report, launch, i, prediction['identity'], case)
            if case.get('closed_by_remote_host') and case['exit_codes'][i] == 255:
                text = (root/f'shard_{i:03d}.log').read_text()
                require('closed by remote host.' in text and 'Traceback (most recent call last)' not in text,
                        'Original transport log differs or includes an unreviewed exception')
        scorer = read(destination/'SCORER_LAUNCH.json')
        local_proofs = [absent_local(l['pid']) for l in launches]+[absent_local(scorer['pid']), absent_local(launches[0]['controller_pid'])]
        remote_proofs = worker_exit_proofs(reports)
        score_log = destination/'canonical_score.log'
        require('RuntimeError: Inference failed after partial scores' in score_log.read_text(), 'Original scorer did not stop at the supported infrastructure boundary')
        recovery = destination/'recoveries'/('completed_transport_'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        recovery.mkdir(parents=True, exist_ok=False)
        evidence = {'request_key': KEY, 'original_classification': failure_value['classification'], 'failure_sha256': FAILURE_SHA,
            'reviewed_classification': 'SSH-255 transport interruption with globally complete original outputs',
            'specific_network_root_cause': ('UNKNOWN: logs show remote-host transport closure; host-internal cause not established'
                if case.get('closed_by_remote_host') else 'UNKNOWN: original SSH exception/traceback not recorded'),
            'original_worker_reports_rewritten': False, 'original_launcher_exit_codes': failure_value['exit_codes'],
            'local_exit_proofs': local_proofs, 'remote_worker_exit_proofs': remote_proofs,
            'canonical_original_log_sha256': digest(score_log), 'producer_sha256': digest(__file__),
            'frozen_dispatcher_sha256': TOOL_SHA, 'CPU_only': True, 'new_GPU_workers': 0, 'started_utc': guard.now()}
        evidence['original_evidence_sha256'] = {str(p): digest(p) for p in [root/'ATTEMPT.json', destination/'identity.json',
            destination/'REGISTRATION.json', destination/'scores/identity.json', *sorted((root/'shards').glob('*.json')),
            *sorted((root/'launches').glob('*.json')), *sorted(root.glob('shard_*.log'))]}
        guard.atomic(recovery/'REGISTRATION.json', evidence)
        archive_state = output/'STATE.json'; (recovery/'ORIGINAL_DISPATCH_STATE.json').write_bytes(archive_state.read_bytes())
        moved = False
        try:
            union = hashlib.sha256()
            with ThreadPoolExecutor(max_workers=2) as pool:
                for count, (token, hashes) in enumerate(pool.map(lambda e: validate_scene(destination, e, prediction, guard.METRICS), index), 1):
                    union.update(json.dumps([token, hashes]).encode())
                    if count % 512 == 0: guard.atomic(recovery/'PROGRESS.json', {'status': 'CPU_VALIDATE', 'scenes_verified': count, 'total': 12146})
            guard.atomic(recovery/'VALIDATION.json', {'passed': True, 'predictions': count, 'q0_scores': count, 'q_final_scores': count,
                'failed_scores': 0, 'membership_identity_payload_proposal_metric_cache_hashes_verified': True,
                'ordered_original_record_hashes_sha256': union.hexdigest(), 'completed_utc': guard.now()})
            require(digest(failure) == FAILURE_SHA, 'Failure changed during validation')
            failure.replace(recovery/'INFERENCE_FAILED.json'); moved = True
            guard.atomic(destination/'COMPLETE.json', {'identity': prediction['identity'], 'scenes': 12146, 'full_population': True,
                'coverage_basis': 'actual exact global membership, all payload hashes and original full canonical score input bindings',
                'recovery_evidence': str(recovery/'VALIDATION.json'), 'attempt_reports': shards, 'final_attempt': ATTEMPT,
                'GPU_hours': None, 'GPU_hours_lower_bound': sum(r.get('inference_seconds', 0) for r in reports)/3600,
                'GPU_cost_status': 'partial worker timing snapshots; complete original GPU cost UNKNOWN', 'completed_utc': guard.now()})
            command = list(scorer['command']); command[command.index('--workers')+1] = '2'
            require(command[3] == str(COLD/'tools/structured_world/stream_navsim_navtest_pdms.py'), 'Original canonical scorer entrypoint differs')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            with (recovery/'canonical_finalize.log').open('ab') as stream:
                child = subprocess.Popen(command, cwd=guard.FROZEN, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                guard.atomic(recovery/'CPU_PROCESS.json', {'pid': child.pid, 'command': command, 'GPU_workers': 0, 'started_utc': guard.now()})
                rc = child.wait()
            guard.atomic(recovery/'CPU_PROCESS_EXIT.json', {'pid': child.pid, 'returncode': rc, 'GPU_hours': 0, 'observed_utc': guard.now()})
            require(rc == 0 and guard.valid_score_completion(destination, request), 'Canonical finalization/strict full-bank validation failed')
            exit_proof = dict(key=KEY, training_identity=request['training_identity'], checkpoint_files_sha256=request['checkpoint_files_sha256'],
                all_gpu_launchers_exited=True, scorer_exited=True, remote_worker_exit_proofs=remote_proofs,
                launchers=[{'pid': l['pid'], 'returncode': rc} for l, rc in zip(launches, failure_value['exit_codes'])],
                original_scorer_exit_observed=True, original_scorer_returncode='UNKNOWN', CPU_finalize_pid=child.pid,
                CPU_finalize_returncode=0, recovery_evidence=str(recovery), observed_utc=guard.now())
            guard.atomic(destination/'OWNED_WORKERS_EXITED.json', exit_proof)
            guard.atomic(output/'completed'/(KEY+'.json'), {'request': request, 'score_complete': str(destination/'scores/COMPLETE.json'),
                'valid': True, 'completed_utc': guard.now(), 'recovery_evidence': str(recovery)})
            guard.atomic(recovery/'COMPLETE.json', {'valid': True, 'score_complete': str(destination/'scores/COMPLETE.json'), 'new_GPU_workers': 0,
                'original_failure_sha256': digest(recovery/'INFERENCE_FAILED.json'), 'completed_utc': guard.now()})
            guard.atomic(output/'STATE.json', {'status': 'RECOVERED_COMPLETE', 'request': request, 'recovery_evidence': str(recovery), 'observed_utc': guard.now()})
            print(json.dumps({'completed': KEY, 'scores': read(destination/'scores/COMPLETE.json'), 'evidence': str(recovery), 'GPU_workers_launched': 0}))
        except BaseException as error:
            if moved and not failure.exists(): failure.write_bytes((recovery/'INFERENCE_FAILED.json').read_bytes())
            guard.atomic(recovery/'FAILED.json', {'valid': False, 'error': repr(error), 'original_failure_preserved': True, 'observed_utc': guard.now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225'))
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--case', choices=sorted(CASES), default='g3_70k')
    args = parser.parse_args(); execute(args.output, args.case)


if __name__ == '__main__': main()
