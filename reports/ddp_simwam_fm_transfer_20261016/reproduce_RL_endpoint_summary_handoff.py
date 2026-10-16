"""Finite CPU-only handoff to the existing completed-RL budget analyzer.

No training, evaluation, policy edits, retries or checkpoint selection. The final
four-model lock remains a separate, stricter gate before any Navtest dispatch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analysis.fm_transfer.final_lock import authenticated, validate_rl_summary
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT = Path('/mnt/project/fm-transfer-20261016')
OUT = ROOT / 'resources/RL_endpoint_summaries_20261016_v1'
SOURCE = ROOT / 'source/d58485b08078a6319b2835793d8c79ffbbd3e9d5'
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
VARIANTS = ('A_DDP_TIME', 'B_SOURCE_ACTION_DIT')


def ready(run):
    paths = [run / f'result_rank{rank}.json' for rank in range(8)]
    if not all(p.exists() for p in paths):
        return False
    rows = [json.loads(p.read_text()) for p in paths]
    if any(r['status'] != 'COMPLETE' for r in rows):
        return False
    for rank, row in enumerate(rows):
        expected = dict(rank=rank, updates=20000, collections=5000,
                        raw_scene_exposures=40000, new_candidates=320000)
        if any(row[key] != value for key, value in expected.items()) or row['qualification']:
            raise ValueError('Completed rank does not match the registered formal budget')
    checkpoint = run / 'checkpoints/update020000_inner0'
    seal = json.loads((checkpoint / 'COMPLETE').read_text())
    if (seal['updates'] != 20000 or seal['phase'] != 'COMPLETE' or
            seal['world'] != 8 or seal['next_inner'] != 0 or
            seal['files_sha256'] != file_sha(checkpoint / 'files.json')):
        raise ValueError('Wrong endpoint checkpoint publication')
    progress = json.loads((run / 'progress.json').read_text())
    if (progress['updates'] != 20000 or progress['phase'] != 'COMPLETE' or
            progress['next_inner'] != 0 or progress['checkpoint'] != str(checkpoint)):
        raise ValueError('Progress does not describe the completed endpoint')
    # The trainer writes result_rank immediately before closing its invocation
    # ledger. Wait for all eight cost records as well, without editing training.
    for rank in range(8):
        path = run/'invocations'/progress['invocation']/f'rank{rank}.json'
        if not path.exists():
            return False
        row = json.loads(path.read_text())
        if row['status'] == 'RUNNING':
            return False
        if (row['status'] != 'COMPLETE' or row['end_update'] != 20000 or
                row['rank'] != rank or row['invocation'] != progress['invocation'] or
                row['last_legal_checkpoint'] != str(checkpoint) or 'end_unix' not in row):
            raise ValueError('Endpoint invocation ledger is not consistently closed')
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'RL_endpoint_summary.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(schema='fm_transfer_RL_endpoint_summary_handoff_v1',
                     pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
                     started_unix=time.time(), deadline_unix=time.time()+24*3600,
                     source=str(SOURCE), script_sha256=file_sha(__file__),
                     summary_source_sha256=file_sha(SOURCE/'scripts/analysis/simwam_recipe/summarize_training.py'),
                     new_candidates=0, optimizer_updates=0, completed={})
        if not args.once:
            if (OUT/'owner.json').exists():
                raise RuntimeError('Existing endpoint-summary attempt; inspect before any restart')
            atomic_json(OUT/'owner.json', state)
        try:
            while time.time() < state['deadline_unix']:
                require_dispatch()
                for variant in VARIANTS:
                    if variant in state['completed']:
                        continue
                    run = ROOT/'runs'/f'{variant}_R_SOURCE_seed42_v1'
                    output = ROOT/'analysis'/f'{variant}_RL_complete_training.json'
                    if not ready(run):
                        continue
                    identity = authenticated(run/'identity.json')
                    if output.exists():
                        validate_rl_summary(json.loads(output.read_text()), identity)
                        state['completed'][variant] = dict(output=str(output), sha256=file_sha(output), reused=True)
                        continue
                    attempt = OUT/variant
                    attempt.mkdir(exist_ok=False)
                    command = [PYTHON, '-m', 'scripts.analysis.simwam_recipe.summarize_training',
                               '--run', str(run), '--output', str(output)]
                    dispatch = dict(status='LAUNCHING', command=command, started_unix=time.time(),
                                    run_identity=identity['sha256'], policy_sha256=file_sha('/mnt/project/server_dispatch_policy.json'))
                    atomic_json(attempt/'dispatch.json', dispatch)
                    require_dispatch()
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(SOURCE), OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    with (attempt/'analysis.log').open('xb') as log:
                        child = subprocess.Popen(command, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    dispatch.update(status='RUNNING', child_pid=child.pid,
                                    child_start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]))
                    atomic_json(attempt/'dispatch.json', dispatch)
                    dispatch['exit_code'] = child.wait()
                    dispatch['ended_unix'] = time.time()
                    if dispatch['exit_code'] != 0:
                        dispatch['status'] = 'FAILED_NO_AUTOMATIC_RETRY'
                        atomic_json(attempt/'dispatch.json', dispatch)
                        raise RuntimeError('Budget analysis failed; preserve evidence and inspect')
                    validate_rl_summary(json.loads(output.read_text()), identity)
                    dispatch.update(status='COMPLETE', output=str(output), output_sha256=file_sha(output))
                    atomic_json(attempt/'COMPLETE.json', dispatch)
                    state['completed'][variant] = dict(output=str(output), sha256=file_sha(output), reused=False)
                state.update(status='COMPLETE' if len(state['completed']) == 2 else 'WAITING_FOR_REGISTERED_RL_ENDPOINTS',
                             observed_unix=time.time())
                atomic_json(OUT/('once.json' if args.once else 'status.json'), state)
                if args.once or state['status'] == 'COMPLETE':
                    return
                time.sleep(30)
            state['status'] = 'WAIT_DEADLINE_NOT_PROOF_OF_TRAINING_STOPPED'
        except BaseException:
            import traceback
            state.update(status='FAILED_NO_AUTOMATIC_RETRY', error=traceback.format_exc())
            raise
        finally:
            atomic_json(OUT/('once.json' if args.once else 'status.json'), state)


if __name__ == '__main__':
    main()
