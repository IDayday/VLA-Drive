"""Finite CPU follow-up for the already registered B-20k milestone only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.analysis.fm_transfer.milestone_controller import live_workers
from scripts.analysis.fm_transfer.final_lock import authenticated, validate_rl_summary
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha

ROOT = Path('/mnt/project/fm-transfer-20261016')
OUT = ROOT/'resources/B_RL_20k_analysis_handoff_20261017'
RUN = ROOT/'runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1'
SOURCE = ROOT/'source/8dbd2bb2c82026d3a8a0b4b96c9b0ba211248f81'
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def validate_endpoint_summary(summary, identity):
    validate_rl_summary(summary, identity)
    if summary['cumulative']['actual_global_optimizer_updates'] != 20000:
        raise ValueError('Wrong actual optimizer update count')
    ledger = summary['cost']['invocation_ledger']
    if ledger['incomplete_invocations'] or ledger['unclosed_invocation_worker_wall_seconds']:
        raise ValueError('Training invocation ledger is not closed')


def validate_checkpoint(checkpoint, update):
    seal = read(checkpoint/'COMPLETE')
    if (seal['phase'] != 'COMPLETE' or seal['updates'] != update or
            seal['world'] != 8 or seal['next_inner'] != 0 or
            file_sha(checkpoint/'files.json') != seal['files_sha256']):
        raise ValueError('Not a sealed registered final checkpoint')


def pending(update):
    """Read-only prerequisite check; no observer or worker is restarted."""
    stem = f'B_SOURCE_ACTION_DIT_RL_update{update:06d}'
    checkpoint = RUN/f'checkpoints/update{update:06d}_inner0'
    needed = [checkpoint/'COMPLETE',
              ROOT/'analysis'/f'{stem}_progress_snapshot.json',
              ROOT/'analysis'/f'{stem}_invocation_cost_snapshot.json',
              ROOT/'analysis'/f'{stem}_saved_batch.json',
              ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_complete_training.json']
    needed += [RUN/f'result_rank{rank}.json' for rank in range(8)]
    for cohort in ('DEV', 'TRAIN_DIAGNOSTIC'):
        needed += [ROOT/'analysis'/f'{stem}_{cohort}.json',
                   ROOT/'analysis'/f'{stem}_{cohort}_paired_own_SFT.json']
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        return missing
    validate_checkpoint(checkpoint, update)
    summary = read(ROOT/'analysis/B_SOURCE_ACTION_DIT_RL_complete_training.json')
    validate_endpoint_summary(summary, authenticated(RUN/'identity.json'))
    for rank in range(8):
        row = read(RUN/f'result_rank{rank}.json')
        expected = dict(rank=rank, status='COMPLETE', updates=20000,
                        collections=5000, raw_scene_exposures=40000,
                        new_candidates=320000, next_inner=0, phase='COMPLETE')
        if any(row[k] != v for k, v in expected.items()) or row['qualification']:
            raise ValueError('Endpoint rank result does not match the registered budget')
    for cohort in ('DEV', 'TRAIN_DIAGNOSTIC'):
        folder = ROOT/'evaluations'/f'{stem}_{cohort}'
        for suffix in ('', '_paired_own_SFT'):
            if read(ROOT/'analysis'/f'{stem}_{cohort}{suffix}.json')['status'] != 'COMPLETE_POPULATION':
                raise ValueError('Incomplete controller analysis')
        invocations = [read(p) for p in (folder/'invocations').glob('*/rank*.json')]
        if (len(invocations) != 8 or
                any(x['status'] != 'ASSIGNED_COMPLETE' or 'end_unix' not in x for x in invocations)):
            missing.append(f'{cohort}: closed invocation ledgers')
        records = [read(p) for p in (folder/'dispatches').glob('*_rank*.json')]
        if len(records) != 8:
            raise ValueError('Unexpected evaluation attempts; inspect without retry')
        for host in {x['host'] for x in records}:
            if any(live_workers(host, [x for x in records if x['host'] == host])):
                missing.append(f'{cohort}: evaluation workers still exiting')
    return missing


def jobs():
    stem = 'B_SOURCE_ACTION_DIT_RL_update020000'
    analysis = ROOT/'analysis'
    result = [
        ('population', SOURCE, 'scripts.analysis.fm_transfer.analyze_rl_population',
         ['--run', str(RUN), '--manifest', str(ROOT/'hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json'),
          '--updates', '20000'], analysis/f'{stem}_population.json')]
    for cohort in ('DEV', 'TRAIN_DIAGNOSTIC'):
        evaluation = ROOT/'evaluations'/f'{stem}_{cohort}'
        result += [
            (f'{cohort}_safety', SOURCE, 'scripts.analysis.fm_transfer.analyze_safety_transitions',
             ['--evaluation', str(evaluation), '--reference',
              str(ROOT/'evaluations'/f'B_SOURCE_ACTION_DIT_SFT_update100000_{cohort}')],
             analysis/f'{stem}_{cohort}_safety_transitions.json'),
            (f'{cohort}_vs15k', SOURCE, 'scripts.analysis.fm_transfer.analyze_evaluation',
             ['--evaluation', str(evaluation), '--reference',
              str(ROOT/'evaluations'/f'B_SOURCE_ACTION_DIT_RL_update015000_{cohort}')],
             analysis/f'{stem}_{cohort}_vs15k.json')]
    result += [
        ('grid_gap', SOURCE, 'scripts.analysis.fm_transfer.analyze_grid_gap',
         ['--evaluation', str(ROOT/'evaluations'/f'{stem}_DEV')],
         analysis/'B_RL20k_DEV_native_minus_shift5.json'),
        ('plot', SOURCE, 'scripts.analysis.fm_transfer.plot_rl_progress',
         ['--variant', 'B_SOURCE_ACTION_DIT'], analysis/'B_RL_DEV_through_20k.png')]
    return result


def stage(name, source, command, output):
    require_dispatch()
    if output.exists():
        raise FileExistsError('Do not overwrite or silently reuse another attempt: '+str(output))
    folder = OUT/name
    folder.mkdir(exist_ok=False)
    record = dict(status='RUNNING', command=command, source=str(source),
                  output=str(output), start_unix=time.time(), policy_sha256=file_sha('/mnt/project/server_dispatch_policy.json'))
    atomic_json(folder/'execution.json', record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr)
        record['child_pid'] = child.pid
        record['child_start_ticks'] = int(Path(f'/proc/{child.pid}/stat').read_text().split(') ')[1].split()[19])
        atomic_json(folder/'execution.json', record)
        record['exit_code'] = child.wait()
    record.update(end_unix=time.time(), status='COMPLETE' if record['exit_code'] == 0 and output.exists() else 'FAILED')
    if output.exists():
        record['output_sha256'] = file_sha(output)
    atomic_json(folder/'execution.json', record)
    if record['status'] != 'COMPLETE':
        raise RuntimeError('Preserved failed stage; no automatic retry: '+name)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require_dispatch()
    for _, source, module, _, _ in jobs():
        seal = read(source/'SOURCE_COMPLETE.json')
        if seal.get('commit') != source.name or not seal.get('git_archive_extraction_complete'):
            raise ValueError('Unsealed analysis source')
        if not (source/Path(*module.split('.')).with_suffix('.py')).is_file():
            raise FileNotFoundError(module)
    if args.once:
        print(json.dumps(dict(status='PREREQUISITE_CHECK_ONLY', pending=pending(20000), analysis_jobs=len(jobs())+1)))
        return
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'owner.json').exists():
            raise RuntimeError('Existing invocation; inspect rather than restart')
        state = dict(status='WAITING_FOR_REGISTERED_20K_EVIDENCE', pid=os.getpid(),
                     start_ticks=int(Path('/proc/self/stat').read_text().split(') ')[1].split()[19]),
                     start_unix=time.time(), deadline_unix=time.time()+21600,
                     script_sha256=file_sha(__file__), new_candidates=0, optimizer_updates=0)
        atomic_json(OUT/'owner.json', state)
        try:
            while time.time() < state['deadline_unix']:
                require_dispatch()
                missing = pending(20000)
                atomic_json(OUT/'status.json', dict(state, missing=missing, observed_unix=time.time()))
                if not missing:
                    break
                time.sleep(20)
            else:
                state['status'] = 'WAIT_DEADLINE_NOT_PROOF_OF_TRAINING_STOPPED'
                return
            state['status'] = 'ANALYZING_REGISTERED_20K'
            atomic_json(OUT/'status.json', state)
            for name, source, module, arguments, output in jobs():
                stage(name, source, [PYTHON, '-m', module, *arguments, '--output', str(output)], output)
            stage('completion_audit', SOURCE, [PYTHON, str(ROOT/'resources/verify_B_RL20k_completion.py')],
                  ROOT/'qualification/B_RL20k_analysis/completion.json')
            state['status'] = 'COMPLETE_20K_ANALYSES_AWAITING_REVIEW_NOT_FINAL_GOAL'
            atomic_json(OUT/'COMPLETE.json', dict(state, end_unix=time.time()))
        except BaseException:
            import traceback
            state.update(status='FAILED_NO_AUTOMATIC_RETRY', error=traceback.format_exc())
            raise
        finally:
            atomic_json(OUT/'status.json', dict(state, end_unix=time.time()))


if __name__ == '__main__':
    main()
