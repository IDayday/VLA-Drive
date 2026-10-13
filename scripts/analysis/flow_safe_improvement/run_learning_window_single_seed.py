"""Adopt the unchanged finite trainer, with user-requested single-seed original DEV.

This composes the qualified trainer, original ODE sampler and canonical scorer.
No new optimizer/model implementation or previous full-training controller is used.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch, safety_decision


def module(name, arguments, *, distributed=False):
    command = [sys.executable]
    if distributed:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8']
    return command + ['-m', 'scripts.analysis.flow_safe_improvement.'+name] + [str(x) for x in arguments]


def launch(command, log, records):
    host = require_dispatch()  # Includes every queued next phase and retry.
    from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
    authorized(host)
    stream = Path(log).open('a')
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    records.append(dict(host=host, pid=process.pid, command=command, log=str(log), started_unix=time.time()))
    return process


def finish(process, record):
    result = process.wait()
    record.update(returncode=result, finished_unix=time.time())
    if result:
        raise RuntimeError('Phase failed; inspect '+record['log'])


def wait_for_owned_process(module_name, option, value, receipt):
    """Adopt an exact owned command without restarting its work or RNG."""
    captured = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = [x for x in (path/'cmdline').read_bytes().decode().split('\0') if x]
            if (module_name in argv and '-u' not in argv
                    and option in argv and argv[argv.index(option)+1] == str(value)):
                start = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
                captured.append(dict(pid=int(path.name), start_ticks=start, command=argv))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if len(captured) > 1:
        raise RuntimeError('Multiple parents target the same owned output')
    if captured:
        atomic_json(receipt, dict(processes=captured, captured_unix=time.time(), restart=False))
    for row in captured:
        while True:
            try:
                fields = Path('/proc', str(row['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
                if fields[19] != row['start_ticks'] or fields[0] == 'Z':
                    break
            except FileNotFoundError:
                break
            time.sleep(5)
    return bool(captured)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('population', 'cohorts', 'targets', 'reference-scores', 'eta-registration',
                 'retention-registration', 'baseline-dev-scores', 'reference-dev-scores', 'dev-cache', 'output', 'checkpoint-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--evaluation-epochs', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--initial-behavior')
    p.add_argument('--initial-behavior-sha256')
    p.add_argument('--previous-safety-decision')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluation-override', required=True)
    p.add_argument('--adopt-existing', action='store_true')
    p.add_argument('--wait-seconds', type=float, default=7200.)
    a = p.parse_args()
    if a.epochs not in (2, 8, 16) or sorted(set(a.evaluation_epochs)) != a.evaluation_epochs or a.evaluation_epochs[-1] != a.epochs:
        raise ValueError('A finite common learning window and final evaluation are required')
    if any(x not in (1, 2, 4, 8, 16) or x > a.epochs for x in a.evaluation_epochs):
        raise ValueError('Only registered milestone epochs may be evaluated')
    from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
    from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds
    pop = json.loads(Path(a.population).read_text())
    seeds, override = evaluation_seeds(pop, a.evaluation_override)
    if seeds != [2101] or a.training_seed != 42:
        raise ValueError('User cancelled all additional seeds')
    host = require_dispatch(); authorized(host)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume or a.adopt_existing)
    records = json.loads((out/'single_seed_phase_records.json').read_text()) if a.resume and (out/'single_seed_phase_records.json').exists() else []
    state = dict(status='WAITING_FOR_COMPLETE_TARGET_BANK', host=host, pid=os.getpid(), completed_epochs=[])
    atomic_json(out/'single_seed_controller_state.json', state)
    deadline = time.monotonic()+a.wait_seconds
    while not (Path(a.targets)/'COMPLETE.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Target bank incomplete; no partial-scene training is allowed')
        time.sleep(5)
    targets = json.loads((Path(a.targets)/'manifest.json').read_text())
    if targets['qualification_only'] or len(targets['tokens']) != 4096:
        raise ValueError('This runner requires the complete formal 4096-scene bank')
    eta, retention = [json.loads(Path(x).read_text()) for x in (a.eta_registration, a.retention_registration)]
    if targets['objective'] != 'nft' and targets['eta'] != eta['selected_eta']:
        raise ValueError('Target eta differs from the common TRAIN-only calibration')
    identity = dict(arguments={k: v for k, v in vars(a).items() if k not in ('resume', 'wait_seconds', 'adopt_existing')},
                    target_sha256=targets['sha256'], host=host, boundary_source_sha256=file_sha(safety_decision.__code__.co_filename),
                    source_files={name: file_sha(Path(__file__).with_name(name+'.py')) for name in
                                  ('run_learning_window_single_seed', 'train_fixed_bank', 'run_single_seed_evaluation', 'sample_original_export', 'score_samples', 'evaluate_policy')})
    identity['sha256'] = fingerprint(identity)
    if (out/'single_seed_window_identity.json').exists() and json.loads((out/'single_seed_window_identity.json').read_text()) != identity:
        raise ValueError('Resume changes window/source/host identity')
    atomic_json(out/'single_seed_window_identity.json', identity)
    previous = json.loads(Path(a.previous_safety_decision).read_text()) if a.previous_safety_decision else None
    train = out/'train'
    if train.exists() and not a.resume and not a.adopt_existing:
        raise ValueError('Adoption of an existing trainer must be explicit')
    wait_for_owned_process('scripts.analysis.flow_safe_improvement.train_fixed_bank', '--output', train,
                           out/'adopted_running_trainer.json')
    try:
        for epoch in a.evaluation_epochs:
            update = 4096*epoch//16
            export = train/f'export_update{update:06d}'
            if not (export/'COMPLETE').exists():
                args = ['--population', a.population, '--cohorts', a.cohorts, '--targets', a.targets,
                        '--reference-scores', a.reference_scores, '--output', train, '--checkpoint-root', a.checkpoint_root,
                        '--epochs', a.epochs, '--training-seed', a.training_seed, '--retention-coefficient', retention['selected_coefficient'],
                        '--retention-registration', a.retention_registration, '--stop-after-updates', update, '--save-every', 512]
                if a.initial_behavior:
                    args += ['--initial-behavior', a.initial_behavior, '--initial-behavior-sha256', a.initial_behavior_sha256]
                latest = train/'latest_checkpoint.json'
                if latest.exists():
                    args += ['--resume', json.loads(latest.read_text())['path']]
                state.update(status='TRAINING', epoch_target=epoch)
                atomic_json(out/'single_seed_controller_state.json', state)
                proc = launch(module('train_fixed_bank', args, distributed=True), out/f'train_to_epoch{epoch}.log', records)
                atomic_json(out/'single_seed_phase_records.json', records)
                finish(proc, records[-1])
            if not (export/'COMPLETE').exists():
                raise RuntimeError('Incomplete milestone export')
            weight = export/'pytorch_model.pt'
            weight_sha = file_sha(weight)
            summary = out/f'DEV_seed2101_original_epoch{epoch}'/'summary'
            if not (summary/'report.json').exists():
                args = ['--export', export, '--population', a.population, '--evaluation-override', a.evaluation_override,
                        '--baseline-scores', a.baseline_dev_scores, '--reference-scores', a.reference_dev_scores,
                        '--dev-cache', a.dev_cache, '--output', summary.parent]
                state.update(status='SINGLE_SEED_ORIGINAL_ODE_DEV', epoch=epoch, behavior_sha256=weight_sha)
                atomic_json(out/'single_seed_controller_state.json', state)
                proc = launch(module('run_single_seed_evaluation', args), out/f'DEV_seed2101_epoch{epoch}.log', records)
                atomic_json(out/'single_seed_phase_records.json', records)
                finish(proc, records[-1])
            report = json.loads((summary/'report.json').read_text())
            decision = safety_decision(report, previous)
            atomic_json(out/f'single_seed_safety_epoch{epoch}.json', decision)
            previous = decision
            state['completed_epochs'].append(epoch)
            state.update(status=decision['status'], last_export=str(export), last_weight_sha256=weight_sha,
                         last_summary=str(summary/'report.json'), last_safety_decision=str(out/f'single_seed_safety_epoch{epoch}.json'))
            atomic_json(out/'single_seed_phase_records.json', records)
            atomic_json(out/'single_seed_controller_state.json', state)
            print(json.dumps(state), flush=True)
            if decision['status'] == 'PAUSE_VARIANT':
                break
        else:
            state['status'] = 'COMPLETE_FIXED_BUDGET_AND_FRESH_DEV'
            atomic_json(out/'single_seed_controller_state.json', state)
    except BaseException as error:
        state.update(status='FAILED_PRESERVE_ALL_ASSETS_NO_AUTOMATIC_RETRY', error=repr(error))
        atomic_json(out/'single_seed_phase_records.json', records)
        atomic_json(out/'single_seed_controller_state.json', state)
        raise


if __name__ == '__main__':
    main()
