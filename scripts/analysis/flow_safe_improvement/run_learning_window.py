"""Execute one finite full-head window and fresh DEV evaluations at saved epochs.

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
    p.add_argument('--wait-seconds', type=float, default=7200.)
    a = p.parse_args()
    if a.epochs not in (2, 8, 16) or sorted(set(a.evaluation_epochs)) != a.evaluation_epochs or a.evaluation_epochs[-1] != a.epochs:
        raise ValueError('A finite common learning window and final evaluation are required')
    if any(x not in (1, 2, 4, 8, 16) or x > a.epochs for x in a.evaluation_epochs):
        raise ValueError('Only registered milestone epochs may be evaluated')
    host = require_dispatch()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume)
    records = json.loads((out/'phase_records.json').read_text()) if a.resume and (out/'phase_records.json').exists() else []
    state = dict(status='WAITING_FOR_COMPLETE_TARGET_BANK', host=host, pid=os.getpid(), completed_epochs=[])
    atomic_json(out/'controller_state.json', state)
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
    identity = dict(arguments={k: v for k, v in vars(a).items() if k not in ('resume', 'wait_seconds')},
                    target_sha256=targets['sha256'], host=host, boundary_source_sha256=file_sha(safety_decision.__code__.co_filename),
                    source_files={name: file_sha(Path(__file__).with_name(name+'.py')) for name in
                                  ('run_learning_window', 'train_fixed_bank', 'sample_frozen', 'score_samples', 'evaluate_policy')})
    identity['sha256'] = fingerprint(identity)
    if (out/'window_identity.json').exists() and json.loads((out/'window_identity.json').read_text()) != identity:
        raise ValueError('Resume changes window/source/host identity')
    atomic_json(out/'window_identity.json', identity)
    previous = json.loads(Path(a.previous_safety_decision).read_text()) if a.previous_safety_decision else None
    train = out/'train'
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
                atomic_json(out/'controller_state.json', state)
                proc = launch(module('train_fixed_bank', args, distributed=True), out/f'train_to_epoch{epoch}.log', records)
                atomic_json(out/'phase_records.json', records)
                finish(proc, records[-1])
            if not (export/'COMPLETE').exists():
                raise RuntimeError('Incomplete milestone export')
            weight = export/'pytorch_model.pt'
            weight_sha = file_sha(weight)
            samples, scores, summary = [out/f'DEV_epoch{epoch}_{part}' for part in ('samples', 'v1', 'summary')]
            if not (summary/'report.json').exists():
                args = ['--population', a.population, '--cohort', 'dev_tokens', '--role', 'evaluation', '--mode', 'ode',
                        '--behavior', weight, '--behavior-sha256', weight_sha, '--output', samples]
                if (samples/'manifest.json').exists():
                    args += ['--resume']
                sampler = launch(module('sample_frozen', args, distributed=True), out/f'DEV_epoch{epoch}_sample.log', records)
                sample_record = records[-1]
                state.update(status='FRESH_ORIGINAL_ODE_DEV', epoch=epoch, behavior_sha256=weight_sha)
                atomic_json(out/'controller_state.json', state)
                deadline = time.monotonic()+a.wait_seconds
                while not (samples/'manifest.json').exists():
                    if sampler.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Sampler failed before publishing its identity')
                    time.sleep(2)
                args = ['--samples', samples, '--population', a.population, '--cache-root', a.dev_cache,
                        '--workers', 16, '--output', scores, '--wait-seconds', a.wait_seconds]
                if (scores/'manifest.json').exists():
                    args += ['--resume']
                scorer = launch(module('score_samples', args), out/f'DEV_epoch{epoch}_score.log', records)
                score_record = records[-1]
                atomic_json(out/'phase_records.json', records)
                finish(sampler, sample_record)
                finish(scorer, score_record)
                args = ['--scores', scores, '--baseline-scores', a.baseline_dev_scores, '--reference-scores', a.reference_dev_scores,
                        '--population', a.population, '--output', summary]
                proc = launch(module('evaluate_policy', args), out/f'DEV_epoch{epoch}_summary.log', records)
                finish(proc, records[-1])
            report = json.loads((summary/'report.json').read_text())
            decision = safety_decision(report, previous)
            atomic_json(out/f'safety_epoch{epoch}.json', decision)
            previous = decision
            state['completed_epochs'].append(epoch)
            state.update(status=decision['status'], last_export=str(export), last_weight_sha256=weight_sha,
                         last_summary=str(summary/'report.json'), last_safety_decision=str(out/f'safety_epoch{epoch}.json'))
            atomic_json(out/'phase_records.json', records)
            atomic_json(out/'controller_state.json', state)
            print(json.dumps(state), flush=True)
            if decision['status'] == 'PAUSE_VARIANT':
                break
        else:
            state['status'] = 'COMPLETE_FIXED_BUDGET_AND_FRESH_DEV'
            atomic_json(out/'controller_state.json', state)
    except BaseException as error:
        state.update(status='FAILED_PRESERVE_ALL_ASSETS_NO_AUTOMATIC_RETRY', error=repr(error))
        atomic_json(out/'phase_records.json', records)
        atomic_json(out/'controller_state.json', state)
        raise


if __name__ == '__main__':
    main()
