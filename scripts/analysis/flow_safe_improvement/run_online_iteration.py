"""Three fresh contextual policy-improvement rounds, including the ODE NFT control.

All variants start at SFT, then inherit the previous round's complete action head.
Each round uses a new fixed bank and a common two-epoch optimizer window. Adam
moments restart only at this declared outer boundary; resumes within a round use
the complete saved optimizer/RNG/data state.
"""
import argparse
import json
from pathlib import Path
import time

from scripts.analysis.flow_safe_improvement.run_learning_window import module, launch, finish
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('population', 'cohorts', 'reference-scores', 'eta-registration', 'retention-registration',
                 'baseline-dev-scores', 'reference-dev-scores', 'dev-cache', 'train-cache', 'output', 'checkpoint-root',
                 'registration', 'batch-qualification', 'physical-fallback-registration'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--variant', choices=['ODE_REWARD_FM', 'ODE_SAFE_FM', 'EXPLORE_SAFE_FM', 'ODE_NFT'], required=True)
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    host = require_dispatch()
    registration = json.loads(Path(a.registration).read_text())
    if registration['rounds'] != 3 or registration['scenes_per_round'] != 4096 or registration['logical_candidates'] != 32:
        raise ValueError('Only the three registered finite rounds are authorized here')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=a.resume)
    identity = dict(arguments={k: v for k, v in vars(a).items() if k != 'resume'}, host=host,
                    registration_sha256=file_sha(a.registration), source_files={name: file_sha(Path(__file__).with_name(name+'.py'))
                    for name in ('run_online_iteration', 'build_candidate_banks', 'prepare_target_bank', 'run_learning_window', 'serve_scoring')})
    identity['sha256'] = fingerprint(identity)
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text()) != identity:
        raise ValueError('Online recovery cannot change source, behavior refresh or quota')
    atomic_json(out/'identity.json', identity)
    records = json.loads((out/'phase_records.json').read_text()) if a.resume and (out/'phase_records.json').exists() else []
    state = dict(status='REGISTERED_THREE_ROUNDS', rounds_complete=[], initial_SFT_sha256=SFT_SHA256)
    behavior, behavior_sha, previous_safety = None, SFT_SHA256, None
    objective = {'ODE_REWARD_FM': 'reward', 'ODE_SAFE_FM': 'safe', 'EXPLORE_SAFE_FM': 'safe', 'ODE_NFT': 'nft'}[a.variant]
    method = 'BEHAVIOR' if a.variant == 'EXPLORE_SAFE_FM' else 'ODE'
    try:
        for iteration in (1, 2, 3):
            round_root = out/f'round{iteration}'; round_root.mkdir(exist_ok=True)
            bank, targets, queue, learning = [round_root/name for name in ('bank', 'targets', 'scoring_queue', 'learning')]
            plan = dict(iteration=iteration, behavior_path=behavior, behavior_sha256=behavior_sha,
                        permanent_reference=SFT_SHA256, seed=a.training_seed, method=method, objective=objective,
                        registration_sha256=identity['registration_sha256'])
            if (round_root/'boundary.json').exists() and json.loads((round_root/'boundary.json').read_text()) != plan:
                raise ValueError('Cannot refresh behavior underneath saved candidates or target weights')
            atomic_json(round_root/'boundary.json', plan)
            if not (targets/'COMPLETE.json').exists():
                ranks = [json.loads(p.read_text()) for p in bank.glob('rank*.json')] if bank.exists() else []
                bank_complete = len(ranks) == 8 and all(x['status'] == 'ASSIGNED_POPULATION_COMPLETE' for x in ranks)
                if not bank_complete:
                    stop_file = round_root/'scoring_complete.json'
                    if stop_file.exists():
                        raise ValueError('Completed scoring marker with an incomplete bank requires inspection')
                    args = ['--queue', queue, '--cache-root', a.train_cache, '--workers', 16,
                            '--max-new-scores', 4096*28, '--max-hours', 8, '--stop-file', stop_file]
                    if (queue/'manifest.json').exists():
                        args += ['--resume']
                    server = launch(module('serve_scoring', args), round_root/'scoring.log', records)
                    server_record = records[-1]
                    deadline = time.monotonic()+180
                    while not (queue/'manifest.json').exists():
                        if server.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Canonical scoring server failed before identity publication')
                        time.sleep(1)
                    args = ['--population', a.population, '--cohorts', a.cohorts, '--stage', 'ONLINE', '--method', method,
                            '--online-registration', a.registration, '--iteration', iteration, '--seed', a.training_seed,
                            '--reference-scores', a.reference_scores, '--queue', queue, '--batch-qualification', a.batch_qualification,
                            '--output', bank]
                    if method == 'BEHAVIOR':
                        args += ['--physical-fallback-registration', a.physical_fallback_registration]
                    if behavior:
                        args += ['--behavior', behavior, '--behavior-sha256', behavior_sha]
                    if (bank/'manifest.json').exists():
                        args += ['--resume']
                    proc = launch(module('build_candidate_banks', args, distributed=True), round_root/'candidates.log', records)
                    state.update(status='FRESH_CURRENT_BEHAVIOR_CANDIDATES', iteration=iteration, behavior_sha256=behavior_sha)
                    atomic_json(out/'controller_state.json', state); atomic_json(out/'phase_records.json', records)
                    finish(proc, records[-1])
                    atomic_json(stop_file, dict(status='BANK_GENERATION_COMPLETE', bank=str(bank)))
                    finish(server, server_record)
                args = ['--bank', bank, '--population', a.population, '--objective', objective, '--output', targets]
                if objective != 'nft':
                    args += ['--eta', registration['eta'], '--eta-registration', a.eta_registration]
                if (targets/'manifest.json').exists():
                    args += ['--resume']
                proc = launch(module('prepare_target_bank', args), round_root/'targets.log', records)
                finish(proc, records[-1])
            target_manifest = json.loads((targets/'manifest.json').read_text())
            if target_manifest['behavior_sha256'] != behavior_sha:
                raise ValueError('Stale teacher/candidate/weight identity at refresh boundary')
            learning_state_path = learning/'controller_state.json'
            learning_state = json.loads(learning_state_path.read_text()) if learning_state_path.exists() else {}
            if learning_state.get('status') not in ('COMPLETE_FIXED_BUDGET_AND_FRESH_DEV', 'PAUSE_VARIANT'):
                args = ['--population', a.population, '--cohorts', a.cohorts, '--targets', targets,
                        '--reference-scores', a.reference_scores, '--eta-registration', a.eta_registration,
                        '--retention-registration', a.retention_registration, '--baseline-dev-scores', a.baseline_dev_scores,
                        '--reference-dev-scores', a.reference_dev_scores, '--dev-cache', a.dev_cache, '--output', learning,
                        '--checkpoint-root', Path(a.checkpoint_root)/f'round{iteration}', '--epochs', 2, '--evaluation-epochs', 2,
                        '--training-seed', a.training_seed]
                if behavior:
                    args += ['--initial-behavior', behavior, '--initial-behavior-sha256', behavior_sha]
                if previous_safety:
                    args += ['--previous-safety-decision', previous_safety]
                if learning.exists():
                    args += ['--resume']
                proc = launch(module('run_learning_window', args), round_root/'learning.log', records)
                state.update(status='FULL_ACTION_HEAD_LEARNING_AND_FRESH_DEV', iteration=iteration)
                atomic_json(out/'controller_state.json', state); atomic_json(out/'phase_records.json', records)
                finish(proc, records[-1])
                learning_state = json.loads(learning_state_path.read_text())
            state['rounds_complete'].append(iteration)
            state.update(status=learning_state['status'], last_round=iteration, last_learning_state=str(learning_state_path))
            atomic_json(out/'controller_state.json', state); atomic_json(out/'phase_records.json', records)
            if state['status'] == 'PAUSE_VARIANT':
                break
            behavior = str(Path(learning_state['last_export'])/'pytorch_model.pt')
            behavior_sha = learning_state['last_weight_sha256']
            if file_sha(behavior) != behavior_sha:
                raise ValueError('Saved next-round current policy changed')
            previous_safety = learning_state['last_safety_decision']
        else:
            state['status'] = 'COMPLETE_THREE_FRESH_POLICY_ITERATIONS'
            atomic_json(out/'controller_state.json', state)
    except BaseException as error:
        state.update(status='FAILED_PRESERVE_BANK_BEHAVIOR_AND_OPTIMIZER_NO_AUTOMATIC_RETRY', error=repr(error))
        atomic_json(out/'controller_state.json', state); atomic_json(out/'phase_records.json', records)
        raise


if __name__ == '__main__':
    main()
