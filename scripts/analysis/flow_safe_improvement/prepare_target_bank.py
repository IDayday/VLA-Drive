"""Solve positive scene-wise targets from a complete sealed candidate bank."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.analysis.flow_safe_improvement.build_candidate_banks import publish_scene
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.candidate_bank import Candidate, deduplicate
from starVLA.rl.safe_flow_improvement.constrained_weights import solve_weights
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, SAFETY_METRICS, file_sha, fingerprint


def unpack(row):
    result = []
    for record in row['candidates']:
        feedback = record['feedback']
        candidate = Candidate(token=row['token'], log_name=row['log_name'], source=record['source'],
                              behavior_sha256=record['behavior_sha256'], normalized_action=np.asarray(record['normalized_action']),
                              requested_trajectory=np.asarray(record['requested_trajectory']), simulated_states=np.asarray(feedback['simulated_states']),
                              score=feedback['metrics']['score'], metrics={k: v for k, v in feedback['metrics'].items() if k != 'score'},
                              protocol_sha256=feedback['protocol_sha256'], noise_keys=record['noise_keys'],
                              source_parameters=record['source_parameters'], is_reference=record['is_reference'])
        candidate.validate(); result.append(candidate)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--objective', choices=['reward', 'safe', 'nft'], required=True)
    p.add_argument('--eta', type=float)
    p.add_argument('--eta-registration', help='Locked training-only calibration receipt; required for formal FM banks')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wait-seconds', type=float, default=0., help='Bounded wait for the existing bank producer to seal missing scenes')
    p.add_argument('--qualification-first-scenes', type=int, help='Bounded real-data optimizer qualification; never a method result')
    a = p.parse_args()
    bank, out = Path(a.bank), Path(a.output)
    manifest, pop = json.loads((bank/'manifest.json').read_text()), json.loads(Path(a.population).read_text())
    if manifest['population_identity'] != pop['identity_sha256']:
        raise ValueError('Population identity differs')
    PopulationContract(*[frozenset(pop[k]) for k in ('train_tokens', 'dev_tokens', 'navtest_tokens')], pop['token_logs']).require_training(manifest['tokens'], 'target fitting')
    if a.objective != 'nft' and (a.eta is None or (a.eta_registration is None and a.qualification_first_scenes is None)):
        raise ValueError('A locked common training-only eta is required before formal fitting')
    if a.eta_registration:
        eta_registration = json.loads(Path(a.eta_registration).read_text())
        if eta_registration['selected_eta'] != a.eta or eta_registration['population_identity'] != pop['identity_sha256']:
            raise ValueError('Eta differs from common calibration')
    tokens = manifest['tokens'][:a.qualification_first_scenes] if a.qualification_first_scenes is not None else manifest['tokens']
    if a.qualification_first_scenes is not None and (a.qualification_first_scenes != 16 or len(tokens) != 16):
        raise ValueError('Qualification uses exactly the first sixteen registered scenes')
    identity = {'schema': 'flow_safe_target_bank_v1', 'candidate_bank_sha256': manifest['sha256'],
                'candidate_bank_path': str(bank.resolve()), 'population_identity': pop['identity_sha256'],
                'tokens': tokens, 'objective': a.objective, 'eta': a.eta,
                'qualification_only': a.qualification_first_scenes is not None,
                'eta_registration_sha256': file_sha(a.eta_registration) if a.eta_registration else None,
                'alpha_max': .5, 'delta_useful': .005, 'source_sha256': file_sha(__file__),
                'permanent_reference_sha256': manifest['reference_SFT'], 'behavior_sha256': manifest['behavior_sha256'],
                'protocol_sha256': manifest['protocol_sha256']}
    from starVLA.rl.safe_flow_improvement import constrained_weights, candidate_bank
    identity['solver_source_sha256'] = file_sha(constrained_weights.__file__)
    identity['candidate_contract_source_sha256'] = file_sha(candidate_bank.__file__)
    identity['sha256'] = fingerprint(identity)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        if not a.resume or json.loads((out/'manifest.json').read_text()) != identity:
            raise ValueError('Target resume changes fixed candidates/objective/solver')
    else:
        atomic_json(out/'manifest.json', identity)
    payloads, records, nft_rewards = {}, [], []
    for token in tokens:
        path = bank/'scenes'/token[:2]/(token+'.pt')
        deadline = time.monotonic() + a.wait_seconds
        while not path.with_suffix('.json').exists() and time.monotonic() < deadline:
            time.sleep(1.)
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != manifest['sha256'] or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Incomplete or changed candidate bank')
        row = torch.load(path, map_location='cpu', weights_only=True)
        if row['invalid_attempts'] and manifest.get('physical_failure_policy') != 'uniform_fixed_reference_fallback':
            # Preserve failures; do not silently redefine u or convert a
            # physical rejection into a fabricated zero-score trajectory.
            raise ValueError('Formal 32-candidate extraction requires a complete scored bank; physical proposal failures need an explicitly registered repaired bank, not silent exclusion')
        if manifest.get('physical_failure_policy') == 'uniform_fixed_reference_fallback':
            if len(row['candidates']) != manifest['logical_budget_per_scene'] or not row['budget'].get('invalid_attempts_are_the_same_slots_as_fallbacks'):
                raise ValueError('Fallback changed the registered candidate-slot budget')
            seen = set()
            for attempt_index, attempt in enumerate(row['invalid_attempts']):
                index = attempt['fallback_candidate_index']
                candidate = row['candidates'][index]
                if index in seen or candidate['source'] != 'reference_fallback' or candidate['source_parameters']['rejected_attempt_index'] != attempt_index:
                    raise ValueError('Rejected physical attempt is missing its explicit fallback identity')
                seen.add(index)
            if len(seen) != sum(c['source'] == 'reference_fallback' for c in row['candidates']):
                raise ValueError('Unaccounted reference fallback')
        payloads[token] = seal['payload_sha256']
        records.append((token, row))
        if a.objective == 'nft':
            current = [c for c in unpack(row) if c.source == 'ode' and not c.is_reference]
            for candidate in current: candidate.require_nft_behavior(manifest['behavior_sha256'])
            if len(current) != manifest['new_draws']:
                raise ValueError('NFT must use all and only actual current ODE draws')
            nft_rewards.extend(c.score for c in current)
    global_nft_std = float(np.std(nft_rewards, dtype=np.float64)) if nft_rewards else None
    diagnostics = []
    seals = {}
    for token, row in records:
        candidates = unpack(row)
        if a.objective == 'nft':
            selected = [c for c in candidates if c.source == 'ode' and not c.is_reference]
            reward = np.array([c.score for c in selected])
            target = np.full(len(selected), 1/len(selected))
            advantage = ((reward-reward.mean())/(global_nft_std+1e-4)).clip(-5., 5.)
            detail = dict(global_reward_std=global_nft_std, advantage_clip=5., epsilon=1e-4,
                          baseline='per_context_mean', normalization='official_NFT_global_reward_std', beta=1.)
        else:
            selected, qref, empirical, groups = deduplicate(candidates)
            reward = np.array([c.score for c in selected])
            costs = np.array([[1-c.metrics[k] for k in SAFETY_METRICS] for c in selected])
            solved = solve_weights(reward, costs, qref, empirical, eta=a.eta, safe=a.objective == 'safe')
            target, advantage, detail = solved.target, None, solved.diagnostics
            detail['dedup_groups'] = groups
        actions = torch.from_numpy(np.stack([c.normalized_action for c in selected]))
        reference_actions = torch.from_numpy(np.stack([c.normalized_action for c in candidates if c.is_reference]))
        value = {'token': token, 'log_name': row['log_name'], 'target_bank_identity': identity['sha256'],
                 'candidate_payload_sha256': payloads[token], 'actions': actions,
                 'target_probabilities': torch.from_numpy(target), 'advantages': torch.from_numpy(advantage) if advantage is not None else None,
                 'reference_actions': reference_actions, 'diagnostics': detail}
        path = out/'scenes'/token[:2]/(token+'.pt')
        if path.with_suffix('.json').exists():
            seal = json.loads(path.with_suffix('.json').read_text())
            if seal['identity_sha256'] != identity['sha256'] or seal['payload_sha256'] != file_sha(path):
                raise ValueError('Saved target changed')
        else:
            publish_scene(path, value, identity['sha256'])
        seals[token] = file_sha(path)
        diagnostics.append({'token': token, **detail})
    atomic_json(out/'diagnostics.json', diagnostics)
    atomic_json(out/'COMPLETE.json', {'identity_sha256': identity['sha256'], 'candidate_payloads_sha256': fingerprint(payloads),
                                     'target_payloads_sha256': fingerprint(seals), 'scenes': len(records), 'global_NFT_reward_std': global_nft_std})
    print(json.dumps({'status': 'COMPLETE', 'scenes': len(records), 'objective': a.objective, 'identity_sha256': identity['sha256']}))


if __name__ == '__main__':
    main()
