"""Check whether the P0/P1 target intervention actually changes training labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.data import keys_for_positions
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import fingerprint, seed_for
from scripts.analysis.flow_safe_improvement.train_fixed_bank import load_sealed


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('left', 'right', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--training-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=8)
    a = p.parse_args()
    left, right = Path(a.left), Path(a.right)
    lm, rm = [json.loads((x/'manifest.json').read_text()) for x in (left, right)]
    for key in ('candidate_bank_sha256', 'tokens', 'population_identity', 'eta', 'alpha_max', 'delta_useful'):
        if lm[key] != rm[key]:
            raise ValueError('This diagnostic isolates weighting on the same candidate bank')
    if len(lm['tokens']) != 4096 or lm['qualification_only'] or rm['qualification_only']:
        raise ValueError('Use the complete fixed learning population')
    rows, l1, max_abs, equal_actions = {}, [], [], True
    for token in lm['tokens']:
        l, r = load_sealed(left, token, lm['sha256']), load_sealed(right, token, rm['sha256'])
        equal_actions &= torch.equal(l['actions'], r['actions'])
        difference = (l['target_probabilities']-r['target_probabilities']).abs()
        l1.append(float(difference.sum())); max_abs.append(float(difference.max()))
        rows[token] = (l['target_probabilities'], r['target_probabilities'])
    if not equal_actions:
        raise ValueError('Candidate action labels/order differ; weighting comparison is confounded')
    different_draws = 0
    for position, (token, _) in enumerate(keys_for_positions(lm['tokens'], a.training_seed, range(len(lm['tokens'])*a.epochs))):
        generation_key = seed_for('extraction_scene', a.training_seed, token, position)
        key = seed_for('candidate_target_draw', generation_key, token)
        l, r = [int(torch.multinomial(weights, 1, generator=torch.Generator().manual_seed(key))) for weights in rows[token]]
        different_draws += l != r
    result = dict(status='COMPLETE_SAME_BANK_INTERVENTION_DIAGNOSTIC', left_identity=lm['sha256'], right_identity=rm['sha256'],
                  scenes=len(lm['tokens']), epochs=a.epochs, training_seed=a.training_seed,
                  max_probability_abs_difference=max(max_abs), mean_probability_L1=float(np.mean(l1)), max_probability_L1=max(l1),
                  scenes_L1_above_1e_9=sum(x>1e-9 for x in l1), same_normalized_actions_and_order=equal_actions,
                  candidate_draws_checked=len(lm['tokens'])*a.epochs, different_actual_training_candidate_draws=different_draws,
                  unchanged_replay_and_retention_settings_required=True,
                  implication='A redundant target intervention cannot identify a benefit from the safety constraints; fresh policy safety remains a separate empirical question.')
    result['sha256'] = fingerprint(result)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
