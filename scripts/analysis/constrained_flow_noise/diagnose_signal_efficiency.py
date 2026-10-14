"""Read saved real groups to separate zero signal, weak signal and refill waste.

Thresholds below are descriptive diagnostics, never new training filters.
No trajectories, rewards, advantages or training configuration are changed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from .summarize_runs import rows


def diagnose(run, collections):
    run = Path(run)
    identity = json.loads((run/'run_identity.json').read_text())
    counts, seconds = Counter(), defaultdict(Counter)
    accepted_rms, accepted_range, first_costs = [], [], []
    archive_hashes = {}
    for index in range(collections):
        path = run/'behavior_collections'/f'{index:06d}.pt'
        receipt = json.loads(path.with_suffix('.json').read_text())
        digest = file_sha(path)
        if digest != receipt['sha256']:
            raise ValueError('Behavior archive changed: ' + str(path))
        archive_hashes[path.name] = digest
        batch = torch.load(path, map_location='cpu', weights_only=False)
        if batch['behavior_identity']['run_identity'] != identity['sha256']:
            raise ValueError('Foreign behavior archive')
        group_rows = batch['rows']
        if len(group_rows) != receipt['raw_groups']:
            raise ValueError('Archive count mismatch')
        first_costs.append(torch.stack([r['costs'].double() for r in group_rows[:16]]).mean((0, 1)).numpy())
        stop = batch['accepted'][15]+1 if len(batch['accepted']) == 16 else len(group_rows)
        counts['tail_groups_after_16th_effective'] += len(group_rows)-max(16, stop)
        for position, row in enumerate(group_rows):
            reason = row['reason']
            counts['raw_groups'] += 1
            counts[reason] += 1
            for key in ('rollout_seconds', 'scoring_seconds', 'reference_seconds', 'scoring_cpu_seconds'):
                seconds[reason][key] += row[key]
            reward = row['rewards'].double()
            cost = row['costs'].double()
            adv = row['advantage'].double()
            z = row['advantage_statistics']['component_z'][0].double()
            varying = (z != 0).any(0)
            reward_range = float(reward.max()-reward.min())
            rms = float(adv.square().mean().sqrt())
            tied = reward_range == 0
            counts['tied_PDMS'] += int(tied)
            counts['tied_PDMS_all_one'] += int(tied and bool((reward == 1).all()))
            counts['tied_PDMS_all_zero'] += int(tied and bool((reward == 0).all()))
            if reason == 'ZERO_FINAL_ADVANTAGE':
                counts['zero_A_all_six_components_constant'] += int(not varying.any())
                weighted = z * torch.cat([torch.ones(1), -batch['lambda_snapshot']]).double()
                counts['zero_A_only_inactive_components_vary'] += int(varying.any() and not (weighted != 0).any())
                counts['zero_A_active_components_cancel'] += int((weighted != 0).any())
                counts['zero_A_still_used_in_first16_retention_dual'] += int(position < 16)
            if reason == 'ACCEPTED':
                accepted_rms.append(rms)
                accepted_range.append(reward_range)
                counts['accepted_A_RMS_below_0_01'] += int(rms < .01)
                counts['accepted_A_RMS_below_0_1'] += int(rms < .1)
                counts['accepted_official_binary_cost_varies'] += int(varying[1:4].any())
                counts['accepted_only_reward_component_varies'] += int(varying[0] and not varying[1:].any())
                counts['accepted_reward_tied_geometric_or_safety_signal'] += int(tied)
                counts['accepted_positive_reward_range_below_0_001'] += int(0 < reward_range < .001)
                counts['accepted_positive_reward_range_below_0_005'] += int(0 < reward_range < .005)
    update_rows = [r for r in rows(run/'training_rank0.jsonl') if r['collection'] < collections]
    latest = update_rows[-1]
    quantile = lambda x: np.quantile(x, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(x) else []
    return dict(variant=identity['variant'], run_identity=identity['sha256'], collections=collections,
                through_update=latest['update'], counts=dict(counts),
                actual_candidate_scores=16*counts['raw_groups'],
                PG_unused_candidate_scores=16*(counts['ZERO_FINAL_ADVANTAGE']+counts['SURPLUS_SAME_BEHAVIOR']),
                accepted_A_RMS_quantiles_0_10_25_50_75_90_100=quantile(accepted_rms),
                accepted_PDMS_range_quantiles_0_10_25_50_75_90_100=quantile(accepted_range),
                first16_unfiltered_cost_means=np.mean(first_costs, axis=0).tolist(),
                last64_first16_unfiltered_cost_means=np.mean(first_costs[-64:], axis=0).tolist(),
                fixed_cost_budgets=latest['dual']['budgets'], lambda_after=latest['dual']['after'],
                measured_worker_seconds_by_reason={k: dict(v) for k, v in seconds.items()},
                time_semantics='Measured worker sums, not exclusive GPU-hours or a prediction of speedup',
                refill_tail_semantics='Read-only prefix counterfactual on each saved collection; smaller refills may lose GPU parallelism and change later policy history',
                archive_hashes_sha256=fingerprint(archive_hashes),
                gradient_limitation='Nonzero/RMS advantages are not measurements of useful actor gradient or deployment gain')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--collections', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_dispatch()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = dict(schema='cfn_real_group_signal_efficiency_v1', source_sha256=file_sha(__file__),
                  variants={v: diagnose(Path(args.run_root)/f'formal_{v}_seed42_v1', args.collections)
                            for v in ('C_SDE', 'N_FLOW_NOISE')},
                  no_new_rollouts_or_updates=True, descriptive_thresholds_not_training_filters=True)
    result['sha256'] = fingerprint(result)
    atomic_json(output, result)
    print(json.dumps(dict(status='COMPLETE', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
