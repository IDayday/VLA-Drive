"""Separate global request drift from context-associated target following.

This is an exploratory read-only diagnostic, not a causal intervention or a
planning score. Five fresh ODE roots define each policy's finite sample mean.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.diagnose_policy_projection import sealed
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, fingerprint, file_sha
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


def centered_dot(x, y):
    return float(np.mean(np.sum((x-x.mean(0))*(y-y.mean(0)), axis=(1, 2))/8))


def summarize(rows):
    actual, wanted = [np.stack([r[k] for r in rows]) for k in ('actual', 'wanted')]
    logs = np.array([r['log'] for r in rows])
    unique = sorted(set(logs))
    groups = [np.flatnonzero(logs == log) for log in unique]
    rng = np.random.default_rng(431)
    boots = []
    for _ in range(2000):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(centered_dot(actual[ids], wanted[ids]))
    bins = np.digitize([r['SFT_endpoint_forward_m'] for r in rows], [0., 5., 15., 30., 60.])
    residual_x, residual_y = actual.copy(), wanted.copy()
    for k in set(bins):
        ids = bins == k
        residual_x[ids] -= actual[ids].mean(0)
        residual_y[ids] -= wanted[ids].mean(0)
    observed = float(np.mean(np.sum(residual_x*residual_y, axis=(1, 2))/8))
    shuffled = []
    for _ in range(2000):
        order = np.arange(len(rows))
        for k in set(bins):
            ids = np.flatnonzero(bins == k); order[ids] = rng.permutation(ids)
        shuffled.append(float(np.mean(np.sum(residual_x*residual_y[order], axis=(1, 2))/8)))
    directions = {}
    for axis, dim, threshold in [('forward', 0, .1), ('lateral', 1, .05)]:
        for label, sign in [('negative', -1), ('positive', 1)]:
            ids = sign*wanted[:, -1, dim] >= threshold
            directions[axis+'_'+label] = dict(scenes=int(ids.sum()),
                actual_direction_agreement=float(np.mean(sign*actual[ids, -1, dim] > 0)) if ids.any() else None,
                mean_actual_endpoint_change_m=float(actual[ids, -1, dim].mean()) if ids.any() else None,
                mean_wanted_endpoint_change_m=float(wanted[ids, -1, dim].mean()) if ids.any() else None)
    return dict(scenes=len(rows), logs=len(unique), raw_mean_dot_m2=float(np.mean(np.sum(actual*wanted, axis=(1, 2))/8)),
        global_mean_update_RMS_m=float(np.sqrt(np.sum(actual.mean(0)**2)/8)),
        global_mean_target_RMS_m=float(np.sqrt(np.sum(wanted.mean(0)**2)/8)),
        centered_dot_m2=centered_dot(actual, wanted), centered_dot_log_bootstrap_CI95=np.quantile(boots, [.025, .975]).tolist(),
        within_SFT_forward_distance_bin_centered_dot_m2=observed,
        within_bin_shuffled_target_dot_CI95=np.quantile(shuffled, [.025, .975]).tolist(),
        exploratory_permutation_upper_tail_p=float((1+np.sum(np.asarray(shuffled) >= observed))/2001),
        direction_groups=directions)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('samples', 'baseline-samples', 'targets', 'population', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--evaluation-override')
    a = p.parse_args(); require_dispatch()
    current, base, target = [json.loads((Path(x)/'manifest.json').read_text()) for x in (a.samples, a.baseline_samples, a.targets)]
    pop = json.loads(Path(a.population).read_text())
    seeds, override = evaluation_seeds(pop, a.evaluation_override)
    if (current['cohort'] != 'diagnostic_tokens' or base['behavior_sha256'] != SFT_SHA256
            or current['execution_profile'] != base['execution_profile'] or current['tokens_sha256'] != base['tokens_sha256']
            or any(not set(seeds) <= set(m['seeds']) <= set(pop['evaluation_seeds']) for m in (current, base))
            or current['role'] != 'evaluation' or current['mode'] != 'ode'
            or target['population_identity'] != pop['identity_sha256'] or target['objective'] not in ('safe', 'reward')
            or not set(target['tokens']) <= set(pop['train_tokens'])):
        raise ValueError('Context diagnostic requires matched TRAIN ODE outputs and locked positive targets')
    selected = [t for t in current['explicit_tokens'] if t in set(target['tokens'])]
    rows, hashes = [], {}
    for token in selected:
        values = [sealed(root, token, manifest['sha256']) for root, manifest in
                  ((a.samples, current), (a.baseline_samples, base), (a.targets, target))]
        x, b, label = [v[0] for v in values]
        for row, manifest in ((x, current), (b, base)):
            indices = [manifest['seeds'].index(seed) for seed in seeds]
            row['requested_trajectories'] = row['requested_trajectories'][indices]
            row['noise_keys'] = [row['noise_keys'][i] for i in indices]
        if x['noise_keys'] != b['noise_keys']:
            raise ValueError('Paired initial ODE roots differ')
        q = label['target_probabilities'].double().numpy()
        if (q < 0).any() or abs(q.sum()-1) > 1e-9:
            raise ValueError('Invalid target distribution')
        candidate, reference = [decode_request(label[k].numpy(), act_norm=1)[..., :2] for k in ('actions', 'reference_actions')]
        wanted = np.sum(q[:, None, None]*candidate, axis=0)-reference.mean(0)
        request, old = [v['requested_trajectories'].double().numpy()[..., :2] for v in (x, b)]
        rows.append(dict(token=token, log=pop['token_logs'][token],
                         stratum='random' if token in pop['diagnostic_random_tokens'] else 'targeted',
                         actual=(request-old).mean(0), wanted=wanted, SFT_endpoint_forward_m=float(old[:, -1, 0].mean())))
        hashes[token] = [v[1] for v in values]
    tables = {s: summarize([r for r in rows if r['stratum'] == s]) for s in ('random', 'targeted') if any(r['stratum'] == s for r in rows)}
    report = dict(status='COMPLETE_EXPLORATORY_CONTEXT_ALIGNMENT', tables=tables, policy_identity=current['sha256'],
                  baseline_identity=base['sha256'], target_identity=target['sha256'], input_payloads_sha256=fingerprint(hashes),
                  source_sha256=file_sha(__file__), forward_distance_bins_m=[0, 5, 15, 30, 60],
                  shuffle_seed=431, repeats=2000, inference_seeds=seeds, evaluation_override=override,
                  inference_and_bootstrap_seeds_are_not_training_repeats=True,
                  preregistered_before_results=False,
                  limitations='Exploratory association after removing global/binned drift, not a causal demonstration of scene reasoning. Bins use SFT requested endpoint distance, not actual initial speed. TRAIN targeted sample is not population-unbiased; random in-bank sample is small. No hyperparameter is fitted or selected.')
    report['sha256'] = fingerprint(report)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    print(json.dumps(dict(status=report['status'], scenes=len(rows))), flush=True)


if __name__ == '__main__':
    main()
