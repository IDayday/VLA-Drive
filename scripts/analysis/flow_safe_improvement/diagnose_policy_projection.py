"""Measure fresh-output drift, diversity and finite-target absorption by context.

This analysis never selects an output or updates a policy. Energy distances and
request geometry complement, and do not replace, canonical planning scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.safe_flow_improvement.evaluation_override import evaluation_seeds


def distance(x, y):
    # Euclidean norm of the complete XY request, scaled to metres per timestep.
    x, y = np.asarray(x, dtype=np.float64)[..., :2], np.asarray(y, dtype=np.float64)[..., :2]
    return np.sqrt(np.square(x[:, None]-y[None]).sum(axis=(-1, -2))/x.shape[-2])


def energy(samples, target, probability):
    q = np.asarray(probability, dtype=np.float64)
    return float(2*np.mean(distance(samples, target)@q)-distance(samples, samples).mean()
                 -q@distance(target, target)@q)


def sealed(root, token, identity):
    path = Path(root)/'scenes'/token[:2]/(token+'.pt')
    marker = json.loads(path.with_suffix('.json').read_text())
    if marker['identity_sha256'] != identity or marker['payload_sha256'] != file_sha(path):
        raise ValueError('Incomplete or changed inputs; cannot drop scenes')
    return torch.load(path, weights_only=True, map_location='cpu'), marker['payload_sha256']


def diversity(request):
    xy = np.asarray(request, dtype=np.float64)[..., :2]
    if len(xy) < 2:
        return {}  # One inference seed cannot identify within-scene diversity.
    pair = np.linalg.norm(xy[:, None]-xy[None], axis=-1)
    upper = np.triu_indices(len(xy), 1)
    centered = xy.reshape(len(xy), -1).copy(); centered -= centered.mean(0)
    power = np.linalg.svd(centered, compute_uv=False)**2
    q = power[power > 0]/power.sum() if power.sum() else np.array([])
    return dict(pairwise_ADE_m=float(pair.mean(-1)[upper].mean()),
                pairwise_FDE_m=float(pair[..., -1][upper].mean()),
                XY_effective_rank=float(np.exp(-np.sum(q*np.log(q)))) if len(q) else 0.)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('samples', 'baseline-samples', 'population', 'cohorts', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--targets', help='Optional fixed TRAIN FM target bank; never use DEV to fit targets')
    p.add_argument('--evaluation-override')
    a = p.parse_args(); require_dispatch()
    pop, cohorts = [json.loads(Path(x).read_text()) for x in (a.population, a.cohorts)]
    manifests = [json.loads((Path(x)/'manifest.json').read_text()) for x in (a.samples, a.baseline_samples)]
    current, base = manifests
    seeds, override = evaluation_seeds(pop, a.evaluation_override)
    fields = ('population_identity', 'tokens_sha256', 'cohort', 'mode', 'role', 'root_precision', 'feature_identity')
    if any(current[k] != base[k] for k in fields) or current.get('execution_profile') != base.get('execution_profile'):
        raise ValueError('Require matched roots, observations and execution profiles')
    if any(not set(seeds) <= set(m['seeds']) <= set(pop['evaluation_seeds']) for m in manifests):
        raise ValueError('Requested fixed evaluation seeds are missing or changed')
    if base['behavior_sha256'] != SFT_SHA256 or current['mode'] != 'ode' or current['role'] != 'evaluation':
        raise ValueError('Require fresh original-ODE samples and the permanent SFT baseline')
    tokens = current.get('explicit_tokens', pop.get(current['cohort']))
    if fingerprint(tokens) != current['tokens_sha256']:
        raise ValueError('Scene denominator changed')
    target_manifest = json.loads((Path(a.targets)/'manifest.json').read_text()) if a.targets else None
    if target_manifest and (current['cohort'] != 'diagnostic_tokens' or target_manifest['objective'] not in ('safe', 'reward')
                            or not set(target_manifest['tokens']) <= set(pop['train_tokens'])):
        raise ValueError('Projection targets are fixed FM TRAIN labels only')
    target_tokens = set(target_manifest['tokens']) if target_manifest else set()
    random_tokens = set(pop['diagnostic_random_tokens'])
    records, hashes = [], {}
    for token in tokens:
        x, xsha = sealed(a.samples, token, current['sha256'])
        b, bsha = sealed(a.baseline_samples, token, base['sha256'])
        for row, manifest in ((x, current), (b, base)):
            indices = [manifest['seeds'].index(seed) for seed in seeds]
            row['requested_trajectories'] = row['requested_trajectories'][indices]
            row['noise_keys'] = [row['noise_keys'][i] for i in indices]
        if x['noise_keys'] != b['noise_keys']:
            raise ValueError('Paired inference roots changed')
        request, old = [row['requested_trajectories'].double().numpy() for row in (x, b)]
        paired = np.linalg.norm(request[..., :2]-old[..., :2], axis=-1)
        dnew, dold = diversity(request), diversity(old)
        values = dict(paired_ADE_m=float(paired.mean()), paired_FDE_m=float(paired[:, -1].mean()),
                      endpoint_forward_shift_m=float((request[:, -1, 0]-old[:, -1, 0]).mean()),
                      endpoint_lateral_shift_m=float((request[:, -1, 1]-old[:, -1, 1]).mean()))
        for key in dnew:
            values['policy_'+key] = dnew[key]; values['SFT_'+key] = dold[key]
            values['change_'+key] = dnew[key]-dold[key]
        hashes[token] = dict(policy=xsha, SFT=bsha)
        if token in target_tokens:
            label, labelsha = sealed(a.targets, token, target_manifest['sha256'])
            target = decode_request(label['actions'].numpy(), act_norm=1)
            reference = decode_request(label['reference_actions'].numpy(), act_norm=1)
            q = label['target_probabilities'].double().numpy()
            if (q < 0).any() or abs(q.sum()-1) > 1e-10:
                raise ValueError('Invalid target distribution')
            values['policy_target_energy_V_m'] = energy(request, target, q)
            values['SFT_target_energy_V_m'] = energy(old, target, q)
            values['change_target_energy_V_m'] = values['policy_target_energy_V_m']-values['SFT_target_energy_V_m']
            wanted = np.sum(q[:, None, None]*target[..., :2], axis=0)-reference[..., :2].mean(0)
            actual = (request[..., :2]-old[..., :2]).mean(0)
            values['target_mean_displacement_RMS_m'] = float(np.sqrt(np.square(wanted).sum()/8))
            values['context_update_target_dot_m2'] = float(np.sum(actual*wanted)/8)
            for dim, axis, threshold in ((0, 'forward', .1), (1, 'lateral', .05)):
                values['target_endpoint_'+axis+'_shift_m'] = float(wanted[-1, dim])
                if abs(wanted[-1, dim]) >= threshold:
                    values['endpoint_'+axis+'_direction_agreement'] = float(actual[-1, dim]*wanted[-1, dim] > 0)
            hashes[token]['target'] = labelsha
        records.append(dict(token=token, log=pop['token_logs'][token],
                            stratum='DEV' if current['cohort'] == 'dev_tokens' else 'random' if token in random_tokens else 'targeted',
                            in_fixed_learning_bank=token in target_tokens, metrics=values))
    tables = {}
    for stratum in sorted({x['stratum'] for x in records}):
        for population in ('all', 'in_fixed_learning_bank') if target_manifest else ('all',):
            subset = [x for x in records if x['stratum'] == stratum and (population == 'all' or x['in_fixed_learning_bank'])]
            if not subset:
                continue
            stats = {}
            for key in sorted(set().union(*(x['metrics'].keys() for x in subset))):
                selected = [x for x in subset if key in x['metrics']]
                stats[key] = log_bootstrap_difference([x['metrics'][key] for x in selected], [x['log'] for x in selected])
            tables[stratum+'/'+population] = dict(scenes=len(subset), statistics=stats)
    report = dict(status='COMPLETE_FRESH_POLICY_GEOMETRY_AND_FINITE_TARGET_DIAGNOSTIC',
                  policy_identity=current['sha256'], baseline_identity=base['sha256'],
                  target_identity=target_manifest['sha256'] if target_manifest else None,
                  input_payloads_sha256=fingerprint(hashes), source_sha256=file_sha(__file__), scenes=len(tokens),
                  inference_seeds=seeds, evaluation_override=override,
                  within_scene_diversity_available=len(seeds) > 1,
                  tables=tables, no_candidate_selection_or_training=True,
                  energy_definition='Empirical V-statistic 2 E||X-Y|| - E||X-Xprime|| - E||Y-Yprime||, XY trajectory norm / sqrt(8). Finite-sample biased; with one seed this is only an output-to-target discrepancy, not a policy-distribution distance.',
                  interpretation='Lower target energy plus canonical improvement supports absorption; geometry alone does not establish safety, mode-collapse causality or useful scenario-dependent behavior.',
                  context_direction_thresholds_m=dict(forward=.1, lateral=.05),
                  TRAIN_targeted_stratum_is_not_unbiased=True)
    report['sha256'] = fingerprint(report)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report); atomic_json(out/'per_scene.json', records)
    print(json.dumps(dict(status=report['status'], scenes=len(tokens))), flush=True)


if __name__ == '__main__':
    main()
