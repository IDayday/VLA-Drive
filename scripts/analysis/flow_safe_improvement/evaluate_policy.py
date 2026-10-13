"""Summarize fresh original-ODE single-candidate evaluations by scene and log.

Generation is performed by sample_frozen --mode ode --role evaluation, and
canonical scoring by score_samples. This entrypoint consumes every sealed scene;
it neither selects candidates nor uses reward to alter actor inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


def load_scores(root, population, role):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    sample = manifest['sample_manifest']
    if sample['mode'] != 'ode' or sample['role'] != role or sample['population_identity'] != population['identity_sha256']:
        raise ValueError('Require original ODE, matching population, and separated reference/evaluation roles')
    seeds = population['evaluation_seeds' if role == 'evaluation' else 'reference_seeds']
    if sample['seeds'] != seeds or sample['root_precision'] != 'original_bfloat16':
        raise ValueError('Deployment seeds or root precision changed')
    tokens = sample['explicit_tokens'] if sample['cohort'] == 'diagnostic_tokens' else population[sample['cohort']]
    if sample['cohort'] == 'diagnostic_tokens' and (len(tokens) != 512 or not set(tokens) <= set(population['train_tokens']) or not sample['cohort_identity']):
        raise ValueError('Full TRAIN diagnostic population changed')
    if sample['tokens_sha256'] != fingerprint(tokens):
        raise ValueError('Evaluation scene denominator changed')
    rows, digests = {}, {}
    for token in tokens:
        path = root/'scenes'/token[:2]/(token+'.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['identity_sha256'] != manifest['sha256'] or seal['payload_sha256'] != file_sha(path):
            raise ValueError('Incomplete or changed evaluation; missing scenes stay in denominator')
        row = torch.load(path, weights_only=True, map_location='cpu')
        feedback = row['feedback']
        if len(feedback) != len(seeds) or any(f['status'] != 'OK' for f in feedback):
            raise ValueError('Failed scoring cannot be dropped or imputed zero')
        rows[token] = {k: np.array([f['metrics'][k] for f in feedback], dtype=np.float64) for k in feedback[0]['metrics']}
        digests[token] = seal['payload_sha256']
    return manifest, rows, fingerprint(digests)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--scores', required=True)
    p.add_argument('--baseline-scores', required=True)
    p.add_argument('--reference-scores', required=True)
    p.add_argument('--population', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    population = json.loads(Path(a.population).read_text())
    manifest, policy, policy_sha = load_scores(a.scores, population, 'evaluation')
    baseline_manifest, baseline, baseline_sha = load_scores(a.baseline_scores, population, 'evaluation')
    reference_manifest, reference, reference_sha = load_scores(a.reference_scores, population, 'reference')
    if baseline_manifest['sample_manifest']['behavior_sha256'] != SFT_SHA256 or reference_manifest['sample_manifest']['behavior_sha256'] != SFT_SHA256:
        raise ValueError('Reference and baseline must both be the permanent original SFT')
    if set(policy) != set(baseline) or set(policy) != set(reference):
        raise ValueError('Unmatched policy/baseline/reference scene populations')
    if len({x['protocol_sha256'] for x in (manifest, baseline_manifest, reference_manifest)}) != 1:
        raise ValueError('Never compare v1/v2 as one reward')
    tokens = list(policy)
    logs = [population['token_logs'][t] for t in tokens]
    metrics = sorted(policy[tokens[0]])
    means, changes = {}, {}
    for key in metrics:
        current = np.array([policy[t][key].mean() for t in tokens])
        original = np.array([baseline[t][key].mean() for t in tokens])
        means[key] = dict(policy=float(current.mean()), SFT=float(original.mean()),
                          policy_scene_quantiles=np.quantile(current, [0, .05, .1, .5, .9, 1]).tolist())
        changes[key] = log_bootstrap_difference(current-original, logs)
    safety = {}
    for key in (*SAFETY_METRICS, 'joint'):
        def passes(row):
            return np.logical_and.reduce([row[k] == 1 for k in SAFETY_METRICS]) if key == 'joint' else row[key] == 1
        safe_tokens = [t for t in tokens if passes(reference[t]).all()]
        safety[key] = dict(fixed_reference_stable_safe_scenes=len(safe_tokens),
                           current_any_seed_violation_in_fixed_safe=sum(not passes(policy[t]).all() for t in safe_tokens),
                           baseline_any_eval_seed_violation_in_fixed_safe=sum(not passes(baseline[t]).all() for t in safe_tokens),
                           newly_violating_vs_matched_baseline=int(sum(passes(baseline[t]).all() and not passes(policy[t]).all() for t in tokens)),
                           recovered_all_eval_seeds=int(sum(not passes(baseline[t]).all() and passes(policy[t]).all() for t in tokens)),
                           current_pass_rate=float(np.mean([passes(policy[t]).mean() for t in tokens])))
    zeros = {}
    for label, test in [('any_seed_zero', lambda x: bool(np.any(x == 0))),
                        ('all_fixed_seeds_zero', lambda x: bool(np.all(x == 0))),
                        ('five_seed_scene_mean_zero', lambda x: bool(x.mean() == 0))]:
        old = np.array([test(baseline[t]['score']) for t in tokens])
        new = np.array([test(policy[t]['score']) for t in tokens])
        zeros[label] = dict(SFT=int(old.sum()), policy=int(new.sum()), new_failures=int((~old & new).sum()), recovered=int((old & ~new).sum()))
    old_score = np.array([baseline[t]['score'].mean() for t in tokens])
    delta = np.array([policy[t]['score'].mean()-baseline[t]['score'].mean() for t in tokens])
    high = old_score >= .9
    opportunities = {}
    for label, source in (('policy', policy), ('SFT', baseline)):
        best = []
        for token in tokens:
            feasible = np.logical_and.reduce([source[token][k] == 1 for k in SAFETY_METRICS])
            if feasible.any():
                best.append(float(source[token]['score'][feasible].max()))
        opportunities[label] = dict(candidates_per_scene=len(population['evaluation_seeds']),
                                    feasible_scenes=len(best), F_feasible=len(best)/len(tokens),
                                    safe_best_mean_on_feasible=float(np.mean(best)) if best else None,
                                    candidate_opportunity_only_not_deployed_policy_score=True)
    report = dict(schema='flow_safe_single_candidate_deployment_v1', status='COMPLETE_MATCHED_EVALUATION',
                  policy_identity=manifest['sha256'], baseline_identity=baseline_manifest['sha256'], reference_identity=reference_manifest['sha256'],
                  payload_hashes=dict(policy=policy_sha, baseline=baseline_sha, reference=reference_sha),
                  scenes=len(tokens), inference_seeds=population['evaluation_seeds'], training_seed_repetitions=1,
                  cohort=manifest['sample_manifest']['cohort'],
                  targeted_TRAIN_diagnostic_is_not_unbiased_population_estimate=manifest['sample_manifest']['cohort'] == 'diagnostic_tokens',
                  primary='mean_of_five_independent_single_candidate_original_10_step_ODE_outputs_no_selection',
                  units='raw_0_to_1', means=means, paired_changes=changes, safety=safety, zero_transitions=zeros,
                  separate_finite_candidate_opportunities=opportunities,
                  high_score_decline=dict(SFT_scene_mean_threshold=.9, scenes=int(high.sum()),
                                          any_negative=int((high & (delta < 0)).sum()),
                                          decline_at_least_001=int((high & (delta <= -.001)).sum()),
                                          decline_at_least_005=int((high & (delta <= -.005)).sum())),
                  no_best_of_N=True, no_search_GT_selection_or_online_scorer_in_actor=True,
                  SFT_may_have_seen_DEV=True, old_RL_saw_DEV=True)
    report['sha256'] = fingerprint(report)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve all evaluation identities')
    out.mkdir(parents=True)
    atomic_json(out/'report.json', report)
    atomic_json(out/'per_scene.json', [dict(token=t, log=population['token_logs'][t],
                                          means={k: float(policy[t][k].mean()) for k in metrics},
                                          delta_score=float(delta[i])) for i, t in enumerate(tokens)])
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
