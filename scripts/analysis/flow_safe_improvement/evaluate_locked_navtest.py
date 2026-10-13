"""Final matched single-seed Navtest; baseline safety is explicitly one-sample."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('final-lock', 'model', 'population', 'scores', 'baseline-scores', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); authorized(require_dispatch())
    pop = json.loads(Path(a.population).read_text())
    lock, lock_sha, selected = locked_contract(a.final_lock, a.model, pop)
    manifest, policy, policy_hash = load_scores(a.scores, pop, 'evaluation', [2101])
    baseline_manifest, baseline, baseline_hash = load_scores(a.baseline_scores, pop, 'evaluation', [2101])
    for sample in (manifest['sample_manifest'], baseline_manifest['sample_manifest']):
        if (sample['final_selection_lock_sha256'] != lock_sha or sample['cohort'] != 'navtest_tokens'
                or sample['execution_profile'] != lock['deployment_execution_profile'] or sample['seeds'] != [2101]):
            raise ValueError('Final sample identity differs from the locked protocol')
    if (manifest['sample_manifest']['behavior_sha256'] != selected['weight_sha256']
            or baseline_manifest['sample_manifest']['behavior_sha256'] != SFT_SHA256
            or manifest['protocol_sha256'] != baseline_manifest['protocol_sha256']
            or manifest['protocol_sha256'] != lock['v1_protocol_sha256']
            or set(policy) != set(baseline) or set(policy) != set(pop['navtest_tokens'])):
        raise ValueError('Unmatched final policy/SFT/population/protocol')
    tokens = list(policy); logs = [pop['token_logs'][t] for t in tokens]
    means, changes, arrays = {}, {}, {}
    for metric in sorted(policy[tokens[0]]):
        new = np.array([policy[t][metric][0] for t in tokens]); old = np.array([baseline[t][metric][0] for t in tokens])
        means[metric] = dict(policy=float(new.mean()), SFT=float(old.mean()),
                            policy_scene_quantiles=np.quantile(new, [0, .05, .1, .5, .9, 1]).tolist())
        changes[metric] = log_bootstrap_difference(new-old, logs); arrays[metric] = (new, old)
    safety = {}
    for metric in (*SAFETY_METRICS, 'joint'):
        keys = SAFETY_METRICS if metric == 'joint' else (metric,)
        new = np.logical_and.reduce([arrays[k][0] == 1 for k in keys])
        old = np.logical_and.reduce([arrays[k][1] == 1 for k in keys])
        safety[metric] = dict(SFT_single_seed_safe_scenes=int(old.sum()),
            newly_violating_SFT_single_seed_safe_scenes=int((old & ~new).sum()),
            recovered_SFT_single_seed_failures=int((~old & new).sum()), policy_pass_rate=float(new.mean()),
            SFT_pass_rate=float(old.mean()), finite_single_sample_definition_not_stable_multi_seed_safety=True)
    new, old = arrays['score']; zero_new, zero_old = new == 0, old == 0; high = old >= .9; delta = new-old
    report = dict(schema='flow_safe_final_single_seed_navtest_v1', status='COMPLETE_LOCKED_FINAL_NAVTEST',
        model=a.model, final_selection_lock_sha256=lock_sha, scenes=len(tokens), logs=len(set(logs)),
        inference_seeds=[2101], training_seeds=[42], multi_seed_cancelled_by_user=True,
        policy_identity=manifest['sha256'], baseline_identity=baseline_manifest['sha256'],
        payload_hashes=dict(policy=policy_hash, SFT=baseline_hash), units='raw_0_to_1', means=means,
        paired_changes=changes, safety=safety,
        zero_transitions=dict(SFT=int(zero_old.sum()), policy=int(zero_new.sum()),
                              new_failures=int((~zero_old & zero_new).sum()), recovered=int((zero_old & ~zero_new).sum()),
                              single_seed_any_all_mean_zero_events_coincide=True),
        high_score_decline=dict(SFT_threshold=.9, scenes=int(high.sum()), any_negative=int((high & (delta < 0)).sum()),
                               decline_at_least_001=int((high & (delta <= -.001)).sum()),
                               decline_at_least_005=int((high & (delta <= -.005)).sum())),
        deployment_execution_profile=lock['deployment_execution_profile'], source_sha256=file_sha(__file__),
        no_best_of_N=True, no_search_GT_or_scorer_in_actor=True, no_fitting_or_selection_on_this_Navtest=True,
        historical_Navtest_has_been_observed_repeatedly_not_fully_blind=True,
        fixed_DEV_four_seed_reference_population_not_redefined=True,
        limitations='One training and inference seed by user request. Log-cluster CI measures scene/log variation only; no seed stability claim. Final safety transfer uses the matched SFT2101 output, not a newly sampled four-seed Navtest reference.')
    report['sha256'] = fingerprint(report); out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    atomic_json(out/'per_scene.json', [dict(token=t, log=logs[i],
        means={k: float(arrays[k][0][i]) for k in arrays}, delta_score=float(delta[i])) for i, t in enumerate(tokens)])
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
