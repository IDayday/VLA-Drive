"""Paired single-ODE comparison using actual common scenes, clustered by log."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analysis.flow_safe_improvement.evaluate_policy import load_scores
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_ode import SAFETY


def compare(left_scores, right_scores, population):
    for root in (left_scores, right_scores):
        if not (Path(root)/'COMPLETE').exists():
            raise ValueError('Both full evaluation populations must be complete')
    lm, left, ls = load_scores(left_scores, population, 'evaluation', [2101])
    rm, right, rs = load_scores(right_scores, population, 'evaluation', [2101])
    if (lm['protocol_sha256'] != rm['protocol_sha256'] or set(left) != set(right)
            or lm['sample_manifest']['execution_profile'] != rm['sample_manifest']['execution_profile']
            or lm['sample_manifest']['seeds'] != [2101] or rm['sample_manifest']['seeds'] != [2101]):
        raise ValueError('Unmatched protocol, population, execution or single-seed contract')
    tokens = list(left)
    logs = [population['token_logs'][t] for t in tokens]
    metrics, safety = {}, {}
    for key in left[tokens[0]]:
        l = np.array([left[t][key][0] for t in tokens])
        r = np.array([right[t][key][0] for t in tokens])
        metrics[key] = dict(left=float(l.mean()), right=float(r.mean()), right_minus_left=log_bootstrap_difference(r-l, logs))
    for key in (*SAFETY, 'joint'):
        fields = SAFETY if key == 'joint' else (key,)
        l = np.array([all(left[t][f][0] == 1 for f in fields) for t in tokens])
        r = np.array([all(right[t][f][0] == 1 for f in fields) for t in tokens])
        safety[key] = dict(left_violations=int((~l).sum()), right_violations=int((~r).sum()),
                           left_pass_right_fail=int((l & ~r).sum()), left_fail_right_pass=int((~l & r).sum()),
                           violation_difference=log_bootstrap_difference((~r).astype(float)-(~l).astype(float), logs))
    l = np.array([left[t]['score'][0] == 0 for t in tokens])
    r = np.array([right[t]['score'][0] == 0 for t in tokens])
    return dict(schema='cfn_paired_ODE_comparison_v1', scenes=len(tokens), logs=len(set(logs)), inference_seed=2101,
                left_behavior_sha256=lm['sample_manifest']['behavior_sha256'], right_behavior_sha256=rm['sample_manifest']['behavior_sha256'],
                left_payloads_sha256=ls, right_payloads_sha256=rs, protocol_sha256=lm['protocol_sha256'],
                cohort=lm['sample_manifest']['cohort'], metrics=metrics, safety=safety,
                zeros=dict(left=int(l.sum()), right=int(r.sum()), nonzero_to_zero=int((~l & r).sum()), zero_to_nonzero=int((l & ~r).sum())),
                units='raw 0-1 scores; multiply means and differences by100 for points',
                uncertainty='paired log-cluster bootstrap with scene weights, excludes training/inference-seed uncertainty',
                deployment='one original 10-step ODE trajectory per scene; no candidate ranking')


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('left-scores', 'right-scores', 'population', 'output', 'left-name', 'right-name'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--qualification', action='store_true', help='Mark plumbing smoke; never a formal method result')
    args = parser.parse_args()
    require_dispatch()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare(args.left_scores, args.right_scores, json.loads(Path(args.population).read_text()))
    result.update(left_name=args.left_name, right_name=args.right_name, qualification_only=args.qualification, source_sha256=file_sha(__file__))
    result['sha256'] = fingerprint(result)
    atomic_json(output, result)
    print(json.dumps(dict(scenes=result['scenes'], right_minus_left_PDMS_points=100*result['metrics']['score']['right_minus_left']['mean'],
                          qualification_only=args.qualification)))


if __name__ == '__main__':
    main()
