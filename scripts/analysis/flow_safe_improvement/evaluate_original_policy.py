"""Enforce matched original-export numerics before using the common statistics."""
import argparse
import json
from pathlib import Path

from scripts.analysis.flow_safe_improvement import evaluate_policy
from scripts.analysis.flow_safe_improvement.sample_original_export import PROFILE
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import fingerprint


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('scores', 'baseline-scores', 'reference-scores', 'population', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    samples = [json.loads((Path(root)/'manifest.json').read_text())['sample_manifest']
               for root in (a.scores, a.baseline_scores, a.reference_scores)]
    left, right = [sample.get('execution_profile', {}) for sample in samples[:2]]
    if left.get('name') != PROFILE or left != right:
        raise ValueError('Primary deployment comparisons require identical original-export numerical profiles')
    evaluate_policy.main()
    path = Path(a.output)/'report.json'
    report = json.loads(path.read_text())
    report.pop('sha256')
    report['deployment_execution_profile'] = left
    report['fixed_reference_safety_definition'] = dict(
        identity=samples[2]['sha256'],
        execution_profile=samples[2].get('execution_profile', 'frozen_training_BF16_feature_cache'),
        semantics='Predeclared finite four-seed SFT set; membership is retained, not reclassified using new policy results. '
                  'Matched SFT and policy mean/transition comparisons both use the original-export profile.')
    report['sha256'] = fingerprint(report)
    atomic_json(path, report)


if __name__ == '__main__':
    main()
