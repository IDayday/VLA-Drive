"""Lock representative/directed cohorts using ONLY fixed SFT training feedback.

Run after reference scoring and geometric auditing, before any search results.
No learned model, DEV scores or search success enter this selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, SAFETY_METRICS, file_sha, fingerprint


def choose_directed(features, excluded, quotas, namespace):
    available = set(features) - set(excluded)
    selected, labels = [], {}
    for label, count in quotas.items():
        def key(token):
            f = features[token]
            rank = {'unstable': (-f['safety_instability'], -f['score_std']),
                    'low_score': (f['mean_score'],),
                    'near_boundary': (f['near_boundary_m'],),
                    'low_progress': (f['mean_EP'],)}[label]
            return (*rank, fingerprint([namespace, label, token]))
        ordered = sorted(available, key=key)[:count]
        if len(ordered) != count:
            raise ValueError('Insufficient directed population')
        for token in ordered:
            labels[token] = label
        selected.extend(ordered)
        available.difference_update(ordered)
    return selected, labels


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--population', required=True)
    p.add_argument('--reference-scores', required=True)
    p.add_argument('--geometry', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    population = json.loads(Path(a.population).read_text())
    contract = PopulationContract(*[frozenset(population[k]) for k in ('train_tokens', 'dev_tokens', 'navtest_tokens')], population['token_logs'])
    score_root, geometry_root = Path(a.reference_scores), Path(a.geometry)
    score_manifest = json.loads((score_root / 'manifest.json').read_text())
    if score_manifest['sample_manifest']['role'] != 'reference' or score_manifest['sample_manifest']['behavior_sha256'] != score_manifest['sample_manifest']['SFT_sha256']:
        raise ValueError('Cohorts require the permanent SFT reference')
    if score_manifest['sample_manifest']['seeds'] != population['reference_seeds']:
        raise ValueError('Reference seeds changed')
    features, seals, geometry_seals = {}, {}, {}
    safe_by_metric = {k: [] for k in SAFETY_METRICS}
    joint_safe, intermittent, failed = [], [], []
    for token in population['reference_screen_tokens']:
        contract.require_training([token], 'reference-based cohort selection')
        path = score_root / 'scenes' / token[:2] / (token + '.pt')
        seal = json.loads(path.with_suffix('.json').read_text())
        if seal['payload_sha256'] != file_sha(path) or seal['identity_sha256'] != score_manifest['sha256']:
            raise ValueError('Reference score publication changed')
        row = torch.load(path, map_location='cpu', weights_only=True)
        feedback = row['feedback']
        if len(feedback) != 4 or any(f['status'] != 'OK' for f in feedback):
            raise ValueError('Incomplete fixed reference; never hide failed scenes')
        metrics = np.array([[f['metrics'][k] for k in SAFETY_METRICS] for f in feedback])
        safety = metrics == 1.
        for j, k in enumerate(SAFETY_METRICS):
            if safety[:, j].all():
                safe_by_metric[k].append(token)
        joint = safety.all(1)
        (joint_safe if joint.all() else failed if not joint.any() else intermittent).append(token)
        geom_path = geometry_root / 'scenes' / token[:2] / (token + '.json')
        geom = json.loads(geom_path.read_text())
        if geom['score_payload_sha256'] != seal['payload_sha256']:
            raise ValueError('Geometry refers to different simulations')
        # Only nonnegative geometric clearance of jointly safe references is
        # used for near-boundary coverage, not as an official penalty/reward.
        distances = []
        for valid, g in zip(joint, geom['feedback']['candidates']):
            if valid:
                distances.extend(x for k in ('body_inside_road_boundary_clearance_m', 'body_to_recorded_object_distance_m')
                                 for x in g[k] if x is not None)
        scores = np.array([f['metrics']['score'] for f in feedback])
        features[token] = {'mean_score': float(scores.mean()), 'score_std': float(scores.std()),
                           'mean_EP': float(np.mean([f['metrics']['ego_progress'] for f in feedback])),
                           'safety_instability': int(np.sum(safety.any(0) != safety.all(0))),
                           'near_boundary_m': min(distances) if distances else 1e6,
                           'joint_safe_seeds': int(joint.sum())}
        seals[token], geometry_seals[token] = seal['payload_sha256'], file_sha(geom_path)
    learning, learning_labels = choose_directed(features, population['learning_random_tokens'],
                                               dict(unstable=512, low_score=512, near_boundary=512, low_progress=512), 'learning_v1')
    diagnostic, diagnostic_labels = choose_directed(features, population['diagnostic_random_tokens'],
                                                    dict(unstable=64, low_score=64, near_boundary=64, low_progress=64), 'diagnostic_v1')
    result = {'schema': 'flow_safe_locked_cohorts_v1', 'population_identity': population['identity_sha256'],
              'reference_score_manifest_sha256': file_sha(score_root / 'manifest.json'),
              'reference_score_payloads_sha256': fingerprint(seals), 'geometry_payloads_sha256': fingerprint(geometry_seals),
              'selection_source_sha256': file_sha(__file__),
              'learning_tokens': population['learning_random_tokens'] + learning,
              'diagnostic_tokens': population['diagnostic_random_tokens'] + diagnostic,
              'learning_targeted_labels': learning_labels, 'diagnostic_targeted_labels': diagnostic_labels,
              'reference_safe_by_metric': safe_by_metric, 'reference_joint_safe': joint_safe,
              'reference_intermittent_failure': intermittent, 'reference_stable_failure': failed,
              'screen_features': features, 'selection_uses_only_SFT_RL_TRAIN_feedback': True,
              'directed_subsets_are_not_population_unbiased': True,
              'no_safe_seed_is_finite_reference_failure_not_impossibility': True}
    assert len(set(result['learning_tokens'])) == 4096
    assert len(set(result['diagnostic_tokens'])) == 512
    result['sha256'] = fingerprint(result)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Do not overwrite locked cohorts')
    out.mkdir(parents=True)
    atomic_json(out / 'manifest.json', result)
    summary = {k: v for k, v in result.items() if k not in ('screen_features', 'learning_tokens', 'diagnostic_tokens', 'learning_targeted_labels', 'diagnostic_targeted_labels', 'reference_safe_by_metric', 'reference_joint_safe', 'reference_intermittent_failure', 'reference_stable_failure')}
    summary['counts'] = {k: len(result[k]) for k in ('learning_tokens', 'diagnostic_tokens', 'reference_joint_safe', 'reference_intermittent_failure', 'reference_stable_failure')}
    summary['metric_safe_counts'] = {k: len(v) for k, v in safe_by_metric.items()}
    atomic_json(out / 'receipt.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
