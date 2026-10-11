"""Summarize existing canonical scene scores without changing the evaluator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from export_experiment_snapshot import scene_csv, require, METRICS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_patterns(table, tokens):
    # These are the canonical multiplicative components, not a new risk score.
    components = ('no_at_fault_collisions', 'drivable_area_compliance',
                  'driving_direction_compliance')
    counts = Counter()
    for token in tokens:
        pattern = '+'.join(key for key in components if float(table[token][key]) == 0)
        counts[pattern or 'no_zero_multiplicative_component'] += 1
    return dict(sorted(counts.items()))


def compare(candidate, baseline, name):
    require(set(candidate) == set(baseline) and
            all(candidate[t]['log'] == baseline[t]['log'] for t in candidate),
            'Scene/log populations differ')
    introduced = [t for t in candidate if float(candidate[t]['score']) == 0 and
                  float(baseline[t]['score']) > 0]
    fixed = [t for t in candidate if float(candidate[t]['score']) > 0 and
             float(baseline[t]['score']) == 0]
    metrics = {}
    for label, key in {**METRICS, 'DDC': 'driving_direction_compliance'}.items():
        delta = [float(candidate[t][key])-float(baseline[t][key]) for t in candidate]
        metrics[label] = {'improved_scenes': sum(d > 1e-12 for d in delta),
                          'worsened_scenes': sum(d < -1e-12 for d in delta),
                          'unchanged_scenes': sum(abs(d) <= 1e-12 for d in delta),
                          'mean_delta_points': sum(delta)/len(delta)*100}
    return {'baseline': name, 'introduced_zero_scenes': len(introduced),
            'introduced_zero_component_patterns': zero_patterns(candidate, introduced),
            'fixed_baseline_zero_scenes': len(fixed),
            'fixed_baseline_zero_component_patterns': zero_patterns(baseline, fixed),
            'metric_transfers': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-manifest', type=Path, required=True)
    parser.add_argument('--group', default='G3_EVENT_LOCAL')
    parser.add_argument('--update', type=int, default=50000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    manifest = json.loads(args.snapshot_manifest.read_text())
    run = next(r for r in manifest['runs'] if r['dataset'] == 'navsim' and r['group'] == args.group)
    require(run['checkpoint_updates'] >= args.update, 'Observation exceeds captured native boundary')
    folder = Path(run['native_run'])/f'dev_{args.update}_canonical'
    source = next(s for s in manifest['evaluation_sources'] if s.get('path') == str(folder))
    require(digest(folder/'COMPLETE.json') == source['complete_sha256'] and
            digest(folder/'identity.json') == source['identity_sha256'], 'Canonical source changed')
    candidate = scene_csv(folder/'q_final/scenes.csv')
    paths = {'same_model_q0': folder/'q0/scenes.csv'}
    for reference in manifest['evaluation_sources']:
        if reference['kind'] == 'historical_NAVSIM' and reference['updates'] == args.update:
            require(digest(reference['path']) == reference['sha256'], 'Historical source changed')
            paths[reference['name']] = Path(reference['path'])
    zero = [t for t in candidate if float(candidate[t]['score']) == 0]
    result = {'schema': 'canonical_NAVSIM_failure_transfer_v1', 'group': args.group,
              'updates': args.update, 'scenes': 1696, 'logs': 16,
              'snapshot_manifest_sha256': digest(args.snapshot_manifest),
              'analyzer_sha256': digest(__file__),
              'scene_csv_sha256': {'q_final': digest(folder/'q_final/scenes.csv'),
                                   **{n: digest(p) for n, p in paths.items()}},
              'q_final_zero_scenes': len(zero),
              'q_final_zero_component_patterns': zero_patterns(candidate, zero),
              'comparisons': [compare(candidate, scene_csv(p), n) for n, p in paths.items()],
              'limits': ['Existing development scores, no additional inference',
                         'Component patterns are evaluator outcomes, not causal explanations',
                         'TTC and progress affect weighted scores; zero multiplicative components are listed separately',
                         'One training seed; historical architectures and compute differ',
                         'No raw scene tokens exported']}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'q_final_zero_scenes': len(zero)}))


if __name__ == '__main__':
    main()
