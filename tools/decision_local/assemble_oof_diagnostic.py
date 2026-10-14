"""Pool each training scene's opposite-fold predictor output exactly once.

This checks readout generalization only: the common S0 was trained on every
original training log. Each fold predictor sees half the original population.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, identity, read, source_sha


def merge_heldout(rows_by_training_fold, log_fold, expected_rows):
    expected = {row['token']: row['log'] for row in expected_rows}
    if len(expected) != len(expected_rows) or set(rows_by_training_fold) != {0, 1}:
        raise ValueError('Unique complete scenes and both folds are required')
    selected = {}
    for fold, rows in rows_by_training_fold.items():
        by_token = {row['token']: row for row in rows}
        if len(by_token) != len(rows) or set(by_token) != set(expected):
            raise ValueError('Each evaluated model must cover the identical diagnostic scenes')
        for token, row in by_token.items():
            if row['log'] != expected[token] or row['log'] not in log_fold:
                raise ValueError('Evaluation log identity differs from the fixed train partition')
            if log_fold[row['log']] == fold:
                continue
            if token in selected:
                raise ValueError('A scene cannot count as held-out twice')
            selected[token] = {**row, 'model_training_fold': fold}
    if set(selected) != set(expected):
        raise ValueError('Held-out coverage is incomplete')
    return [selected[row['token']] for row in expected_rows]


def summarize(rows):
    result = {'scenes': len(rows), 'logs': len({row['log'] for row in rows})}
    for channel in ('road_margin_m', 'current_clearance_m', 'future_clearance_m'):
        for suffix, num, den in [
                ('MAE_m', '_abs_error_sum', '_valid_points'),
                ('pair_MAE_m', '_pair_abs_error_sum', '_pair_valid_times'),
                ('direction_accuracy', '_direction_correct_pairs', '_direction_valid_pairs')]:
            numerator = sum(row[channel+num] for row in rows)
            denominator = sum(row[channel+den] for row in rows)
            result[channel+'_'+suffix] = numerator/denominator if denominator else None
            result[channel+'_'+suffix+'_denominator'] = denominator
    for channel in ('current_overlap', 'future_overlap'):
        counts = {key: sum(row[channel+'_'+key] for row in rows) for key in ('TP', 'FN', 'FP', 'TN')}
        result[channel+'_counts'] = counts
        for key, den in [('precision', counts['TP']+counts['FP']), ('recall', counts['TP']+counts['FN'])]:
            result[channel+'_'+key] = counts['TP']/den if den else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous OOF evidence')
    root = args.artifacts/'conditional_oof_v1'
    partition = read(root/'PARTITION.json')
    registry = read(root/'EVALUATIONS.json')
    trigger = read(root/'TRIGGER.json')
    if not trigger['required']:
        raise ValueError('OOF fits must follow the registered diagnostic condition')
    feature = read(args.artifacts/'anchor_diagnostic4096_v1/identity.json')
    label = read(args.artifacts/'labels_diagnostic4096_current_static_ground_kinv2/identity.json')
    expected = feature['index']
    if len(expected) != 4096 or feature['split'] != 'train':
        raise ValueError('Use the original fixed4096 training diagnostic set')
    metadata, pooled, predictions = {}, {}, {}
    initialization = set()
    for kind in ('P_ABS', 'P_DIFF'):
        rows_by_fold = {}
        arrays = {}
        for fold in (0, 1):
            name = f'fold{fold}_{kind}'
            job = next(x for x in registry['jobs'] if x['id'] == name)
            run = root/name
            trained = read(run/'identity.json')
            state = read(run/'COMPLETE.json')
            control = trained['predictor_OOF_control']
            if trained['kind'] != kind or trained['seed'] != 42 or control['training_fold'] != fold:
                raise ValueError('Different registered predictor/fold')
            if control['partition_identity'] != identity(partition):
                raise ValueError('Training partition changed')
            if state['epoch'] != 24 or state['exposure'] != 24*control['training_scenes']:
                raise ValueError('Incomplete fold training is not a24epoch result')
            evaluation = Path(job['output'])
            done = read(evaluation/'COMPLETE.json')
            if done['checkpoint_sha256'] != digest(job['checkpoint']):
                raise ValueError('Evaluation checkpoint identity differs')
            if done['features'] != feature['identity'] or done['labels'] != label['identity'] or done['scenes'] != 4096:
                raise ValueError('Evaluation population differs')
            initialization.add(trained['initial_model_sha256'])
            rows_by_fold[fold] = [x for line in (evaluation/'metrics.jsonl').read_text().splitlines()
                                  for x in [json.loads(line)] if x['condition'] == 'normal']
            with np.load(evaluation/'predictions.npz') as bank:
                if list(bank['tokens']) != [x['token'] for x in expected]:
                    raise ValueError('Prediction arrays are not in the registered scene order')
                arrays[fold] = bank['relations'].copy()
            metadata[name] = {'checkpoint_sha256': done['checkpoint_sha256'],
                'evaluation_complete_sha256': digest(evaluation/'COMPLETE.json'),
                'metrics_sha256': digest(evaluation/'metrics.jsonl'),
                'training_source_sha': trained['source_sha'], 'completed_updates': state['completed'],
                'training_scenes': control['training_scenes'], 'exposure': state['exposure'],
                'GPU_hours': state['GPU_hours']}
        pooled[kind] = merge_heldout(rows_by_fold, partition['log_fold'], expected)
        predictions[kind] = np.stack([arrays[row['model_training_fold']][i] for i, row in enumerate(pooled[kind])])
    if len(initialization) != 1:
        raise ValueError('ABS/DIFF fold controls must share the same initial parameters')
    args.output.mkdir(parents=True)
    for kind, rows in pooled.items():
        with (args.output/(kind+'_metrics.jsonl')).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        np.savez_compressed(args.output/(kind+'_predictions.npz'), relations=predictions[kind],
                            tokens=np.array([x['token'] for x in expected]))
    atomic_json(args.output/'summary.json', {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'partition_sha256': digest(root/'PARTITION.json'), 'trigger_sha256': digest(root/'TRIGGER.json'),
        'models': metadata, 'results': {kind: summarize(rows) for kind, rows in pooled.items()},
        'aggregation': 'One opposite-fold output per original scene;4096 scenes, not8192 independent observations',
        'limits': ['Only the predictor readout is out-of-fold; S0 saw all original training logs',
                   'Half-population fits differ from full-population primary fits; this is not a pure full-data overfit decomposition',
                   'No new refiner was trained and no OOF result was used to select a main checkpoint']})


if __name__ == '__main__':
    main()
