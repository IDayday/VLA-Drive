"""Read-only projection coverage versus actual selection errors on stage_val.

Projection coverage is neither occlusion truth nor future-obstacle visibility.
The fixed bins describe evidence; they never change sampling or model selection.
No image features, model weights, optimizer, new official score or test data are
needed. The existing calibration sidecar still validates current observations.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, lock, read_json
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.metrics import select
from .calibration_bank import PreparedCalibrationReader
from .full import load_config, resource_check
from .geometry import candidate_anchors, project_points


def coverage_groups(fraction):
    return {
        'none': fraction == 0,
        'some_below_half': (fraction > 0) & (fraction < .5),
        'half_to_below_all': (fraction >= .5) & (fraction < 1),
        'all': fraction == 1,
    }


def selection_accounting(values, bank, metadata, waypoint_fraction):
    choices = select(values, bank['valid'], expert_ids=metadata['provenance']['sources'])
    index = np.arange(len(values))
    scores = bank['scores'].astype(np.float64)
    rewards = scores[index, choices]
    gain = rewards - scores[:, 0]
    positive, negative = gain.clip(min=0), (-gain).clip(min=0)
    high_zero = (scores[:, 0] >= .8) & (rewards <= 1e-8)
    wrong = gain < -1e-8
    fraction = waypoint_fraction[index, choices]
    rows = []
    for name, subset in coverage_groups(fraction).items():
        count = int(subset.sum())
        rows.append(dict(group=name, scenes=count, wrong_replacements=int((wrong & subset).sum()),
                         high_to_zero=int((high_zero & subset).sum()),
                         positive_gain_population_01=float(positive[subset].sum()/len(values)),
                         negative_cost_population_01=float(negative[subset].sum()/len(values)),
                         negative_cost_share=float(negative[subset].sum()/negative.sum())
                         if negative.sum() else None,
                         mean_selected_01=float(rewards[subset].mean()) if count else None))
    require(sum(x['scenes'] for x in rows) == len(values), 'coverage groups must partition all scenes')
    require(abs(sum(x['negative_cost_population_01'] for x in rows)-negative.mean()) < 1e-12,
            'negative cost accounting failed')
    return dict(rows=rows, wrong_replacements=int(wrong.sum()), high_to_zero=int(high_zero.sum()),
                selected_01=float(rewards.mean()), gain_01=float(gain.mean()),
                positive_gain_01=float(positive.mean()), negative_cost_01=float(negative.mean()))


def run(config):
    resource_check(config)
    torch.set_num_threads(2)
    source = config.source()
    choice_path = config.root/'MODEL_CHOICE.json'
    choice = read_json(choice_path)
    require(choice['selection_role'] == 'stage_val' and not choice['dev_used_for_this_choice'],
            'requires the already locked validation choice')
    step = choice['step']
    prediction = config.root/f'scorer/predictions_stage_val_{step:06d}.npz'
    validation = read_json(config.root/f'scorer/validation_{step:06d}.json')
    require(file_hash(prediction) == validation['prediction_hash'], 'prediction checksum mismatch')
    metadata, bank = load_bank(source, 'deploy', 'stage_val')
    require(metadata['hash'] == validation['bank_hash'] and bank['valid'].all(),
            'audit requires the complete fixed valid bank')
    scenes = scenes_by_role(source)['stage_val']
    require([s.scene_id for s in scenes] == metadata['scene_ids'], 'scene order mismatch')
    identity = dict(role='stage_val', scenes=len(scenes), bank_hash=metadata['hash'],
                    choice_hash=file_hash(choice_path), prediction_hash=file_hash(prediction),
                    calibration_hash=file_hash(Path(config.calibration_bank)/'MANIFEST.json'),
                    control_receipt_hash=file_hash(config.root/'qualification/global_S11_control/COMPLETE.json'),
                    code_hash=file_hash(__file__), bins=['0', '(0,.5)', '[.5,1)', '1'],
                    prediction_overestimate_threshold=.8, zero_epsilon=1e-8,
                    visibility_definition='each waypoint has at least one z=0/1 anchor in any original camera',
                    scope='descriptive association, not causal attribution or a selection rule',
                    no_model_updates=True, no_new_scores=True, no_dev_or_test=True)
    output = config.root/'qualification/visibility_risk'
    with lock(output/'RUN.lock'):
        atomic_json(output/'PLAN.json', identity, immutable=True)
        if (output/'REPORT.json').exists():
            report = read_json(output/'REPORT.json')
            require(report['identity'] == identity, 'stale risk-coverage report')
            return report
        reader = PreparedCalibrationReader(source, config.calibration_bank)
        waypoint_fraction, anchor_fraction = [], []
        for start in range(0, len(scenes), 64):
            batch = scenes[start:start+64]
            camera = reader.batch(batch, 'cpu')
            trajectory = torch.from_numpy(bank['trajectories'][start:start+len(batch)])
            points = candidate_anchors(trajectory)
            _, inside = project_points(points, camera['camera_intrinsics'], camera['camera_to_ego'],
                                       camera['camera_image_sizes'])
            anchors = inside.any(2)
            waypoint_fraction.append(anchors.reshape(len(batch), trajectory.shape[1], 8, 2)
                                     .any(-1).float().mean(-1).numpy())
            anchor_fraction.append(anchors.float().mean(-1).numpy())
        waypoint_fraction = np.concatenate(waypoint_fraction)
        anchor_fraction = np.concatenate(anchor_fraction)
        with np.load(prediction) as f:
            values = f['values'].copy()
        require(values.shape == bank['scores'].shape == waypoint_fraction.shape, 'coverage alignment')
        candidate_rows = []
        zero = bank['scores'] <= 1e-8
        for name, subset in coverage_groups(waypoint_fraction).items():
            count = int(subset.sum())
            zeros = zero & subset
            failures = {}
            for j, metric in enumerate(metadata['components']):
                valid = subset & bank['component_valid'][..., j]
                failures[metric] = dict(valid=int(valid.sum()),
                                       below_full=int((valid & (bank['components'][..., j] < 1-1e-8)).sum()))
            candidate_rows.append(dict(group=name, candidates=count, zero_candidates=int(zeros.sum()),
                                       high_prediction_zero=int((zeros & (values > .8)).sum()),
                                       mean_prediction_on_zero=float(values[zeros].mean()) if zeros.any() else None,
                                       components=failures))
        require(sum(r['candidates'] for r in candidate_rows) == values.size, 'candidate partition')
        selected = selection_accounting(values, bank, metadata, waypoint_fraction)
        require(abs(selected['selected_01']-validation['selected_01']) < 1e-10,
                'selection reproduction differs')
        control_root = config.root/'qualification/global_S11_control'
        receipt = read_json(control_root/'COMPLETE.json')
        require(receipt['bank_hash'] == metadata['hash'] and
                file_hash(control_root/'predictions.npz') == receipt['prediction_hash'], 'control identity')
        with np.load(control_root/'predictions.npz') as f:
            control = selection_accounting(f['values'], bank, metadata, waypoint_fraction)
        report = dict(status='COMPLETE', identity=identity, utc=datetime.now(timezone.utc).isoformat(),
                      selected_step=step, candidates=candidate_rows, geometric=selected, global_S11=control,
                      average_anchor_fraction=anchor_fraction.mean(0).tolist(),
                      average_waypoint_fraction=waypoint_fraction.mean(0).tolist(),
                      limitations=['in-frame does not prove unoccluded, correctly located or future-visible',
                                   'height hypotheses are not measured road surface',
                                   'coverage correlates with trajectory length and scene difficulty',
                                   'this audit does not establish causality or permission to tune on dev'])
        atomic_json(output/'REPORT.json', report, immutable=True)
        return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    print(json.dumps(run(load_config(a.config)), allow_nan=False))


if __name__ == '__main__':
    main()
