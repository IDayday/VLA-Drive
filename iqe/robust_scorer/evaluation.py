"""Read-only model choice, calibration and complete-population reporting.

No optimizer is constructed here. All four heads must finish before the choice
is locked. Development labels are loaded only after that immutable lock.
"""
from pathlib import Path
import csv
import io
import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_bytes, atomic_json, read_json, file_hash, digest, lock
from iqe.evaluation.retention import module_hash
from iqe.internal_baseline.evaluation import paired_bootstrap
from iqe.specialist_study.bank import load_bank, pool
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.metrics import report_metrics, select
from iqe.specialist_study.training import runtime
from .config import CELLS
from .runner import binding, create, predict_bank


def chosen_endpoint(config, cell, device='cpu'):
    folder = config.root/'scorers'/cell
    result = read_json(folder/'result.json')
    require(result['status'] == 'COMPLETE' and result['optimizer_steps'] == config.updates,
            'fixed-budget cell incomplete')
    require(result['termination']['terminal'], 'Scorer has not reached a terminal checkpoint')
    require(file_hash(folder/'best.pt') == result['best_checkpoint_hash'], 'best snapshot checksum changed')
    model, identity = create(config, cell)
    require(identity == read_json(folder/'SCORER_INITIALIZATION.json'), 'Scorer initializer/lineage changed')
    model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False), result


def lock_choice(config):
    bound = binding(config)
    with lock(config.root/'MODEL_CHOICE.lock'):
        rows = []
        for cell in CELLS:
            model, result = chosen_endpoint(config, cell)
            folder = config.root/'scorers'/cell
            validations = [(step, read_json(folder/f'validation_{step:06d}.json'))
                           for step in config.validation_steps]
            step, stats = max(validations, key=lambda x: (x[1]['selection_key'], -x[0]))
            require(stats['scorer_state_hash'] == module_hash(model), 'best model/validation identity mismatch')
            require(stats['selection_key'] == result['best_selection_key'], 'best validation changed')
            rows.append(dict(cell=cell, step=step, checkpoint=str(folder/'best.pt'),
                             checkpoint_hash=result['best_checkpoint_hash'], state_hash=module_hash(model),
                             metrics=stats, validation_hash=file_hash(folder/f'validation_{step:06d}.json')))
        winner = max(rows, key=lambda r: (r['metrics']['selection_key'], -r['step'], -list(CELLS).index(r['cell'])))
        result = dict(experiment_hash=digest(bound), selection_role='stage_val', cells=rows, selected=winner,
                      pool_hash=pool(config.source())['hash'], dev_used_for_selection=False,
                      tie_break='fewer new NC+DAC, earlier endpoint, fixed cell order')
        atomic_json(config.root/'MODEL_CHOICE.json', result, immutable=True)
        return result


def score_diagnostics(prediction, bank):
    p = np.asarray(prediction['values'], dtype=np.float64)
    r = np.asarray(bank['scores'], dtype=np.float64)
    valid = np.asarray(bank['valid'], bool) & np.isfinite(r)
    require(p.shape == r.shape and np.isfinite(p[valid]).all(), 'invalid scoring predictions')
    result = {'valid_candidates': int(valid.sum()), 'mse': float(((p-r)[valid]**2).mean()),
              'mean_error': float((p-r)[valid].mean()), 'reliability_bins': [],
              'zero_true_mean_prediction': float(p[valid & (r <= 1e-8)].mean()) if (valid & (r <= 1e-8)).any() else None}
    clipped = np.clip(p, 1e-7, 1-1e-7)
    result['soft_target_cross_entropy'] = float((-(r*np.log(clipped)+(1-r)*np.log1p(-clipped)))[valid].mean())
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = valid & (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        result['reliability_bins'].append({'lower': float(lo), 'upper': float(hi), 'count': int(mask.sum()),
                                         'prediction': float(p[mask].mean()) if mask.any() else None,
                                         'truth': float(r[mask].mean()) if mask.any() else None})
    result['binned_mean_calibration_error'] = sum(
        row['count']*abs(row['prediction']-row['truth']) for row in result['reliability_bins'] if row['count'])/int(valid.sum())
    result['candidate_columns'] = []
    for index in range(r.shape[1]):
        good = valid[:, index]
        zero = good & (r[:, index] <= 1e-8)
        result['candidate_columns'].append(dict(index=index, valid=int(good.sum()), zeros=int(zero.sum()),
            predicted_mean=float(p[good, index].mean()) if good.any() else None,
            true_mean=float(r[good, index].mean()) if good.any() else None,
            mean_prediction_on_zero=float(p[zero, index].mean()) if zero.any() else None))
    i, j = np.triu_indices(r.shape[1], 1)
    gap = r[:, i]-r[:, j]
    good = valid[:, i] & valid[:, j] & (np.abs(gap) > .01)
    pair_count = valid.sum(1)*(valid.sum(1)-1)/2
    weight = np.where(good, np.abs(gap)/np.maximum(pair_count[:, None], 1), 0)
    correct = (p[:, i]-p[:, j])*gap > 0
    result['comparison_pairs'] = int(good.sum())
    result['gap_weighted_pair_accuracy'] = float((weight*correct).sum()/weight.sum()) if weight.sum() else None
    if 'nonzero_logits' in prediction:
        logits = np.asarray(prediction['nonzero_logits'], dtype=np.float64)
        probability = 1/(1+np.exp(-np.clip(logits, -80, 80)))
        result['joint_nonzero_brier'] = float(((probability-(r > 1e-8))[valid]**2).mean())
    return result


def information_diagnostics(config, device):
    """Read-only corruptions, never new trajectory labels or selection data."""
    from iqe.specialist_study.scorer_training import predict
    choice = read_json(config.root/'MODEL_CHOICE.json')
    cell = choice['selected']['cell']
    dependencies = dict(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
                        checkpoint_hash=choice['selected']['checkpoint_hash'])
    path = config.root/'INFORMATION_DIAGNOSTICS.json'
    if path.exists():
        result = read_json(path)
        require(result['dependencies'] == dependencies, 'stale information diagnostic')
        return result
    model, _ = chosen_endpoint(config, cell, device)
    source = config.source()
    rows = {}
    for kind in (None, 'scene_mismatch', 'trajectory_time_shuffle', 'remove_W'):
        meta, bank, values = predict(source, model, 'stage_val', device, diagnostic=kind)
        rows[kind or 'correct_pairing'] = statistics(config, meta, bank, {'values': values})
    result = dict(dependencies=dependencies, role='fixed_stage_val_after_choice_lock', rows=rows,
                  new_optimizer_steps=0, used_to_change_choice_or_recipe=False,
                  labels='original physical candidate bank; predictions alone are deliberately corrupted',
                  trajectory_corruptions_are_new_policy_results=False)
    atomic_json(path, result, immutable=True)
    return result


def predictions(config, cell, role, device):
    require(role in ('scorer_fit', 'stage_val', 'selector_cal', 'dev_report'), 'no test access in this experiment')
    selection = read_json(config.root/'MODEL_CHOICE.json')
    require(selection['experiment_hash'] == digest(binding(config)), 'selection experiment changed')
    row = next(x for x in selection['cells'] if x['cell'] == cell)
    source = config.source()
    meta, bank = load_bank(source, 'deploy', role)
    scenes = scenes_by_role(source)[role]
    require(meta['scene_ids'] == [s.scene_id for s in scenes], 'prediction/scene join mismatch')
    identity = dict(cell=cell, role=role, checkpoint_hash=row['checkpoint_hash'], bank_hash=meta['hash'],
                    selection_hash=file_hash(config.root/'MODEL_CHOICE.json'), requested_scenes=len(scenes))
    folder = config.root/'evaluation'/cell/role
    with lock(folder/'PREDICT.lock'):
        done = folder/'COMPLETE.json'
        path = folder/'predictions.npz'
        if done.exists():
            receipt = read_json(done)
            require(receipt['identity'] == identity and file_hash(path) == receipt['checksum'], 'stale/corrupt prediction cache')
            with np.load(path) as f:
                return meta, bank, {k: f[k].copy() for k in f.files}
        model, _ = chosen_endpoint(config, cell, device)
        pred = predict_bank(model, scenes, bank, ReusedContext(source, 128), device)
        require(len(pred['values']) == len(scenes), 'incomplete prediction population')
        stream = io.BytesIO()
        np.savez(stream, **pred)
        atomic_bytes(path, stream.getvalue(), immutable=True)
        atomic_json(done, dict(identity=identity, checksum=file_hash(path), all_predictions_finite=True), immutable=True)
        return meta, bank, pred


def statistics(config, meta, bank, prediction, margin=0.):
    source = config.source()
    chosen = select(prediction['values'], bank['valid'], margin, expert_ids=meta['provenance']['sources'])
    result = report_metrics(bank['scores'], bank['components'], chosen, meta['source_logs'],
                            seed=source.bootstrap_seed, repetitions=source.bootstrap_repetitions)
    fixed_index = int(bank['scores'].mean(0).argmax())
    delta = bank['scores'][np.arange(len(chosen)), chosen]-bank['scores'][:, fixed_index]
    result['gain_over_best_fixed_ci'] = paired_bootstrap(delta, meta['source_logs'], source.bootstrap_repetitions, source.bootstrap_seed)
    result['prediction_diagnostics'] = score_diagnostics(prediction, bank)
    return result


def calibrate(config, device='cuda:0'):
    choice = read_json(config.root/'MODEL_CHOICE.json')
    rules = dict(grid=[0., .01, .02, .05, .1], new_NC=0, new_DAC=0, high_to_zero=0,
                 role='selector_cal', rule='best mean; fewer switches; larger margin',
                 fallback_when_no_feasible_margin='always Base, not a Scorer success',
                 unknown_scene_safety_guarantee=False, choice_hash=file_hash(config.root/'MODEL_CHOICE.json'))
    atomic_json(config.root/'MARGIN_RULES.json', rules, immutable=True)
    meta, bank, pred = predictions(config, choice['selected']['cell'], 'selector_cal', device)
    rows = []
    for delta in rules['grid']:
        stats = statistics(config, meta, bank, pred, delta)
        feasible = stats['high_to_zero'] == 0 and all(stats['safety'][k]['new_failures'] == 0
                    for k in ('no_at_fault_collisions', 'drivable_area_compliance'))
        rows.append(dict(delta=delta, feasible=feasible, metrics=stats))
    valid = [r for r in rows if r['feasible']]
    best = max(valid, key=lambda r: (r['metrics']['selected_01'], -r['metrics']['switch_count'], r['delta'])) if valid else None
    result = dict(cell=choice['selected']['cell'], choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
                  rules_hash=file_hash(config.root/'MARGIN_RULES.json'), selected_delta=best['delta'] if best else None,
                  always_Base=best is None or best['metrics']['switch_count'] == 0, rows=rows)
    atomic_json(config.root/'SELECTOR_MARGIN.json', result, immutable=True)
    return result


def evaluate(config, device='cuda:0'):
    device = runtime(device)
    choice = lock_choice(config)
    margin = calibrate(config, device)
    information_diagnostics(config, device)
    source = config.source()
    frozen = dict(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'), margin_hash=file_hash(config.root/'SELECTOR_MARGIN.json'),
                  dev_scene_ids_hash=digest([s.scene_id for s in scenes_by_role(source)['dev_report']]),
                  training_completed=True, test_access=False)
    atomic_json(config.root/'FINAL_EVALUATION_LOCK.json', frozen, immutable=True)
    result = {'role': 'entire_fixed_dev_report', 'cells': {}, 'fit_diagnostics': {}, 'bindings': frozen, 'units': '[0,1]'}
    for cell in CELLS:
        meta, bank, pred = predictions(config, cell, 'dev_report', device)
        result['cells'][cell] = statistics(config, meta, bank, pred)
        fm, fb, fp = predictions(config, cell, 'scorer_fit', device)
        result['fit_diagnostics'][cell] = statistics(config, fm, fb, fp)
        atomic_json(config.root/'evaluation'/cell/'DEV_RESULT.json', result['cells'][cell], immutable=True)
        atomic_json(config.root/'evaluation'/cell/'FIT_DIAGNOSTIC.json', result['fit_diagnostics'][cell], immutable=True)
    cell = choice['selected']['cell']
    meta, bank, pred = predictions(config, cell, 'dev_report', device)
    source = config.source()
    selected = np.zeros(len(bank['scores']), dtype=int) if margin['selected_delta'] is None else select(
        pred['values'], bank['valid'], margin['selected_delta'], expert_ids=meta['provenance']['sources'])
    result['margin'] = report_metrics(bank['scores'], bank['components'], selected, meta['source_logs'],
                                      seed=source.bootstrap_seed, repetitions=source.bootstrap_repetitions)
    atomic_json(config.root/'FINAL_RESULTS.json', result, immutable=True)
    rows = []
    fields = ['cell', 'base_01', 'oracle_01', 'selected_01', 'selected_gain_01', 'selected_over_best_fixed_01',
              'oracle_capture', 'regret_01', 'selected_zero', 'high_to_zero', 'switch_rate']
    for name, stats in result['cells'].items():
        rows.append(dict(cell=name, **{key: stats[key] for key in fields[1:]}))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(config.root/'FINAL_RESULTS.csv', stream.getvalue().encode(), immutable=True)
    raw = result['cells'][cell]
    risk = raw['high_to_zero'] > 0 or any(raw['safety'][k]['new_failures'] for k in raw['safety'])
    verdict = dict(selected_cell=cell, argmax_gain_01=raw['selected_gain_01'],
                   gain_over_best_fixed_01=raw['selected_over_best_fixed_01'], risk_regression=risk,
                   log_CI_excludes_zero=raw['gain_ci']['lower_95'] > 0,
                   comparison_signal='POSITIVE_HELDOUT_SIGNAL' if raw['selected_over_best_fixed_01'] > 0 else 'CANDIDATE_GAIN_NOT_REALIZED',
                   empirical_heldout_qualification=(raw['gain_ci']['lower_95'] > 0
                       and raw['gain_over_best_fixed_ci']['lower_95'] > 0 and not risk),
                   universal_robustness_claim=False, multi_seed='UNTESTED', ACTIVE_updated=False,
                   external_scorer_weights_loaded=False, new_navtest=False)
    atomic_json(config.root/'VERDICT.json', verdict, immutable=True)
    return verdict


def main(argv=None):
    import argparse
    import json
    from .config import load
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['lock-choice', 'calibrate', 'evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args(argv)
    config = load(a.config)
    result = {'lock-choice': lambda: lock_choice(config), 'calibrate': lambda: calibrate(config, runtime(a.device)),
              'evaluate': lambda: evaluate(config, a.device)}[a.command]()
    print(json.dumps({'status': 'COMPLETE', 'result': result}, allow_nan=False))


if __name__ == '__main__':
    main()
