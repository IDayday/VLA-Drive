"""Physical paired drift matching and ability-retention summaries, not output KL."""
import math
import numpy as np


def physical_drift(reference, current):
    old, new = np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if old.shape != new.shape or old.ndim != 3 or old.shape[1:] != (8, 3) or not (np.isfinite(old).all() and np.isfinite(new).all()):
        raise ValueError('Paired common-root physical requested trajectories [scene,8,3] required')
    heading = new[..., 2]-old[..., 2]
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    return dict(mean_ADE_m=float(np.linalg.norm(new[..., :2]-old[..., :2], axis=-1).mean()),
                RMS_heading_rad=float(np.sqrt(np.mean(heading**2))), scenes=len(old),
                interpretation='common-root requested trajectory drift proxy; not output-distribution KL')


def select_conservative_trial(trials, target, *, relative_tolerance=.2, absolute_tolerance=1e-5, maximum_trials=2):
    if not 1 <= len(trials) <= maximum_trials:
        raise ValueError('TRAIN-only bounded P1 calibration, not unlimited search')
    keys = ('mean_ADE_m', 'RMS_heading_rad')
    if any(t.get('split') != 'TRAIN' for t in trials):
        raise ValueError('No DEV-based conservative update matching')
    def error(trial):
        return max(abs(trial['drift'][k]-target[k])/max(abs(target[k]), absolute_tolerance) for k in keys)
    if any(not math.isfinite(error(t)) for t in trials):
        raise ValueError('Invalid measured drift')
    best = min(trials, key=lambda t: (error(t), t['learning_rate']))
    return dict(status='MATCHED' if error(best) <= relative_tolerance else 'NOT_MATCHED',
                selected_learning_rate=best['learning_rate'], selected_experiment_id=best['experiment_id'],
                maximum_relative_error=error(best), tolerance=relative_tolerance,
                target=target, selected_drift=best['drift'], trials=len(trials), split='TRAIN',
                interpretation='If NOT_MATCHED the slower-learning explanation remains unresolved')


def retention_summary(reference_rewards, current_rewards, reference_safe, *, high_threshold=.9, useful_delta=.005, fixed_high_mask=None):
    old, new, safe = np.asarray(reference_rewards), np.asarray(current_rewards), np.asarray(reference_safe, dtype=bool)
    if old.shape != new.shape or old.shape != safe.shape or old.ndim != 1 or not (np.isfinite(old).all() and np.isfinite(new).all()):
        raise ValueError('Paired fixed reference population required')
    if (old < 0).any() or (old > 1).any() or (new < 0).any() or (new > 1).any():
        raise ValueError('Retention uses original 0-1 score units')
    delta = new-old;high = old >= high_threshold if fixed_high_mask is None else np.asarray(fixed_high_mask, dtype=bool)
    if high.shape != old.shape:
        raise ValueError('Fixed multi-draw reference strata must preserve scene identity')
    return dict(scenes=len(old), fixed_high_scenes=int(high.sum()), fixed_safe_scenes=int(safe.sum()),
                high_to_zero=int((high & (new == 0)).sum()), high_any_drop=int((high & (delta < 0)).sum()),
                high_practical_drop=int((high & (delta <= -useful_delta)).sum()),
                old_low_practical_improved=int(((old < .6) & (delta >= useful_delta)).sum()),
                old_low_to_high=int(((old < .6) & (new >= high_threshold)).sum()),
                zero_to_nonzero=int(((old == 0) & (new > 0)).sum()),
                G_plus_raw_sum=float(delta[delta > 0].sum()), G_minus_raw_sum=float(-delta[delta < 0].sum()),
                net_mean_raw=float(delta.mean()), net_PDMS_points=float(100*delta.mean()),
                high_mask_source='fixed_multi_draw_stratum' if fixed_high_mask is not None else 'single_reference_score',
                reference_population_redefined=False, zero_events_not_a_zero_failure_probability=True)
