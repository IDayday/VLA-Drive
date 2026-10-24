"""Observed canonical process feedback; missing is never an absence of risk."""
import numpy as np


SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound')
NOVELTY_NAMES = ('raw_progress_m', 'mean_future_speed_mps', 'first_braking_seconds_or_horizon', 'maximum_jerk_mps3')


def process_summary(record):
    if record.get('status') != 'OK':
        raise ValueError('Infrastructure failure cannot be used as physical feedback')
    metrics = record['metrics']
    if any(k not in metrics or not np.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1 for k in SAFETY):
        raise ValueError('Official safety components are missing or invalid')
    available = dict(full_window_collision_mask=False, full_window_ttc_mask=False,
                     official_first_collision='official_first_at_fault_collision_seconds' in record,
                     official_first_ttc_infraction='official_first_ttc_infraction_seconds' in record,
                     non_drivable_mask='non_drivable_body_by_time' in record,
                     simulated_dynamics='simulated_states' in record,
                     continuous_geometric_margins=False,
                     raw_progress='raw_progress_meters' in record)
    values = dict(official_first_at_fault_collision_seconds=record.get('official_first_at_fault_collision_seconds'),
                  official_first_ttc_infraction_seconds=record.get('official_first_ttc_infraction_seconds'))
    # These canonical event times are diagnostics only. In particular, the TTC
    # field is a first RULE-INFRACTION time, not a minimum time-to-collision.
    if not available['non_drivable_mask'] or not available['simulated_dynamics']:
        raise ValueError('F3 requires canonical full-window body mask and simulated states; no zero fill')
    mask = np.asarray(record['non_drivable_body_by_time'])
    states = np.asarray(record['simulated_states'], dtype=np.float64)
    if mask.shape != (41,) or states.shape != (41, 11) or not np.isfinite(states).all():
        raise ValueError('Canonical physical-time feedback must have 41 states at 0.1 seconds')
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('Invalid drivable body mask')
    speed = np.linalg.norm(states[:, 3:5], axis=1)
    acceleration = states[:, 5:7]
    jerk = np.gradient(acceleration, .1, axis=0, edge_order=2)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    # t=0 is immutable and excluded from all action-dependent means.
    outside = float(mask[1:].mean())
    dynamic_excess = np.clip((acc_magnitude[1:]-8.)/8., 0., 1.)
    dynamic_excess = np.maximum(dynamic_excess, np.clip((jerk_magnitude[1:]-8.)/8., 0., 1.))
    longitudinal_acc = np.sum(acceleration*states[:, 3:5], axis=1)/np.maximum(speed, .1)
    braking = np.flatnonzero(longitudinal_acc[1:] < -.5)
    first_braking = float((braking[0]+1)*.1) if len(braking) else None
    values.update(non_drivable_future_fraction=outside, non_drivable_future_duration_seconds=outside*4.,
                  mean_future_speed_mps=float(speed[1:].mean()), stop_future_fraction=float((speed[1:] < .2).mean()),
                  maximum_acceleration_mps2=float(acc_magnitude[1:].max()), maximum_jerk_mps3=float(jerk_magnitude[1:].max()),
                  mean_bounded_dynamic_excess=float(dynamic_excess.mean()), first_braking_seconds=first_braking,
                  driving_time_seconds=np.arange(41).astype(float).tolist())
    values['driving_time_seconds'] = (np.arange(41)*.1).tolist()
    costs = [1-float(metrics[k]) for k in SAFETY]+[outside, float(dynamic_excess.mean())]
    feasible = all(float(metrics[k]) == 1. for k in SAFETY)
    novelty = None
    if available['raw_progress']:
        progress = float(record['raw_progress_meters'])
        if not np.isfinite(progress):
            raise ValueError('Nonfinite native progress')
        values['raw_progress_m'] = progress
        novelty = [progress, values['mean_future_speed_mps'], first_braking if first_braking is not None else 4.,
                   values['maximum_jerk_mps3']]
    return dict(available_mask=available, physical_feedback=values, feasible=feasible,
                search_violation_cost=float(np.mean(costs)), search_cost_components=costs,
                novelty_features=novelty,
                semantics='Native metric deficits, full future body-outside duration and bounded simulated dynamic excess; not a risk probability or a modified official PDMS')


class NoveltyScale:
    """Fixed TRAIN-only robust scales; dimensions are not fitted on evaluation."""
    def __init__(self, center, scale):
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.center.shape != (4,) or self.scale.shape != (4,) or not np.isfinite([self.center, self.scale]).all() or (self.scale <= 0).any():
            raise ValueError('Frozen four-feature TRAIN normalization required')

    @classmethod
    def fit_train(cls, rows, *, split):
        if split != 'TRAIN':
            raise ValueError('Novelty scaling cannot fit DEV or Navtest')
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows) or not np.isfinite(rows).all():
            raise ValueError('Invalid TRAIN novelty calibration')
        return cls(np.median(rows, axis=0), np.maximum(np.quantile(rows, .95, axis=0)-np.quantile(rows, .05, axis=0), [1., .5, .25, .5]))

    def encode(self, values):
        return (np.asarray(values, dtype=np.float64)-self.center)/self.scale

    def identity(self):
        return dict(features=list(NOVELTY_NAMES), center=self.center.tolist(), scale=self.scale.tolist(), fitted_split='TRAIN')
