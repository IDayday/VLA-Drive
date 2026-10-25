"""Read existing canonical feedback without inventing missing repair labels.

This module is a label contract, not an actor, learned critic, or new scorer.
Requested local poses and simulated global states remain different variables.
"""
from dataclasses import dataclass
import numpy as np


SAFETY = ('no_at_fault_collisions', 'drivable_area_compliance',
          'time_to_collision_within_bound')
METRICS = (*SAFETY, 'ego_progress', 'comfort', 'driving_direction_compliance', 'score')


@dataclass
class ObservedFeedback:
    requested_local_poses: np.ndarray
    simulated_global_states: np.ndarray
    official_metrics: dict
    available: dict
    labels: dict


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Invalid {name}: expected finite {shape}')
    return result


def _binary(value, name):
    result = _array(value, (41,), name)
    if not np.isin(result, [0., 1.]).all():
        raise ValueError(f'Nonbinary {name}')
    return result.astype(bool)


def _event(record, key):
    # Present null means the official scorer observed no such event. Absent
    # means there is no annotation; it is never silently treated as no event.
    if key not in record:
        return False, None
    value = record[key]
    if value is not None:
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 4 or abs(value * 10 - round(value * 10)) > 1e-7:
            raise ValueError(f'Invalid official event time: {key}')
    return True, value


def read_feedback(record):
    if record.get('status') != 'OK':
        raise ValueError('Infrastructure errors are not negative driving labels')
    metrics = {key: float(record['metrics'][key]) for key in METRICS}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in metrics.values()):
        raise ValueError('Invalid canonical components')
    expected = (metrics[SAFETY[0]] * metrics[SAFETY[1]] *
                (5 * metrics['ego_progress'] + 5 * metrics[SAFETY[2]] + 2 * metrics['comfort']) / 12)
    if abs(expected - metrics['score']) > 1e-10:
        raise ValueError('Not the registered canonical 5:5:2 readout')
    requested = _array(record['requested_trajectory'], (8, 3), 'requested local poses')
    simulated = _array(record['simulated_states'], (41, 11), 'simulated global states')
    available = dict(requested_local_poses=True, simulated_global_states=True,
                     official_subscores=True, road_body_mask=False,
                     first_at_fault_collision=False, first_ttc_rule_infraction=False,
                     full_collision_mask=False, full_ttc_rule_mask=False,
                     continuous_geometric_margins=False, raw_progress_m=False,
                     progress_time_profile=False, verified_edit_direction=False)
    labels = dict(physical_time_seconds=np.arange(41, dtype=float) * .1,
                  request_time_seconds=np.arange(1, 9, dtype=float) * .5,
                  road_body_outside_mask=None, first_at_fault_collision_seconds=None,
                  first_ttc_rule_infraction_seconds=None, full_collision_mask=None,
                  full_ttc_rule_mask=None, raw_progress_m=None,
                  repair_direction=None)
    if 'non_drivable_body_by_time' in record:
        labels['road_body_outside_mask'] = _binary(record['non_drivable_body_by_time'], 'road mask')
        available['road_body_mask'] = True
        # Official DAC includes t=0; future action-dependent summaries do not.
        if (labels['road_body_outside_mask'].any()) != (metrics[SAFETY[1]] < 1):
            raise ValueError('Official DAC and body mask disagree')
    for source_key, flag, target, metric in [
        ('official_first_at_fault_collision_seconds', 'first_at_fault_collision',
         'first_at_fault_collision_seconds', SAFETY[0]),
        ('official_first_ttc_infraction_seconds', 'first_ttc_rule_infraction',
         'first_ttc_rule_infraction_seconds', SAFETY[2]),
    ]:
        present, value = _event(record, source_key)
        available[flag], labels[target] = present, value
        if present and ((value is not None) != (metrics[metric] < 1)):
            raise ValueError(f'Official {metric} and event time disagree')
    # Only explicit full-window annotations qualify. A first event is not a
    # duration mask and cannot recover a full collision/TTC event sequence.
    for source, key, metric, first in [
        ('official_at_fault_collision_by_time', 'full_collision_mask', SAFETY[0], 'first_at_fault_collision_seconds'),
        ('official_ttc_rule_infraction_by_time', 'full_ttc_rule_mask', SAFETY[2], 'first_ttc_rule_infraction_seconds'),
    ]:
        if key in record:
            raise ValueError('Ambiguous event mask: require explicit official rule semantics')
        if source in record:
            labels[key] = _binary(record[source], source)
            available[key] = True
            indices = np.flatnonzero(labels[key])
            if bool(len(indices)) != (metrics[metric] < 1):
                raise ValueError('Full official mask and metric disagree')
            if labels[first] is not None and (not len(indices) or abs(indices[0] * .1 - labels[first]) > 1e-7):
                raise ValueError('Full official mask and first event disagree')
    if 'raw_progress_meters' in record:
        progress = float(record['raw_progress_meters'])
        if not np.isfinite(progress) or progress < 0:
            raise ValueError('Invalid native raw progress')
        available['raw_progress_m'] = True
        labels['raw_progress_m'] = progress
    return ObservedFeedback(requested, simulated, metrics, available, labels)


def candidate_evidence(records, useful_gain=.005):
    """Finite-pool evidence, not a verified local edit or deployment success.

    No alternative found is UNKNOWN, not proof that a scene cannot be repaired.
    A slower original is never labelled wrong solely because it is slower.
    """
    if not np.isfinite(useful_gain) or useful_gain <= 0:
        raise ValueError('Expected a positive raw-score gain threshold')
    parsed = [read_feedback(r) for r in records]
    score = np.array([r.official_metrics['score'] for r in parsed])
    progress = np.array([r.official_metrics['ego_progress'] for r in parsed])
    feasible = np.array([all(r.official_metrics[k] == 1 for k in SAFETY) for r in parsed], dtype=bool)
    if not len(parsed):
        raise ValueError('Empty scene is not a failed trajectory')
    has_better = np.zeros(len(parsed), dtype=bool)
    has_better_progress = np.zeros(len(parsed), dtype=bool)
    for i in range(len(parsed)):
        # Same-scene alternatives are independently scored; these comparisons
        # never change the official EP proposal normalization contract.
        qualified = feasible & (score >= score[i] + useful_gain)
        has_better[i] = qualified.any()
        has_better_progress[i] = (qualified & (progress > progress[i] + 1e-12)).any()
    return dict(parsed=parsed, feasible=feasible, score=score,
                has_safe_higher_scoring_alternative=has_better,
                has_safe_higher_scoring_higher_EP_alternative=has_better_progress)
