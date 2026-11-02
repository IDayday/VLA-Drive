"""Real-field availability and strict feasible precedence, CPU only."""
import copy
import pytest
import torch

from starVLA.rl.qwen_world_performance.temporal_credit import (
    safetyep_parts, temporal_credit_parts, real_temporal_cost)
from starVLA.rl.qwen_world_safety_exploration.credit import raw_advantage_parts


def records(components):
    names = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'ego_progress')
    groups = []
    for group in components:
        rows = []
        for i, c in enumerate(group):
            flags = torch.zeros(41, dtype=torch.bool)
            if c[1] < 1:flags[1:2+i % 16] = True
            rows.append(dict(status='OK', metrics=dict(zip(names, c.tolist())),
                             non_drivable_body_by_time=flags,
                             official_first_at_fault_collision_seconds=.5+.1*i if c[0] < 1 else None,
                             official_first_ttc_infraction_seconds=.4+.1*i if c[2] < 1 else None))
        groups.append(rows)
    return groups


def test_all_feasible_and_mixed_feasible_exact_unsafe_precedence_G16():
    c = torch.ones(3, 16, 4, dtype=torch.float64)
    c[:, :, 3] = torch.linspace(0, 1, 16, dtype=torch.float64)
    c[1, 12:, 1] = 0
    c[2, :, 1] = 0
    baseline = safetyep_parts(c)
    for key in ('A_safe', 'A_EP', 'A_raw'):
        assert torch.equal(baseline[key], raw_advantage_parts(c)[key])
    out = temporal_credit_parts(c, records(c))
    assert torch.equal(out['A_raw'][0], baseline['A_raw'][0])
    feasible = baseline['feasible'][1]
    assert torch.equal(out['A_raw'][1, feasible], baseline['A_raw'][1, feasible])
    assert out['A_raw'][1, ~feasible].lt(0).all()
    assert out['A_raw'][1, feasible].min() > out['A_raw'][1, ~feasible].max()
    assert out['temporal_delta'][1, ~feasible].ne(0).any()
    assert out['A_raw'][2].ne(0).any()  # same coarse failure, actual duration direction
    assert out['A_raw'].sum(1).abs().max() < 1e-12
    assert out['temporal_audit']['applied_groups'] == 2


def test_missing_terms_keep_indicators_and_t0_is_audited_not_skipped():
    c = torch.ones(1, 16, 4, dtype=torch.float64); c[:, :, 1] = 0
    base = safetyep_parts(c)
    feedback = records(c)
    feedback[0][3].pop('non_drivable_body_by_time')
    out = temporal_credit_parts(c, feedback)
    assert out['cost'][0, 3] == .35
    detail = out['temporal_audit']['groups'][0]['candidate_feedback'][3]
    assert 'DAC_duration' in detail['omitted_unknown_terms']
    offset = out['cost'][0]-out['cost'][0, :1]
    torch.testing.assert_close(out['A_raw'][0], offset.mean()-offset, atol=0, rtol=0)
    feedback = records(c);feedback[0][0]['non_drivable_body_by_time'][0] = True
    t0 = temporal_credit_parts(c, feedback)
    assert t0['temporal_audit']['groups'][0]['candidate_feedback'][0]['initially_non_drivable'] is True
    assert t0['temporal_audit']['applied_groups'] == 1
    collision = torch.tensor([0., 1., 1., .5], dtype=torch.float64)
    row = dict(status='OK', metrics=dict(no_at_fault_collisions=0., drivable_area_compliance=1., time_to_collision_within_bound=1., ego_progress=.5),
               official_first_at_fault_collision_seconds=None)
    value, evidence = real_temporal_cost(row, collision)
    assert value == .40 and evidence['collision_time_term'] is None
    assert evidence['collision_time_status'] == 'FAILED_INDICATOR_EVENT_TIME_UNKNOWN'
    row['official_first_at_fault_collision_seconds'] = float('nan')
    with pytest.raises(ValueError):real_temporal_cost(row, collision)


def test_G32_allfailed_centers_full_user_cost_and_records_immutable():
    c = torch.ones(1, 32, 4, dtype=torch.float64);c[:, :, 1] = 0;c[:, 16:, 0] = 0
    feedback = records(c);before = copy.deepcopy(feedback)
    out = temporal_credit_parts(c, feedback)
    offset = out['cost'][0]-out['cost'][0, :1]
    torch.testing.assert_close(out['A_raw'][0], offset.mean()-offset, atol=0, rtol=0)
    assert not torch.equal(out['A_raw'], safetyep_parts(c)['A_raw'])
    assert out['A_raw'].sum().abs() < 1e-12
    for left, right in zip(feedback[0], before[0]):
        assert torch.equal(left['non_drivable_body_by_time'], right['non_drivable_body_by_time'])
        assert left['metrics'] == right['metrics']
    with pytest.raises(ValueError):temporal_credit_parts(c, feedback, refinement_weight=.5)


def test_exact_user_indicator_and_all_actual_time_terms():
    component = torch.tensor([.5, 0., .5, .7], dtype=torch.float64)
    flags = torch.zeros(41, dtype=torch.bool);flags[1:21] = True;flags[0] = True
    row = dict(status='OK', metrics=dict(no_at_fault_collisions=.5, drivable_area_compliance=0.,
        time_to_collision_within_bound=.5, ego_progress=.7), non_drivable_body_by_time=flags,
        official_first_at_fault_collision_seconds=1.2, official_first_ttc_infraction_seconds=2.)
    value, detail = real_temporal_cost(row, component)
    assert value == pytest.approx(.40*.5+.35+.25*.5+.15*(1-1.2/4)+.10*(1-2/4)+.10*.5)
    assert detail['initially_non_drivable'] is True
    row['official_first_at_fault_collision_seconds'] = 0.
    at_zero, detail = real_temporal_cost(row, component)
    assert at_zero-value == pytest.approx(.15*1.2/4)
    assert detail['first_at_fault_collision_seconds'] == 0.
