import pytest
from scripts.analysis.fm_transfer.repartition_cost_report import aggregate_attempts


def row(invocation, calls, wall, start, end, status='ASSIGNED_COMPLETE'):
    return dict(invocation=invocation, physical_scoring_calls=calls, wall_seconds=wall,
                scoring_worker_seconds=wall*3, started_utc=start, ended_utc=end, status=status)


def test_old_drain_cost_is_retained_and_not_equal_to_elapsed():
    old = row('old', 4, 5, '2026-10-18T00:00:00+00:00', '2026-10-18T00:00:05+00:00', 'FAILED_PRESERVED')
    new = row('new', 8, 10, '2026-10-18T00:00:20+00:00', '2026-10-18T00:00:30+00:00')
    x = aggregate_attempts([new], [old])
    assert x['physical_scoring_calls'] == 12
    assert x['cumulative_invocation_wall_seconds'] == 15
    assert x['first_start_to_last_end_seconds'] == 30
    assert x['uncommitted_inflight_call_cost'] == 'UNKNOWN_NOT_ZERO'


def test_duplicate_or_running_invocation_rejected():
    x = row('x', 4, 5, '2026-10-18T00:00:00+00:00', '2026-10-18T00:00:05+00:00')
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_attempts([x], [x])
    with pytest.raises(ValueError, match='open'):
        aggregate_attempts([dict(x, status='RUNNING')], [])
