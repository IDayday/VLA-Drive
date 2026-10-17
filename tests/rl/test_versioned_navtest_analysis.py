import numpy as np
import pytest
from scripts.analysis.fm_transfer.analyze_versioned_navtest import summarize, validate_origin_ranks


def test_gain_loss_and_partial_collision_are_separate():
    base = np.ones((4, 7)); current = base.copy()
    base[:, 0] = [.9, 1, 0, .5]; current[:, 0] = [1, .5, .8, 0]
    current[1, 1] = .5; base[2, 3] = 0
    result = summarize(current, ['a', 'a', 'b', 'b'], base)
    paired = result['paired']
    assert paired['positive_gain_sum'] == pytest.approx(.9)
    assert paired['negative_degradation_sum'] == pytest.approx(1.)
    assert paired['PDMS_log_cluster_bootstrap']['mean'] == pytest.approx(-.025)
    assert paired['nonzero_to_zero'] == 1 and paired['zero_to_nonzero'] == 1
    assert paired['safety_transitions']['no_at_fault_collisions']['new_violation'] == 1
    assert paired['safety_transitions']['time_to_collision_within_bound']['recovery'] == 1
    assert paired['reference_safe_new_joint_violation'] == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.01, 1.01])
def test_invalid_score_not_removed(bad):
    values = np.ones((2, 7)); values[0, 0] = bad
    with pytest.raises(ValueError): summarize(values, ['a', 'b'])


def test_shape_or_denominator_mismatch_rejected():
    with pytest.raises(ValueError): summarize(np.ones((1, 7)), ['a', 'b'])
    with pytest.raises(ValueError): summarize(np.ones((2, 6)), ['a', 'b'])


def test_exact_legacy_summary_schema_still_requires_complete_counts():
    source = dict(source_sha256='f9bfa2d205bd9548beca787730c8cf7508f2412998e72f7665848e062b218163',
                  shards=2, scenes=5, sha256='manifest')
    ranks = [dict(status='ASSIGNED_COMPLETE', new=n, reused=0, direct_original_check=True) for n in (3, 2)]
    assert validate_origin_ranks(source, ranks) == ['exact_legacy_worker_shard_counts_and_per_scene_seals']
    ranks[1]['new'] = 1
    with pytest.raises(ValueError): validate_origin_ranks(source, ranks)


def test_missing_identity_not_permitted_for_a_new_worker():
    source = dict(source_sha256='new', shards=1, scenes=2, sha256='manifest')
    ranks = [dict(status='ASSIGNED_COMPLETE', new=2, reused=0, direct_original_check=True)]
    with pytest.raises(ValueError): validate_origin_ranks(source, ranks)
    ranks[0]['identity_sha256'] = 'other'
    with pytest.raises(ValueError): validate_origin_ranks(source, ranks)
    ranks[0]['identity_sha256'] = 'manifest'
    assert validate_origin_ranks(source, ranks) == ['rank_identity_and_per_scene_seals']
