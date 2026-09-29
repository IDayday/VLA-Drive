import pytest
from tools.foresight.student_state import validate_rank_batches
from tools.ddpolicy_vehicle.training_state import epoch_batches


def test_sixteen_rank_navsim_tail_has_no_loss_or_duplication():
    validate_rank_batches(101592, 32, 16, 2)
    batches = epoch_batches(101592, 32, 42, 0)
    assert len(batches) == 3175 and len(batches[-1]) == 24
    tail = [batches[-1][rank::16] for rank in range(16)]
    assert [len(v) for v in tail] == [2] * 8 + [1] * 8
    assert sorted(sum(tail, [])) == sorted(batches[-1])
    assert len(set(sum(batches, []))) == 101592


def test_true_global_mean_with_unequal_rank_counts():
    # Each rank contributes world_size * local numerator / actual global24;
    # the distributed gradient average therefore equals the single-batch mean.
    values = list(range(24))
    gradients = [16 * sum(values[rank::16]) / 24 for rank in range(16)]
    assert sum(gradients) / 16 == pytest.approx(sum(values) / 24)


@pytest.mark.parametrize('args', [(8, 32, 16, 2), (24, 32, 16, 1)])
def test_reject_empty_rank_or_mismatched_collectives(args):
    with pytest.raises(ValueError):validate_rank_batches(*args)


@pytest.mark.parametrize('world,micro', [(4,8),(8,4),(16,2)])
def test_supported_full_optimizer_batch(world,micro):
    validate_rank_batches(101592,32,world,micro)
