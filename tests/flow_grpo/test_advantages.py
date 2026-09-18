"""Production advantage code, analytic oracle and actual two-rank collectives."""
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.rl.flow_grpo.advantages import behavior_advantages, assign_behavior_advantages
from starVLA.rl.flow_grpo.math import group_advantages


@pytest.mark.parametrize("size", [3, 8, 16])
@pytest.mark.parametrize("value", [5/7, 13/14, .91, .95])
def test_constant_nonbinary_reward_has_exactly_zero_advantage(size, value):
    rewards = torch.full((2, size), value)
    assert torch.equal(group_advantages(rewards), torch.zeros_like(rewards))
    for mode in ("group", "global_batch", "none"):
        actual, _ = behavior_advantages(rewards, normalization=mode)
        assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("g", [8, 16])
def test_matches_upstream_global_std_oracle_and_keeps_scene_baseline(g):
    rewards = torch.stack([torch.linspace(.949, .951, g),
                           torch.linspace(0., 1., g), torch.full((g,), .73)])
    actual, stats = behavior_advantages(rewards, normalization="global_batch")
    values = rewards.numpy().astype(np.float64)
    expected = (values-values.mean(1, keepdims=True))/(values.std()+1e-6)
    expected = np.clip(expected, -5, 5)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)
    assert actual[-1].eq(0).all()
    assert stats["global_candidate_count"] == 3*g
    assert stats["global_reward_std"] == pytest.approx(values.std(), rel=1e-12)
    # Unit group scaling gives these wildly different spreads similar influence.
    grouped = group_advantages(rewards)
    assert grouped[0].abs().mean() / grouped[1].abs().mean() > .99
    assert actual[0].abs().mean() / actual[1].abs().mean() < .003


def test_invalid_mask_and_almost_constant_rewards():
    r = torch.tensor([[.91, float("nan"), .91], [.91, .9100001, float("nan")]])
    mask = torch.isfinite(r)
    a = group_advantages(r, mask)
    assert a[0].eq(0).all() and a[1, 2] == 0
    assert a[1, 0] < 0 < a[1, 1]
    assert a[1].sum() == 0
    with pytest.raises(FloatingPointError):
        group_advantages(r)


def test_assignment_once_and_pending_buffer_serialization(tmp_path):
    r = torch.tensor([[.1, .3], [.9, 1.]])
    buffers = [SimpleNamespace(rewards=x[None], advantages=None) for x in r]
    algorithm = {"advantage_normalization": "global_batch", "advantage_epsilon": 1e-6,
                 "advantage_clip": 5.}
    assign_behavior_advantages(buffers, algorithm, "cpu")
    expected, stats = behavior_advantages(r, normalization="global_batch")
    torch.testing.assert_close(torch.cat([b.advantages for b in buffers]), expected)
    assert buffers[0].advantage_statistics == buffers[1].advantage_statistics == stats
    path = tmp_path / "pending.pt"
    torch.save({"next_inner": 1, "buffers": buffers}, path)
    resumed = torch.load(path, weights_only=False)
    assert resumed["next_inner"] == 1
    for a, b in zip(resumed["buffers"], buffers):
        assert torch.equal(a.advantages, b.advantages)
        assert a.advantage_statistics == b.advantage_statistics
    with pytest.raises(RuntimeError, match="fresh unassigned"):
        assign_behavior_advantages(resumed["buffers"], algorithm, "cpu")


def _distributed_worker(rank, rendezvous, output, failure):
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    # Different rank counts AND different microbatch counts; pooling rank stds fails.
    rewards = [torch.tensor([[.94, .95, .96]]),
               torch.tensor([[0., .2, 1.], [.75, .75, .75]])][rank]
    if failure and rank == 1:
        rewards[0, 0] = float("nan")
    buffers = [SimpleNamespace(rewards=x[None], advantages=None) for x in rewards]
    algorithm = {"advantage_normalization": "global_batch", "advantage_epsilon": 1e-6,
                 "advantage_clip": 5.}
    try:
        assign_behavior_advantages(buffers, algorithm, "cpu")
        value = {"advantage": torch.cat([b.advantages for b in buffers]),
                 "stats": buffers[0].advantage_statistics}
    except RuntimeError as exc:
        if not failure:
            raise
        value = {"error": str(exc)}
    torch.save(value, output+str(rank)+".pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True])
def test_real_two_rank_global_moments_or_collective_error(tmp_path, failure):
    output = str(tmp_path / "result")
    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), output, failure),
             nprocs=2, join=True)
    results = [torch.load(output+str(r)+".pt", weights_only=True) for r in range(2)]
    if failure:
        assert all("nonfinite" in r["error"] for r in results)
        return
    rewards = torch.tensor([[.94, .95, .96], [0., .2, 1.], [.75, .75, .75]])
    expected, stats = behavior_advantages(rewards, normalization="global_batch")
    torch.testing.assert_close(torch.cat([r["advantage"] for r in results]), expected,
                               rtol=0, atol=0)
    assert all(r["stats"] == stats for r in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA arithmetic comparison requires a GPU")
@pytest.mark.parametrize("g", [8, 16])
def test_cuda_reward_arithmetic_and_cpu_oracle(g):
    r = torch.stack([torch.full((g,), 5/7), torch.linspace(.9, .9001, g),
                     torch.linspace(0, 1, g)])
    for mode in ("group", "global_batch", "none"):
        expected, _ = behavior_advantages(r, normalization=mode)
        actual, _ = behavior_advantages(r.cuda(), normalization=mode)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-7)
