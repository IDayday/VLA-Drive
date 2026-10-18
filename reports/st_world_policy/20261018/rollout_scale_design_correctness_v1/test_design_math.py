"""Independent CPU mathematics only; this is not a production implementation."""
from pathlib import Path
import importlib.util
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "reviewed_fpo", ROOT / "starVLA/model/modules/flow_posttraining/fpo_objective.py")
FPO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FPO
SPEC.loader.exec_module(FPO)
torch.set_num_threads(1)


def reference(shards):
    """Two-pass detached mathematical reference; no distributed/runtime claim."""
    if any(x.requires_grad or not torch.isfinite(x).all() for x in shards):
        raise ValueError("detached finite complete advantages required")
    count = sum(x.numel() for x in shards)
    if count < 2:
        raise ValueError("population too small")
    mean = sum(x.double().sum() for x in shards) / count
    m2 = sum((x.double() - mean).square().sum() for x in shards)
    std = (m2 / (count - 1)).sqrt()
    if not torch.isfinite(std) or (std == 0 and any(x.ne(0).any() for x in shards)):
        raise ValueError("invalid scale")
    result = [(x.double() / (std + 1e-10)).float().detach() for x in shards]
    return result, std, count


def centered(rewards):
    return FPO.group_centered_advantages(rewards, torch.ones_like(rewards, dtype=torch.bool))


def test_roundoff_counterexample_and_zero_groups():
    rewards = torch.tensor([[1., 1., 1., 1.-2.**-24], [.5]*4], dtype=torch.float32)
    a = centered(rewards)
    assert torch.equal(a, torch.tensor([[0., 0., 0., -2.**-24], [0.]*4]))
    normalized, std, n = reference([a])
    assert n == 8 and std > 0
    assert torch.equal(normalized[0][1], torch.zeros(4))
    incorrectly_recentered = (a.double() - a.double().mean()) / (std + 1e-10)
    assert (incorrectly_recentered[1] > 0).all()
    allzero, std, n = reference([centered(torch.tensor([[0.]*4, [.5]*4, [1.]*4]))])
    assert n == 12 and std == 0 and torch.equal(allzero[0], torch.zeros(3, 4))


def test_complete_feedback_and_invalid_input_rejections():
    r = torch.ones(2, 4)
    v = torch.ones_like(r, dtype=torch.bool)
    v[0, 0] = False
    with pytest.raises(ValueError):
        FPO.group_centered_advantages(r, v)
    for shards in ([torch.empty(0)], [torch.ones(1)], [torch.ones(2)],
                   [torch.tensor([float("nan"), 0.])], [torch.ones(2, requires_grad=True)]):
        with pytest.raises(ValueError):
            reference(shards)


@pytest.mark.parametrize("scenes", [32, 24, 5])
def test_full_rollout_std_and_shards_with_tail_and_empty_local_shard(scenes):
    rewards = ((torch.arange(scenes * 4).reshape(scenes, 4) * 17) % 31).float() / 31
    a = centered(rewards)
    full, std, count = reference([a])
    shards = [a[r::4] for r in range(4)] + [torch.empty(0, 4)]
    partitioned, shard_std, shard_count = reference(shards)
    assert count == shard_count == scenes * 4
    torch.testing.assert_close(std, a.double().std(correction=1), rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(std, shard_std, rtol=1e-15, atol=1e-15)
    for rank in range(4):
        assert torch.equal(partitioned[rank], full[0][rank::4])
    assert partitioned[-1].numel() == 0
    population_std = a.double().std(correction=0)
    torch.testing.assert_close(std / population_std,
        torch.tensor((count / (count-1))**.5, dtype=torch.float64), rtol=1e-15, atol=1e-15)


def test_surrogate_positive_scale_and_microbatch_ddp_math():
    a = centered(torch.tensor([[.1, .3, .8, 1.], [.6, .6, .6, .6], [0., .1, .2, .4]]))
    normalized, std, n = reference([a])
    cfg = FPO.FPOConfig(mc_samples=2)
    old = torch.zeros(n, 2, dtype=torch.float64)
    current = torch.linspace(-.8, .8, n).double().reshape(-1, 1).expand(-1, 2).clone().requires_grad_()
    valid = torch.ones(n, dtype=torch.bool)
    raw_loss, _ = FPO.fpo_surrogate(old, current, a.flatten().double(), valid, cfg, global_count=n, world_size=1)
    exact_scaled = a.flatten().double() / (std + 1e-10)
    loss, _ = FPO.fpo_surrogate(old, current, exact_scaled, valid, cfg, global_count=n, world_size=1)
    raw_g = torch.autograd.grad(raw_loss, current, retain_graph=True)[0]
    g = torch.autograd.grad(loss, current, retain_graph=True)[0]
    torch.testing.assert_close(loss, raw_loss / (std + 1e-10), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(g, raw_g / (std + 1e-10), rtol=1e-14, atol=1e-14)
    split_loss = 0.
    # Unequal rank counts and uneven microbatches; average rank gradients once.
    for indices in (torch.arange(0, 8), torch.arange(8, 12)):
        for chunk in indices.split(3):
            term, _ = FPO.fpo_surrogate(old[chunk], current[chunk], exact_scaled[chunk], valid[chunk],
                                       cfg, global_count=n, world_size=2)
            split_loss = split_loss + term / 2
    split_g = torch.autograd.grad(split_loss, current)[0]
    torch.testing.assert_close(split_loss, loss, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(split_g, g, rtol=1e-14, atol=1e-14)
    assert not normalized[0].requires_grad


def test_batch_weights_change_and_microbatch_scale_is_not_equivalent():
    pattern = torch.tensor([[-1., -.5, .5, 1.]])
    low, low_s, _ = reference([pattern * .001])
    high, high_s, _ = reference([pattern * .1])
    torch.testing.assert_close(low_s / high_s, torch.tensor(.01, dtype=torch.float64), rtol=1e-7, atol=0)
    # Absolute original gap ratio 100 is almost removed across updates.
    torch.testing.assert_close(low[0], high[0], rtol=2e-7, atol=2e-7)
    pooled, _, _ = reference([torch.cat([pattern * .001, pattern * .1])])
    assert not torch.allclose(pooled[0], torch.cat([low[0], high[0]]))
    sparse, sparse_s, n = reference([torch.cat([pattern, torch.zeros(3, 4)])])
    dense, dense_s, _ = reference([pattern])
    assert n == 16 and sparse_s < dense_s
    assert sparse[0][0].abs().max() > dense[0][0].abs().max()
    assert torch.equal(sparse[0][1:], torch.zeros(3, 4))


def test_additive_epsilon_not_variance_epsilon_or_clamp():
    a = torch.tensor([[-1., -.5, .5, 1.]]) * 1e-12
    normalized, std, _ = reference([a])
    expected = (a.double() / (std + 1e-10)).float()
    assert torch.equal(normalized[0], expected)
    assert not torch.allclose(expected, (a.double() / std.clamp_min(1e-10)).float(), rtol=1e-4, atol=0)
    assert not torch.allclose(expected, (a.double() / (std.square() + 1e-10).sqrt()).float(), rtol=1e-4, atol=0)
