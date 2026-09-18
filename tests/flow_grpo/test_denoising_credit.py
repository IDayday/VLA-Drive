"""Tests exercise the real loss wrapper; CPU fixtures are not model acceptance."""
from types import SimpleNamespace
import math
import pytest
import torch
from torch import nn
from starVLA.rl.flow_grpo.credit import transition_credit_weights, validate_discount
from starVLA.rl.flow_grpo.model import FlowGRPOActor
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
from starVLA.rl.flow_grpo.config import config_hash, resolve_config


def test_time_direction_mask_and_scene_normalization():
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    valid[1, 0, 1] = False
    weights = transition_credit_weights(valid, .5, dtype=torch.float64, normalization='scene_mean_one')
    raw = torch.tensor([.25, .5, 1.], dtype=torch.float64)
    torch.testing.assert_close(weights[0], raw.expand(2, 3)*6/3.5)
    torch.testing.assert_close(weights[1], raw.expand(2, 3)*valid[1]*5/3.)
    torch.testing.assert_close(weights.sum((1, 2)), valid.sum((1, 2)).double())
    assert weights[0, 0, 0] < weights[0, 0, -1]
    assert weights[1, 0, 1] == 0


def test_uniform_default_is_bitwise_original_loss_and_gradient():
    gen = torch.Generator().manual_seed(201)
    x = torch.randn(3, 16, 10, generator=gen, requires_grad=True)
    valid = torch.rand(3, 16, 10, generator=gen) > .2
    old = ((x*valid).sum((1, 2))/valid.sum((1, 2))).mean()
    new = ((x*transition_credit_weights(valid)).sum((1, 2))/valid.sum((1, 2))).mean()
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, x)[0], torch.autograd.grad(new, x)[0])


@pytest.mark.parametrize('gamma', [0, -.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_discount(gamma):
    with pytest.raises(ValueError, match='discount'):
        validate_discount(gamma)


def test_empty_masks_underflow_and_low_precision_rejected():
    with pytest.raises(ValueError, match='no valid'):
        transition_credit_weights(torch.zeros(1, 2, 3, dtype=torch.bool), .6)
    with pytest.raises(ValueError, match='underflows'):
        transition_credit_weights(torch.ones(1, 2, 100, dtype=torch.bool), 1e-8)
    with pytest.raises(ValueError, match='precision'):
        transition_credit_weights(torch.ones(1, 2, 3, dtype=torch.bool), .6, dtype=torch.bfloat16)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        self.calls = 0

    def compute_sft_losses(self, examples):
        self.calls += len(examples)
        return {'action_loss': (self.weight-.4).square()}


def actor_loss(monkeypatch, gamma):
    p = Policy()
    shape = (1, 2, 3, 2, 2)
    factors = torch.tensor([1., 2., 3.], dtype=torch.float64).reshape(1, 1, 3)
    logs = p.weight*factors.expand(1, 2, 3)
    ratios = torch.tensor([1.01, 1.03, .97], dtype=torch.float64).expand(1, 2, 3)
    old = logs.detach()-ratios.log()
    old_copy = old.clone()
    monkeypatch.setattr('starVLA.rl.flow_grpo.model.evaluate_transitions',
        lambda *a, **k: {'elementwise_logprob': logs[..., None, None].expand(shape),
                         'mean': p.weight.expand(shape), 'std': torch.ones(shape, dtype=torch.float64)})
    cfg = {'runtime': {'activation_checkpointing': True},
           'algorithm': {'ppo_clip_range': .02, 'reference_kl_coefficient': .01, 'denoising_discount': gamma},
           'retention': {'original_sft_coefficient': .1}}
    rollout = SimpleNamespace(observation=None, old_logprob=old, dimension_mask=None,
        spec=SimpleNamespace(reduction='flow_grpo_dimension_mean'),
        advantages=torch.tensor([[1., -1.]], dtype=torch.float64),
        reference_mean=torch.zeros(shape, dtype=torch.float64), reference_std=torch.ones(shape, dtype=torch.float64),
        transition_mask=torch.ones((1, 2, 3), dtype=torch.bool))
    result = FlowGRPOActor(p, cfg)(mode='update', rollout=rollout, replay=[{}, {}], diagnostic_outputs=True)
    assert torch.equal(old, old_copy) and p.calls == 2
    return p, result


def test_production_actor_weighted_clipping_gradients_and_replay_reference_unchanged(monkeypatch):
    p, result = actor_loss(monkeypatch, .5)
    # Hand derivative: positive advantage active at ratios1.01/.97; negative
    # active at1.01/1.03. The .97 negative branch and1.03 positive are clipped.
    weights = [.25, .5, 1.]
    expected = (-1.01*weights[0] - .97*3*weights[2]
                +1.01*weights[0] +1.03*2*weights[1])/6
    torch.testing.assert_close(torch.autograd.grad(result['grpo'], p.weight)[0],
                               torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    _, plain = actor_loss(monkeypatch, 1.)
    assert torch.equal(plain['reference'], result['reference'])
    assert torch.equal(plain['sft'], result['sft'])
    torch.testing.assert_close(result['loss'], result['grpo']+.01*result['reference']+.1*result['sft'])


def test_raw_discount_matches_locked_dppo_simwam_direction_and_scale():
    valid = torch.ones(2, 16, 10, dtype=torch.bool)
    actual = transition_credit_weights(valid, .6, dtype=torch.float64)
    expected = torch.tensor([.6**(10-i-1) for i in range(10)], dtype=torch.float64)
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-15, atol=1e-15)
    assert actual[0, 0, -1] == 1
    assert math.isclose(float(actual.mean()), .2484883456, rel_tol=1e-9)
    with pytest.raises(ValueError, match='normalization'):
        transition_credit_weights(valid, .6, normalization='silently_rescale')


def test_resolved_config_identity_and_formal_rejection():
    path = 'configs/flow_grpo/frozen_research_g16_group.yaml'
    cfg, _ = resolve_config(path)
    changed, _ = resolve_config(path, overrides=['algorithm.denoising_discount=0.6'])
    assert config_hash(cfg) != config_hash(changed)  # exact-resume contract changes
    with pytest.raises(ValueError, match='denoising credit'):
        enforce_training_budget({'runtime': {'run_mode':'formal', 'max_updates':8},
                                 'algorithm': {'denoising_discount': .6}, 'sampling':{}})
    with pytest.raises(ValueError, match='discount'):
        resolve_config(path, overrides=['algorithm.denoising_discount=0'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_weights_remain_fp32_under_bf16_autocast():
    valid = torch.ones(2, 16, 10, dtype=torch.bool, device='cuda')
    expected = transition_credit_weights(valid, .6)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = transition_credit_weights(valid, .6)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
