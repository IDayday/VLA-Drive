"""Exercise actual actor packing, immutable chains and fail-closed layout gates."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
from starVLA.rl.flow_grpo.config import validate_transition_layout, config_hash
from starVLA.rl.flow_grpo.model import FlowGRPOActor
from starVLA.rl.flow_grpo.rollout import SamplingSpec, sample_chain, evaluate_transitions
from tests.analysis.test_action_batch_probe import TimeScenePolicy


class ReplayPolicy(TimeScenePolicy):
    def __init__(self):
        super().__init__()
        self.replay_calls = 0

    def compute_sft_losses(self, examples):
        self.replay_calls += len(examples)
        return {'action_loss': (self.action_model.weight - .2).square()}


@pytest.mark.parametrize('checkpoint', [True, False])
@pytest.mark.parametrize('group,steps', [(2, 3), (16, 10)])
def test_actual_actor_same_loss_gradient_masks_and_replay(checkpoint, group, steps):
    policy = ReplayPolicy()
    obs = SimpleNamespace(tokens=('a', 'b'))
    rollout = sample_chain(policy, obs, SamplingSpec(group_size=group, num_steps=steps,
                          temporal_noise_correlation=.8), 0, 42, {})
    with torch.no_grad():
        ref = evaluate_transitions(policy, obs, rollout)
    rollout.reference_mean, rollout.reference_std = ref['mean'], ref['std']
    rollout.advantages = torch.linspace(-1, 1, group).expand(2, group)
    rollout.transition_mask[:, 0, 0] = False
    saved = [x.clone() for x in (rollout.chain, rollout.old_elementwise_logprob, rollout.advantages)]
    cfg = {'runtime': {'activation_checkpointing': checkpoint},
           'algorithm': {'ppo_clip_range': .02, 'reference_kl_coefficient': .04},
           'retention': {'original_sft_coefficient': .1}}
    results, gradients = [], []
    for layout in ('serial', 'flat_saved_chain'):
        cfg['runtime']['transition_evaluation'] = layout
        result = FlowGRPOActor(policy, cfg)(mode='update', rollout=rollout, replay=[{}, {}])
        results.append(result)
        gradients.append(torch.autograd.grad(result['loss'], tuple(policy.parameters())))
        probe = FlowGRPOActor(policy, cfg)(mode='transitions', observation=obs, rollout=rollout)
        assert probe['velocity'].shape == (2, group, steps, 8, 4)
    for name in ('grpo', 'reference', 'sft', 'loss', 'ratio'):
        torch.testing.assert_close(results[0][name], results[1][name], atol=0, rtol=0)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
    assert policy.replay_calls == 4
    for before, after in zip(saved, (rollout.chain, rollout.old_elementwise_logprob, rollout.advantages)):
        assert torch.equal(before, after)


@pytest.mark.parametrize('mode', ['formal', 'paired_short', 'bounded_research', 'full_navtrain_epoch'])
def test_every_formal_gate_rejects_unqualified_layout(mode):
    cfg = {'trainable_policy': 'action_head', 'runtime': {'transition_evaluation': 'flat_saved_chain',
           'run_mode': mode, 'max_updates': 2}}
    with pytest.raises(ValueError, match='NOT_READY'):
        enforce_training_budget(cfg)


def test_layout_bound_and_identity():
    cfg = {'trainable_policy': 'action_head', 'runtime': {'transition_evaluation': 'flat_saved_chain',
           'run_mode': 'diagnostic', 'max_updates': 2}}
    validate_transition_layout(cfg)
    serial = deepcopy(cfg); serial['runtime']['transition_evaluation'] = 'serial'
    assert config_hash(cfg) != config_hash(serial)
    cfg['runtime']['max_updates'] = 9
    with pytest.raises(ValueError, match='bounded'): validate_transition_layout(cfg)
    cfg['runtime']['max_updates'] = 2
    cfg['trainable_policy'] = 'inherit_sft'
    with pytest.raises(ValueError, match='action-head'): validate_transition_layout(cfg)
    cfg['runtime']['transition_evaluation'] = 'typo'
    with pytest.raises(ValueError, match='unknown'): validate_transition_layout(cfg)


def test_flat_probe_does_not_reuse_rollout_shape_graph(monkeypatch):
    p = TimeScenePolicy(); obs = SimpleNamespace(tokens=('a',))
    chain = sample_chain(p, obs, SamplingSpec(group_size=16), 0, 42, {})
    p._flow_velocity_graph_enabled = True
    def forbidden(*args): raise AssertionError('rollout graph used with wrong shape')
    monkeypatch.setattr('starVLA.rl.flow_grpo.velocity_graph.graph_velocity', forbidden)
    with torch.no_grad():
        evaluate_transitions(p, obs, chain, layout='flat_saved_chain')
