"""Directed CPU math and native-payload tests; no production/GPU score claim."""
from contextlib import contextmanager
import copy
import io

import pytest
import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_world_performance import awfm, fpo_adapter as fpo
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from test_qwen_world_lastdet_contract import recipe as source_recipe


class TinyActor(torch.nn.Module):
    """Reduced differentiable action+memory fixture, never production scope."""
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor(.19))
        self.memory = torch.nn.Parameter(torch.tensor(.07))
        self.device = torch.device('cpu')
        self.grpo_kl_beta = 50.
        self.grpo_kl_type = 'x'
        self.grpo_bc_coeff = .1
        self.grpo_denoising_discount = 1.
        self.context_calls = 0

    @contextmanager
    def _context_call(self, descriptor):
        self.context_calls += 1
        yield {'memory': self.memory+descriptor['feature']}

    def action_velocity(self, state, timestep, condition, use_ref=False):
        assert not use_ref
        return self.action*state+condition['memory']+timestep[:, None, None]*1e-4

    def action_chain_logprobs(self, condition, chain, times, deltas, *, stoch_idx=None, return_terms=False, use_ref=False):
        value = chain[:, stoch_idx]
        mean = value*.1+.03 if use_ref else self.action*value+self.memory
        velocity = mean*.3
        logp = -mean.square().mean((-1, -2))
        terms = dict(mean=mean, vel=velocity, std=torch.full((len(stoch_idx),), .2))
        return (logp, terms) if return_terms else logp

    def kl_weight_matrix(self, **kwargs):
        return torch.tensor([.2, 1., 1.]).expand(8, 3)


def rollout(g=16):
    gen = torch.Generator().manual_seed(7)
    chain = torch.randn(g, 11, 8, 3, generator=gen)*.1
    requested = ActionCodec.denorm_odo(chain[:, -1]).numpy()
    return dict(chain=chain, timesteps=torch.arange(10.), deltas=torch.ones(10), stoch_idx=torch.tensor([7, 8, 9]),
                old_logp=torch.zeros(g, 3),
                token='test_scene', log='test_physical_log', protocol_sha256='actual_protocol_fixture',
                behavior_version=dict(run='test_run', update=0, adapter_sha256='initial'),
                cond_bg=dict(identity='condition_fixture', feature=.04), rewards=torch.full((1, g), .5),
                advantages=torch.linspace(-1, 1, g), adv_mask=torch.ones(g),
                records=[dict(status='OK', token='test_scene', log='test_physical_log', protocol_sha256='actual_protocol_fixture',
                              requested_trajectory=requested[i], metrics={'score': .5}) for i in range(g)])


def test_awfm_negative_advantage_velocity_hessian_lower_bound_and_detached_targets():
    target = torch.linspace(-.4, .8, 24).reshape(1, 1, 8, 3).double().requires_grad_()
    behavior = torch.full_like(target, .1, requires_grad=True)
    for advantage in [-5., -2., 0., 2., 5.]:
        a = torch.tensor([advantage], dtype=torch.float64, requires_grad=True)
        def objective(vector):
            return awfm.conservative_awfm_objective(vector.reshape(1, 1, 8, 3), target, behavior, a, torch.ones(1))[0]
        vector = torch.linspace(-.3, .5, 24, dtype=torch.float64, requires_grad=True)
        hessian = torch.autograd.functional.hessian(objective, vector)
        eigen = torch.linalg.eigvalsh(hessian)
        assert float(eigen.min()) >= 2/24-2e-8
        expected = 2/24*(1+max(.25*advantage, 0))
        torch.testing.assert_close(eigen, torch.full_like(eigen, expected), atol=2e-8, rtol=1e-6)
        objective(vector).backward()
        assert target.grad is None and behavior.grad is None and a.grad is None


def test_fpo_generic_G16_perMC_clip_detached_old_and_energy_units():
    velocity = torch.ones(16, 8, 3)
    assert torch.equal(fpo.cfm_energy(velocity, torch.zeros_like(velocity)), torch.full((16,), 8.))
    assert torch.equal(fpo.cfm_energy(velocity, torch.zeros_like(velocity), reduction='mean24'), torch.ones(16))
    old = torch.ones(2, 16, requires_grad=True)
    current = old.detach().clone().requires_grad_()
    a = torch.tensor([1., -1.]+[0.]*14)
    loss, metrics = fpo.per_mc_fpo_objective(current, old, a, torch.ones(16))
    loss.backward()
    assert current.grad[:, 0].gt(0).all() and current.grad[:, 1].lt(0).all()
    assert old.grad is None
    assert metrics['fpo_proxy_ratio_mean'] == 1
    clipped = old.detach().clone()
    clipped[:, 0] -= torch.log(torch.tensor(1.02))
    clipped[:, 1] -= torch.log(torch.tensor(.98))
    clipped.requires_grad_()
    loss, metrics = fpo.per_mc_fpo_objective(clipped, old, a, torch.ones(16))
    loss.backward()
    assert clipped.grad[:, :2].eq(0).all()
    assert metrics['fpo_eligible_blocked_fraction'] == 1


@pytest.mark.parametrize('module,loss_class,attach,validate,key,mc', [
    (awfm, awfm.AWFMLoss, awfm.attach_awfm_material, awfm.validate_awfm_material, awfm.MATERIAL_KEY, 4),
    (fpo, fpo.FPOAdapterLoss, fpo.attach_fpo_material, fpo.validate_fpo_material, fpo.MATERIAL_KEY, 2)])
@pytest.mark.parametrize('g', [16, 32])
def test_loss_material_current_memory_anchor_parity_and_pending_resume(module, loss_class, attach, validate, key, mc, g):
    cfg = source_recipe(False)
    cfg['grpo']['sample']['group_size'] = g
    cfg['grpo']['train']['num_inner_epochs'] = 2  # inner count is not MC count
    actor, out = TinyActor(), rollout(g)
    before_rng = torch.get_rng_state().clone()
    attach(actor, out, rank=3)
    assert torch.equal(torch.get_rng_state(), before_rng)
    validate(out, rank=3)
    assert out[key]['sigma'].shape == (mc, g)
    assert out[key]['behavior_velocity'].shape == (mc, g, 8, 3)
    assert not out[key]['old_energy'].requires_grad and not out[key]['behavior_velocity'].requires_grad
    digest = fpo.material_digest(out[key])
    original_anchor = copy.deepcopy(out)
    original_anchor['advantages'].zero_(); original_anchor['adv_mask'].zero_()
    anchor, _ = ResearchLoss(cfg)._policy_loss(actor, original_anchor)
    anchor_grad = torch.autograd.grad(anchor, tuple(actor.parameters()))
    captured = []
    loss = loss_class(cfg)
    loss.record_policy_components = lambda pg, weighted_anchor: captured.append((pg, weighted_anchor))
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    total, metrics = loss._policy_loss(actor, out)
    assert len(captured) == 1
    torch.testing.assert_close(total, sum(captured[-1]), rtol=0, atol=0)
    torch.testing.assert_close(captured[-1][1], anchor, rtol=0, atol=0)
    got = torch.autograd.grad(captured[-1][1], tuple(actor.parameters()), retain_graph=True)
    for actual, expected in zip(got, anchor_grad):torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    opt.step(); opt.zero_grad()
    stream = io.BytesIO()
    torch.save(dict(actor=actor.state_dict(), optimizer=opt.state_dict(), rollout=out), stream)
    stream.seek(0); saved = torch.load(stream, weights_only=False)
    resumed = TinyActor(); resumed.load_state_dict(saved['actor'])
    restored_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-5); restored_opt.load_state_dict(saved['optimizer'])
    validate(saved['rollout'], rank=3)
    assert fpo.material_digest(saved['rollout'][key]) == digest
    left, _ = loss._policy_loss(actor, out)
    right, _ = loss_class(cfg)._policy_loss(resumed, saved['rollout'])
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.backward(); right.backward()
    for p, q in zip(actor.parameters(), resumed.parameters()):torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert actor.context_calls >= 3  # behavior plus fresh current memory per inner
    damaged = copy.deepcopy(out); damaged[key]['old_energy'][0, 0] += .01
    with pytest.raises(ValueError):validate(damaged, rank=3)
