"""B qualification: a cached directed trunk must match a full masked trunk."""
import copy

import pytest
import torch

from starVLA.model.modules.qwen_world_compare.masks import reasoning_mask
from starVLA.model.modules.qwen_world_compare.shared_actor import ArchitectureB


@pytest.fixture
def model():
    torch.manual_seed(812)
    torch.set_num_threads(1)
    value = ArchitectureB(hidden_dim=24, num_layers=3, num_heads=3, ffn_dim=48,
                          p_tokens=4, r_tokens=3)
    # Test nontrivial trained gates as well as the separately tested zero init.
    with torch.no_grad():
        for block in value.blocks:
            block.modulation[-1].weight.normal_(std=0.015)
            block.modulation[-1].bias.normal_(std=0.02)
    return value


def inputs(batch=2):
    torch.manual_seed(320)
    scene = torch.randn(batch, 7, 24)
    valid = torch.ones(batch, 7, dtype=torch.bool)
    valid[:, -2:] = False
    return torch.randn(batch, 8, 3), scene, valid, torch.randn(batch, 8)


def test_directional_mask_semantics():
    m = reasoning_mask(4, 8, 3)
    assert m.dtype == torch.bool and m.shape == (15, 15)
    assert m[:4, :4].all() and not m[:4, 4:].any()
    assert m[4:12, :12].all() and not m[4:12, 12:].any()
    assert m[12:].all() and m.diagonal().all()


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('sigma_value', [0.0, 0.37])
def test_cached_masked_actions_and_pre_features_match(model, batch, sigma_value):
    state, scene, valid, ego = inputs(batch)
    sigma = torch.full((batch,), sigma_value)
    prefix = model.prepare_prefix(scene, valid, ego)
    actual = model.denoise_x0(state, sigma, prefix)
    expected, p, r = model.forward_full(state, sigma, scene, valid, ego)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(prefix.p_hidden, p, rtol=2e-5, atol=2e-6)
    if sigma_value == 0:
        post = model.replay_candidates(state, torch.arange(batch), prefix)
        torch.testing.assert_close(post, r, rtol=2e-5, atol=2e-6)


def test_cached_full_gradients_match_with_candidate_replay(model):
    other = copy.deepcopy(model)
    state, scene, valid, ego = inputs()
    state.requires_grad_(); scene.requires_grad_()
    state2 = state.detach().clone().requires_grad_(); scene2 = scene.detach().clone().requires_grad_()
    sigma = torch.zeros(2)
    prefix = model.prepare_prefix(scene, valid, ego)
    a = model.denoise_x0(state, sigma, prefix)
    r = model.replay_candidates(state, torch.arange(2), prefix)
    loss = a.square().mean() + 0.3 * prefix.p_hidden.square().mean() + r[..., :7].sin().mean()
    loss.backward()
    a2, p2, r2 = other.forward_full(state2, sigma, scene2, valid, ego)
    (a2.square().mean() + 0.3 * p2.square().mean() + r2[..., :7].sin().mean()).backward()
    torch.testing.assert_close(state.grad, state2.grad, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(scene.grad, scene2.grad, rtol=2e-4, atol=2e-6)
    for (name, p), (_, p2) in zip(model.named_parameters(), other.named_parameters()):
        assert (p.grad is None) == (p2.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, p2.grad, rtol=3e-4, atol=3e-6, msg=name)


def test_prefix_and_actions_do_not_read_post_queries(model):
    state, scene, valid, ego = inputs()
    p1 = model.prepare_prefix(scene, valid, ego)
    x1 = model.denoise_x0(state, torch.full((2,), 0.9), p1)
    with torch.no_grad():
        model.r_queries.add_(torch.randn_like(model.r_queries) * 100)
    p2 = model.prepare_prefix(scene, valid, ego)
    x2 = model.denoise_x0(state, torch.full((2,), 0.9), p2)
    torch.testing.assert_close(x1, x2, rtol=0, atol=0)
    torch.testing.assert_close(p1.p_hidden, p2.p_hidden, rtol=0, atol=0)
    # P API does not accept candidate state or sigma; all its layer KVs are stable.
    model.denoise_x0(state * 30, torch.full((2,), 0.01), p2)
    for v1, v2 in zip(p1.p_kv, p2.p_kv):
        for a, b in zip(v1, v2):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_post_replay_updates_shared_trunk_and_candidate_embedding(model):
    state, scene, valid, ego = inputs()
    state.requires_grad_(); scene.requires_grad_()
    p = model.prepare_prefix(scene, valid, ego)
    r = model.replay_candidates(state, torch.arange(2), p)
    r[..., :5].square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert model.action_input.weight.grad.abs().sum() > 0
    assert model.blocks[0].self_attn.k.weight.grad.abs().sum() > 0
    assert model.blocks[0].ffn[0].weight.grad.abs().sum() > 0
    assert model.action_output.weight.grad is None  # post task reads hidden, not x0 output


def test_candidate_scene_mapping_and_padding_are_exact(model):
    state, scene, valid, ego = inputs()
    scene[:, -2:] = 10000  # masked current memory cannot affect any role
    prefix = model.prepare_prefix(scene, valid, ego)
    ids = torch.tensor([1, 0, 1], dtype=torch.long)
    candidates = state.index_select(0, ids)
    batch = model.replay_candidates(candidates, ids, prefix)
    for i, index in enumerate(ids.tolist()):
        local = model.prepare_prefix(scene[index:index + 1], valid[index:index + 1], ego[index:index + 1])
        expected = model.replay_candidates(candidates[i:i + 1], torch.tensor([0]), local)
        torch.testing.assert_close(batch[i:i + 1], expected, rtol=2e-5, atol=2e-6)
    clean = scene.clone(); clean[:, -2:] = -20000
    again = model.prepare_prefix(clean, valid, ego)
    torch.testing.assert_close(prefix.p_hidden, again.p_hidden, rtol=0, atol=0)


def test_zero_initialization_and_all_invalid_scene_are_finite():
    torch.manual_seed(42)
    m = ArchitectureB(hidden_dim=24, num_layers=2, num_heads=3, ffn_dim=48, p_tokens=4, r_tokens=3)
    state, scene, valid, ego = inputs(); valid.zero_()
    p = m.prepare_prefix(scene, valid, ego)
    a = m.denoise_x0(state, torch.ones(2), p)
    post = m.replay_candidates(state, torch.arange(2), p)
    assert torch.isfinite(a).all() and torch.isfinite(post).all()
    (a.square().mean() + post.square().mean()).backward()
    assert m.blocks[0].cross_attn.o.weight.grad is not None


def test_zero_gates_unlock_candidate_post_gradient_after_an_update():
    torch.manual_seed(42)
    m = ArchitectureB(hidden_dim=24, num_layers=2, num_heads=3, ffn_dim=48, p_tokens=4, r_tokens=3)
    state, scene, valid, ego = inputs()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = m.prepare_prefix(scene, valid, ego)
    action = m.denoise_x0(state, torch.ones(2) * 0.4, p)
    r = m.replay_candidates(state, torch.arange(2), p)
    (action.square().mean() + (r[..., :5] - 1).square().mean()).backward()
    assert m.blocks[0].modulation[-1].weight.grad.abs().sum() > 0
    opt.step(); opt.zero_grad(set_to_none=True)
    state = state.detach().requires_grad_()
    p = m.prepare_prefix(scene, valid, ego)
    m.replay_candidates(state, torch.arange(2), p)[..., :5].square().mean().backward()
    assert state.grad.abs().sum() > 0
    assert m.action_input.weight.grad.abs().sum() > 0


def test_eval_checkpoint_preserves_gradients_and_prefix_updates(model):
    other = copy.deepcopy(model); other.activation_checkpointing = True
    model.eval(); other.eval()
    state, scene, valid, ego = inputs()
    for m in (model, other):
        p = m.prepare_prefix(scene, valid, ego)
        (m.denoise_x0(state, torch.full((2,), 0.2), p).square().mean()
         + m.replay_candidates(state, torch.arange(2), p)[..., :5].square().mean()).backward()
    for (name, p), (_, p2) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, p2.grad, rtol=2e-5, atol=1e-6, msg=name)
    before = model.prepare_prefix(scene, valid, ego)
    with torch.no_grad():
        model.blocks[0].self_attn.k.weight.add_(0.3 * torch.randn_like(model.blocks[0].self_attn.k.weight))
    after = model.prepare_prefix(scene, valid, ego)
    assert not torch.equal(before.p_hidden, after.p_hidden)


def test_deployment_removes_R_and_keeps_complete_ode_identical(model):
    other = copy.deepcopy(model).strip_training_only()
    assert other.r_queries is None and other.role_embedding.weight.shape[0] == 2
    state, scene, valid, ego = inputs()
    u = torch.linspace(1, 0, 11)
    schedule = 5 * u / (1 + 4 * u)
    trajectories = []
    with torch.no_grad():
        for m in (model, other):
            p = m.prepare_prefix(scene, valid, ego)
            x = state.clone()
            for sigma, next_sigma in zip(schedule[:-1], schedule[1:]):
                pred = m.denoise_x0(x, sigma.expand(2), p)
                a = next_sigma / sigma
                x = a * x + (1 - a) * pred
            trajectories.append(x)
    torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=0)
