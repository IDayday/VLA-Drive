"""Finite CPU qualifications for the actual A/C modules, with small widths."""
import copy

import pytest
import torch

from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_compare.independent_actor import ArchitectureA, ArchitectureC


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def model(kind, checkpointing=False):
    return kind(actor_config=dict(hidden_dim=24, num_layers=4, num_heads=4,
            ffn_dim=72, activation_checkpointing=checkpointing),
        world_config=dict(hidden_dim=12, num_layers=4, num_heads=3,
            ffn_dim=36, bridge_groups=2, activation_checkpointing=checkpointing))


def inputs(batch=2, grad=False):
    scene = torch.randn(batch, 160, 24, requires_grad=grad)
    valid = torch.ones(batch, 160, dtype=torch.bool)
    valid[0, -7:] = False
    ego = torch.randn(batch, 8)
    state = torch.randn(batch, 8, 3)
    sigma = torch.linspace(.1, .8, batch)
    return scene, valid, ego, state, sigma


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_shapes_finite_and_prefix_current_only(kind):
    network = model(kind).eval()
    scene, valid, ego, action, sigma = inputs()
    prefix = network.prepare_prefix(scene, valid, ego)
    frozen = [x.detach().clone() for x in prefix.world.layer_states]
    first = network.denoise_x0(action, sigma, prefix)
    assert first.shape == (2, 8, 3) and first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert network.pre_reasoning_states(prefix).shape == (2, 32, 12)
    result = network.replay_candidates(action.flip(0) * 2, torch.tensor([1, 0]), prefix)
    assert result.shape == (2, 16, 12) and torch.isfinite(result).all()
    network.denoise_x0(action + 4, sigma.flip(0), prefix)
    assert all(torch.equal(x, y) for x, y in zip(frozen, prefix.world.layer_states))
    assert torch.equal(first, network.denoise_x0(action, sigma, prefix))


def test_a_preserves_original_actor_and_does_not_read_world():
    network = model(ArchitectureA).eval()
    reference = CleanXActor(hidden_dim=24, num_layers=4, num_heads=4, ffn_dim=72).eval()
    reference.load_state_dict(network.actor.state_dict(), strict=True)
    scene, valid, ego, action, sigma = inputs()
    prefix = network.prepare_prefix(scene, valid, ego)
    expected = reference(action, sigma, reference.prepare_context(scene, valid, ego))
    actual = network.denoise_x0(action, sigma, prefix)
    assert torch.equal(actual, expected)
    with torch.no_grad():
        for parameter in network.world_stream.parameters():
            parameter.add_(torch.randn_like(parameter))
    changed_world_prefix = network.prepare_prefix(scene, valid, ego)
    assert not torch.equal(prefix.world.final, changed_world_prefix.world.final)
    assert torch.equal(actual, network.denoise_x0(action, sigma, changed_world_prefix))
    no_auxiliary = network.prepare_prefix(scene, valid, ego, need_auxiliary=False)
    assert no_auxiliary.world is None
    assert torch.equal(actual, network.denoise_x0(action, sigma, no_auxiliary))


def test_c_zero_gate_equals_base_including_shared_o_bias():
    network = model(ArchitectureC).eval()
    reference = CleanXActor(hidden_dim=24, num_layers=4, num_heads=4, ffn_dim=72).eval()
    state = network.actor.state_dict()
    reference.load_state_dict({name: state[name] for name in reference.state_dict()}, strict=True)
    with torch.no_grad():
        network.actor.world_gate.zero_()
        for block, reference_block in zip(network.actor.blocks, reference.blocks):
            block.cross_attn.o.bias.fill_(.17)
            reference_block.cross_attn.o.bias.fill_(.17)
    scene, valid, ego, action, sigma = inputs()
    prefix = network.prepare_prefix(scene, valid, ego)
    actual = network.denoise_x0(action, sigma, prefix)
    expected = reference(action, sigma, reference.prepare_context(scene, valid, ego))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_post_reaches_actor_scene_and_world_not_action_output(kind):
    network = model(kind)
    scene, valid, ego, action, _ = inputs(grad=True)
    prefix = network.prepare_prefix(scene, valid, ego)
    post = network.replay_candidates(action, torch.tensor([0, 1]), prefix)
    weight = torch.randn_like(post)
    (post * weight).mean().backward()
    assert scene.grad is not None and scene.grad.abs().sum() > 0
    assert network.actor.action_input.weight.grad.abs().sum() > 0
    assert network.actor.blocks[0].cross_attn.q.weight.grad.abs().sum() > 0
    assert network.actor.action_output.weight.grad is None
    assert network.world_stream.blocks[-1].ffn[0].weight.grad.abs().sum() > 0
    assert network.world_stream.post_action_bridges[0][1].weight.grad.abs().sum() > 0


def test_c_world_path_active_and_no_actor_bridge_alias():
    network = model(ArchitectureC)
    scene, valid, ego, action, sigma = inputs(grad=True)
    assert (network.actor.world_gate != 0).all()
    assert len(network.actor.world_kv_bridges) == 2
    prefix = network.prepare_prefix(scene, valid, ego)
    loss = network.denoise_x0(action, sigma, prefix).square().mean()
    loss.backward()
    assert network.actor.world_gate.grad.abs().sum() > 0
    assert network.actor.world_kv_bridges[0].k.weight.grad.abs().sum() > 0
    assert network.world_stream.prefix_queries.grad.abs().sum() > 0
    assert network.world_stream.post_queries.grad is None
    assert not ({id(p) for p in network.actor.parameters()} &
                {id(p) for p in network.world_stream.parameters()})


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_selected_cached_prefix_matches_per_scene_recompute(kind):
    network = model(kind).eval()
    scene, valid, ego, action, sigma = inputs()
    indices = torch.tensor([1, 0, 1])
    prefix = network.prepare_prefix(scene, valid, ego)
    actual = network.denoise_x0(action.index_select(0, indices),
        sigma.index_select(0, indices), prefix.index_select(indices))
    rows = []
    for index in indices:
        i = int(index)
        local = network.prepare_prefix(scene[i:i+1], valid[i:i+1], ego[i:i+1])
        rows.append(network.denoise_x0(action[i:i+1], sigma[i:i+1], local))
    torch.testing.assert_close(actual, torch.cat(rows), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_deployment_pruning_keeps_exact_action(kind):
    network = model(kind).eval()
    scene, valid, ego, action, sigma = inputs()
    before = network.denoise_x0(action, sigma, network.prepare_prefix(scene, valid, ego))
    count = sum(p.numel() for p in network.parameters())
    network.prune_training_components()
    after = network.denoise_x0(action, sigma, network.prepare_prefix(scene, valid, ego,
                                                                   need_auxiliary=False))
    assert torch.equal(before, after)
    assert sum(p.numel() for p in network.parameters()) < count
    if kind is ArchitectureA:
        assert network.world_stream is None
    else:
        assert network.world_stream.post_queries is None
        assert network.world_stream.post_action_bridges is None


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_checkpoint_eval_grad_matches_eager(kind):
    eager = model(kind).eval()
    checked = model(kind, checkpointing=True).eval()
    checked.load_state_dict(eager.state_dict(), strict=True)
    scene, valid, ego, action, sigma = inputs()
    def run(network):
        p = network.prepare_prefix(scene, valid, ego)
        x = network.denoise_x0(action, sigma, p)
        r = network.replay_candidates(action, torch.tensor([0, 1]), p)
        (x.square().mean() + (r * torch.arange(12).float()).mean()).backward()
        return x, r
    a, b = run(eager), run(checked)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    for (name, x), (_, y) in zip(eager.named_parameters(), checked.named_parameters()):
        assert (x.grad is None) == (y.grad is None), name
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6, msg=name)


def test_invalid_current_scene_and_wrong_family_rejected():
    network = model(ArchitectureA)
    scene, valid, ego, action, sigma = inputs()
    valid.zero_()
    with pytest.raises(ValueError, match='at least one current token'):
        network.prepare_prefix(scene, valid, ego)
    valid.fill_(True)
    other = model(ArchitectureC)
    with pytest.raises(ValueError, match='another model family'):
        network.denoise_x0(action, sigma, other.prepare_prefix(scene, valid, ego))


@pytest.mark.parametrize('kind', [ArchitectureA, ArchitectureC])
def test_legacy_memory_or_predictive192_never_accepted(kind):
    from types import SimpleNamespace
    network = model(kind)
    scene, valid, ego, _, _ = inputs()
    with pytest.raises(ValueError, match='C160'):
        network.prepare_prefix(SimpleNamespace(actor_memory=scene), valid, ego,
                               need_auxiliary=False)
    expanded = torch.randn(2, 192, 24)
    with pytest.raises(ValueError, match='C160'):
        network.prepare_prefix(expanded, torch.ones(2, 192, dtype=torch.bool), ego,
                               need_auxiliary=False)

