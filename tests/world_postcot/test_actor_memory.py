import copy

import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor, PreparedContext
from starVLA.model.modules.qwen_world_postcot.flow import (
    CleanXFlowPolicy, clean_x_l1, clean_x_step, clean_x_to_velocity, sample_sigma, shift_sigma,
)
from starVLA.model.modules.qwen_world_postcot.memory import PredictiveMemory, SceneMemoryProvider


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def actor(**kwargs):
    return CleanXActor(hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=128, **kwargs)


def unlock(value):
    """A trained nonzero-gate state for full-path comparisons."""
    with torch.no_grad():
        for block in value.blocks:
            block.modulation[-1].bias[64:96].fill_(0.1)
            block.modulation[-1].bias[160:192].fill_(0.1)
    return value


def inputs(batch=3):
    memory = torch.randn(batch, 7, 32, requires_grad=True)
    valid = torch.ones(batch, 7, dtype=torch.bool)
    valid[:, -2:] = False
    ego = torch.randn(batch, 8)
    state = torch.randn(batch, 8, 3)
    sigma = torch.linspace(0.1, 0.9, batch)
    return memory, valid, ego, state, sigma


def test_real_full_actor_parameter_budget_and_projection_names():
    with torch.device('meta'):
        model = CleanXActor()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 467119107
    assert 460_000_000 <= count <= 500_000_000
    assert len(model.blocks) == 20 and model.action_dim == 3
    for block in model.blocks:
        assert block.self_attn.heads == 16 and block.self_attn.head_dim == 64
        for attention in (block.self_attn, block.cross_attn):
            assert all(isinstance(getattr(attention, name), nn.Linear) for name in ('q', 'k', 'v', 'o'))
    # Shape accounting without allocating a 467M model or using a GPU.
    key = torch.empty((1, 16, 192, 64), device='meta', dtype=torch.bfloat16)
    context = PreparedContext(tuple((key, key) for _ in range(20)),
                              torch.ones((1, 192), device='meta', dtype=torch.bool),
                              torch.empty((1, 1024), device='meta'))
    assert context.logical_nbytes == 15 * 1024**2


def test_shift_formula_and_generator_identity():
    u = torch.tensor([0.0, 0.25, 0.5, 1.0])
    torch.testing.assert_close(shift_sigma(u), 5 * u / (1 + 4 * u), rtol=0, atol=0)
    first = sample_sigma(20, 'cpu', generator=torch.Generator().manual_seed(9))
    second = sample_sigma(20, 'cpu', generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, second)
    assert bool(((first >= 0) & (first < 1)).all())


def test_clean_x_transition_matches_velocity_once_and_endpoint():
    x, x0 = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
    sigma = torch.tensor([1.0, 0.25])
    next_sigma = torch.tensor([0.5, 0.0])
    velocity = clean_x_to_velocity(x, x0, sigma)
    expected = x + (next_sigma - sigma)[:, None, None] * velocity
    actual = clean_x_step(x, x0, sigma, next_sigma)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[1], x0[1])
    with pytest.raises(ValueError, match='strictly positive'):
        clean_x_to_velocity(x, x0, torch.tensor([0.0, 0.25]))
    with pytest.raises(ValueError, match='positive current'):
        clean_x_step(x, x0, 0, 0)


def test_ode_never_calls_sigma_zero_and_preserves_noise():
    class KnownClean(nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmas = []

        def forward(self, x, sigma, context):
            assert x.dtype == sigma.dtype == torch.float32
            self.sigmas.append(sigma.clone())
            return context

    clean = torch.randn(2, 8, 3)
    noise = torch.randn_like(clean)
    original = noise.clone()
    network = KnownClean()
    result = CleanXFlowPolicy(network).sample_ode(clean, noise)
    assert torch.equal(result, clean) and torch.equal(noise, original)
    assert len(network.sigmas) == 10 and all(bool((sigma > 0).all()) for sigma in network.sigmas)


def test_clean_x_l1_mask_and_zero_denominator_are_exact():
    prediction = torch.ones(2, 8, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[0, :2] = True
    target[1] = float('nan')
    loss, stats = clean_x_l1(prediction, target, valid, return_statistics=True)
    assert loss.item() == 1.0 and stats['denominator'].item() == 6
    loss.backward()
    assert prediction.grad[0, :2].count_nonzero() == 6
    assert prediction.grad[1].count_nonzero() == 0 and torch.isfinite(prediction.grad).all()
    prediction.grad = None
    empty = clean_x_l1(prediction, target, valid & False)
    empty.backward()
    assert empty.item() == 0 and prediction.grad.count_nonzero() == 0


def test_policy_is_clean_x_l1_not_velocity_mse():
    class Capture(nn.Module):
        def forward(self, x, sigma, context):
            self.state, self.sigma = x.clone(), sigma.clone()
            return context
    network = Capture()
    policy = CleanXFlowPolicy(network)
    target = torch.randn(2, 8, 3)
    noise = torch.randn_like(target)
    sigma = torch.tensor([0.25, 0.75])
    prediction = torch.randn_like(target, requires_grad=True)
    loss, _ = policy.loss(prediction, target, noise=noise, sigma=sigma)
    assert torch.equal(network.sigma, sigma)
    torch.testing.assert_close(network.state, (1 - sigma[:, None, None]) * target + sigma[:, None, None] * noise)
    torch.testing.assert_close(loss, (prediction - target).abs().mean())


def test_cached_context_forward_gradient_and_projection_count():
    torch.manual_seed(3)
    cached = unlock(actor())
    uncached = copy.deepcopy(cached)
    memory, valid, ego, state, sigma = inputs()
    other_memory = memory.detach().clone().requires_grad_()
    selected = torch.tensor([1, 1, 2, 2])
    candidate = torch.randn(4, 8, 3)
    counts = [0, 0]
    handles = []
    for index, model in enumerate((cached, uncached)):
        for block in model.blocks:
            for name in ('k', 'v'):
                def count(_module, _inputs, _output, index=index):
                    counts[index] += 1
                handles.append(getattr(block.cross_attn, name).register_forward_hook(count))
    prepared = cached.prepare_context(memory, valid, ego)
    main = cached(state, sigma, prepared)
    aux = cached.forward_features(candidate, torch.zeros(4), prepared.index_select(selected))
    (main.square().mean() + aux.square().mean()).backward()
    first = uncached(state, sigma, uncached.prepare_context(other_memory, valid, ego))
    second = uncached.forward_features(candidate, torch.zeros(4), uncached.prepare_context(
        other_memory.index_select(0, selected), valid.index_select(0, selected), ego.index_select(0, selected)))
    (first.square().mean() + second.square().mean()).backward()
    torch.testing.assert_close(main, first, rtol=0, atol=0)
    torch.testing.assert_close(aux, second, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(memory.grad, other_memory.grad, rtol=1e-5, atol=1e-6)
    for (name, one), (other_name, two) in zip(cached.named_parameters(), uncached.named_parameters()):
        assert name == other_name
        if one.grad is not None:
            torch.testing.assert_close(one.grad, two.grad, rtol=1e-4, atol=2e-6, msg=name)
    assert counts == [4, 8]  # two blocks * two context projections, once vs twice
    assert memory.grad[1:].abs().sum() > 0 and memory.grad[:, -2:].count_nonzero() == 0
    for handle in handles:
        handle.remove()


def test_no_cross_candidate_attention_and_sigma_zero_encode_only():
    model = unlock(actor()).eval()
    memory, valid, ego, _, _ = inputs(batch=1)
    ctx = model.prepare_context(memory, valid, ego).index_select(torch.tensor([0, 0, 0]))
    one, two = torch.randn(1, 8, 3), torch.randn(1, 8, 3)
    candidate = torch.cat((one, two, one), dim=0)
    hidden = model.forward_features(candidate, torch.zeros(3), ctx)
    order = torch.tensor([1, 2, 0])
    permuted = model.forward_features(candidate[order], torch.zeros(3), ctx.index_select(order))
    torch.testing.assert_close(hidden[order], permuted, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden[0], hidden[2], rtol=0, atol=0)
    calls = []
    handle = model.action_output.register_forward_hook(lambda *args: calls.append(True))
    model.forward_features(candidate, torch.zeros(3), ctx)
    assert calls == []
    handle.remove()


def test_post_features_backprop_without_action_output_and_gates_unlock():
    torch.manual_seed(5)
    model = actor()
    memory, valid, ego, state, _ = inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = model.blocks[0].ffn[0].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        memory.grad = None
        features = model.forward_features(state, torch.zeros(3), model.prepare_context(memory, valid, ego))
        # A fixed readout loss has a nonuniform derivative through layer norm.
        features[..., :7].square().mean().backward()
        assert model.action_output.weight.grad is None
        assert model.action_input.weight.grad.abs().sum() > 0
        assert model.blocks[0].cross_attn.q.weight.grad.abs().sum() > 0
        assert memory.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(original, model.blocks[0].ffn[0].weight)


def test_cpu_bf16_compute_preserves_fp32_parameters_state_and_kv_dtype():
    model = actor()
    memory, valid, ego, state, sigma = inputs()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        context = model.prepare_context(memory, valid, ego)
        features = model.forward_features(state, sigma, context)
        prediction = model.predict_x0(features)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(value.dtype == torch.bfloat16 for pair in context.layer_kv for value in pair)
    assert features.dtype == torch.bfloat16
    assert prediction.dtype == state.dtype == torch.float32


def test_nonreentrant_action_checkpoint_has_matching_gradient():
    torch.manual_seed(2)
    normal = unlock(actor())
    recomputed = copy.deepcopy(normal)
    recomputed.activation_checkpointing = True
    memory, valid, ego, state, sigma = inputs()
    other = memory.detach().clone().requires_grad_()
    a = normal(state, sigma, normal.prepare_context(memory, valid, ego))
    b = recomputed(state, sigma, recomputed.prepare_context(other, valid, ego))
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(memory.grad, other.grad, rtol=0, atol=0)
    for (_, first), (_, second) in zip(normal.named_parameters(), recomputed.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def provider_input():
    selected = {key: torch.randn(2, 12, 16, requires_grad=True) for key in (8, 16, 28)}
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, -2:] = False
    return dict(selected_hidden_states=selected, final_hidden=selected[28],
        visual_indices=torch.arange(6)[None].expand(2, -1), visual_mask=valid,
        text_indices=torch.arange(6, 12)[None].expand(2, -1), text_mask=torch.ones(2, 6, dtype=torch.bool),
        camera_ids=torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, -1),
        visual_positions=torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]]]).expand(2, -1, -1),
        merged_visual_features=torch.randn(2, 6, 16, requires_grad=True))


def test_current_only_provider_predictive_layout_and_gradient():
    provider = SceneMemoryProvider(input_dim=16, hidden_dim=32, num_heads=4,
                                   spatial_grid=(2, 2), global_tokens=4)
    predictor = PredictiveMemory(input_dim=32, hidden_dim=16, num_heads=4, tokens_per_interval=2)
    features = provider_input()
    ego = torch.randn(2, 8)
    current = provider(features, ego)
    assert current.spatial.shape == (2, 12, 32) and current.global_memory.shape == (2, 4, 32)
    assert current.predictive.shape == (2, 0, 32)
    assert not current.spatial_valid[1, -4:].any() and current.spatial[1, -4:].count_nonzero() == 0
    memory = predictor(current)
    assert memory.actor_memory.shape == (2, 24, 32) and memory.actor_valid.shape == (2, 24)
    assert torch.equal(memory.metadata['predictive_interval_ids'], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    weights = torch.linspace(0.1, 1, 32)
    (memory.actor_memory * weights).square().mean().backward()
    for value in features['selected_hidden_states'].values():
        assert value.grad is not None and value.grad.abs().sum() > 0
    assert features['merged_visual_features'].grad.abs().sum() > 0
    assert provider.ego_projection[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        altered = dict(features, GT_future=torch.randn(2, 8, 3), metric_labels=torch.zeros(2, 6))
        a, b = predictor(provider(features, ego)), predictor(provider(altered, ego))
        assert torch.equal(a.actor_memory, b.actor_memory)


def test_provider_rejects_missing_or_wrong_identity_features():
    provider = SceneMemoryProvider(input_dim=16, hidden_dim=32, num_heads=4, spatial_grid=(2, 2), global_tokens=4)
    features = provider_input()
    ego = torch.randn(2, 8)
    missing = dict(features)
    missing.pop('merged_visual_features')
    with pytest.raises(ValueError, match='merged_visual_features'):
        provider(missing, ego)
    wrong = dict(features, selected_hidden_states={0: features['selected_hidden_states'][8]})
    with pytest.raises(ValueError, match='block identities'):
        provider(wrong, ego)
    with pytest.raises(ValueError, match='Navigation comes'):
        provider(features, ego, navigation=torch.ones(2, 4))
