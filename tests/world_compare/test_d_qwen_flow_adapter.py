"""Native tiny-Qwen oracle tests for architecture D, no model downloads/GPU."""
import copy

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import QwenFlowAdapter
from starVLA.model.modules.qwen_world_postcot.actor import timestep_embedding


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def model(checkpointing=False):
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=1024, attention_dropout=0.0,
            rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
            num_heads=4, out_hidden_size=32, patch_size=16,
            spatial_merge_size=2, temporal_patch_size=2,
            num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62,
        vision_end_token_id=63)
    native = Qwen3VLModel._from_config(cfg, attn_implementation='sdpa', dtype=torch.float32)
    return QwenFlowAdapter(native, qualification=True,
                           activation_checkpointing=checkpointing).eval()


def batch(vision=False, padded=False):
    ids = torch.tensor([[3, 5, 7, 9, 11], [6, 8, 10, 12, 14]])
    if vision:
        ids = torch.tensor([[3, 62, 60, 60, 60, 60, 63, 7],
                            [8, 62, 60, 60, 60, 60, 63, 9]])
    valid = torch.ones_like(ids)
    if padded:
        valid[1, -1] = 0
        ids[1, -1] = 0
    result = dict(input_ids=ids, attention_mask=valid, ego=torch.randn(2, 8),
                  scene_identity=('scene0', 'scene1'))
    if vision:
        result['image_grid_thw'] = torch.tensor([[1, 4, 4], [1, 4, 4]])
        result['pixel_values'] = torch.randn(32, 3*2*16*16)
    return result


def full_reference(m, current, state, sigma, include_r=True):
    """Uncached C/E/P/A/R with one directional mask and native layer ops."""
    b, c, _ = current.inputs_embeds.shape
    start = current.positions.masked_fill(~current.valid[None], -1).amax((0, 2))+1
    ego = m.ego_input(current.ego)[:, None]+m.role_embedding.weight[0][None, None]
    p = m.p_embeddings[None].expand(b, -1, -1)+m.role_embedding.weight[1][None, None]
    a = m._action_embeds(state)
    parts = [current.inputs_embeds, ego, p, a]
    if include_r:
        parts.append(m.r_embeddings[None].expand(b, -1, -1)+m.role_embedding.weight[3][None, None])
    hidden = torch.cat(parts, 1)
    total = hidden.shape[1]
    allowed = torch.zeros(b, total, total, dtype=torch.bool)
    allowed[:, :c, :c] = torch.ones(c, c, dtype=torch.bool).tril()[None]&current.valid[:, None]
    allowed[:, c:c+1, :c+1] = True
    allowed[:, c+1:c+33, :c+33] = True
    allowed[:, c+33:c+41, :c+41] = True
    allowed[:, c+41:, :] = True
    allowed[:, c:, :c] &= current.valid[:, None]
    positions = torch.cat((current.positions, m._suffix_positions(start, total-c)), -1)
    action_rows = torch.zeros(total)
    action_rows[c+33:c+41] = 1
    visual_mask = torch.cat((current.visual_mask, torch.zeros(b, total-c, dtype=torch.bool)), -1)
    time = m.time_embedding(timestep_embedding(sigma, m.time_dim))
    h, kv, layers = m._run(hidden, positions, allowed, time=time,
        action_rows=action_rows, visual_mask=visual_mask, deepstack=current.deepstack)
    return h, kv, layers


@pytest.mark.parametrize('vision,padded', [(False, False), (False, True), (True, False), (True, True)])
def test_native_current_forward_and_deepstack_parity(vision, padded):
    m = model()
    inputs = batch(vision=vision, padded=padded)
    current = m.encode_current(inputs)
    n = current.valid.shape[1]
    allowed = torch.ones(n, n, dtype=torch.bool).tril()[None]&current.valid[:, None]
    ours, _, _ = m._run(current.inputs_embeds, current.positions, allowed,
        visual_mask=current.visual_mask, deepstack=current.deepstack)
    kwargs = {k: v for k, v in inputs.items() if k in
              ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
    with torch.no_grad():
        native = m.qwen_model(**kwargs, use_cache=False, return_dict=True).last_hidden_state
    torch.testing.assert_close(ours[current.valid], native[current.valid], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('vision,padded', [(False, False), (False, True), (True, False), (True, True)])
def test_cached_directional_prefix_matches_full_mask(vision, padded):
    m = model()
    current = m.encode_current(batch(vision=vision, padded=padded))
    state, sigma = torch.randn(2, 8, 3), torch.tensor([0.4, 0.8])
    prefix = m.prepare_prefix(current, policy_version='initial')
    cached, _, _ = m._candidate_suffix(state, sigma, prefix, include_r=True)
    full, _, _ = full_reference(m, current, state, sigma)
    torch.testing.assert_close(prefix.reasoning, full[:, current.valid.shape[1]+1:current.valid.shape[1]+33],
                               atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(cached, full[:, -24:], atol=3e-6, rtol=3e-5)


def test_prefix_is_immutable_across_noise_sigma_and_replay_and_no_old_action_keys():
    m = model()
    current = m.encode_current(batch())
    prefix = m.prepare_prefix(current, policy_version='v0')
    copies = tuple((k.detach().clone(), v.detach().clone()) for k, v in prefix.layer_kv)
    expected = current.valid.shape[1]+33
    for _ in range(3):
        out = m.denoise_x0(torch.randn(2, 8, 3), torch.rand(2), prefix)
        assert out.shape == (2, 8, 3) and out.dtype == torch.float32
        assert all(k.shape[-2] == expected for k, _ in prefix.layer_kv)
    m.replay_candidates(torch.randn(3, 8, 3), torch.tensor([0, 0, 1]), prefix)
    for old, new in zip(copies, prefix.layer_kv):
        torch.testing.assert_close(old[0], new[0], atol=0, rtol=0)
        torch.testing.assert_close(old[1], new[1], atol=0, rtol=0)
    again = m.prepare_prefix(current)
    torch.testing.assert_close(again.reasoning, prefix.reasoning, atol=0, rtol=0)


def test_r_cannot_change_candidate_a_or_deployment_output():
    m = model()
    p = m.prepare_prefix(m.encode_current(batch()))
    state, sigma = torch.randn(2, 8, 3), torch.zeros(2)
    action, _, _ = m._candidate_suffix(state, sigma, p)
    post, _, _ = m._candidate_suffix(state, sigma, p, include_r=True)
    altered, _, _ = m._candidate_suffix(state, sigma, p, include_r=True,
                                       r_override=torch.randn(2, 16, 32)*10)
    torch.testing.assert_close(post[:, :8], action, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(altered[:, :8], action, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(altered[:, 8:], post[:, 8:])


def test_cached_full_gradients_and_explicit_eval_checkpoint():
    left, right = model(), model(checkpointing=True)
    right.load_state_dict(left.state_dict())
    # A nonzero FiLM tests masking rather than only zero-initialized coincidence.
    for pair in left.time_films:
        for film in pair:
            nn = torch.randn_like(film.weight)*0.005
            with torch.no_grad(): film.weight.copy_(nn)
    right.load_state_dict(left.state_dict())
    inputs, state, sigma = batch(vision=True, padded=True), torch.randn(2, 8, 3), torch.rand(2)
    p = left.prepare_prefix(left.encode_current(inputs))
    h, _, _ = left._candidate_suffix(state, sigma, p, include_r=True)
    loss = (h.float().square().mean()+0.3*p.reasoning.float().square().mean())
    loss.backward()
    cur = right.encode_current(inputs)
    full, _, _ = full_reference(right, cur, state, sigma)
    c = cur.valid.shape[1]
    loss_full = full[:, -24:].float().square().mean()+0.3*full[:, c+1:c+33].float().square().mean()
    loss_full.backward()
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=3e-3, msg=name)


def test_interface_freezes_base_but_retains_new_input_and_post_gradients():
    m = model(checkpointing=True)
    for param in m.qwen_model.parameters():
        param.requires_grad_(False)
    p = m.prepare_prefix(m.encode_current(batch(vision=True)))
    r = m.replay_candidates(torch.randn(3, 8, 3), torch.tensor([0, 0, 1]), p)
    # A feature component avoids the constant norm-square of RMS-normalized output.
    r[..., 0].mean().backward()
    assert m.action_input.weight.grad is not None and m.action_input.weight.grad.abs().sum() > 0
    assert m.p_embeddings.grad is not None and m.p_embeddings.grad.abs().sum() > 0
    assert m.r_embeddings.grad is not None and m.r_embeddings.grad.abs().sum() > 0
    assert m.action_output.weight.grad is None  # post loss is hidden replay, not output regression.
    assert all(p.grad is None for p in m.qwen_model.parameters())


def test_current_language_prefix_has_grad_and_changes_when_language_attention_changes():
    m = model()
    cur = m.encode_current(batch()).detached()
    p = m.prepare_prefix(cur, policy_version='v0')
    m.denoise_x0(torch.randn(2, 8, 3), torch.ones(2)*0.4, p).mean().backward()
    parameter = m.language_model.layers[0].self_attn.k_proj.weight
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    old = p.layer_kv[0][0].detach().clone()
    with torch.no_grad(): parameter.add_(0.01*torch.randn_like(parameter))
    changed = m.prepare_prefix(cur, policy_version='v1')
    assert not torch.allclose(old, changed.layer_kv[0][0])


def test_candidate_selection_and_explicit_parameter_roles():
    m = model()
    p = m.prepare_prefix(m.encode_current(batch(padded=True)))
    selected = p.index_select(torch.tensor([1, 1, 1]))
    assert selected.layer_kv[0][0].stride(0) == 0
    assert selected.scene_identity == ('scene1',)*3
    state = torch.randn(3, 8, 3)
    together = m.replay_candidates(state, torch.tensor([1, 1, 0]), p)
    separated = torch.cat([m.replay_candidates(state[i:i+1], torch.tensor([s]), p)
                          for i, s in enumerate([1, 1, 0])], 0)
    torch.testing.assert_close(together, separated, atol=3e-6, rtol=3e-5)
    roles = m.named_parameter_roles()
    assert roles['action_input.weight'] == 'new_action_interface'
    assert roles['time_films.0.0.weight'] == 'new_time_modulation'
    assert roles['qwen_model.visual.merger.linear_fc1.weight'] == 'pretrained_merger'
    assert roles['qwen_model.language_model.layers.0.self_attn.q_proj.weight'] == 'pretrained_language'
    assert len(m.rl_attention_paths()) == 12
    assert m.deployment_components()['qwen_text_layers_per_action_step'] == 3


def test_current_contract_rejects_future_supervision_and_bad_sigma():
    m = model()
    with pytest.raises(ValueError, match='supervision'):
        m.encode_current(dict(batch(), reward=torch.ones(2)))
    p = m.prepare_prefix(m.encode_current(batch()))
    with pytest.raises(ValueError, match='Sigma'):
        m.denoise_x0(torch.randn(2, 8, 3), torch.tensor([float('nan'), 0.2]), p)


def test_eager_native_c_prefill_remains_causal():
    m = model()
    m.qwen_model.config.text_config._attn_implementation = 'eager'
    inputs = batch()
    current = m.encode_current(inputs)
    prefix = m.prepare_prefix(current)
    state, sigma = torch.randn(2, 8, 3), torch.rand(2)
    cached, _, _ = m._candidate_suffix(state, sigma, prefix, include_r=True)
    full, _, _ = full_reference(m, current, state, sigma)
    torch.testing.assert_close(cached, full[:, -24:], atol=3e-6, rtol=3e-5)


def test_deployment_removes_r_without_changing_ode_denoiser():
    m = model()
    p = m.prepare_prefix(m.encode_current(batch()))
    state, sigma = torch.randn(2, 8, 3), torch.rand(2)
    before = m.denoise_x0(state, sigma, p)
    assert m.discard_training_only() == ('r_embeddings',)
    assert 'r_embeddings' not in m.state_dict()
    after = m.denoise_x0(state, sigma, p)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='no training-only R'):
        m.replay_candidates(state, torch.arange(2), p)


@pytest.mark.parametrize('include_r', [False, True])
def test_shared_candidate_packing_matches_independent_batch_and_gradients(include_r):
    packed, independent = model(checkpointing=True), model(checkpointing=True)
    independent.load_state_dict(packed.state_dict())
    independent.pack_shared_candidates = False
    inputs = batch(vision=True, padded=True)
    state, sigma = torch.randn(4, 8, 3), torch.rand(4)
    outputs = []
    for m in (packed, independent):
        prefix = m.prepare_prefix(m.encode_current(inputs)).index_select(torch.tensor([1, 1, 1, 1]))
        hidden, kv, _ = m._candidate_suffix(state, sigma, prefix, include_r=include_r)
        outputs.append(hidden)
        hidden[..., 0].mean().backward()
        if m is packed:
            # The temporary native attention K/V has one CP, not G CP copies.
            n = 24 if include_r else 8
            assert kv[0][0].shape[0] == 1
            assert kv[0][0].shape[-2] == prefix.valid.shape[1]+4*n
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-6, rtol=3e-5)
    for (name, a), (_, b) in zip(packed.named_parameters(), independent.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4, msg=name)
