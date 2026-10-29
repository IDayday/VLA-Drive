"""Actual A/B/C/D modules on the inherited source math; tiny CPU qualification."""
import copy
import math

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLModel

from starVLA.model.modules.qwen_world_compare.contracts import ArchitecturePrefix, CurrentContext
from starVLA.model.modules.qwen_world_compare.independent_actor import ArchitectureA, ArchitectureC
from starVLA.model.modules.qwen_world_compare.shared_actor import ArchitectureB
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import QwenFlowAdapter
from starVLA.rl.qwen_world_compare.rl_scope import scope_targets, install_lora
from starVLA.rl.qwen_world_compare.policy_adapter import FourFrameworkRecipeActor, CONDITION_SCHEMA
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


class TinyDeployment(nn.Module):
    """Same public adapter API without loading the full Qwen weights in CPU tests."""
    def __init__(self, architecture):
        super().__init__()
        self.architecture_id = architecture
        actor = dict(hidden_dim=16, num_layers=2, num_heads=4, ffn_dim=64)
        world = dict(hidden_dim=12, num_layers=2, num_heads=3, ffn_dim=48, bridge_groups=1)
        if architecture in ('A', 'C'):
            self.decision = {'A': ArchitectureA, 'C': ArchitectureC}[architecture](
                actor_config=actor, world_config=world)
        elif architecture == 'B':
            self.decision = ArchitectureB(**actor)
        else:
            cfg = Qwen3VLConfig(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, attention_dropout=0.,
                rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
                vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                    out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                    num_position_embeddings=16, deepstack_visual_indexes=[0, 1]),
                image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
            qwen = Qwen3VLModel._from_config(cfg, attn_implementation='sdpa', dtype=torch.float32)
            self.decision = QwenFlowAdapter(qwen, qualification=True, activation_checkpointing=True)
        # Represent a learned IL initialization with active self/FFN gates, not
        # the unrelated generic-init zero-gate special case.
        for name, parameter in self.decision.named_parameters():
            if name.endswith('modulation.1.bias'):
                with torch.no_grad(): parameter.add_(0.05)

    def prepare_prefix(self, current, auxiliary=False):
        if self.architecture_id == 'D':
            value = self.decision.prepare_prefix(current.encoded, policy_version=current.policy_version)
        elif self.architecture_id == 'A':
            value = self.decision.prepare_prefix(current.scene, current.valid, current.ego,
                                                 need_auxiliary=auxiliary)
        else:
            value = self.decision.prepare_prefix(current.scene, current.valid, current.ego)
        return ArchitecturePrefix(self.architecture_id, current.scene_identity, current.policy_version,
            current.execution_identity, current.position_mask_identity, value)

    def denoise_x0(self, state, sigma, prefix):
        return self.decision.denoise_x0(state, sigma, prefix.tensors)


def condition(model, group=8):
    ego = torch.randn(1, 8)
    if model.architecture_id == 'D':
        raw = dict(input_ids=torch.tensor([[3, 5, 7, 9]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                   scene_identity=('current0',))
        encoded = model.decision.encode_current(raw, ego).detached()
        current = CurrentContext(None, None, ego, encoded, ('current0',), 'current', 'CPU_tiny', 'native')
    else:
        current = CurrentContext(torch.randn(1, 160, 16), torch.ones(1, 160, dtype=torch.bool),
            ego, None, ('current0',), 'current', 'CPU_tiny', 'scene160')
    return dict(schema=CONDITION_SCHEMA, context=torch.zeros(group, 1, 1), current=current)


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_exact_projection_scope_and_permanent_reference(architecture):
    model = TinyDeployment(architecture)
    paths = scope_targets(model)
    actor = FourFrameworkRecipeActor(model, expected_cleanx_recipe())
    assert len(actor.scope_manifest) == len(paths)
    assert all(isinstance(actor.action_expert.get_submodule(p), LoRALinear) for p in paths)
    assert not any(p.requires_grad for p in actor.ref_action_expert.parameters())
    assert set(actor.closure['trainable_names']) == {n for n, p in actor.named_parameters() if p.requires_grad}
    assert not any('post_action_bridges' in p or 'readout' in p for p in paths)
    if architecture == 'C':
        assert any('.world_kv_bridges.' in p for p in paths)
        assert any('world_stream.blocks.' in p for p in paths)
    if architecture == 'D':
        assert all('qwen_model.language_model.layers.' in p for p in paths)
        assert not any(p.requires_grad for p in actor.action_expert.decision.qwen_model.visual.parameters())


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_real_sampler_terms_and_fresh_ratio_match_and_prefix_is_per_call(architecture):
    model = TinyDeployment(architecture)
    cfg = expected_cleanx_recipe()
    actor = FourFrameworkRecipeActor(model, cfg)
    cond = condition(model)
    actor.record_sampling_terms = True
    chain, times, deltas, steps = actor.sample_action_chain(cond, 8,
        generator=torch.Generator().manual_seed(42))
    assert chain.shape == (8, 11, 8, 3) and steps.tolist() == [7, 8, 9]
    assert actor.prepared_context_builds['current'] == 1
    assert len(actor.sampling_terms) == 3
    old, terms = actor.action_chain_logprobs(cond, chain, times, deltas,
                                           stoch_idx=steps, return_terms=True)
    for i, (sampled_mean, sampled_std) in enumerate(actor.sampling_terms):
        torch.testing.assert_close(sampled_mean, terms['mean'][:, i], atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(sampled_std), terms['std'][i], atol=0, rtol=0)
    new = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=steps)
    assert (torch.exp(new-old)-1).abs().max() < 1e-4
    assert actor.prepared_context_builds['current'] == 3
    assert actor._prepared_scope is None
    reference = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=steps, use_ref=True)
    torch.testing.assert_close(new, reference, atol=0, rtol=0)
    assert actor.prepared_context_builds['reference'] == 1


@pytest.mark.parametrize('architecture', ['A', 'B', 'C', 'D'])
def test_policy_gradients_update_only_lora_and_recompute_current_prefix(architecture):
    actor = FourFrameworkRecipeActor(TinyDeployment(architecture), expected_cleanx_recipe())
    cond = condition(actor.action_expert)
    chain, times, deltas, steps = actor.sample_action_chain(cond, 8,
        generator=torch.Generator().manual_seed(13))
    old = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=steps).detach()
    rollout = dict(cond_bg=cond, chain=chain, timesteps=times, deltas=deltas, stoch_idx=steps,
        old_logp=old.clone(), advantages=torch.linspace(-1, 1, 8), adv_mask=torch.ones(8))
    reference = {n: p.detach().clone() for n, p in actor.ref_action_expert.named_parameters()}
    frozen = {n: p.detach().clone() for n, p in actor.action_expert.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=5e-5,
                                  betas=(.9, .95), weight_decay=.01)
    # Four real inner steps on one fixed behavior chain, as in the formal recipe.
    before_builds = actor.prepared_context_builds['current']
    for inner in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = actor(rollout)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1.)
        optimizer.step()
        actor.set_policy_version(f'inner{inner+1}')
    assert actor.prepared_context_builds['current'] == before_builds+4
    assert actor.prepared_context_builds['reference'] == 4
    assert actor._prepared_scope is None
    torch.testing.assert_close(rollout['old_logp'], old, atol=0, rtol=0)
    for n, p in actor.ref_action_expert.named_parameters():
        torch.testing.assert_close(p, reference[n], atol=0, rtol=0)
    for n, p in actor.action_expert.named_parameters():
        if not p.requires_grad:torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)
    assert all(state['step'] == 4 for state in optimizer.state.values())
    if architecture in ('B', 'C', 'D'):
        with actor._context_call(cond):
            prepared = actor._prepare(cond, False)
        tree = prepared.tensors
        key = (tree.p_kv[0][0] if architecture == 'B' else
               tree.world_kv[0][0] if architecture == 'C' else tree.layer_kv[0][0])
        assert key.requires_grad


def test_shared_linear_aliases_receive_one_adapter_object():
    model = TinyDeployment('B')
    model.decision.extra_alias = model.decision.blocks[0].cross_attn.q
    paths = scope_targets(model)
    manifest = install_lora(model, paths=paths)
    first = model.decision.blocks[0].cross_attn.q
    assert model.decision.extra_alias is first
    row = next(r for r in manifest if 'decision.extra_alias' in r['aliases'])
    assert row['aliases'] == ['decision.blocks.0.cross_attn.q', 'decision.extra_alias']
    assert len(manifest) == len(paths)


def test_reject_old_condition_graph_and_privileged_labels():
    actor = FourFrameworkRecipeActor(TinyDeployment('B'), expected_cleanx_recipe())
    cond = condition(actor.action_expert)
    with pytest.raises(ValueError, match='current deployment'):
        actor.materialize(dict(cond, reward=torch.ones(8)))
    cond['current'].scene.requires_grad_(True)
    with pytest.raises(ValueError, match='old graph'):
        actor.materialize(cond)


@pytest.mark.parametrize('nonzero_lora', [False, True])
@pytest.mark.parametrize('padded', [False, True])
def test_D_native_three_view_sample_terms_match_no_grad_old_and_grad_first_inner(nonzero_lora, padded):
    """Actual rollout mode differs from first-inner grad/checkpoint execution.

    Native tiny vision/merger/DeepStack run on three camera images. This checks
    CPU BF16 arithmetic only; the full own-parent CUDA gate is still required.
    """
    model = TinyDeployment('D')
    # An own IL parent has also learned the new time interfaces. Avoid relying
    # on the unrelated generic-init all-zero FiLM special case in this gate.
    with torch.no_grad():
        for pair in model.decision.time_films:
            for film in pair:
                film.weight.normal_(mean=0., std=.001)
                film.bias.normal_(mean=0., std=.001)
    actor = FourFrameworkRecipeActor(model, expected_cleanx_recipe())
    if nonzero_lora:
        # Represent an already adapted checkpoint, without making a new method
        # or relying only on the zero-B initialization special case.
        with torch.no_grad():
            for module in actor.action_expert.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(mean=0., std=.001)
    image = [62, 60, 60, 60, 60, 63]
    ids = torch.tensor([[3]+image+image+image+[7, 0]])
    valid = torch.ones_like(ids)
    if padded:valid[:, -1] = 0
    raw = dict(input_ids=ids, attention_mask=valid,
        image_grid_thw=torch.tensor([[1, 4, 4]]*3), pixel_values=torch.randn(48, 1536),
        scene_identity=('native_three_view',))
    ego = torch.randn(1, 8)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        encoded = actor.action_expert.decision.encode_current(raw, ego).detached()
    assert encoded.visual_mask.sum() == 12 and len(encoded.deepstack) == 2
    assert encoded.inputs_embeds.dtype == torch.float32
    assert all(value.dtype == torch.bfloat16 for value in encoded.deepstack)
    current = CurrentContext(None, None, ego, encoded, ('native_three_view',),
                             'current', 'CPU_BF16_native_vision', 'native')
    cond = dict(schema=CONDITION_SCHEMA, context=torch.zeros(8, 1, 1), current=current)
    generator = torch.Generator().manual_seed(2142)
    actor.record_sampling_terms = True
    with torch.no_grad():
        chain, times, deltas, steps = actor.sample_action_chain(cond, 8, generator=generator)
        actual_terms = copy.deepcopy(actor.sampling_terms)
        old = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=steps)
        old_audit = copy.deepcopy(actor._probability_audit['current'])
    assert steps.tolist() == [7, 8, 9] and len(actual_terms) == 3
    # Reconstruct probabilities from the terms that actually produced x_next,
    # not from a second model pass or from old/current self-consistency alone.
    for i, (mean, std) in enumerate(actual_terms):
        torch.testing.assert_close(mean, old_audit['mean'][:, i], atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(std), old_audit['std'][i], atol=0, rtol=0)
        raw_logp = torch.distributions.Normal(mean.float(),
            torch.tensor(std, dtype=torch.float32)).log_prob(chain[:, int(steps[i])+1].float())
        # Source uses the Python-double std for relative clamp bounds.
        peak = -math.log(std)-.5*math.log(2*math.pi)
        clipped_logp = raw_logp.clamp(min=peak-7., max=peak)
        torch.testing.assert_close(raw_logp, old_audit['raw_gaussian_logprob'][:, i], atol=0, rtol=0)
        torch.testing.assert_close(clipped_logp, old_audit['source_clamped_logprob'][:, i], atol=0, rtol=0)
        torch.testing.assert_close(clipped_logp.mean((-1, -2)), old[:, i], atol=0, rtol=0)
    # Optional diagnostic recording consumes no RNG and changes no sample.
    actor.record_sampling_terms = False
    unrecorded_generator = torch.Generator().manual_seed(2142)
    with torch.no_grad():
        unrecorded = actor.sample_action_chain(cond, 8, generator=unrecorded_generator)
    for left, right in zip((chain, times, deltas, steps), unrecorded):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert torch.equal(generator.get_state(), unrecorded_generator.get_state())
    first_inner = actor.action_chain_logprobs(cond, chain, times, deltas, stoch_idx=steps)
    assert (torch.exp(first_inner-old)-1).abs().max() < 1e-4
    torch.testing.assert_close(actor._probability_audit['current']['mean'], old_audit['mean'], atol=0, rtol=0)
    first_inner.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in actor.parameters() if p.requires_grad)
    assert actor.prepared_context_builds['current'] == 4
    assert actor._prepared_scope is None
