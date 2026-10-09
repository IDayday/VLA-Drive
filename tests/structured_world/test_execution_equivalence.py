"""Checkpoint removal must preserve native stochastic Qwen/DiT computations."""
import pytest
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


@pytest.mark.parametrize('steps', [6, 8])
def test_native_Qwen_DiT_checkpoint_modes_preserve_dropout_gradients_and_RNG(steps):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cfg = Qwen3VLTextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rope_scaling={'rope_type': 'default', 'mrope_section': [2, 3, 3], 'mrope_interleaved': True},
            attention_dropout=.1, use_cache=False)
        cfg._attn_implementation = 'sdpa'
        language = Qwen3VLTextModel(cfg).float().train()
        action_cfg = OmegaConf.create({'framework': {'qwenvl': {'vl_hidden_dim': 64}, 'action_model': {
            'hidden_size': 64, 'action_dim': 4, 'action_horizon': steps, 'num_inference_timesteps': 10,
            'noise_beta_alpha': 1.5, 'noise_beta_beta': 1., 'noise_s': .999, 'num_timestep_buckets': 1000,
            'add_pos_embed': True, 'max_seq_len': 32,
            'DiTConfig': {'num_layers': 2, 'input_embedding_dim': 64, 'attention_head_dim': 16, 'num_attention_heads': 4},
            'diffusion_model_cfg': {'cross_attention_dim': 64, 'dropout': .2, 'final_dropout': True,
                'interleave_self_attention': True, 'norm_type': 'ada_norm', 'output_dim': 64, 'positional_embeddings': None}}}})
        action = FlowmatchingActionHead(action_cfg).float().train()
        inputs = torch.randn(2, 7, 64)
        ego, noise, times = torch.randn(16, steps, 4), torch.randn(16, steps, 4), torch.rand(16)
        rng = torch.random.get_rng_state().clone()
        snapshots = []
        for checkpointed in (True, False):
            language.zero_grad(set_to_none=True); action.zero_grad(set_to_none=True)
            language.gradient_checkpointing_enable() if checkpointed else language.gradient_checkpointing_disable()
            action.model.gradient_checkpointing = checkpointed
            torch.random.set_rng_state(rng)
            current = inputs.detach().clone().requires_grad_(True)
            hidden = language(inputs_embeds=current, use_cache=False).last_hidden_state
            loss = action(hidden[:, -5:].repeat(8, 1, 1), ego, noise=noise, times=times)
            loss.backward()
            grads = {'input': current.grad.detach().clone()}
            for prefix, module in [('language', language), ('action', action)]:
                grads.update({prefix+'/'+name: p.grad.detach().clone() for name, p in module.named_parameters() if p.grad is not None})
            snapshots.append((hidden.detach().clone(), loss.detach().clone(), grads, torch.random.get_rng_state().clone()))
        torch.testing.assert_close(snapshots[0][0], snapshots[1][0], rtol=0, atol=0)
        torch.testing.assert_close(snapshots[0][1], snapshots[1][1], rtol=0, atol=0)
        assert torch.equal(snapshots[0][3], snapshots[1][3])
        assert snapshots[0][2].keys() == snapshots[1][2].keys()
        for name in snapshots[0][2]:
            torch.testing.assert_close(snapshots[0][2][name], snapshots[1][2][name], rtol=1e-6, atol=1e-7)
