from omegaconf import OmegaConf
import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


def _config(use_scene=True):
    return OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 16},
                "action_model": {
                    "hidden_size": 64,
                    "qwen_input_dim": 16,
                    "use_global_scene_tokens": use_scene,
                    "scene_dim": 32,
                    "action_dim": 4,
                    "action_horizon": 8,
                    "state_dim": 4,
                    "num_inference_timesteps": 2,
                    "num_target_vision_tokens": 2,
                    "max_seq_len": 32,
                    "add_pos_embed": True,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "DiTConfig": {
                        "num_layers": 2,
                        "input_embedding_dim": 64,
                        "attention_head_dim": 64,
                        "num_attention_heads": 1,
                    },
                    "diffusion_model_cfg": {
                        "num_layers": 2,
                        "cross_attention_dim": 64,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "output_dim": 64,
                        "positional_embeddings": None,
                    },
                },
            }
        }
    )


def test_scene_condition_is_appended_and_old_signature_works():
    head = LayerwiseFlowmatchingActionHead(_config()).eval()
    layerwise = [torch.randn(2, 5, 16) for _ in range(2)]
    scene = torch.randn(2, 4, 32)
    projected = head._project_condition_memories(layerwise, scene)
    assert all(value.shape == (2, 9, 64) for value in projected)
    loss = head(layerwise, torch.randn(2, 8, 4), torch.randn(2, 1, 4))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_multi_candidate_noise_reproducible_and_chunk_invariant():
    torch.manual_seed(5)
    head = LayerwiseFlowmatchingActionHead(_config()).eval()
    layerwise = [torch.randn(2, 5, 16) for _ in range(2)]
    scene = torch.randn(2, 4, 32)
    state = torch.randn(2, 1, 4)
    noise = torch.randn(2, 4, 8, 4)
    first = head.predict_multi_action(
        layerwise, state, scene, 4, candidate_chunk_size=1, initial_noise=noise
    )
    second = head.predict_multi_action(
        layerwise, state, scene, 4, candidate_chunk_size=4, initial_noise=noise
    )
    assert first.shape == (2, 4, 8, 4)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    assert torch.unique(first[0].reshape(4, -1), dim=0).shape[0] == 4


def test_single_sample_and_k1_multi_sampler_are_seed_equivalent():
    head = LayerwiseFlowmatchingActionHead(_config()).eval()
    layerwise = [torch.randn(2, 5, 16) for _ in range(2)]
    state = torch.randn(2, 1, 4)
    torch.manual_seed(29)
    single = head.predict_action(layerwise, state)
    torch.manual_seed(29)
    multi = head.predict_multi_action(
        layerwise, state, num_candidates=1, candidate_chunk_size=1
    )[:, 0]
    torch.testing.assert_close(single, multi, rtol=0.0, atol=0.0)


def test_flow_loss_reaches_scene_projection():
    head = LayerwiseFlowmatchingActionHead(_config()).train()
    layerwise = [torch.randn(2, 5, 16) for _ in range(2)]
    scene = torch.randn(2, 4, 32, requires_grad=True)
    loss = head(layerwise, torch.randn(2, 8, 4), torch.randn(2, 1, 4), scene)
    loss.backward()
    assert scene.grad is not None and torch.count_nonzero(scene.grad)
    assert head.scene_proj.layer1.weight.grad is not None
