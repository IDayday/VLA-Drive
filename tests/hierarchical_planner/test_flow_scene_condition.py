from omegaconf import OmegaConf
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
)
import starVLA.model.modules.action_model.GR00T_ActionHeader as action_header


def _config(use_scene=True):
    return OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 16},
                "action_model": {
                    "hidden_size": 64,
                    "use_global_scene_tokens": use_scene,
                    "scene_dim": 32,
                    "action_dim": 4,
                    "action_horizon": 8,
                    "state_dim": 4,
                    "num_inference_timesteps": 2,
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


def test_scene_condition_is_appended_and_legacy_signature_works():
    head = FlowmatchingActionHead(_config()).eval()
    action_queries = torch.randn(2, 8, 16)
    scene = torch.randn(2, 4, 32)
    projected = head._project_condition(action_queries, scene)
    assert projected.shape == (2, 12, 64)

    legacy = FlowmatchingActionHead(_config(use_scene=False)).eval()
    loss = legacy(action_queries, torch.randn(2, 8, 4))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert legacy.scene_to_dit is None


def test_scene_projection_is_256_to_1536_in_main_shape(monkeypatch):
    class TinyDiT(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(kwargs["num_layers"])]
            )

    monkeypatch.setattr(action_header, "DiT", TinyDiT)
    config = _config()
    config.framework.action_model.hidden_size = 1536
    config.framework.action_model.scene_dim = 256
    config.framework.action_model.DiTConfig.input_embedding_dim = 1536
    config.framework.action_model.DiTConfig.attention_head_dim = 64
    config.framework.action_model.DiTConfig.num_attention_heads = 24
    config.framework.action_model.diffusion_model_cfg.cross_attention_dim = 1536
    config.framework.action_model.diffusion_model_cfg.output_dim = 1536
    # Keep two blocks here; production's 24-block assertion is exercised from YAML.
    head = FlowmatchingActionHead(config)
    assert head.scene_to_dit[0].in_features == 256
    assert head.scene_to_dit[0].out_features == 1536
    assert isinstance(head.scene_to_dit[1], torch.nn.LayerNorm)


def test_multi_candidate_noise_reproducible_and_chunk_invariant():
    torch.manual_seed(5)
    head = FlowmatchingActionHead(_config()).eval()
    action_queries = torch.randn(2, 8, 16)
    scene = torch.randn(2, 4, 32)
    noise = torch.randn(2, 4, 8, 4)
    first = head.predict_multi_action(
        action_queries,
        global_scene_tokens=scene,
        num_candidates=4,
        candidate_chunk_size=1,
        initial_noise=noise,
    )
    second = head.predict_multi_action(
        action_queries,
        global_scene_tokens=scene,
        num_candidates=4,
        candidate_chunk_size=4,
        initial_noise=noise,
    )
    assert first.shape == (2, 4, 8, 4)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    assert torch.unique(first[0].reshape(4, -1), dim=0).shape[0] == 4


def test_single_sample_and_k1_multi_sampler_are_seed_equivalent():
    head = FlowmatchingActionHead(_config()).eval()
    action_queries = torch.randn(2, 8, 16)
    scene = torch.randn(2, 4, 32)
    torch.manual_seed(29)
    single = head.predict_action(
        action_queries, global_scene_tokens=scene
    )
    torch.manual_seed(29)
    multi = head.predict_multi_action(
        action_queries,
        global_scene_tokens=scene,
        num_candidates=1,
        candidate_chunk_size=1,
    )[:, 0]
    torch.testing.assert_close(single, multi, rtol=0.0, atol=0.0)


def test_flow_loss_reaches_scene_projection():
    head = FlowmatchingActionHead(_config()).train()
    action_queries = torch.randn(2, 8, 16)
    scene = torch.randn(2, 4, 32, requires_grad=True)
    loss = head(
        action_queries,
        torch.randn(2, 8, 4),
        global_scene_tokens=scene,
    )
    loss.backward()
    assert scene.grad is not None and torch.count_nonzero(scene.grad)
    assert head.scene_to_dit[0].weight.grad is not None


def test_flow_train_repeats_sample_independent_noise_and_timestep(monkeypatch):
    head = FlowmatchingActionHead(_config(use_scene=False)).train()
    seen = {}
    conditions = torch.randn(2, 8, 16).repeat(8, 1, 1)
    actions = torch.zeros(2, 8, 4).repeat(8, 1, 1)

    def deterministic_noise(shape, *, device, dtype):
        value = torch.arange(
            int(torch.tensor(shape).prod()), device=device, dtype=dtype
        ).reshape(shape)
        seen["noise"] = value
        return value

    def deterministic_time(batch_size, device, dtype):
        value = torch.linspace(0.05, 0.95, batch_size, device=device, dtype=dtype)
        seen["time"] = value
        return value

    monkeypatch.setattr(torch, "randn", deterministic_noise)
    monkeypatch.setattr(head, "sample_time", deterministic_time)
    loss = head(conditions, actions)
    assert torch.isfinite(loss)
    assert seen["noise"].shape[0] == 16
    assert torch.unique(seen["noise"].reshape(16, -1), dim=0).shape[0] == 16
    assert torch.unique(seen["time"]).numel() == 16
