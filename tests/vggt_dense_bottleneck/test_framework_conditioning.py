from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.QwenOFT_VGGT_Bottleneck import (
    Qwenvl_OFT_VGGT_Bottleneck,
    apply_dense_intervention,
    pad_dense_geometry_payloads,
)
from starVLA.model.modules.vggt_query.planning_heads import AuxiliaryTrajectoryHead
from starVLA.model.modules.vggt_query.task_geometry_bottleneck import (
    PlanningConditionedDenseVGGTBottleneck,
)
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


def _payload(value: float, count: int, dim: int = 24):
    return {
        "features": torch.full((count, dim), value, dtype=torch.bfloat16),
        "valid_mask": torch.ones(count, dtype=torch.bool),
        "view_ids": torch.arange(count, dtype=torch.int16).remainder(3),
        "uv_coords": torch.full((count, 2), value, dtype=torch.float16),
        "ray_features": torch.full((count, 6), value, dtype=torch.float32),
        "patch_grid_hw": torch.tensor([[1, count], [0, 0], [0, 0]], dtype=torch.int16),
    }


def _harness():
    model = Qwenvl_OFT_VGGT_Bottleneck.__new__(Qwenvl_OFT_VGGT_Bottleneck)
    nn.Module.__init__(model)
    model.baseline_weight = nn.Parameter(torch.ones(()))
    model.vggt_bottleneck_enabled = True
    model._vggt_intervention_mode = "real"
    model.vggt_dense_bottleneck = PlanningConditionedDenseVGGTBottleneck(
        planning_dim=32,
        source_dim=24,
        bottleneck_dim=16,
        expected_horizons=8,
        slots_per_horizon=4,
        num_heads=4,
        attention_dropout=0.0,
    )
    model.vggt_bottleneck_aux_plan_head = AuxiliaryTrajectoryHead(16, 16, 4)
    model.config = SimpleNamespace()
    return model


def test_framework_inherits_action_only_and_source_has_no_vggt_text_tokens():
    assert issubclass(Qwenvl_OFT_VGGT_Bottleneck, Qwenvl_OFT)
    assert "_build_action_prompt_suffix" not in Qwenvl_OFT_VGGT_Bottleneck.__dict__
    source = Path(
        "starVLA/model/framework/QwenOFT_VGGT_Bottleneck.py"
    ).read_text(encoding="utf-8")
    assert "build_vggt_global_query_tokens" not in source
    assert "extra_context=task" not in source


def test_dense_bottleneck_overlay_keeps_the_action_only_contract():
    from omegaconf import OmegaConf

    overlay = OmegaConf.load(
        "starVLA/config/training/vggt_dense_bottleneck.yaml"
    )
    assert overlay.framework.name == "QwenOFT_VGGT_Bottleneck"
    assert overlay.framework.action_prompt_mode == "minimal"
    assert overlay.framework.vggt_bottleneck.cache.component == "vggt_dense"
    assert overlay.framework.vggt_bottleneck.teacher.layer_index == 23
    assert (
        overlay.framework.vggt_bottleneck.teacher.representation
        == "full_aggregated_feature"
    )
    assert overlay.framework.vggt_bottleneck.teacher.include_special_tokens is False
    assert overlay.framework.vggt_bottleneck.teacher.preprocess_mode == "crop"
    assert overlay.trainer.freeze_modules == "qwen_vl_interface"
    assert "vggt" not in overlay.framework


def test_variable_length_dense_batch_padding_contract():
    batch = pad_dense_geometry_payloads(
        [_payload(1.0, 5), _payload(2.0, 8)], device=torch.device("cpu")
    )
    assert batch["features"].shape == (2, 8, 24)
    assert batch["valid_mask"].tolist() == [
        [True] * 5 + [False] * 3,
        [True] * 8,
    ]
    assert torch.count_nonzero(batch["features"][0, 5:]) == 0
    assert torch.count_nonzero(batch["uv_coords"][0, 5:]) == 0
    assert torch.count_nonzero(batch["ray_features"][0, 5:]) == 0
    assert torch.count_nonzero(batch["view_ids"][0, 5:]) == 0


def test_shuffled_intervention_rolls_the_entire_payload_together():
    dense = pad_dense_geometry_payloads(
        [_payload(1.0, 5), _payload(2.0, 5)], device=torch.device("cpu")
    )
    shuffled, diagnostics = apply_dense_intervention(dense, "shuffled")
    for key in (
        "features",
        "valid_mask",
        "view_ids",
        "uv_coords",
        "ray_features",
        "patch_grid_hw",
    ):
        torch.testing.assert_close(shuffled[key][0], dense[key][1])
        torch.testing.assert_close(shuffled[key][1], dense[key][0])
    assert diagnostics["intervention_skipped"].item() == 0

    singleton, diagnostics = apply_dense_intervention(
        {key: value[:1] for key, value in dense.items()}, "shuffled"
    )
    torch.testing.assert_close(singleton["features"], dense["features"][:1])
    assert diagnostics["intervention_skipped"].item() == 1


def test_training_and_inference_conditioning_share_the_same_bottleneck():
    model = _harness().eval()
    planning = torch.randn(2, 8, 32)
    examples = [
        {"vggt_dense_feature_cache": _payload(1.0, 5), "action": torch.zeros(8, 4)},
        {"vggt_dense_feature_cache": _payload(2.0, 8), "action": torch.zeros(8, 4)},
    ]
    extension = model._compute_query_extension(
        torch.empty(0), {}, examples, input_ids=None, image_grid_thw=None
    )
    training, training_context, _ = model._condition_action_queries(
        planning, extension
    )
    inference, inference_context, _ = model._condition_inference_action_queries(
        torch.empty(0), torch.empty(0, dtype=torch.long), planning, examples=examples
    )
    torch.testing.assert_close(training, inference)
    assert training_context is None
    assert inference_context is None
    torch.testing.assert_close(training, planning, rtol=0.0, atol=0.0)


def test_old_action_checkpoint_missing_key_whitelist_is_strict():
    model = _harness()
    full = model.state_dict()
    old = {
        key: value
        for key, value in full.items()
        if not key.startswith(("vggt_dense_bottleneck.", "vggt_bottleneck_aux_plan_head."))
    }
    model.load_state_dict(old, strict=True)

    missing_baseline = dict(old)
    missing_baseline.pop("baseline_weight")
    with pytest.raises(RuntimeError, match="baseline_weight"):
        model.load_state_dict(missing_baseline, strict=True)

    broken = dict(old)
    broken["unexpected.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        model.load_state_dict(broken, strict=True)


def test_new_modules_receive_their_explicit_optimizer_groups():
    model = _harness()
    cfg = SimpleNamespace(
        trainer={
            "freeze_modules": "",
            "learning_rate": {
                "base": 1.0e-5,
                "vggt_dense_bottleneck": 5.0e-5,
                "vggt_bottleneck_aux_plan_head": 5.0e-5,
            },
        }
    )
    # The production trainer receives an OmegaConf node; use the same access
    # contract instead of relying on Python attribute dictionaries.
    from omegaconf import OmegaConf

    groups = build_param_lr_groups(model, OmegaConf.create(cfg.__dict__))
    by_name = {group["name"]: group for group in groups}
    assert by_name["vggt_dense_bottleneck"]["lr"] == 5.0e-5
    assert by_name["vggt_bottleneck_aux_plan_head"]["lr"] == 5.0e-5
    parameter_ids = [
        id(parameter) for group in groups for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_zero_init_matches_action_only_through_the_unchanged_dit():
    from omegaconf import OmegaConf

    from starVLA.model.modules.action_model.GR00T_ActionHeader import (
        FlowmatchingActionHead,
        merge_action_context,
    )

    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 32},
                "action_model": {
                    "hidden_size": 16,
                    "action_dim": 4,
                    "action_horizon": 8,
                    "num_inference_timesteps": 2,
                    "num_timestep_buckets": 1000,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "add_pos_embed": True,
                    "max_seq_len": 16,
                    "DiTConfig": {
                        "input_embedding_dim": 16,
                        "num_attention_heads": 2,
                        "attention_head_dim": 8,
                    },
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 16,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "num_layers": 1,
                        "output_dim": 16,
                        "positional_embeddings": None,
                    },
                },
            }
        }
    )
    torch.manual_seed(9)
    action_only_head = FlowmatchingActionHead(cfg).eval()
    bottleneck_head = FlowmatchingActionHead(cfg).eval()
    bottleneck_head.load_state_dict(action_only_head.state_dict(), strict=True)
    model = _harness().eval()
    planning = torch.randn(2, 8, 32)
    examples = [
        {"vggt_dense_feature_cache": _payload(1.0, 5)},
        {"vggt_dense_feature_cache": _payload(2.0, 8)},
    ]
    dense = model._dense_batch_from_examples(examples, planning.device)

    model.vggt_bottleneck_enabled = False
    disabled, disabled_context, _ = model._condition_action_queries(planning, {})
    model.vggt_bottleneck_enabled = True
    enabled, enabled_context, _ = model._condition_action_queries(
        planning, {"dense_geometry": dense, "losses": {}, "metrics": {}}
    )
    assert disabled_context is enabled_context is None
    assert disabled.shape[1] == enabled.shape[1] == 8

    noisy_action = torch.randn(2, 8, 4)
    timestep = torch.tensor([123, 456], dtype=torch.long)

    def velocity(head, action_context):
        qwen_context = head.qwen_proj(
            merge_action_context(action_context, extra_context=None)
        )
        action_features = head.action_encoder(noisy_action, timestep)
        positions = torch.arange(8, dtype=torch.long)
        action_features = action_features + head.position_embedding(positions)[None]
        hidden = head.model(
            hidden_states=action_features,
            encoder_hidden_states=qwen_context,
            timestep=timestep,
        )
        return head.action_decoder(hidden)

    baseline_velocity = velocity(action_only_head, planning)
    disabled_velocity = velocity(bottleneck_head, disabled)
    enabled_velocity = velocity(bottleneck_head, enabled)
    torch.testing.assert_close(disabled_velocity, baseline_velocity, rtol=0.0, atol=0.0)
    torch.testing.assert_close(enabled_velocity, baseline_velocity, rtol=0.0, atol=0.0)
