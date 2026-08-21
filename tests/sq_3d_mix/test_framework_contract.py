from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.QwenOFT_SQ3DMix import (
    Qwenvl_OFT_SQ3DMix,
    apply_sq3dmix_intervention,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    merge_action_context,
)
from starVLA.model.modules.vggt_query.scene_query_compressor import (
    SceneQueryCompressor,
)
from starVLA.model.modules.vggt_query.sq_3d_mix import (
    SceneConditionedGatedFusion,
)
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_per_view,
)
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


ACTION_IDS = tuple(range(100, 108))


def _payload(value: float, feature_dim: int = 12):
    grids = torch.tensor([[2, 3], [2, 3], [2, 3]], dtype=torch.int16)
    return {
        "features": torch.full((18, feature_dim), value),
        "valid_mask": torch.ones(18, dtype=torch.bool),
        "patch_grid_hw": grids,
    }


def _harness(mode: str, batch_size: int = 2) -> Qwenvl_OFT_SQ3DMix:
    torch.manual_seed(13)
    model = Qwenvl_OFT_SQ3DMix.__new__(Qwenvl_OFT_SQ3DMix)
    nn.Module.__init__(model)
    model.baseline_weight = nn.Parameter(torch.ones(()))
    model.fusion_mode = mode
    model.scene_query_compressor = SceneQueryCompressor(
        input_dim=16,
        hidden_dim=8,
        num_queries=16,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        query_init_std=1e-6,
    )
    model.vggt_feature_dim = 12
    model.vggt_view_count = 3
    model.vggt_view_order = ["cam_f0", "cam_l0", "cam_r0"]
    model.vggt_output_hw = (6, 10)
    model.gated_fusion = SceneConditionedGatedFusion(scene_dim=16, vggt_dim=12)
    model._sq3dmix_intervention_mode = "real"
    model._sq3dmix_intervention_seed = 20260821
    model._special_token_ids = {"action": ACTION_IDS}
    model._use_named_loss_contract = True
    model._configure_fusion_trainability()
    model.eval()
    model._test_batch_size = batch_size
    return model


def _inputs(batch_size: int = 2):
    sequence_length = 24
    hidden = torch.randn(batch_size, sequence_length, 16)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    action_positions = torch.arange(8, 16).expand(batch_size, -1).clone()
    input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    for offset, token_id in enumerate(ACTION_IDS):
        input_ids[:, 8 + offset] = token_id
    action_queries = hidden.gather(
        1,
        action_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
    )
    examples = [
        {"vggt_dense_feature_cache": _payload(float(index + 1))}
        for index in range(batch_size)
    ]
    return hidden, attention_mask, action_positions, input_ids, action_queries, examples


class _NoCacheAccess(dict):
    def get(self, key, default=None):
        raise AssertionError(f"scene_only attempted to read {key}")


def test_scene_only_context_contract():
    model = _harness("scene_only")
    hidden, attention, positions, _, actions, _ = _inputs()
    context, _ = model._build_sq3dmix_context(
        hidden,
        attention,
        positions,
        [_NoCacheAccess(), _NoCacheAccess()],
    )

    conditioned, returned_context, _ = model._condition_action_queries(
        actions, {"action_context": context, "metrics": {}}
    )
    assert conditioned is actions
    assert returned_context.shape == (2, 16, 16)
    torch.testing.assert_close(returned_context, context)


def test_projected_concat_context_contract():
    model = _harness("projected_concat")
    hidden, attention, positions, _, _, examples = _inputs()
    context, _ = model._build_sq3dmix_context(
        hidden, attention, positions, examples
    )
    scene_mask = model._build_scene_memory_mask(attention, positions)
    scene, _ = model.scene_query_compressor(hidden, scene_mask)
    pooled = pool_dense_vggt_per_view(
        [example["vggt_dense_feature_cache"] for example in examples],
        output_hw=(6, 10),
        dtype=scene.dtype,
    )
    projected = model.gated_fusion.project_geometry(pooled)

    assert context.shape == (2, 196, 16)
    torch.testing.assert_close(context[:, :16], scene)
    torch.testing.assert_close(context[:, 16:], projected)


def test_gated_context_contract():
    model = _harness("gated")
    hidden, attention, positions, _, _, examples = _inputs()
    context, _ = model._build_sq3dmix_context(
        hidden, attention, positions, examples
    )
    scene_mask = model._build_scene_memory_mask(attention, positions)
    scene, _ = model.scene_query_compressor(hidden, scene_mask)
    pooled = pool_dense_vggt_per_view(
        [example["vggt_dense_feature_cache"] for example in examples],
        output_hw=(6, 10),
        dtype=scene.dtype,
    )
    fused, _ = model.gated_fusion(scene, pooled)

    assert context.shape == (2, 196, 16)
    torch.testing.assert_close(context[:, :16], scene)
    torch.testing.assert_close(context[:, 16:], fused)


def test_final_action_condition_order():
    actions = torch.full((2, 8, 16), -1.0)
    scene = torch.full((2, 16, 16), 2.0)
    geometry = torch.full((2, 180, 16), 3.0)
    condition = merge_action_context(actions, torch.cat([scene, geometry], dim=1))

    assert condition.shape == (2, 204, 16)
    torch.testing.assert_close(condition[:, :8], actions)
    torch.testing.assert_close(condition[:, 8:24], scene)
    torch.testing.assert_close(condition[:, 24:], geometry)


def test_action_queries_are_not_modified():
    model = _harness("gated")
    actions = torch.randn(2, 8, 16)
    context = torch.randn(2, 196, 16)

    conditioned, returned_context, _ = model._condition_action_queries(
        actions,
        {"action_context": context, "metrics": {}},
    )

    assert conditioned is actions
    torch.testing.assert_close(conditioned, actions, rtol=0.0, atol=0.0)
    assert returned_context is context


def test_action_queries_do_not_enter_gate():
    model = _harness("gated")
    hidden, attention, positions, _, _, examples = _inputs()
    changed = hidden.clone()
    changed.scatter_(
        1,
        positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        torch.randn(2, 8, 16) * 1000,
    )

    expected, _ = model._build_sq3dmix_context(
        hidden, attention, positions, examples
    )
    actual, _ = model._build_sq3dmix_context(
        changed, attention, positions, examples
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_train_inference_context_consistency():
    model = _harness("gated")
    hidden, attention, positions, input_ids, actions, examples = _inputs()
    extension = model._compute_query_extension(
        hidden,
        {"action": positions},
        examples,
        input_ids=input_ids,
        attention_mask=attention,
    )
    train_actions, train_context, _ = model._condition_action_queries(
        actions, extension
    )
    infer_actions, infer_context, _ = model._condition_inference_action_queries(
        hidden,
        input_ids,
        actions,
        attention_mask=attention,
        examples=examples,
    )

    torch.testing.assert_close(train_actions, infer_actions, rtol=0.0, atol=0.0)
    torch.testing.assert_close(train_context, infer_context, rtol=0.0, atol=0.0)


def test_zero_intervention():
    tokens = torch.randn(2, 180, 12)
    intervened, metrics = apply_sq3dmix_intervention(tokens, "zero", 3)

    assert torch.count_nonzero(intervened) == 0
    assert metrics == {}


def test_gaussian_intervention_is_reproducible():
    tokens = torch.zeros(2, 180, 12)
    first, _ = apply_sq3dmix_intervention(tokens, "gaussian", 17)
    second, _ = apply_sq3dmix_intervention(tokens, "gaussian", 17)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(first) > 0


def test_shuffled_intervention_moves_scene_geometry_together():
    tokens = torch.stack(
        [torch.full((180, 12), 1.0), torch.full((180, 12), 2.0)]
    )
    shuffled, metrics = apply_sq3dmix_intervention(tokens, "shuffled", 3)

    torch.testing.assert_close(shuffled[0], tokens[1])
    torch.testing.assert_close(shuffled[1], tokens[0])
    assert metrics == {}


def test_shuffled_batch_one_records_skip():
    tokens = torch.randn(1, 180, 12)
    shuffled, metrics = apply_sq3dmix_intervention(tokens, "shuffled", 3)

    torch.testing.assert_close(shuffled, tokens, rtol=0.0, atol=0.0)
    assert metrics["sq3dmix/intervention_skipped"].item() == 1


def test_action_only_checkpoint_compatibility():
    model = _harness("gated")
    full = model.state_dict()
    action_only = {
        key: value
        for key, value in full.items()
        if not key.startswith(("scene_query_compressor.", "gated_fusion."))
    }

    model.load_state_dict(action_only, strict=True)
    missing_baseline = dict(action_only)
    missing_baseline.pop("baseline_weight")
    with pytest.raises(RuntimeError, match="baseline_weight"):
        model.load_state_dict(missing_baseline, strict=True)
    unexpected = dict(action_only)
    unexpected["unexpected.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        model.load_state_dict(unexpected, strict=True)


def test_missing_vggt_cache_fails_closed():
    model = _harness("gated")
    hidden, attention, positions, _, _, _ = _inputs()

    with pytest.raises(RuntimeError, match="missing batch indices"):
        model._build_sq3dmix_context(
            hidden,
            attention,
            positions,
            [{}, {}],
        )


def test_qwen_oft_baseline_contract_unchanged():
    model = Qwenvl_OFT.__new__(Qwenvl_OFT)
    nn.Module.__init__(model)
    actions = torch.randn(2, 8, 16)
    extension = model._compute_query_extension(
        torch.randn(2, 20, 16),
        {"action": torch.arange(8).expand(2, -1)},
        [{}, {}],
        attention_mask=torch.ones(2, 20, dtype=torch.long),
    )
    conditioned, context, metrics = model._condition_action_queries(actions, extension)
    inferred, inference_context, inference_metrics = (
        model._condition_inference_action_queries(
            torch.randn(2, 20, 16),
            torch.zeros(2, 20, dtype=torch.long),
            actions,
            attention_mask=torch.ones(2, 20, dtype=torch.long),
        )
    )

    assert conditioned is inferred is actions
    assert context is inference_context is None
    assert metrics == inference_metrics == {}
    assert merge_action_context(actions, None) is actions


def test_mode_trainability_and_named_action_loss_only():
    expected_trainable = {
        "scene_only": set(),
        "projected_concat": {"vggt_projection"},
        "gated": {
            "vggt_projection",
            "gate_projection",
            "semantic_projection",
            "geometry_projection",
        },
    }
    for mode, expected in expected_trainable.items():
        model = _harness(mode)
        actual = {
            name
            for name in (
                "vggt_projection",
                "gate_projection",
                "semantic_projection",
                "geometry_projection",
            )
            if any(parameter.requires_grad for parameter in getattr(model.gated_fusion, name).parameters())
        }
        assert actual == expected
        output = model._attach_framework_outputs(
            {"action_loss": torch.ones(())},
            torch.ones(()),
            {"losses": {}, "metrics": {}},
            {},
        )
        assert set(output["losses"]) == {"action"}


def test_planning_usage_metrics_reports_nonzero_sq3dmix_gradients():
    model = _harness("gated")
    hidden, attention, positions, _, _, examples = _inputs()
    context, _ = model._build_sq3dmix_context(
        hidden,
        attention,
        positions,
        examples,
    )

    context.square().mean().backward()
    metrics = model.get_planning_usage_metrics()

    assert set(metrics) == {
        "sq3dmix/scene_input_projection_grad_norm",
        "sq3dmix/scene_output_projection_grad_norm",
        "sq3dmix/scene_query_grad_norm",
        "sq3dmix/vggt_projection_grad_norm",
        "sq3dmix/gate_projection_grad_norm",
        "sq3dmix/semantic_projection_grad_norm",
        "sq3dmix/geometry_projection_grad_norm",
    }
    assert all(metric.item() > 0 for metric in metrics.values())


def test_optimizer_groups_and_overlay_preserve_baseline_learning_rates():
    overlay = OmegaConf.load("starVLA/config/training/sq_3d_mix.yaml")
    assert "qwen_vl_interface" not in overlay.trainer.learning_rate
    assert "action_model" not in overlay.trainer.learning_rate
    assert "freeze_modules" not in overlay.trainer
    model = _harness("gated")
    cfg = OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "",
                "learning_rate": {
                    "base": 1e-5,
                    "scene_query_compressor": 3e-5,
                    "gated_fusion": 1e-4,
                },
            }
        }
    )
    groups = {group["name"]: group for group in build_param_lr_groups(model, cfg)}
    assert groups["scene_query_compressor"]["lr"] == 3e-5
    assert groups["gated_fusion"]["lr"] == 1e-4


def test_forbidden_legacy_vggt_components_are_not_reused():
    source = Path("starVLA/model/framework/QwenOFT_SQ3DMix.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "PlanningConditionedDenseVGGTBottleneck",
        "V3ResidualGeometryFusion",
        "WaypointGeometryReader",
        "VGGTQueryAligner",
        "PhysicalGeometryHead",
        "AuxiliaryTrajectoryHead",
    ):
        assert forbidden not in source
