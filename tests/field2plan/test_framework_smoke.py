from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.QwenOFT_Field2Plan import Qwenvl_OFT_Field2Plan
from starVLA.model.modules.field2plan.types import CameraBatch


def _config(enabled: bool):
    return SimpleNamespace(
        field2plan=SimpleNamespace(
            enabled=enabled,
            proposal=SimpleNamespace(freeze_base_planner=True, stop_gradient=True, online_fallback=False),
            visual_tap=SimpleNamespace(enabled=True, mode="explicit", num_views=1, view_order=[0]),
            geometry=SimpleNamespace(
                enabled=True, input_channels=4, channels=8, field_size=[8, 8],
                x_range_m=[0.0, 8.0], y_range_m=[-4.0, 4.0], height_anchors_m=[0.0]
            ),
            semantics=SimpleNamespace(enabled=False, input_dim=0, channels=0, num_tokens=0),
            reader=SimpleNamespace(
                output_dim=8, lateral_offsets_m=[0.0], longitudinal_offsets_m=[0.0]
            ),
            refiner=SimpleNamespace(hidden_dim=16, num_layers=1, max_delta_xy_m=4.0, max_delta_heading_rad=0.5),
            controls=SimpleNamespace(disable_access=False),
        )
    )


class FakeBaseline(nn.Module):
    def __init__(self, camera: CameraBatch) -> None:
        super().__init__()
        self.camera = camera
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.sentinel = {"action_loss": torch.tensor(2.0)}
        self.forward_calls = 0
        self.predict_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        if batch == "legacy":
            return self.sentinel
        batch_size = len(batch)
        return {
            "action_loss": self.anchor * 0.0,
            "visual_features": torch.randn(batch_size, 1, 4, 8, 8),
            "camera": self.camera,
        }

    def predict_action_infer_1d(self, batch):
        self.predict_calls += 1
        batch_size = len(batch)
        draft = torch.zeros(batch_size, 8, 4)
        draft[..., 3] = 1.0
        return {
            "normalized_actions": draft.numpy(),
            "visual_features": torch.randn(batch_size, 1, 4, 8, 8),
            "camera": self.camera,
        }


def _camera(batch_size=2):
    k = torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    return CameraBatch(
        intrinsics=k.reshape(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
        ego_to_camera=transform.reshape(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
        image_hw=torch.tensor([8.0, 8.0]).reshape(1, 1, 2).repeat(batch_size, 1, 1),
        view_names=("cam_f0",),
        frame_index=3,
    )


def test_disabled_framework_delegates_without_changing_output() -> None:
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(_config(False), baseline_model=baseline)
    assert framework("legacy") is baseline.sentinel


def test_frozen_proposal_stays_in_eval_mode_without_changing_disabled_path() -> None:
    enabled_baseline = FakeBaseline(_camera())
    enabled = Qwenvl_OFT_Field2Plan(
        _config(True), baseline_model=enabled_baseline
    )
    assert not enabled_baseline.training

    enabled.train()
    assert enabled.training
    assert enabled.geometry_field_writer.training
    assert not enabled_baseline.training

    disabled_baseline = FakeBaseline(_camera())
    disabled = Qwenvl_OFT_Field2Plan(
        _config(False), baseline_model=disabled_baseline
    )
    disabled.train()
    assert disabled_baseline.training


def test_no_teacher_framework_forward_backward_smoke() -> None:
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(_config(True), baseline_model=baseline)
    draft = torch.zeros(8, 4)
    draft[:, 3] = 1.0
    target = draft.clone()
    target[:, 0] += 0.1
    batch = [
        {"proposal": {"draft_action": draft}, "action": target, "token": f"t{i}"}
        for i in range(2)
    ]

    output = framework(batch)

    assert output["draft_action"].shape == (2, 1, 8, 4)
    assert torch.equal(output["final_action"], output["draft_action"])
    assert set(output["losses"]) == {"plan", "delta_reg"}
    sum(output["losses"].values()).backward()
    assert framework.trajectory_refiner.output_projection.weight.grad is not None


def test_enabled_inference_runs_proposal_once_and_zero_init_returns_draft() -> None:
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(_config(True), baseline_model=baseline)
    batch = [{"token": "a"}, {"token": "b"}]

    output = framework.predict_action_infer_1d(batch)

    assert baseline.predict_calls == 1
    assert baseline.forward_calls == 0
    assert output["normalized_actions"].shape == (2, 8, 4)
    expected = torch.zeros(2, 8, 4)
    expected[..., 3] = 1.0
    assert torch.equal(torch.from_numpy(output["normalized_actions"]), expected)


def test_non_strict_baseline_checkpoint_mismatch_is_not_silent(tmp_path) -> None:
    baseline = FakeBaseline(_camera())
    checkpoint = tmp_path / "bad.pt"
    torch.save({"wrong.weight": torch.ones(1)}, checkpoint)
    config = _config(True)
    config.field2plan.proposal.checkpoint = str(checkpoint)
    config.field2plan.proposal.checkpoint_strict = False

    with pytest.raises(RuntimeError, match="baseline checkpoint mismatch"):
        Qwenvl_OFT_Field2Plan(config, baseline_model=baseline)


def test_explicit_checkpoint_prefix_allowlist_accepts_removed_heads(tmp_path) -> None:
    baseline = FakeBaseline(_camera())
    checkpoint = tmp_path / "partial.pt"
    torch.save(
        {
            "anchor": torch.tensor(3.0),
            "rgb_model.unused": torch.ones(1),
        },
        checkpoint,
    )
    config = _config(True)
    config.field2plan.proposal.checkpoint = str(checkpoint)
    config.field2plan.proposal.checkpoint_strict = False
    config.field2plan.proposal.allowed_unexpected_prefixes = ["rgb_model."]

    framework = Qwenvl_OFT_Field2Plan(config, baseline_model=baseline)

    assert framework.baseline_model.anchor.item() == 3.0


def _teacher(depth=4.0):
    return {
        "depth_m": torch.full((1, 8, 8), depth),
        "confidence": torch.ones(1, 8, 8),
        "valid_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        "source_image_hw": torch.tensor([[8, 8]], dtype=torch.int64),
        "depth_hw": torch.tensor([[8, 8]], dtype=torch.int64),
        "resize_scale_xy": torch.ones(1, 2),
        "view_names": ("cam_f0",),
        "coordinate_frame": "camera_optical_z_depth_m",
    }


def test_geometry_supervision_is_structured_spatial_and_has_gradient() -> None:
    config = _config(True)
    config.field2plan.geometry.supervision = SimpleNamespace(
        enabled=True,
        build_head=True,
        hidden_channels=12,
        occupancy_threshold_m=0.5,
        free_space_margin_m=0.5,
        relative_depth_scale_m=4.0,
        max_depth_residual_m=20.0,
        min_teacher_depth_m=0.1,
        max_teacher_depth_m=100.0,
    )
    config.field2plan.controls.random_teacher = False
    config.field2plan.controls.shuffle_teacher_across_batch = False
    config.field2plan.controls.gt_mlp_teacher = False
    config.field2plan.controls.equal_capacity_no_teacher = False
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(config, baseline_model=baseline)
    draft = torch.zeros(8, 4)
    draft[:, 3] = 1.0
    batch = [
        {
            "proposal": {"draft_action": draft},
            "action": draft.clone(),
            "state": torch.zeros(1, 4),
            "geometry_teacher": _teacher(depth=4.0 + i),
            "token": f"t{i}",
        }
        for i in range(2)
    ]

    output = framework(batch)

    assert set(output["losses"]) == {
        "plan",
        "delta_reg",
        "geometry_depth",
        "geometry_occupancy",
        "geometry_free_space",
        "geometry_relative",
    }
    assert output["metrics"]["geometry_valid_ratio"] > 0
    sum(output["losses"].values()).backward()
    assert any(
        parameter.grad is not None
        for parameter in framework.geometry_supervision_head.parameters()
    )


def test_equal_capacity_no_teacher_keeps_head_but_requires_no_cache() -> None:
    config = _config(True)
    config.field2plan.geometry.supervision = SimpleNamespace(
        enabled=False,
        build_head=True,
        hidden_channels=12,
        max_depth_residual_m=20.0,
    )
    config.field2plan.controls.equal_capacity_no_teacher = True
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(config, baseline_model=baseline)
    draft = torch.zeros(8, 4)
    draft[:, 3] = 1.0
    batch = [
        {
            "proposal": {"draft_action": draft},
            "action": draft.clone(),
            "state": torch.zeros(1, 4),
            "token": f"t{i}",
        }
        for i in range(2)
    ]

    output = framework(batch)

    assert framework.geometry_supervision_head is not None
    assert output["losses"]["geometry_depth"].item() == 0.0
    sum(output["losses"].values()).backward()
    assert all(
        parameter.grad is not None
        for parameter in framework.geometry_supervision_head.parameters()
    )


def test_gt_mlp_control_rejects_external_geometry_supervision() -> None:
    config = _config(True)
    config.field2plan.geometry.supervision = SimpleNamespace(enabled=True)
    config.field2plan.controls.gt_mlp_teacher = True
    with pytest.raises(ValueError, match="GT-MLP"):
        Qwenvl_OFT_Field2Plan(config, baseline_model=FakeBaseline(_camera()))


def _temporal_sample() -> dict:
    future_camera = {
        "view_names": ["cam_f0"],
        "frame_indices": torch.arange(4, 12),
        "intrinsics": _camera(batch_size=8).intrinsics[:, 0].reshape(8, 1, 3, 3),
        "ego_to_camera": _camera(batch_size=8).ego_to_camera[:, 0].reshape(8, 1, 4, 4),
        "image_hw": _camera(batch_size=8).image_hw[:, 0].reshape(8, 1, 2),
        "valid_mask": torch.ones(8, 1, dtype=torch.bool),
    }
    return {
        "current_frame_index": 3,
        "history_frame_indices": torch.arange(4),
        "future_frame_indices": torch.arange(4, 12),
        "frame_times_s": (torch.arange(12) - 3) * 0.5,
        "current_from_ego": torch.eye(4).reshape(1, 4, 4).repeat(12, 1, 1),
        "ego_from_current": torch.eye(4).reshape(1, 4, 4).repeat(12, 1, 1),
        "valid_mask": torch.ones(12, dtype=torch.bool),
        "future_camera": future_camera,
    }


def _dynamics_teacher() -> dict:
    return {
        "features": torch.randn(8, 1, 6, 8, 8).half(),
        "confidence": torch.ones(8, 1, 8, 8),
        "valid_mask": torch.ones(8, 1, 8, 8, dtype=torch.bool),
        "frame_indices": torch.arange(4, 12),
        "frame_times_s": torch.arange(1, 9) * 0.5,
        "source_image_hw": torch.full((8, 1, 2), 8, dtype=torch.int64),
        "feature_hw": torch.full((8, 1, 2), 8, dtype=torch.int64),
        "view_names": ("cam_f0",),
        "future_frame_indices": tuple(range(4, 12)),
        "spatial_layout": "per_view_patch_grid",
        "feature_normalization": "l2",
    }


def test_dynamics_framework_is_action_free_zero_init_and_has_auxiliary_gradient() -> None:
    config = _config(True)
    config.field2plan.dynamics = SimpleNamespace(
        enabled=True,
        access_enabled=True,
        channels=12,
        horizon=8,
        history_length=4,
        hidden_channels=16,
        frame_interval_s=0.5,
        history_frame_indices=[0, 1, 2, 3],
        future_frame_indices=list(range(4, 12)),
        temporal_interpolation="linear",
        source_dropout_p=0.0,
        supervision=SimpleNamespace(
            enabled=True,
            build_head=True,
            teacher_channels=6,
            hidden_channels=12,
            temporal_contrast_margin=0.05,
            normalize_features=True,
        ),
    )
    config.field2plan.controls.dynamics_random_teacher = False
    config.field2plan.controls.dynamics_shuffle_teacher_across_batch = False
    config.field2plan.controls.temporal_shuffle_teacher = False
    baseline = FakeBaseline(_camera())
    framework = Qwenvl_OFT_Field2Plan(config, baseline_model=baseline)
    draft = torch.zeros(8, 4)
    draft[:, 3] = 1.0
    batch = [
        {
            "proposal": {"draft_action": draft},
            "action": draft.clone(),
            "state": torch.zeros(1, 4),
            "token": f"d{i}",
            "temporal": _temporal_sample(),
            "dynamics_teacher": _dynamics_teacher(),
        }
        for i in range(2)
    ]

    output = framework(batch)

    assert torch.equal(output["final_action"], output["draft_action"])
    assert output["metrics"]["dynamics_valid_ratio"] > 0
    assert "dynamics_cosine" in output["losses"]
    sum(output["losses"].values()).backward()
    assert any(
        parameter.grad is not None
        for parameter in framework.dynamics_field_writer.parameters()
    )
