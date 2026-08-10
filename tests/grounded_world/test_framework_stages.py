from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from starVLA.model.framework.QwenOFT_GroundedWorld import (
    Qwenvl_OFT_GroundedWorld,
)


class FakePlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.predict_calls = 0
        self.forward_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        return {"action_loss": self.anchor * 0.0}

    def predict_action_infer_1d(self, batch):
        self.predict_calls += 1
        action = torch.zeros(len(batch), 8, 4)
        action[..., 3] = 1.0
        return {"normalized_actions": action.numpy()}


class FeaturePlanner(FakePlanner):
    """Expose features only from the differentiable training forward."""

    def forward(self, batch):
        output = super().forward(batch)
        output["visual_features"] = self.anchor * torch.ones(
            len(batch), 1, 4, 4, 4
        )
        output["visual_view_names"] = ("cam_f0",)
        return output


def _config(stage: str, future: bool, source: str = "driving_jepa"):
    return SimpleNamespace(
        grounded_world=SimpleNamespace(
            enabled=True,
            training=SimpleNamespace(
                stage=stage,
                phase="A",
                init_checkpoint=None,
                direct_init=False,
            ),
            visual_tap=SimpleNamespace(
                mode="explicit",
                input_channels=4,
                view_names=["cam_f0"],
                view_order=[0],
            ),
            memory=SimpleNamespace(
                geometry_channels=[16, 24, 32],
                geometry_scale_factors=[1, 2, 4],
                dynamics_channels=12,
                history_length=4,
                horizon=8,
                hidden_channels=24,
                field_size=[16, 16],
                x_range_m=[-8.0, 56.0],
                y_range_m=[-32.0, 32.0],
                height_anchors_m=[0.0],
            ),
            prior=SimpleNamespace(
                source=source,
                teacher_mode="none" if source == "none" else "real",
                retention_enabled=stage != "prior",
                teacher_channels=6,
                control_seed=9,
            ),
            future=SimpleNamespace(
                enabled=future,
                target=SimpleNamespace(
                    source="student_ema",
                    shared_across_teacher_controls=True,
                ),
            ),
            planner=SimpleNamespace(
                enabled=stage == "planning",
                baseline_checkpoint=None,
                world_access=True,
                tube=SimpleNamespace(
                    output_dim=20,
                    lateral_offsets_m=[-1.0, 0.0, 1.0],
                    longitudinal_offsets_m=[0.0, 2.5],
                ),
                refiner=SimpleNamespace(
                    hidden_dim=32,
                    num_layers=2,
                    max_delta_xy_m=4.0,
                    max_delta_heading_rad=0.5,
                ),
            ),
            consequence=SimpleNamespace(
                enabled=False, inference_enabled=False, hidden_dim=24
            ),
        )
    )


def _base_sample(index: int) -> dict:
    return {
        "token": f"token-{index}",
        "world_geometry": torch.randn(16, 16, 16),
        "history_current_from_ego": torch.eye(4).repeat(4, 1, 1),
        "grounded_world_prior": {
            "features": torch.randn(4, 3, 6, 2, 2),
            "confidence": torch.ones(4, 3, 2, 2),
            "teacher": "driving_jepa",
        },
    }


def _camera_sample(index: int) -> dict:
    sample = _base_sample(index)
    sample.pop("world_geometry")
    sample["camera"] = {
        "view_names": ["cam_f0"],
        "frame_index": 3,
        "intrinsics": torch.tensor(
            [[[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        "ego_to_camera": torch.tensor(
            [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.5],
              [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
        "image_hw": torch.tensor([[4.0, 4.0]]),
        "transform_status": "explicit",
    }
    return sample


def test_stage1_does_not_call_planner_or_require_draft() -> None:
    planner = FakePlanner()
    model = Qwenvl_OFT_GroundedWorld(
        _config("prior", future=False), baseline_model=planner, load_checkpoints=False
    )
    output = model([_base_sample(0), _base_sample(1)])
    assert planner.predict_calls == 0
    assert set(output["losses"]) == {"prior_cosine", "prior_smooth_l1"}
    sum(output["losses"].values()).backward()
    assert model.world_core.current_dynamics_encoder.spatial_projection[0].weight.grad is not None


def test_stage2_uses_shared_ema_future_target_without_planner() -> None:
    planner = FakePlanner()
    model = Qwenvl_OFT_GroundedWorld(
        _config("predictive", future=True),
        baseline_model=planner,
        load_checkpoints=False,
    )
    batch = [_base_sample(0), _base_sample(1)]
    for sample in batch:
        sample["grounded_world_future_target"] = {
            "features": torch.randn(8, 12, 16, 16),
            "valid_mask": torch.ones(8, 16, 16, dtype=torch.bool),
            "source": "student_ema",
        }
    output = model(batch)
    assert planner.predict_calls == 0
    assert {"future_cosine", "future_smooth_l1", "future_temporal_contrast"}.issubset(
        output["losses"]
    )


def test_stage3_generates_internal_draft_once_and_zero_init_is_exact() -> None:
    planner = FakePlanner()
    model = Qwenvl_OFT_GroundedWorld(
        _config("planning", future=False, source="none"),
        baseline_model=planner,
        load_checkpoints=False,
    )
    batch = [_base_sample(0), _base_sample(1)]
    for sample in batch:
        action = torch.zeros(8, 4)
        action[..., 3] = 1.0
        sample["action"] = action
    output = model(batch)
    assert planner.predict_calls == 1
    assert torch.equal(output["final_action"], output["draft_action"])
    assert set(output["losses"]) == {"plan", "delta_reg"}


def test_stage3_inference_needs_no_gt_or_teacher_target() -> None:
    planner = FakePlanner()
    config = _config("planning", future=True, source="none")
    config.grounded_world.consequence.enabled = True
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=planner, load_checkpoints=False
    )
    output = model.predict_action_infer_1d([_base_sample(0), _base_sample(1)])
    assert output["normalized_actions"].shape == (2, 8, 4)
    assert planner.predict_calls == 1
    assert set(output["diagnostics"]) == {
        "draft_action",
        "final_action",
        "delta_physical",
        "source_gates",
        "tube_valid_mask",
        "tube_points",
    }
    assert output["diagnostics"]["source_gates"].shape[:2] == (2, 8)


def test_same_checkpoint_inference_can_remove_world_access() -> None:
    config = _config("planning", future=False, source="none")
    config.grounded_world.diagnostics = SimpleNamespace(
        inference_intervention="disable_access"
    )
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=FakePlanner(), load_checkpoints=False
    )
    output = model.predict_action_infer_1d([_base_sample(0), _base_sample(1)])
    assert not output["diagnostics"]["source_gates"].any()


def test_stage3_phase_b_runs_flow_training_forward_once() -> None:
    planner = FakePlanner()
    config = _config("planning", future=False, source="none")
    config.grounded_world.training.phase = "B"
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=planner, load_checkpoints=False
    )
    batch = [_base_sample(0), _base_sample(1)]
    for sample in batch:
        action = torch.zeros(8, 4)
        action[..., 3] = 1.0
        sample["action"] = action
    output = model(batch)
    assert planner.forward_calls == 1
    assert planner.predict_calls == 1
    assert "baseline_plan" in output["losses"]


def test_stage3_phase_b_world_loss_reuses_differentiable_forward_features() -> None:
    planner = FeaturePlanner()
    config = _config("planning", future=False, source="none")
    config.grounded_world.training.phase = "B"
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=planner, load_checkpoints=False
    )
    sample = _camera_sample(0)
    action = torch.zeros(8, 4)
    action[..., 3] = 1.0
    sample["action"] = action
    output = model([sample])
    sum(output["losses"].values()).backward()
    assert planner.forward_calls == 1
    assert planner.anchor.grad is not None


def test_gt_task_control_uses_current_state_and_route_without_future() -> None:
    planner = FakePlanner()
    config = _config("prior", future=False, source="gt_task_mlp")
    config.grounded_world.prior.teacher_mode = "gt_task_mlp"
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=planner, load_checkpoints=False
    )
    batch = [_base_sample(0), _base_sample(1)]
    for sample in batch:
        sample.pop("grounded_world_prior")
        sample["state"] = torch.zeros(4)
        sample["navigation_command"] = "keep straight"
    output = model(batch)
    assert set(output["losses"]) == {"prior_cosine", "prior_smooth_l1"}


def test_gt_task_control_can_keep_the_same_vggt_geometry_supervision() -> None:
    config = _config("prior", future=False, source="vggt_gt_task_mlp")
    config.grounded_world.prior.teacher_mode = "gt_task_mlp"
    model = Qwenvl_OFT_GroundedWorld(
        config, baseline_model=FakePlanner(), load_checkpoints=False
    )
    assert model.geometry_supervision_enabled
    assert model.dynamics_prior_supervision_enabled


def test_stage1_ema_updates_without_becoming_trainable() -> None:
    model = Qwenvl_OFT_GroundedWorld(
        _config("prior", future=False),
        baseline_model=FakePlanner(),
        load_checkpoints=False,
    )
    source = next(model.world_core.parameters())
    target = next(model.ema_world_core.parameters())
    before = target.detach().clone()
    with torch.no_grad():
        source.add_(1.0)
    model.update_ema()
    assert not torch.equal(target, before)
    assert all(not parameter.requires_grad for parameter in model.ema_world_core.parameters())


def _save_state(path: Path, module: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def test_stage3_phase_a_loads_stage1_world_with_ema_and_pure_baseline(
    tmp_path: Path,
) -> None:
    stage1 = Qwenvl_OFT_GroundedWorld(
        _config("prior", future=False),
        baseline_model=FakePlanner(),
        load_checkpoints=False,
    )
    world_path = tmp_path / "stage1.pt"
    _save_state(world_path, stage1)
    pure = FakePlanner()
    with torch.no_grad():
        pure.anchor.fill_(7.0)
    baseline_path = tmp_path / "baseline.pt"
    _save_state(baseline_path, pure)

    config = _config("planning", future=False)
    config.grounded_world.training.init_checkpoint = str(world_path)
    config.grounded_world.planner.baseline_checkpoint = str(baseline_path)
    model = Qwenvl_OFT_GroundedWorld(
        config,
        baseline_model=FakePlanner(),
        load_checkpoints=True,
    )
    assert model.baseline_model.anchor.item() == 7.0
    assert all(
        key.startswith(("ema_geometry_grounder.", "ema_world_core."))
        for key in model.world_checkpoint_report["unexpected_allowed"]
    )


def test_stage3_phase_b_direct_init_loads_world_and_pure_baseline(
    tmp_path: Path,
) -> None:
    stage1 = Qwenvl_OFT_GroundedWorld(
        _config("prior", future=False),
        baseline_model=FakePlanner(),
        load_checkpoints=False,
    )
    world_path = tmp_path / "stage1.pt"
    _save_state(world_path, stage1)
    pure = FakePlanner()
    with torch.no_grad():
        pure.anchor.fill_(9.0)
    baseline_path = tmp_path / "baseline.pt"
    _save_state(baseline_path, pure)

    config = _config("planning", future=False)
    config.grounded_world.training.phase = "B"
    config.grounded_world.training.direct_init = True
    config.grounded_world.training.init_checkpoint = str(world_path)
    config.grounded_world.planner.baseline_checkpoint = str(baseline_path)
    model = Qwenvl_OFT_GroundedWorld(
        config,
        baseline_model=FakePlanner(),
        load_checkpoints=True,
    )
    assert model.baseline_model.anchor.item() == 9.0
    assert model.baseline_model.anchor.requires_grad
