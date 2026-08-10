"""Revised GroundedWorld-VLA framework with separated prior/future supervision."""

from __future__ import annotations

from contextlib import nullcontext
import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.QwenOFT_Field2Plan import (
    Qwenvl_OFT_Field2Plan,
    _select,
)
from starVLA.model.modules.field2plan.geometry_field_writer import GeometryFieldWriter
from starVLA.model.modules.field2plan.geometry_supervision import (
    GeometrySupervisionHead,
    build_geometry_targets,
    geometry_supervision_losses,
)
from starVLA.model.modules.field2plan.losses import trajectory_refinement_losses
from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.field2plan.trajectory_refiner import TrajectoryRefiner
from starVLA.model.modules.field2plan.types import RefinerOutput
from starVLA.model.modules.field2plan.visual_feature_tap import VisualFeatureTap
from starVLA.model.modules.grounded_world.consequence import (
    ConsequenceTargets,
    PlanningConsequenceHead,
    consequence_losses,
)
from starVLA.model.modules.grounded_world.core import (
    GroundedWorldCore,
    GroundedWorldMemoryOutput,
)
from starVLA.model.modules.grounded_world.prior_adapter import ExternalPriorAdapter
from starVLA.model.modules.grounded_world.supervision import (
    FeatureAlignmentTarget,
    FutureTargetContract,
    future_prediction_losses,
    global_alignment_losses,
)
from starVLA.model.modules.grounded_world.trajectory_tube_reader import (
    MultiScaleTrajectoryTubeReader,
)


@FRAMEWORK_REGISTRY.register("QwenOFT_GroundedWorld")
class Qwenvl_OFT_GroundedWorld(Qwenvl_OFT_Field2Plan):
    """Three-stage GroundedWorld algorithm from the revised research plan.

    Stage I/II never request a proposal trajectory. Stage III obtains one
    first-pass trajectory from this framework's VLM+DiT planner, then performs
    one zero-initialized world-grounded refinement. External prior supervision
    is current/history-only; future targets are a separate shared student/EMA
    cache and cannot be replaced by an external future teacher.
    """

    def __init__(
        self,
        config: Any,
        accelerator: Any = None,
        baseline_model: Optional[nn.Module] = None,
        load_checkpoints: bool = True,
    ) -> None:
        # The parent is deliberately initialized with no field2plan section.
        # This reuses its tested Qwen visual/camera helpers without constructing
        # the legacy Field2Plan draft/cache modules.
        super().__init__(config, accelerator, baseline_model=baseline_model)
        self.enabled = bool(_select(config, "grounded_world.enabled", False))
        if not self.enabled:
            return
        self.stage = str(_select(config, "grounded_world.training.stage", "prior"))
        if self.stage not in {"prior", "predictive", "planning"}:
            raise ValueError("grounded_world.training.stage is invalid")
        self.stage3_phase = str(
            _select(config, "grounded_world.training.phase", "A")
        ).upper()
        if self.stage == "planning" and self.stage3_phase not in {"A", "B"}:
            raise ValueError("GroundedWorld planning phase must be A or B")
        self.stage3_direct_init = bool(
            _select(config, "grounded_world.training.direct_init", False)
        )
        if self.stage3_direct_init and not (
            self.stage == "planning" and self.stage3_phase == "B"
        ):
            raise ValueError("direct_init is supported only for planning Phase B")

        self.planner_enabled = bool(
            _select(config, "grounded_world.planner.enabled", False)
        )
        if self.planner_enabled != (self.stage == "planning"):
            raise ValueError("planner.enabled must be true only for planning stage")
        self.future_enabled = bool(
            _select(config, "grounded_world.future.enabled", False)
        )
        if self.stage == "prior" and self.future_enabled:
            raise ValueError("Stage I cannot enable future prediction")
        if self.stage == "predictive" and not self.future_enabled:
            raise ValueError("Stage II requires future prediction")
        target_source = str(
            _select(config, "grounded_world.future.target.source", "student_ema")
        )
        shared_target = bool(
            _select(
                config,
                "grounded_world.future.target.shared_across_teacher_controls",
                True,
            )
        )
        self.future_target_contract = FutureTargetContract(
            source=target_source,
            target_id="grounded-world-student-ema-v1",
            shared_across_teacher_controls=shared_target,
        ).validate()

        self.freeze_base_planner = self.stage != "planning" or self.stage3_phase == "A"
        if self.freeze_base_planner:
            for parameter in self.baseline_model.parameters():
                parameter.requires_grad_(False)
            self.baseline_model.eval()
        self.stop_gradient = True
        self.online_fallback = True

        view_names = tuple(
            _select(
                config,
                "grounded_world.visual_tap.view_names",
                ["cam_f0", "cam_l0", "cam_r0"],
            )
        )
        view_order = tuple(
            int(value)
            for value in _select(
                config,
                "grounded_world.visual_tap.view_order",
                list(range(len(view_names))),
            )
        )
        self.visual_feature_tap = VisualFeatureTap(
            enabled=True,
            mode=str(_select(config, "grounded_world.visual_tap.mode", "hook")),
            num_views=len(view_names),
            view_order=view_order,
            view_names=view_names,
        )
        self.visual_output_view_names = tuple(view_names[index] for index in view_order)

        geometry_channels = tuple(
            int(value)
            for value in _select(
                config,
                "grounded_world.memory.geometry_channels",
                [128, 192, 256],
            )
        )
        scale_factors = tuple(
            int(value)
            for value in _select(
                config,
                "grounded_world.memory.geometry_scale_factors",
                [1, 2, 4],
            )
        )
        dynamics_channels = int(
            _select(config, "grounded_world.memory.dynamics_channels", 192)
        )
        self.dynamics_channels = dynamics_channels
        self.dynamics_history_length = int(
            _select(config, "grounded_world.memory.history_length", 4)
        )
        self.dynamics_horizon = int(
            _select(config, "grounded_world.memory.horizon", 8)
        )
        self.dynamics_history_indices = tuple(range(self.dynamics_history_length))
        self.dynamics_future_indices = tuple(
            range(self.dynamics_history_length, self.dynamics_history_length + self.dynamics_horizon)
        )
        field_size = tuple(
            int(value)
            for value in _select(config, "grounded_world.memory.field_size", [64, 64])
        )
        x_range = _select(config, "grounded_world.memory.x_range_m", [-8.0, 56.0])
        y_range = _select(config, "grounded_world.memory.y_range_m", [-32.0, 32.0])
        self.geometry_grounder = GeometryFieldWriter(
            input_channels=int(
                _select(config, "grounded_world.visual_tap.input_channels", 2048)
            ),
            output_channels=geometry_channels[0],
            field_size=field_size,
            x_range_m=x_range,
            y_range_m=y_range,
            height_anchors_m=_select(
                config,
                "grounded_world.memory.height_anchors_m",
                [0.0, 1.0, 2.0],
            ),
        )
        self.world_core = GroundedWorldCore(
            geometry_input_channels=geometry_channels[0],
            geometry_channels=geometry_channels,
            scale_factors=scale_factors,
            dynamics_channels=dynamics_channels,
            history_length=self.dynamics_history_length,
            horizon=self.dynamics_horizon,
            future_enabled=self.future_enabled,
            hidden_channels=int(
                _select(config, "grounded_world.memory.hidden_channels", 256)
            ),
            x_range_m=x_range,
            y_range_m=y_range,
        )
        self.ema_decay = float(
            _select(config, "grounded_world.future.target.ema_decay", 0.996)
        )
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("GroundedWorld EMA decay must be in (0,1)")
        self.ema_geometry_grounder = None
        self.ema_world_core = None
        if self.stage == "prior":
            self.ema_geometry_grounder = copy.deepcopy(self.geometry_grounder)
            self.ema_world_core = copy.deepcopy(self.world_core)
            self._set_requires_grad(self.ema_geometry_grounder, False)
            self._set_requires_grad(self.ema_world_core, False)
        self.external_prior_source = str(
            _select(config, "grounded_world.prior.source", "none")
        )
        self.teacher_mode = str(
            _select(config, "grounded_world.prior.teacher_mode", "real")
        )
        if self.teacher_mode not in {"real", "none", "random", "scene_shuffled", "gt_task_mlp"}:
            raise ValueError("unsupported GroundedWorld teacher_mode")
        self.retention_enabled = bool(
            _select(config, "grounded_world.prior.retention_enabled", False)
        )
        self.geometry_supervision_enabled = "vggt" in self.external_prior_source
        self.dynamics_prior_supervision_enabled = (
            "jepa" in self.external_prior_source
            or "random_frozen" in self.external_prior_source
            or "gt_task_mlp" in self.external_prior_source
        )
        self.geometry_supervision_head = GeometrySupervisionHead(
            input_channels=geometry_channels[0],
            num_views=len(self.visual_output_view_names),
            num_heights=len(self.geometry_grounder.height_anchors_m),
            hidden_channels=int(
                _select(config, "grounded_world.prior.geometry_hidden_channels", 128)
            ),
            max_depth_residual_m=float(
                _select(config, "grounded_world.prior.max_depth_residual_m", 50.0)
            ),
        )
        self.prior_adapter = ExternalPriorAdapter(
            teacher_channels=int(
                _select(config, "grounded_world.prior.teacher_channels", 96)
            ),
            output_channels=dynamics_channels,
        )
        # Target-side adapters are fixed: alignment losses detach all teacher
        # targets, so leaving these parameters trainable would create unused
        # optimizer state without changing the objective.
        self._set_requires_grad(self.prior_adapter, False)
        gt_task_input_dim = int(
            _select(config, "grounded_world.prior.gt_task_input_dim", 8)
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                int(_select(config, "grounded_world.prior.control_seed", 20260810))
            )
            self.gt_task_projection = nn.Sequential(
                nn.Linear(gt_task_input_dim, dynamics_channels),
                nn.GELU(),
                nn.Linear(dynamics_channels, dynamics_channels),
            )
        self._set_requires_grad(self.gt_task_projection, False)
        self.control_seed = int(
            _select(config, "grounded_world.prior.control_seed", 20260810)
        )

        self.codec = TrajectoryCodec()
        self.world_access = bool(
            _select(config, "grounded_world.planner.world_access", True)
        )
        self.refiner_enabled = bool(
            _select(config, "grounded_world.planner.refiner_enabled", True)
        )
        self.inference_intervention = str(
            _select(
                config,
                "grounded_world.diagnostics.inference_intervention",
                "none",
            )
        )
        if self.inference_intervention not in {"none", "disable_access"}:
            raise ValueError(
                "GroundedWorld inference_intervention must be none or disable_access"
            )
        reader_dim = int(
            _select(config, "grounded_world.planner.tube.output_dim", 256)
        )
        self.trajectory_tube_reader = MultiScaleTrajectoryTubeReader(
            geometry_channels=geometry_channels,
            dynamics_channels=dynamics_channels,
            output_dim=reader_dim,
            x_range_m=x_range,
            y_range_m=y_range,
            lateral_offsets_m=_select(
                config,
                "grounded_world.planner.tube.lateral_offsets_m",
                [-1.0, 0.0, 1.0],
            ),
            longitudinal_offsets_m=_select(
                config,
                "grounded_world.planner.tube.longitudinal_offsets_m",
                [0.0, 2.5],
            ),
        )
        self.trajectory_refiner = TrajectoryRefiner(
            context_dim=reader_dim,
            hidden_dim=int(
                _select(config, "grounded_world.planner.refiner.hidden_dim", 512)
            ),
            num_layers=int(
                _select(config, "grounded_world.planner.refiner.num_layers", 4)
            ),
            max_delta_xy_m=float(
                _select(
                    config,
                    "grounded_world.planner.refiner.max_delta_xy_m",
                    4.0,
                )
            ),
            max_delta_heading_rad=float(
                _select(
                    config,
                    "grounded_world.planner.refiner.max_delta_heading_rad",
                    0.5,
                )
            ),
        )
        self.consequence_enabled = bool(
            _select(config, "grounded_world.consequence.enabled", False)
        )
        if bool(
            _select(config, "grounded_world.consequence.inference_enabled", False)
        ):
            raise ValueError("GroundedWorld consequence head is training-only")
        self.consequence_head = PlanningConsequenceHead(
            reader_dim,
            int(_select(config, "grounded_world.consequence.hidden_dim", 256)),
        )
        consequence_scales = torch.as_tensor(
            _select(
                config,
                "grounded_world.consequence.target_scales",
                [10.0, 4.0, 1.0, 5.0, 20.0, 1.0],
            ),
            dtype=torch.float32,
        )
        if consequence_scales.shape != (6,) or not torch.isfinite(
            consequence_scales
        ).all() or (consequence_scales <= 0).any():
            raise ValueError(
                "grounded_world.consequence.target_scales must be positive [6]"
            )
        self.register_buffer(
            "consequence_target_scales", consequence_scales, persistent=True
        )
        self._configure_stage_trainability()
        init_checkpoint = _select(
            config, "grounded_world.training.init_checkpoint", None
        )
        if load_checkpoints and self.stage in {"predictive", "planning"}:
            if not init_checkpoint:
                raise ValueError(f"GroundedWorld {self.stage} stage requires init_checkpoint")
            self.world_checkpoint_report = self._load_declared_checkpoint(
                self,
                init_checkpoint,
                allowed_missing_prefixes=(
                    "world_core.predictive_memory.",
                )
                if self.stage == "predictive"
                else (),
                allowed_unexpected_prefixes=(
                    "ema_geometry_grounder.",
                    "ema_world_core.",
                )
                if self.stage == "predictive"
                or (
                    self.stage == "planning"
                    and (self.stage3_phase == "A" or self.stage3_direct_init)
                )
                else (),
            )
        baseline_checkpoint = _select(
            config, "grounded_world.planner.baseline_checkpoint", None
        )
        if load_checkpoints and self.stage == "planning" and (
            self.stage3_phase == "A" or self.stage3_direct_init
        ):
            if not baseline_checkpoint:
                raise ValueError(
                    "planning Phase A/direct Phase B requires planner.baseline_checkpoint"
                )
            self.baseline_checkpoint_report = self._load_declared_checkpoint(
                self.baseline_model,
                baseline_checkpoint,
                allowed_unexpected_prefixes=tuple(
                    _select(
                        config,
                        "grounded_world.planner.allowed_unexpected_prefixes",
                        [],
                    )
                ),
            )

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _configure_stage_trainability(self) -> None:
        """Apply the Stage I/II/III-A trainable-module contract."""

        planning_modules = (
            self.trajectory_tube_reader,
            self.trajectory_refiner,
            self.consequence_head,
        )
        if self.stage in {"prior", "predictive"}:
            for module in planning_modules:
                self._set_requires_grad(module, False)
        if self.stage == "planning" and self.stage3_phase == "A":
            for module in (
                self.geometry_grounder,
                self.world_core,
                self.prior_adapter,
                self.geometry_supervision_head,
            ):
                self._set_requires_grad(module, False)
        if not self.consequence_enabled:
            self._set_requires_grad(self.consequence_head, False)
        if not self.refiner_enabled:
            self._set_requires_grad(self.trajectory_tube_reader, False)
            self._set_requires_grad(self.trajectory_refiner, False)
        if not self.geometry_supervision_enabled:
            self._set_requires_grad(self.geometry_supervision_head, False)

    @staticmethod
    def _checkpoint_file(path_value: Any) -> Path:
        path = Path(str(path_value))
        if path.is_dir():
            candidates = (
                path / "pytorch_model.pt",
                path / "final_model" / "pytorch_model.pt",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"no pytorch_model.pt under checkpoint: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    @classmethod
    def _load_declared_checkpoint(
        cls,
        module: nn.Module,
        path_value: Any,
        *,
        allowed_missing_prefixes: Sequence[str] = (),
        allowed_unexpected_prefixes: Sequence[str] = (),
    ) -> dict[str, tuple[str, ...]]:
        """Load a checkpoint and reject every undeclared key mismatch."""

        path = cls._checkpoint_file(path_value)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise TypeError(f"checkpoint must contain a state-dict mapping: {path}")
        missing, unexpected = module.load_state_dict(payload, strict=False)
        unresolved_missing = [
            key
            for key in missing
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        unresolved_unexpected = [
            key
            for key in unexpected
            if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        ]
        if unresolved_missing or unresolved_unexpected:
            raise RuntimeError(
                "GroundedWorld checkpoint mismatch: "
                f"missing={unresolved_missing[:20]}, "
                f"unexpected={unresolved_unexpected[:20]}"
            )
        return {
            "missing_allowed": tuple(missing),
            "unexpected_allowed": tuple(unexpected),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "enabled", False) and getattr(
            self, "freeze_base_planner", False
        ):
            self.baseline_model.eval()
        if getattr(self, "ema_geometry_grounder", None) is not None:
            self.ema_geometry_grounder.eval()
        if getattr(self, "ema_world_core", None) is not None:
            self.ema_world_core.eval()
        return self

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update the Stage-I student/EMA target encoder after an optimizer step."""

        if self.ema_geometry_grounder is None or self.ema_world_core is None:
            return
        pairs = (
            (self.ema_geometry_grounder, self.geometry_grounder),
            (self.ema_world_core, self.world_core),
        )
        for target_module, source_module in pairs:
            target_parameters = dict(target_module.named_parameters())
            for name, source in source_module.named_parameters():
                target = target_parameters[name]
                target.mul_(self.ema_decay).add_(
                    source.detach().to(device=target.device, dtype=target.dtype),
                    alpha=1.0 - self.ema_decay,
                )
            target_buffers = dict(target_module.named_buffers())
            for name, source in source_module.named_buffers():
                target_buffers[name].copy_(source.detach().to(target_buffers[name]))

    def _history_batch(
        self, batch: Sequence[Mapping[str, Any]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direct = [sample.get("history_current_from_ego") for sample in batch]
        if all(value is not None for value in direct):
            transforms = torch.stack([torch.as_tensor(value) for value in direct])
            valid = torch.ones(
                transforms.shape[:2], device=device, dtype=torch.bool
            )
            return transforms.to(device=device, dtype=torch.float32), valid
        temporals = [sample.get("temporal") for sample in batch]
        if any(value is None for value in temporals):
            raise RuntimeError("GroundedWorld samples require temporal metadata")
        transforms = torch.stack(
            [torch.as_tensor(value["current_from_ego"]) for value in temporals]
        ).to(device=device, dtype=torch.float32)
        valid = torch.stack(
            [torch.as_tensor(value["valid_mask"]) for value in temporals]
        ).to(device=device, dtype=torch.bool)
        indices = torch.tensor(
            self.dynamics_history_indices, device=device, dtype=torch.long
        )
        return transforms.index_select(1, indices), valid.index_select(1, indices)

    def _extract_geometry(
        self,
        batch: Sequence[Mapping[str, Any]],
        baseline_output: Optional[Mapping[str, Any]] = None,
        captured: Any = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Mapping[str, Any], Any]:
        direct = [sample.get("world_geometry") for sample in batch]
        if all(value is not None for value in direct):
            finest = torch.stack([torch.as_tensor(value) for value in direct]).float()
            direct_history = [sample.get("world_history_geometry") for sample in batch]
            history = (
                torch.stack([torch.as_tensor(value) for value in direct_history]).float()
                if all(value is not None for value in direct_history)
                else None
            )
            if any(value is not None for value in direct_history) and history is None:
                raise ValueError("batch cannot mix present/missing world_history_geometry")
            return finest, history, baseline_output or {}, captured
        if any(value is not None for value in direct):
            raise ValueError("batch cannot mix direct and visual world geometry")
        if baseline_output is None:
            baseline_output, captured = self._run_visual_only(batch)
        if not isinstance(baseline_output, Mapping):
            raise TypeError("GroundedWorld visual output must be a mapping")
        visual, camera = self._resolve_visual_and_camera(
            baseline_output, captured, batch
        )
        geometry = self.geometry_grounder(visual, camera)
        history = self._history_visual_geometry(batch, geometry.field)
        return geometry.field, history, baseline_output, captured

    def _history_visual_geometry(
        self,
        batch: Sequence[Mapping[str, Any]],
        current_geometry: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Encode past images once each and align them later in the world core."""

        histories = [sample.get("world_history_images") for sample in batch]
        if all(value is None for value in histories):
            return None
        if any(value is None for value in histories):
            raise ValueError("batch cannot mix present/missing world_history_images")
        if any(len(value) != self.dynamics_history_length for value in histories):
            raise ValueError("world_history_images must contain Th frames")
        fields = []
        for history_position in range(self.dynamics_history_length - 1):
            temporal_samples = []
            for sample, images in zip(batch, histories):
                temporal = sample.get("temporal")
                if temporal is None or temporal.get("history_camera") is None:
                    raise RuntimeError("history images require calibrated history_camera")
                camera_tree = temporal["history_camera"]
                temporal_samples.append(
                    {
                        "image": images[history_position],
                        "lang": str(sample.get("lang", "")),
                        "camera": {
                            "view_names": camera_tree["view_names"],
                            "frame_index": int(
                                camera_tree["frame_indices"][history_position]
                            ),
                            "intrinsics": camera_tree["intrinsics"][history_position],
                            "ego_to_camera": camera_tree["ego_to_camera"][history_position],
                            "image_hw": camera_tree["image_hw"][history_position],
                        },
                    }
                )
            output, captured = self._run_visual_only(temporal_samples)
            visual, camera = self._resolve_visual_and_camera(
                output, captured, temporal_samples
            )
            fields.append(self.geometry_grounder(visual, camera).field)
        fields.append(current_geometry)
        return torch.stack(fields, dim=1)

    def _memory(
        self,
        batch: Sequence[Mapping[str, Any]],
        baseline_output: Optional[Mapping[str, Any]] = None,
        captured: Any = None,
    ) -> GroundedWorldMemoryOutput:
        finest, history, _, _ = self._extract_geometry(
            batch, baseline_output, captured
        )
        transforms, valid = self._history_batch(batch, finest.device)
        core_stage = "predictive" if self.future_enabled else "prior"
        return self.world_core(
            finest,
            transforms,
            history_valid_mask=valid,
            history_geometry=history,
            stage=core_stage,
        )

    def _prior_losses(
        self,
        memory: GroundedWorldMemoryOutput,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        prediction_field = memory.current_dynamics.field
        prediction = prediction_field.mean(dim=(-2, -1))
        linked_zero = (
            prediction_field.sum()
            + sum(parameter.sum() for parameter in self.prior_adapter.parameters())
        ) * 0.0
        if (
            not self.dynamics_prior_supervision_enabled
            or self.external_prior_source == "none"
            or self.teacher_mode == "none"
        ):
            return {"prior_capacity_zero": linked_zero}, {
                "prior_supervision_enabled": linked_zero.detach()
            }
        if self.teacher_mode == "gt_task_mlp":
            state_rows = []
            command_names = ("turn left", "keep straight", "turn right", "unknown")
            for sample in batch:
                state = torch.as_tensor(sample["state"], dtype=torch.float32).reshape(-1)
                if state.numel() != 4:
                    raise ValueError("GT-task control requires current state shape [4]")
                command = str(sample.get("navigation_command", "unknown"))
                one_hot = torch.zeros(4, dtype=torch.float32)
                one_hot[command_names.index(command) if command in command_names else 3] = 1.0
                state_rows.append(torch.cat((state, one_hot)))
            inputs = torch.stack(state_rows).to(device=prediction.device)
            target_vector = self.gt_task_projection(inputs).detach()
            weights = prediction.new_ones(prediction.shape[0])
            prefix = "retention" if self.stage != "prior" else "prior"
            return global_alignment_losses(
                prediction,
                target=target_vector,
                weights=weights,
                target_id="gt-current-task-mlp-v1",
                prefix=prefix,
            )
        payloads = [sample.get("grounded_world_prior") for sample in batch]
        if any(payload is None for payload in payloads):
            raise RuntimeError("external prior cache entry is missing")
        features = torch.stack(
            [torch.as_tensor(payload["features"]) for payload in payloads]
        ).to(device=prediction.device, dtype=torch.float32)
        confidence = torch.stack(
            [torch.as_tensor(payload["confidence"]) for payload in payloads]
        ).to(device=prediction.device, dtype=torch.float32)
        if self.teacher_mode == "scene_shuffled":
            features = torch.roll(features, shifts=1, dims=0)
            confidence = torch.roll(confidence, shifts=1, dims=0)
        elif self.teacher_mode == "random":
            generator = torch.Generator(device=prediction.device)
            generator.manual_seed(self.control_seed)
            features = torch.randn(
                features.shape,
                device=features.device,
                dtype=features.dtype,
                generator=generator,
            )
        target, weights = self.prior_adapter(features, confidence)
        prefix = "retention" if self.stage != "prior" else "prior"
        return global_alignment_losses(
            prediction,
            target=target,
            weights=weights,
            target_id=str(payloads[0].get("teacher", "external-prior")),
            prefix=prefix,
        )

    def _geometry_losses(
        self,
        memory: GroundedWorldMemoryOutput,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.geometry_supervision_enabled or self.teacher_mode == "none":
            return {}, {}
        device = memory.geometry.finest.device
        depth, confidence, source_hw, names = self._teacher_batch(
            batch, device, self.visual_output_view_names
        )
        if self.teacher_mode == "scene_shuffled":
            depth = torch.roll(depth, shifts=1, dims=0)
            confidence = torch.roll(confidence, shifts=1, dims=0)
        elif self.teacher_mode == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(self.control_seed)
            depth = torch.rand(
                depth.shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ) * 79.0 + 1.0
        current_camera = self._camera_from_batch(batch, device, names)
        teacher_camera = self._teacher_camera_from_batch(
            batch, current_camera, source_hw, names
        )
        targets = build_geometry_targets(
            depth_m=depth,
            confidence=confidence,
            camera=teacher_camera,
            field_size=memory.geometry.finest.shape[-2:],
            x_range_m=_select(
                self.config, "grounded_world.memory.x_range_m", [-8.0, 56.0]
            ),
            y_range_m=_select(
                self.config, "grounded_world.memory.y_range_m", [-32.0, 32.0]
            ),
            height_anchors_m=self.geometry_grounder.height_anchors_m,
            occupancy_threshold_m=float(
                _select(
                    self.config,
                    "grounded_world.prior.occupancy_threshold_m",
                    0.75,
                )
            ),
            free_space_margin_m=float(
                _select(
                    self.config,
                    "grounded_world.prior.free_space_margin_m",
                    0.75,
                )
            ),
            relative_depth_scale_m=float(
                _select(
                    self.config,
                    "grounded_world.prior.relative_depth_scale_m",
                    10.0,
                )
            ),
        )
        prediction = self.geometry_supervision_head(memory.geometry.finest)
        return geometry_supervision_losses(prediction, targets)

    def _future_losses(
        self,
        memory: GroundedWorldMemoryOutput,
        batch: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if memory.predictive is None:
            return {}, {}
        payloads = [sample.get("grounded_world_future_target") for sample in batch]
        if any(payload is None for payload in payloads):
            raise RuntimeError("student/EMA future target cache entry is missing")
        target = torch.stack(
            [torch.as_tensor(payload["features"]) for payload in payloads]
        ).to(device=memory.predictive.future.device, dtype=torch.float32)
        weights = torch.stack(
            [torch.as_tensor(payload["valid_mask"]) for payload in payloads]
        ).to(device=memory.predictive.future.device, dtype=torch.float32)
        return future_prediction_losses(
            memory.predictive.future,
            FeatureAlignmentTarget(
                target,
                weights,
                target_id=self.future_target_contract.target_id,
            ),
            contract=self.future_target_contract,
        )

    def _planning_forward(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        compute_losses: bool = True,
    ) -> dict[str, Any]:
        joint_planner_output = None
        joint_captured = None
        if compute_losses and self.stage3_phase == "B":
            # Reuse the differentiable training forward for the shared visual
            # path. The later action sampler remains detached and exists only
            # to provide a physically meaningful read trajectory.
            joint_planner_output, joint_captured = self._run_baseline_method(
                batch, "forward"
            )
            if not isinstance(joint_planner_output, Mapping):
                raise TypeError("joint planner forward must return a mapping")
        baseline_output, captured = self._run_baseline_method(
            batch, "predict_action_infer_1d"
        )
        if not isinstance(baseline_output, Mapping) or "normalized_actions" not in baseline_output:
            raise RuntimeError("planner must return normalized_actions")
        device = (
            captured.features.device
            if captured is not None
            else next(self.world_core.parameters()).device
        )
        draft = torch.as_tensor(
            baseline_output["normalized_actions"], device=device, dtype=torch.float32
        )
        if draft.ndim != 3 or draft.shape[-2:] != (8, 4):
            raise ValueError("planner draft must have shape [B,8,4]")
        draft = draft[:, None].detach()
        memory = self._memory(
            batch,
            joint_planner_output if joint_planner_output is not None else baseline_output,
            joint_captured if joint_captured is not None else captured,
        )
        physical = self.codec.decode_action(draft)
        if not isinstance(physical, torch.Tensor):
            raise TypeError("GroundedWorld planner expects torch trajectories")
        dynamics = (
            memory.predictive.future
            if memory.predictive is not None
            else memory.current_dynamics.field
        )
        intervention_disables_access = (
            not compute_losses
            and self.inference_intervention == "disable_access"
        )
        readout = self.trajectory_tube_reader(
            memory.geometry,
            physical,
            current_dynamics=dynamics if dynamics.ndim == 4 else None,
            future_dynamics=dynamics if dynamics.ndim == 5 else None,
            disable_access=not self.world_access or intervention_disables_access,
        )
        if self.refiner_enabled:
            refined = self.trajectory_refiner(draft, readout.waypoint_context)
        else:
            zero_delta = physical.new_zeros((*draft.shape[:-1], 3))
            refined = RefinerOutput(
                final_action=draft,
                delta_physical=zero_delta,
                delta_norm=zero_delta.sum(),
            )
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {
            "world_delta_norm": refined.delta_norm.detach(),
            "world_access_enabled": torch.tensor(
                float(self.world_access and not intervention_disables_access),
                device=refined.final_action.device,
            ),
            "world_tube_valid_ratio": readout.tube_valid_mask.float().mean().detach(),
        }
        for source_index in range(readout.source_gates.shape[-1]):
            metrics[f"world_source_gate_{source_index}"] = (
                readout.source_gates[..., source_index].mean().detach()
            )
        if memory.predictive is not None:
            metrics["future_log_variance_mean"] = (
                memory.predictive.log_variance.mean().detach()
            )
        if compute_losses:
            targets = torch.stack(
                [torch.as_tensor(sample["action"]) for sample in batch]
            ).to(device=refined.final_action.device, dtype=torch.float32)
            losses.update(
                trajectory_refinement_losses(
                    refined.final_action, targets, refined.delta_physical
                )
            )
            if joint_planner_output is not None:
                baseline_loss = joint_planner_output.get("action_loss")
                if (
                    not isinstance(baseline_loss, torch.Tensor)
                    or baseline_loss.numel() != 1
                ):
                    raise ValueError("joint planner action_loss must be a scalar tensor")
                losses["baseline_plan"] = baseline_loss
            prior_losses, prior_metrics = self._prior_losses(memory, batch)
            if self.retention_enabled and self.external_prior_source != "none":
                losses.update(prior_losses)
            metrics.update(prior_metrics)
            geometry_losses, geometry_metrics = self._geometry_losses(memory, batch)
            if self.retention_enabled:
                losses.update(geometry_losses)
            metrics.update(geometry_metrics)
            future_losses, future_metrics = self._future_losses(memory, batch)
            losses.update(future_losses)
            metrics.update(future_metrics)
        if compute_losses and self.consequence_enabled:
            payloads = [sample.get("planning_consequence") for sample in batch]
            if any(payload is None for payload in payloads):
                raise RuntimeError("planning consequence labels are missing")
            consequence_physical = torch.stack(
                [torch.as_tensor(payload["physical_trajectories"]) for payload in payloads]
            ).to(device=physical.device, dtype=torch.float32)
            consequence_readout = self.trajectory_tube_reader(
                memory.geometry,
                consequence_physical,
                current_dynamics=dynamics if dynamics.ndim == 4 else None,
                future_dynamics=dynamics if dynamics.ndim == 5 else None,
                disable_access=False,
            )
            prediction = self.consequence_head(
                consequence_readout.waypoint_context,
                consequence_readout.tube_valid_mask.any(dim=-1),
            )
            consequence_loss_map, consequence_metrics = consequence_losses(
                prediction,
                ConsequenceTargets(
                    values=torch.stack(
                        [torch.as_tensor(payload["values"]) for payload in payloads]
                    ).to(device=physical.device, dtype=torch.float32),
                    valid_mask=torch.stack(
                        [torch.as_tensor(payload["valid_mask"]) for payload in payloads]
                    ).to(device=physical.device, dtype=torch.bool),
                ),
                scales=self.consequence_target_scales,
            )
            losses.update(consequence_loss_map)
            metrics.update(consequence_metrics)
        return {
            "losses": losses,
            "metrics": metrics,
            "draft_action": draft,
            "final_action": refined.final_action,
            "delta_physical": refined.delta_physical,
            "source_gates": readout.source_gates,
            "tube_valid_mask": readout.tube_valid_mask,
            "tube_points": readout.tube_points,
        }

    def forward(self, batch):
        if not self.enabled:
            return self.baseline_model(batch)
        if self.stage == "planning":
            return self._planning_forward(batch)
        memory = self._memory(batch)
        prior_losses, prior_metrics = self._prior_losses(memory, batch)
        losses = prior_losses
        metrics = prior_metrics
        geometry_losses, geometry_metrics = self._geometry_losses(memory, batch)
        losses.update(geometry_losses)
        metrics.update(geometry_metrics)
        if self.stage == "predictive":
            future_losses, future_metrics = self._future_losses(memory, batch)
            losses.update(future_losses)
            metrics.update(future_metrics)
        return {"losses": losses, "metrics": metrics, "world_memory": memory}

    @torch.no_grad()
    def predict_action_infer_1d(self, batch):
        if not self.enabled:
            return self.baseline_model.predict_action_infer_1d(batch)
        if self.stage != "planning":
            raise RuntimeError("action inference requires planning stage")
        output = self._planning_forward(batch, compute_losses=False)
        return {
            "normalized_actions": output["final_action"][:, 0]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "diagnostics": {
                "draft_action": output["draft_action"][:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "final_action": output["final_action"][:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "delta_physical": output["delta_physical"][:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "source_gates": output["source_gates"][:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "tube_valid_mask": output["tube_valid_mask"][:, 0]
                .detach()
                .cpu()
                .numpy(),
                "tube_points": output["tube_points"][:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
            },
        }
