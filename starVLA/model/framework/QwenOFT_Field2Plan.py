"""Backward-compatible Draft-Read-Refine framework for Field2Plan Phase 1."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.field2plan.diagnostics import collect_mvp_metrics
from starVLA.model.modules.field2plan.controls import (
    GTMLPFieldControl,
    apply_dynamics_teacher_controls,
    apply_geometry_teacher_controls,
)
from starVLA.model.modules.field2plan.dynamics_field_writer import (
    ActionFreeDynamicsFieldWriter,
)
from starVLA.model.modules.field2plan.dynamics_supervision import (
    DynamicsSupervisionHead,
    build_dynamics_targets,
    dynamics_supervision_losses,
)
from starVLA.model.modules.field2plan.geometry_field_writer import GeometryFieldWriter
from starVLA.model.modules.field2plan.geometry_supervision import (
    GeometrySupervisionHead,
    build_geometry_targets,
    geometry_supervision_losses,
)
from starVLA.model.modules.field2plan.losses import trajectory_refinement_losses
from starVLA.model.modules.field2plan.semantic_writer import SemanticWriter
from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.field2plan.trajectory_refiner import TrajectoryRefiner
from starVLA.model.modules.field2plan.trajectory_tube_reader import TrajectoryTubeReader
from starVLA.model.modules.field2plan.types import (
    CameraBatch,
    DynamicsFieldOutput,
    GeometryFieldOutput,
    TemporalCameraBatch,
    VisualFeatureOutput,
)
from starVLA.model.modules.field2plan.visual_feature_tap import VisualFeatureTap


def _select(config: Any, path: str, default: Any = None) -> Any:
    """OmegaConf/dict/namespace compatible config selection."""

    if OmegaConf.is_config(config):
        return OmegaConf.select(config, path, default=default)
    value = config
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return default
    return value


@FRAMEWORK_REGISTRY.register("QwenOFT_Field2Plan")
class Qwenvl_OFT_Field2Plan(nn.Module):
    """Compose legacy QwenOFT with a teacher-free Field2Plan MVP.

    With ``field2plan.enabled=false``, ``forward`` directly returns the legacy
    model output object. With the feature enabled, drafts come from the sample
    proposal cache unless the explicit ``online_debug`` fallback is selected.
    The geometry writer consumes only current visual features and camera data.
    """

    def __init__(
        self,
        config: Any,
        accelerator: Any = None,
        baseline_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.enabled = bool(_select(config, "field2plan.enabled", False))
        if baseline_model is None:
            from starVLA.model.framework.QwenOFT import Qwenvl_OFT

            baseline_model = Qwenvl_OFT(config, accelerator, infer_not_load_wan=1)
        self.baseline_model = baseline_model
        if not self.enabled:
            return

        proposal_qwen_forward_mode = str(
            _select(
                config,
                "field2plan.proposal.qwen_forward_mode",
                "optimized",
            )
        )
        if proposal_qwen_forward_mode not in {"legacy", "optimized"}:
            raise ValueError(
                "field2plan.proposal.qwen_forward_mode must be legacy or optimized"
            )
        self.baseline_model._inference_qwen_forward_mode = (
            proposal_qwen_forward_mode
        )

        self.freeze_base_planner = bool(
            _select(config, "field2plan.proposal.freeze_base_planner", True)
        )
        self.stop_gradient = bool(
            _select(config, "field2plan.proposal.stop_gradient", True)
        )
        self.online_fallback = bool(
            _select(config, "field2plan.proposal.online_fallback", False)
        )
        if self.freeze_base_planner:
            for parameter in self.baseline_model.parameters():
                parameter.requires_grad_(False)
            self.baseline_model.eval()

        view_order = tuple(
            _select(config, "field2plan.visual_tap.view_order", [0, 1, 2])
        )
        num_views = int(
            _select(config, "field2plan.visual_tap.num_views", len(view_order))
        )
        self.visual_feature_tap = VisualFeatureTap(
            enabled=bool(_select(config, "field2plan.visual_tap.enabled", True)),
            mode=str(_select(config, "field2plan.visual_tap.mode", "hook")),
            num_views=num_views,
            view_order=view_order,
            view_names=tuple(
                _select(
                    config,
                    "field2plan.visual_tap.view_names",
                    ["cam_f0", "cam_l0", "cam_r0"][:num_views],
                )
            ),
        )
        if not self.visual_feature_tap.enabled:
            raise ValueError(
                "field2plan.enabled=true requires visual_tap.enabled=true in the MVP"
            )
        self.visual_output_view_names = tuple(
            self.visual_feature_tap.view_names[index] for index in view_order
        )
        geometry_channels = int(_select(config, "field2plan.geometry.channels", 128))
        self.geometry_field_writer = GeometryFieldWriter(
            input_channels=int(
                _select(config, "field2plan.geometry.input_channels", 2048)
            ),
            output_channels=geometry_channels,
            field_size=_select(config, "field2plan.geometry.field_size", [32, 32]),
            x_range_m=_select(
                config, "field2plan.geometry.x_range_m", [-8.0, 56.0]
            ),
            y_range_m=_select(
                config, "field2plan.geometry.y_range_m", [-32.0, 32.0]
            ),
            height_anchors_m=_select(
                config,
                "field2plan.geometry.height_anchors_m",
                [0.0, 1.0, 2.0],
            ),
        )
        self.geometry_supervision_enabled = bool(
            _select(
                config,
                "field2plan.geometry.supervision.enabled",
                False,
            )
        )
        self.random_teacher = bool(
            _select(config, "field2plan.controls.random_teacher", False)
        )
        self.shuffle_teacher_across_batch = bool(
            _select(
                config,
                "field2plan.controls.shuffle_teacher_across_batch",
                False,
            )
        )
        self.gt_mlp_teacher = bool(
            _select(config, "field2plan.controls.gt_mlp_teacher", False)
        )
        self.equal_capacity_no_teacher = bool(
            _select(
                config,
                "field2plan.controls.equal_capacity_no_teacher",
                False,
            )
        )
        if self.random_teacher and self.shuffle_teacher_across_batch:
            raise ValueError("random and shuffled teacher controls are mutually exclusive")
        if (self.random_teacher or self.shuffle_teacher_across_batch) and not self.geometry_supervision_enabled:
            raise ValueError("random/shuffled teacher controls require geometry supervision")
        if self.gt_mlp_teacher and self.geometry_supervision_enabled:
            raise ValueError("GT-MLP control cannot consume external geometry supervision")
        if self.gt_mlp_teacher and (
            self.random_teacher
            or self.shuffle_teacher_across_batch
            or self.equal_capacity_no_teacher
        ):
            raise ValueError("GT-MLP is mutually exclusive with other teacher controls")
        if self.equal_capacity_no_teacher and self.geometry_supervision_enabled:
            raise ValueError("equal-capacity-no-teacher cannot enable teacher supervision")
        self.teacher_control_seed = int(
            _select(config, "field2plan.controls.teacher_seed", 0)
        )
        build_geometry_head = bool(
            _select(
                config,
                "field2plan.geometry.supervision.build_head",
                self.geometry_supervision_enabled,
            )
        )
        if self.geometry_supervision_enabled and not build_geometry_head:
            raise ValueError("geometry supervision requires build_head=true")
        self.geometry_supervision_head = None
        if build_geometry_head:
            self.geometry_supervision_head = GeometrySupervisionHead(
                input_channels=geometry_channels,
                num_views=len(self.visual_output_view_names),
                num_heights=len(self.geometry_field_writer.height_anchors_m),
                hidden_channels=int(
                    _select(
                        config,
                        "field2plan.geometry.supervision.hidden_channels",
                        128,
                    )
                ),
                max_depth_residual_m=float(
                    _select(
                        config,
                        "field2plan.geometry.supervision.max_depth_residual_m",
                        50.0,
                    )
                ),
            )
        self.gt_mlp_field_control = None
        if self.gt_mlp_teacher:
            self.gt_mlp_field_control = GTMLPFieldControl(
                state_dim=int(
                    _select(config, "field2plan.controls.gt_mlp_state_dim", 4)
                ),
                output_channels=geometry_channels,
                field_size=self.geometry_field_writer.field_size,
                hidden_dim=int(
                    _select(config, "field2plan.controls.gt_mlp_hidden_dim", 128)
                ),
            )

        self.dynamics_enabled = bool(
            _select(config, "field2plan.dynamics.enabled", False)
        )
        self.dynamics_access_enabled = bool(
            _select(config, "field2plan.dynamics.access_enabled", True)
        )
        self.dynamics_supervision_enabled = bool(
            _select(
                config,
                "field2plan.dynamics.supervision.enabled",
                False,
            )
        )
        if self.dynamics_supervision_enabled and not self.dynamics_enabled:
            raise ValueError(
                "dynamics supervision requires field2plan.dynamics.enabled=true"
            )
        self.dynamics_random_teacher = bool(
            _select(
                config,
                "field2plan.controls.dynamics_random_teacher",
                False,
            )
        )
        self.dynamics_shuffle_teacher_across_batch = bool(
            _select(
                config,
                "field2plan.controls.dynamics_shuffle_teacher_across_batch",
                False,
            )
        )
        self.temporal_shuffle_teacher = bool(
            _select(
                config,
                "field2plan.controls.temporal_shuffle_teacher",
                False,
            )
        )
        dynamics_control_count = sum(
            int(value)
            for value in (
                self.dynamics_random_teacher,
                self.dynamics_shuffle_teacher_across_batch,
                self.temporal_shuffle_teacher,
            )
        )
        if dynamics_control_count > 1:
            raise ValueError("dynamics teacher controls are mutually exclusive")
        if dynamics_control_count and not self.dynamics_supervision_enabled:
            raise ValueError("dynamics teacher controls require dynamics supervision")

        self.dynamics_history_indices = tuple(
            int(value)
            for value in _select(
                config,
                "field2plan.dynamics.history_frame_indices",
                [0, 1, 2, 3],
            )
        )
        self.dynamics_future_indices = tuple(
            int(value)
            for value in _select(
                config,
                "field2plan.dynamics.future_frame_indices",
                list(range(4, 12)),
            )
        )
        self.dynamics_horizon = int(
            _select(
                config,
                "field2plan.dynamics.horizon",
                len(self.dynamics_future_indices),
            )
        )
        self.dynamics_history_length = int(
            _select(
                config,
                "field2plan.dynamics.history_length",
                len(self.dynamics_history_indices),
            )
        )
        if self.dynamics_horizon != len(self.dynamics_future_indices):
            raise ValueError("dynamics horizon must equal future_frame_indices length")
        if self.dynamics_history_length != len(self.dynamics_history_indices):
            raise ValueError(
                "dynamics history_length must equal history_frame_indices length"
            )
        self.dynamics_channels = int(
            _select(config, "field2plan.dynamics.channels", 192)
        )
        self.dynamics_field_writer = None
        self.dynamics_supervision_head = None
        if self.dynamics_enabled:
            self.dynamics_field_writer = ActionFreeDynamicsFieldWriter(
                input_channels=geometry_channels,
                output_channels=self.dynamics_channels,
                horizon=self.dynamics_horizon,
                history_length=self.dynamics_history_length,
                hidden_channels=int(
                    _select(
                        config,
                        "field2plan.dynamics.hidden_channels",
                        256,
                    )
                ),
            )
            build_dynamics_head = bool(
                _select(
                    config,
                    "field2plan.dynamics.supervision.build_head",
                    self.dynamics_supervision_enabled,
                )
            )
            if self.dynamics_supervision_enabled and not build_dynamics_head:
                raise ValueError("dynamics supervision requires build_head=true")
            if build_dynamics_head:
                self.dynamics_supervision_head = DynamicsSupervisionHead(
                    input_channels=self.dynamics_channels,
                    teacher_channels=int(
                        _select(
                            config,
                            "field2plan.dynamics.supervision.teacher_channels",
                            96,
                        )
                    ),
                    hidden_channels=int(
                        _select(
                            config,
                            "field2plan.dynamics.supervision.hidden_channels",
                            self.dynamics_channels,
                        )
                    ),
                )

        semantics_enabled = bool(
            _select(config, "field2plan.semantics.enabled", False)
        )
        semantic_channels = int(
            _select(config, "field2plan.semantics.channels", 0)
        )
        self.semantic_writer = None
        if semantics_enabled:
            self.semantic_writer = SemanticWriter(
                int(_select(config, "field2plan.semantics.input_dim", 2048)),
                semantic_channels,
                int(_select(config, "field2plan.semantics.num_tokens", 4)),
            )
        reader_dim = int(_select(config, "field2plan.reader.output_dim", 128))
        self.trajectory_tube_reader = TrajectoryTubeReader(
            geometry_channels=geometry_channels,
            output_dim=reader_dim,
            x_range_m=_select(
                config, "field2plan.geometry.x_range_m", [-8.0, 56.0]
            ),
            y_range_m=_select(
                config, "field2plan.geometry.y_range_m", [-32.0, 32.0]
            ),
            lateral_offsets_m=_select(
                config, "field2plan.reader.lateral_offsets_m", [-1.0, 0.0, 1.0]
            ),
            longitudinal_offsets_m=_select(
                config, "field2plan.reader.longitudinal_offsets_m", [0.0, 2.5]
            ),
            semantic_channels=semantic_channels if semantics_enabled else None,
            dynamics_channels=(
                self.dynamics_channels if self.dynamics_enabled else None
            ),
            temporal_interpolation=str(
                _select(
                    config,
                    "field2plan.dynamics.temporal_interpolation",
                    "linear",
                )
            ),
            source_dropout_p=float(
                _select(
                    config,
                    "field2plan.dynamics.source_dropout_p",
                    0.0,
                )
            ),
        )
        self.trajectory_refiner = TrajectoryRefiner(
            context_dim=reader_dim,
            hidden_dim=int(_select(config, "field2plan.refiner.hidden_dim", 256)),
            num_layers=int(_select(config, "field2plan.refiner.num_layers", 2)),
            max_delta_xy_m=float(
                _select(config, "field2plan.refiner.max_delta_xy_m", 4.0)
            ),
            max_delta_heading_rad=float(
                _select(
                    config, "field2plan.refiner.max_delta_heading_rad", 0.5
                )
            ),
        )
        self.codec = TrajectoryCodec()

        baseline_checkpoint = _select(
            config, "field2plan.proposal.checkpoint", None
        )
        if baseline_checkpoint:
            state = torch.load(
                str(baseline_checkpoint), map_location="cpu", weights_only=True
            )
            if not isinstance(state, Mapping):
                raise TypeError("baseline checkpoint must contain a state-dict mapping")
            missing, unexpected = self.baseline_model.load_state_dict(
                state,
                strict=False,
            )
            strict_checkpoint = bool(
                _select(config, "field2plan.proposal.checkpoint_strict", False)
            )
            allowed_missing = tuple(
                _select(
                    config,
                    "field2plan.proposal.allowed_missing_prefixes",
                    [],
                )
            )
            allowed_unexpected = tuple(
                _select(
                    config,
                    "field2plan.proposal.allowed_unexpected_prefixes",
                    [],
                )
            )
            unresolved_missing = [
                key
                for key in missing
                if strict_checkpoint
                or not any(key.startswith(prefix) for prefix in allowed_missing)
            ]
            unresolved_unexpected = [
                key
                for key in unexpected
                if strict_checkpoint
                or not any(
                    key.startswith(prefix) for prefix in allowed_unexpected
                )
            ]
            if unresolved_missing or unresolved_unexpected:
                raise RuntimeError(
                    "baseline checkpoint mismatch: "
                    f"missing={unresolved_missing[:20]}, "
                    f"unexpected={unresolved_unexpected[:20]}"
                )
            self.baseline_checkpoint_report = {
                "missing_allowed": tuple(missing),
                "unexpected_allowed": tuple(unexpected),
            }

    def train(self, mode: bool = True):
        """Set train mode while keeping a frozen proposal network deterministic."""

        super().train(mode)
        if self.enabled and getattr(self, "freeze_base_planner", False):
            self.baseline_model.eval()
        return self

    def _visual_module(self) -> Optional[nn.Module]:
        try:
            return self.baseline_model.qwen_vl_interface.model.model.visual
        except AttributeError:
            return None

    def _run_baseline_method(self, batch, method_name: str):
        if not hasattr(self.baseline_model, method_name):
            raise RuntimeError(f"baseline model has no {method_name} interface")
        visual_module = self._visual_module()
        tap_context = (
            self.visual_feature_tap.capture(visual_module)
            if visual_module is not None
            and self.visual_feature_tap.enabled
            and self.visual_feature_tap.mode == "hook"
            else nullcontext()
        )
        grad_context = torch.no_grad() if self.freeze_base_planner else nullcontext()
        with tap_context:
            with grad_context:
                output = getattr(self.baseline_model, method_name)(batch)
            captured = (
                self.visual_feature_tap.consume(visual_module)
                if visual_module is not None
                and self.visual_feature_tap.mode == "hook"
                and self.visual_feature_tap.has_pending_capture
                else None
            )
        return output, captured

    def _run_visual_only(self, batch):
        """Run the current-image visual encoder once for cached-draft training.

        This path deliberately skips the frozen language/action heads. It is
        available only for the real Qwen wrapper; fake/specialized baselines
        fall back to their ordinary forward interface in unit tests.
        """

        visual_module = self._visual_module()
        interface = getattr(self.baseline_model, "qwen_vl_interface", None)
        if visual_module is None or interface is None:
            return self._run_baseline_method(batch, "forward")
        if any(sample.get("qwen_feature_cache") is not None for sample in batch):
            raise RuntimeError(
                "the legacy Qwen feature cache has no [B,V,C,H,W] feature map; "
                "disable the qwen cache component for Field2Plan Phase 1"
            )
        images = [sample.get("image") for sample in batch]
        if any(not image for image in images):
            raise RuntimeError("Field2Plan visual extraction requires current images")
        instructions = [str(sample.get("lang", "")) for sample in batch]
        qwen_inputs = interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
        )
        pixel_values = qwen_inputs.get("pixel_values")
        grid_thw = qwen_inputs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            raise RuntimeError("Qwen processor did not return visual inputs")
        grad_context = torch.no_grad() if self.freeze_base_planner else nullcontext()
        device_type = pixel_values.device.type
        autocast_enabled = device_type == "cuda"
        with grad_context:
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                visual_output = visual_module(
                    hidden_states=pixel_values,
                    grid_thw=grid_thw,
                )
        captured = self.visual_feature_tap.from_visual_output(
            visual_module,
            visual_output,
            grid_thw,
        )
        return {}, captured

    def _draft_from_batch(
        self,
        batch,
        device: torch.device,
        online_prediction: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        drafts = []
        missing = False
        for sample in batch:
            proposal = sample.get("proposal") if isinstance(sample, Mapping) else None
            draft = proposal.get("draft_action") if proposal else None
            if draft is None:
                missing = True
                break
            tensor = torch.as_tensor(draft, device=device, dtype=torch.float32)
            if tensor.ndim == 2:
                tensor = tensor[None]
            if tensor.ndim != 3 or tensor.shape[-2:] != (8, 4):
                raise ValueError("proposal draft_action must have shape [M,8,4]")
            drafts.append(tensor)
        if missing:
            if not self.online_fallback:
                raise RuntimeError(
                    "Field2Plan draft is missing; online proposal is debug-only and disabled"
                )
            if online_prediction is None:
                raise RuntimeError(
                    "online proposal must be produced by the same visual forward"
                )
            prediction = online_prediction
            online = torch.as_tensor(
                prediction["normalized_actions"], device=device, dtype=torch.float32
            )
            drafts = [item[None] for item in online]
        candidate_counts = {draft.shape[0] for draft in drafts}
        if len(candidate_counts) != 1:
            raise ValueError("all samples must have the same candidate count M")
        result = torch.stack(drafts)
        return result.detach() if self.stop_gradient else result

    @staticmethod
    def _camera_from_batch(
        batch,
        device: torch.device,
        expected_view_names: Optional[Sequence[str]] = None,
    ) -> CameraBatch:
        cameras = [sample.get("camera") for sample in batch]
        if any(camera is None for camera in cameras):
            raise RuntimeError("Field2Plan samples must contain camera metadata")
        if any(camera.get("ego_to_camera") is None for camera in cameras):
            statuses = [camera.get("transform_status") for camera in cameras]
            raise RuntimeError(
                "camera ego_to_camera is unresolved; configure lidar_to_planning_ego "
                f"or an explicit identity assumption. statuses={statuses}"
            )
        source_names = tuple(cameras[0]["view_names"])
        if any(tuple(camera["view_names"]) != source_names for camera in cameras):
            raise ValueError("camera view order differs across batch")
        names = tuple(expected_view_names or source_names)
        if set(names) != set(source_names) or len(names) != len(source_names):
            raise ValueError(
                f"visual/camera views differ: visual={names}, camera={source_names}"
            )
        indices = [source_names.index(name) for name in names]
        return CameraBatch(
            intrinsics=torch.stack(
                [torch.as_tensor(camera["intrinsics"])[indices] for camera in cameras]
            ).to(device=device, dtype=torch.float32),
            ego_to_camera=torch.stack(
                [torch.as_tensor(camera["ego_to_camera"])[indices] for camera in cameras]
            ).to(device=device, dtype=torch.float32),
            image_hw=torch.stack(
                [torch.as_tensor(camera["image_hw"])[indices] for camera in cameras]
            ).to(device=device, dtype=torch.float32),
            view_names=names,
            frame_index=int(cameras[0]["frame_index"]),
        ).validate()

    @staticmethod
    def _reorder_camera(
        camera: CameraBatch,
        expected_view_names: Sequence[str],
    ) -> CameraBatch:
        camera.validate()
        names = tuple(expected_view_names)
        if names == camera.view_names:
            return camera
        if set(names) != set(camera.view_names) or len(names) != len(camera.view_names):
            raise ValueError(
                f"visual/camera views differ: visual={names}, camera={camera.view_names}"
            )
        indices = [camera.view_names.index(name) for name in names]
        return CameraBatch(
            camera.intrinsics[:, indices],
            camera.ego_to_camera[:, indices],
            camera.image_hw[:, indices],
            names,
            camera.frame_index,
        ).validate()

    def _resolve_visual_and_camera(
        self,
        baseline_output: Mapping[str, Any],
        captured: Optional[VisualFeatureOutput],
        batch,
    ):
        direct_features = baseline_output.get("visual_features")
        if direct_features is not None:
            visual = direct_features
            view_names = tuple(
                baseline_output.get(
                    "visual_view_names", self.visual_output_view_names
                )
            )
        elif captured is not None:
            visual = captured.features
            view_names = captured.view_names
        else:
            raise RuntimeError(
                "no visual features were exposed; use the scoped hook or an explicit baseline output"
            )
        if not isinstance(visual, torch.Tensor) or visual.ndim != 5:
            raise ValueError("visual features must have shape [B,V,C,H,W]")
        if len(view_names) != visual.shape[1]:
            raise ValueError("visual view_names length must equal V")
        direct_camera = baseline_output.get("camera")
        camera = (
            self._reorder_camera(direct_camera, view_names)
            if isinstance(direct_camera, CameraBatch)
            else self._camera_from_batch(batch, visual.device, view_names)
        )
        return visual, camera

    def _history_motion_from_batch(self, batch, device: torch.device):
        """Return only historical transforms consumed by the dynamics writer."""

        temporals = [sample.get("temporal") for sample in batch]
        if any(temporal is None for temporal in temporals):
            raise RuntimeError("dynamics-enabled samples must contain temporal metadata")
        expected_history = self.dynamics_history_indices
        expected_future = self.dynamics_future_indices
        for temporal in temporals:
            history = tuple(
                int(value)
                for value in torch.as_tensor(
                    temporal["history_frame_indices"]
                ).tolist()
            )
            future = tuple(
                int(value)
                for value in torch.as_tensor(
                    temporal["future_frame_indices"]
                ).tolist()
            )
            if history != expected_history or future != expected_future:
                raise ValueError("sample temporal indices differ from dynamics config")
        all_current_from_ego = torch.stack(
            [
                torch.as_tensor(temporal["current_from_ego"], dtype=torch.float32)
                for temporal in temporals
            ]
        ).to(device=device, dtype=torch.float32)
        all_valid = torch.stack(
            [
                torch.as_tensor(temporal["valid_mask"], dtype=torch.bool)
                for temporal in temporals
            ]
        ).to(device=device, dtype=torch.bool)
        history_indices = torch.tensor(
            expected_history, device=device, dtype=torch.long
        )
        history_transforms = all_current_from_ego.index_select(1, history_indices)
        history_valid = all_valid.index_select(1, history_indices)
        all_times = torch.stack(
            [
                torch.as_tensor(temporal["frame_times_s"], dtype=torch.float32)
                for temporal in temporals
            ]
        ).to(device=device, dtype=torch.float32)
        if not torch.allclose(all_times, all_times[:1].expand_as(all_times)):
            raise ValueError("temporal frame times differ across the batch")
        future_indices = torch.tensor(
            expected_future, device=device, dtype=torch.long
        )
        future_times = all_times[0].index_select(0, future_indices)
        return history_transforms, history_valid, future_times

    def _future_camera_from_batch(
        self,
        batch,
        device: torch.device,
        expected_view_names: Sequence[str],
    ) -> TemporalCameraBatch:
        """Build future cameras used exclusively for teacher target alignment."""

        temporals = [sample.get("temporal") for sample in batch]
        if any(temporal is None for temporal in temporals):
            raise RuntimeError("dynamics supervision requires temporal metadata")
        cameras = [temporal.get("future_camera") for temporal in temporals]
        if any(camera is None for camera in cameras):
            raise RuntimeError("temporal metadata lacks future_camera")
        source_names = tuple(cameras[0]["view_names"])
        if any(tuple(camera["view_names"]) != source_names for camera in cameras):
            raise ValueError("future camera view order differs across batch")
        names = tuple(expected_view_names)
        if set(names) != set(source_names) or len(names) != len(source_names):
            raise ValueError(
                f"dynamics teacher/future camera views differ: {names} vs {source_names}"
            )
        view_indices = [source_names.index(name) for name in names]
        frame_indices = tuple(
            int(value)
            for value in torch.as_tensor(cameras[0]["frame_indices"]).tolist()
        )
        if frame_indices != self.dynamics_future_indices:
            raise ValueError("future camera indices differ from dynamics config")
        if any(
            tuple(
                int(value)
                for value in torch.as_tensor(camera["frame_indices"]).tolist()
            )
            != frame_indices
            for camera in cameras
        ):
            raise ValueError("future camera frame indices differ across batch")
        future_index_tensor = torch.tensor(
            frame_indices, device=device, dtype=torch.long
        )
        current_to_ego = torch.stack(
            [
                torch.as_tensor(
                    temporal["ego_from_current"], dtype=torch.float32
                )
                for temporal in temporals
            ]
        ).to(device=device, dtype=torch.float32).index_select(1, future_index_tensor)
        return TemporalCameraBatch(
            intrinsics=torch.stack(
                [
                    torch.as_tensor(camera["intrinsics"], dtype=torch.float32)[
                        :, view_indices
                    ]
                    for camera in cameras
                ]
            ).to(device=device, dtype=torch.float32),
            ego_to_camera=torch.stack(
                [
                    torch.as_tensor(camera["ego_to_camera"], dtype=torch.float32)[
                        :, view_indices
                    ]
                    for camera in cameras
                ]
            ).to(device=device, dtype=torch.float32),
            image_hw=torch.stack(
                [
                    torch.as_tensor(camera["image_hw"], dtype=torch.float32)[
                        :, view_indices
                    ]
                    for camera in cameras
                ]
            ).to(device=device, dtype=torch.float32),
            current_to_ego=current_to_ego,
            valid_mask=torch.stack(
                [
                    torch.as_tensor(camera["valid_mask"], dtype=torch.bool)[
                        :, view_indices
                    ]
                    for camera in cameras
                ]
            ).to(device=device, dtype=torch.bool),
            view_names=names,
            frame_indices=frame_indices,
        ).validate()

    def _dynamics_teacher_batch(
        self,
        batch,
        device: torch.device,
        expected_view_names: Sequence[str],
    ):
        teachers = [sample.get("dynamics_teacher") for sample in batch]
        if any(teacher is None for teacher in teachers):
            missing = [
                str(sample.get("token", index))
                for index, (sample, teacher) in enumerate(zip(batch, teachers))
                if teacher is None
            ]
            raise RuntimeError(
                f"dynamics supervision cache entry is missing for tokens={missing}"
            )
        source_names = tuple(teachers[0]["view_names"])
        if any(tuple(teacher["view_names"]) != source_names for teacher in teachers):
            raise ValueError("dynamics teacher view order differs across batch")
        names = tuple(expected_view_names)
        if set(names) != set(source_names) or len(names) != len(source_names):
            raise ValueError("dynamics teacher and future camera views differ")
        view_indices = [source_names.index(name) for name in names]
        for teacher in teachers:
            frame_indices = tuple(
                int(value)
                for value in torch.as_tensor(teacher["frame_indices"]).tolist()
            )
            if frame_indices != self.dynamics_future_indices:
                raise ValueError("dynamics teacher frame indices differ from config")
            if teacher.get("spatial_layout") != "per_view_patch_grid":
                raise ValueError("unsupported dynamics teacher spatial layout")
            if teacher.get("feature_normalization") not in {"none", "l2"}:
                raise ValueError("unsupported dynamics teacher normalization")
        features = torch.stack(
            [
                torch.as_tensor(teacher["features"])[:, view_indices]
                for teacher in teachers
            ]
        ).to(device=device, dtype=torch.float32)
        confidence = torch.stack(
            [
                torch.as_tensor(teacher["confidence"])[:, view_indices]
                * torch.as_tensor(teacher["valid_mask"])[:, view_indices]
                for teacher in teachers
            ]
        ).to(device=device, dtype=torch.float32)
        frame_times = torch.stack(
            [
                torch.as_tensor(teacher["frame_times_s"], dtype=torch.float32)
                for teacher in teachers
            ]
        ).to(device=device, dtype=torch.float32)
        if not torch.allclose(frame_times, frame_times[:1].expand_as(frame_times)):
            raise ValueError("dynamics teacher frame times differ across batch")
        return features, confidence, frame_times[0], names

    def _refine_draft(
        self,
        baseline_output: Mapping[str, Any],
        captured: Optional[VisualFeatureOutput],
        batch,
        draft_action: torch.Tensor,
    ):
        visual, camera = self._resolve_visual_and_camera(
            baseline_output, captured, batch
        )
        geometry = self.geometry_field_writer(visual, camera)
        if self.gt_mlp_field_control is not None:
            current_state = torch.stack(
                [
                    torch.as_tensor(sample["state"], dtype=torch.float32)
                    for sample in batch
                ]
            ).to(device=geometry.field.device)
            control_field = self.gt_mlp_field_control(current_state)
            geometry = GeometryFieldOutput(
                field=control_field,
                valid_ratio=geometry.valid_ratio,
                projection_valid=geometry.projection_valid,
            )
        dynamics: Optional[DynamicsFieldOutput] = None
        future_times = None
        if self.dynamics_field_writer is not None:
            history_transforms, history_valid, future_times = (
                self._history_motion_from_batch(batch, geometry.field.device)
            )
            dynamics = self.dynamics_field_writer(
                geometry.field,
                history_transforms,
                history_valid_mask=history_valid,
            )
        draft_action = draft_action.to(device=geometry.field.device, dtype=torch.float32)
        draft_physical = self.codec.decode_action(draft_action)
        if not isinstance(draft_physical, torch.Tensor):
            raise TypeError("framework expects torch draft tensors")

        semantic_tokens = None
        if self.semantic_writer is not None:
            semantic_hidden = baseline_output.get("semantic_hidden_states")
            if semantic_hidden is None:
                raise RuntimeError(
                    "semantics enabled but baseline exposed no semantic hidden states"
                )
            semantic_tokens = self.semantic_writer(semantic_hidden)
        readout = self.trajectory_tube_reader(
            geometry.field,
            draft_physical,
            semantic_tokens=semantic_tokens,
            disable_access=bool(
                _select(self.config, "field2plan.controls.disable_access", False)
            ),
            dynamics_field=dynamics.field if dynamics is not None else None,
            dynamics_times_s=future_times,
            waypoint_times_s=future_times,
            disable_geometry_access=not bool(
                _select(self.config, "field2plan.geometry.access_enabled", True)
            ),
            disable_dynamics_access=not self.dynamics_access_enabled,
            disable_semantic_access=not bool(
                _select(self.config, "field2plan.semantics.access_enabled", True)
            ),
        )
        refinement = self.trajectory_refiner(
            draft_action, readout.waypoint_context
        )
        return geometry, dynamics, readout, refinement, camera

    @staticmethod
    def _teacher_batch(
        batch,
        device: torch.device,
        expected_view_names: Sequence[str],
    ):
        teachers = [sample.get("geometry_teacher") for sample in batch]
        if any(teacher is None for teacher in teachers):
            missing = [
                str(sample.get("token", index))
                for index, (sample, teacher) in enumerate(zip(batch, teachers))
                if teacher is None
            ]
            raise RuntimeError(
                f"geometry supervision cache entry is missing for tokens={missing}"
            )
        source_names = tuple(teachers[0]["view_names"])
        if any(tuple(teacher["view_names"]) != source_names for teacher in teachers):
            raise ValueError("geometry teacher view order differs across batch")
        names = tuple(expected_view_names)
        if set(names) != set(source_names) or len(names) != len(source_names):
            raise ValueError(
                f"teacher/visual views differ: teacher={source_names}, visual={names}"
            )
        if any(
            teacher.get("coordinate_frame") != "camera_optical_z_depth_m"
            for teacher in teachers
        ):
            raise ValueError("geometry teacher coordinate frame is unsupported")
        indices = [source_names.index(name) for name in names]
        depth = torch.stack(
            [torch.as_tensor(teacher["depth_m"])[indices] for teacher in teachers]
        ).to(device=device, dtype=torch.float32)
        confidence = torch.stack(
            [
                torch.as_tensor(teacher["confidence"])[indices]
                * torch.as_tensor(teacher["valid_mask"])[indices]
                for teacher in teachers
            ]
        ).to(device=device, dtype=torch.float32)
        source_hw = torch.stack(
            [
                torch.as_tensor(teacher["source_image_hw"])[indices]
                for teacher in teachers
            ]
        ).to(device=device, dtype=torch.float32)
        return depth, confidence, source_hw, names

    @staticmethod
    def _teacher_camera_from_batch(
        batch,
        current_camera: CameraBatch,
        source_image_hw: torch.Tensor,
        expected_view_names: Sequence[str],
    ) -> CameraBatch:
        camera_dicts = [sample.get("camera") for sample in batch]
        if all(camera is None for camera in camera_dicts):
            if not torch.equal(
                current_camera.image_hw.to(source_image_hw), source_image_hw
            ):
                raise ValueError("teacher source image size differs from direct camera")
            return current_camera
        if any(camera is None for camera in camera_dicts):
            raise ValueError("camera metadata is missing for part of the batch")
        source_names = tuple(camera_dicts[0]["view_names"])
        names = tuple(expected_view_names)
        if any(tuple(camera["view_names"]) != source_names for camera in camera_dicts):
            raise ValueError("camera view order differs across batch")
        if set(names) != set(source_names) or len(names) != len(source_names):
            raise ValueError("raw camera and teacher views differ")
        indices = [source_names.index(name) for name in names]
        raw_intrinsics = torch.stack(
            [
                torch.as_tensor(camera["raw_intrinsics"])[indices]
                for camera in camera_dicts
            ]
        ).to(device=current_camera.intrinsics.device, dtype=torch.float32)
        raw_hw = torch.stack(
            [
                torch.as_tensor(camera["raw_image_hw"])[indices]
                for camera in camera_dicts
            ]
        ).to(device=current_camera.intrinsics.device, dtype=torch.float32)
        if not torch.equal(raw_hw, source_image_hw.to(raw_hw)):
            raise ValueError("teacher source_image_hw does not match raw camera metadata")
        return CameraBatch(
            intrinsics=raw_intrinsics,
            ego_to_camera=current_camera.ego_to_camera,
            image_hw=raw_hw,
            view_names=names,
            frame_index=current_camera.frame_index,
        ).validate()

    def _geometry_auxiliary(
        self,
        geometry: GeometryFieldOutput,
        camera: CameraBatch,
        batch,
    ):
        if self.geometry_supervision_head is None:
            return {}, {}
        prediction = self.geometry_supervision_head(geometry.field)
        if not self.geometry_supervision_enabled:
            linked_zero = (
                prediction.depth_residual_m.sum()
                + prediction.relative_geometry.sum()
                + prediction.occupancy_logits.sum()
                + prediction.free_space_logits.sum()
            ) * 0.0
            losses = {
                "geometry_depth": linked_zero,
                "geometry_occupancy": linked_zero,
                "geometry_free_space": linked_zero,
                "geometry_relative": linked_zero,
            }
            return losses, {
                "geometry_valid_ratio": linked_zero.detach(),
                "geometry_control_no_teacher": torch.ones(
                    (), device=geometry.field.device
                ),
            }

        depth, confidence, source_hw, view_names = self._teacher_batch(
            batch, geometry.field.device, camera.view_names
        )
        controlled = apply_geometry_teacher_controls(
            depth,
            confidence,
            tuple(str(sample.get("token", index)) for index, sample in enumerate(batch)),
            seed=self.teacher_control_seed,
            random_teacher=self.random_teacher,
            shuffle_teacher_across_batch=self.shuffle_teacher_across_batch,
            random_depth_range_m=tuple(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.random_depth_range_m",
                    [1.0, 80.0],
                )
            ),
        )
        teacher_camera = self._teacher_camera_from_batch(
            batch, camera, source_hw, view_names
        )
        targets = build_geometry_targets(
            depth_m=controlled.depth_m,
            confidence=controlled.confidence,
            camera=teacher_camera,
            field_size=self.geometry_field_writer.field_size,
            x_range_m=self.geometry_field_writer.x_range_m,
            y_range_m=self.geometry_field_writer.y_range_m,
            height_anchors_m=self.geometry_field_writer.height_anchors_m,
            occupancy_threshold_m=float(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.occupancy_threshold_m",
                    0.75,
                )
            ),
            free_space_margin_m=float(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.free_space_margin_m",
                    0.75,
                )
            ),
            relative_depth_scale_m=float(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.relative_depth_scale_m",
                    10.0,
                )
            ),
            min_teacher_depth_m=float(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.min_teacher_depth_m",
                    0.1,
                )
            ),
            max_teacher_depth_m=float(
                _select(
                    self.config,
                    "field2plan.geometry.supervision.max_teacher_depth_m",
                    200.0,
                )
            ),
        )
        losses, metrics = geometry_supervision_losses(prediction, targets)
        metrics[f"geometry_teacher_{controlled.mode}"] = torch.ones(
            (), device=geometry.field.device
        )
        return losses, metrics

    def _dynamics_auxiliary(
        self,
        dynamics: Optional[DynamicsFieldOutput],
        batch,
    ):
        if self.dynamics_supervision_head is None:
            return {}, {}
        if dynamics is None:
            raise RuntimeError("dynamics supervision head exists without a dynamics field")
        prediction = self.dynamics_supervision_head(dynamics.field)
        if not self.dynamics_supervision_enabled:
            linked_zero = prediction.features.sum() * 0.0
            losses = {
                "dynamics_cosine": linked_zero,
                "dynamics_smooth_l1": linked_zero,
                "dynamics_temporal_contrast": linked_zero,
                "dynamics_uncertainty": linked_zero,
            }
            return losses, {
                "dynamics_valid_ratio": linked_zero.detach(),
                "dynamics_control_no_teacher": torch.ones(
                    (), device=dynamics.field.device
                ),
            }

        features, confidence, _, view_names = self._dynamics_teacher_batch(
            batch,
            dynamics.field.device,
            expected_view_names=tuple(
                batch[0]["temporal"]["future_camera"]["view_names"]
            ),
        )
        controlled = apply_dynamics_teacher_controls(
            features,
            confidence,
            tuple(
                str(sample.get("token", index))
                for index, sample in enumerate(batch)
            ),
            seed=self.teacher_control_seed,
            random_teacher=self.dynamics_random_teacher,
            shuffle_teacher_across_batch=self.dynamics_shuffle_teacher_across_batch,
            temporal_shuffle=self.temporal_shuffle_teacher,
        )
        future_camera = self._future_camera_from_batch(
            batch,
            dynamics.field.device,
            expected_view_names=view_names,
        )
        targets = build_dynamics_targets(
            teacher_features=controlled.features,
            confidence=controlled.confidence,
            camera=future_camera,
            field_size=self.geometry_field_writer.field_size,
            x_range_m=self.geometry_field_writer.x_range_m,
            y_range_m=self.geometry_field_writer.y_range_m,
            height_anchors_m=_select(
                self.config,
                "field2plan.dynamics.supervision.height_anchors_m",
                [0.0, 1.0, 2.0],
            ),
            normalize_features=bool(
                _select(
                    self.config,
                    "field2plan.dynamics.supervision.normalize_features",
                    True,
                )
            ),
        )
        losses, metrics = dynamics_supervision_losses(
            prediction,
            targets,
            log_variance=dynamics.log_variance,
            temporal_contrast_margin=float(
                _select(
                    self.config,
                    "field2plan.dynamics.supervision.temporal_contrast_margin",
                    0.05,
                )
            ),
        )
        metrics[f"dynamics_teacher_{controlled.mode}"] = torch.ones(
            (), device=dynamics.field.device
        )
        return losses, metrics

    def forward(self, batch):
        if not self.enabled:
            return self.baseline_model(batch)
        has_cached_drafts = all(
            isinstance(sample, Mapping)
            and isinstance(sample.get("proposal"), Mapping)
            and sample["proposal"].get("draft_action") is not None
            for sample in batch
        )
        if has_cached_drafts:
            baseline_output, captured = self._run_visual_only(batch)
            online_prediction = None
        else:
            if not self.online_fallback:
                raise RuntimeError(
                    "Field2Plan draft is missing; online proposal is debug-only and disabled"
                )
            baseline_output, captured = self._run_baseline_method(
                batch, "predict_action_infer_1d"
            )
            online_prediction = baseline_output
        if not isinstance(baseline_output, Mapping):
            raise TypeError("Field2Plan baseline output must be a mapping")
        draft_device = (
            captured.features.device
            if captured is not None
            else baseline_output["visual_features"].device
        )
        draft_action = self._draft_from_batch(
            batch, draft_device, online_prediction=online_prediction
        )
        geometry, dynamics, readout, refinement, camera = self._refine_draft(
            baseline_output, captured, batch, draft_action
        )
        targets = torch.stack(
            [torch.as_tensor(sample["action"], dtype=torch.float32) for sample in batch]
        ).to(device=refinement.final_action.device)
        losses = trajectory_refinement_losses(
            refinement.final_action, targets, refinement.delta_physical
        )
        geometry_losses, geometry_metrics = self._geometry_auxiliary(
            geometry, camera, batch
        )
        losses.update(geometry_losses)
        dynamics_losses, dynamics_metrics = self._dynamics_auxiliary(
            dynamics, batch
        )
        losses.update(dynamics_losses)
        metrics = collect_mvp_metrics(geometry, readout, refinement)
        metrics.update(geometry_metrics)
        metrics.update(dynamics_metrics)
        return {
            "losses": losses,
            "metrics": metrics,
            "draft_action": draft_action,
            "final_action": refinement.final_action,
        }

    @torch.no_grad()
    def predict_action_infer_1d(self, batch):
        """Return legacy output when disabled; Phase 1 diagnostics when enabled."""

        if not self.enabled:
            return self.baseline_model.predict_action_infer_1d(batch)
        baseline_output, captured = self._run_baseline_method(
            batch, "predict_action_infer_1d"
        )
        if not isinstance(baseline_output, Mapping):
            raise TypeError("baseline inference output must be a mapping")
        if "normalized_actions" not in baseline_output:
            raise KeyError("baseline inference output lacks normalized_actions")
        draft_action = torch.as_tensor(
            baseline_output["normalized_actions"], dtype=torch.float32
        )
        if draft_action.ndim != 3 or draft_action.shape[-2:] != (8, 4):
            raise ValueError("baseline normalized_actions must have shape [B,8,4]")
        draft_action = draft_action[:, None].detach()
        geometry, dynamics, readout, refinement, _ = self._refine_draft(
            baseline_output, captured, batch, draft_action
        )
        final = refinement.final_action[:, 0].detach().float().cpu().numpy()
        return {
            "normalized_actions": np.asarray(final, dtype=np.float32),
            "diagnostics": {
                "draft_action": draft_action[:, 0].detach().float().cpu().numpy(),
                "final_action": final,
                "delta_physical": refinement.delta_physical[:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "source_gates": readout.source_gates[:, 0]
                .detach()
                .float()
                .cpu()
                .numpy(),
                "field_valid_ratio": geometry.valid_ratio.detach()
                .float()
                .cpu()
                .numpy(),
                **(
                    {
                        "dynamics_log_variance_mean": dynamics.log_variance.detach()
                        .mean(dim=(-2, -1))
                        .float()
                        .cpu()
                        .numpy()
                    }
                    if dynamics is not None
                    else {}
                ),
            },
        }
