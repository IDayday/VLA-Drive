"""Action-only Qwen planner with a dense offline VGGT bottleneck."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from omegaconf import OmegaConf

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vggt_query.planning_heads import AuxiliaryTrajectoryHead
from starVLA.model.modules.vggt_query.task_geometry_bottleneck import (
    PlanningConditionedDenseVGGTBottleneck,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


DENSE_PAYLOAD_KEYS = (
    "features",
    "valid_mask",
    "view_ids",
    "uv_coords",
    "ray_features",
    "patch_grid_hw",
)


def pad_dense_geometry_payloads(
    payloads: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pad variable-length cache records once for both train and inference."""

    if not payloads:
        raise ValueError("Dense VGGT conditioning requires a non-empty batch")
    for batch_index, payload in enumerate(payloads):
        missing = set(DENSE_PAYLOAD_KEYS) - set(payload)
        if missing:
            raise RuntimeError(
                f"Dense VGGT payload {batch_index} is missing keys: {sorted(missing)}"
            )
    feature_dims = {int(payload["features"].shape[-1]) for payload in payloads}
    if len(feature_dims) != 1:
        raise RuntimeError(f"Dense VGGT batch mixes feature dimensions: {feature_dims}")
    feature_dim = feature_dims.pop()
    lengths = [int(payload["features"].shape[0]) for payload in payloads]
    if any(length <= 0 for length in lengths):
        raise RuntimeError("Every dense VGGT payload needs at least one token")
    max_length = max(lengths)
    batch_size = len(payloads)
    feature_dtype = payloads[0]["features"].dtype
    features = torch.zeros(
        batch_size,
        max_length,
        feature_dim,
        dtype=feature_dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        batch_size, max_length, dtype=torch.bool, device=device
    )
    view_ids = torch.zeros(
        batch_size, max_length, dtype=torch.long, device=device
    )
    uv_coords = torch.zeros(
        batch_size, max_length, 2, dtype=torch.float32, device=device
    )
    ray_features = torch.zeros(
        batch_size, max_length, 6, dtype=torch.float32, device=device
    )
    patch_grid_hw = torch.zeros(
        batch_size, 3, 2, dtype=torch.int16, device=device
    )
    for batch_index, (payload, length) in enumerate(zip(payloads, lengths)):
        payload_features = payload["features"]
        payload_mask = payload["valid_mask"].bool()
        payload_views = payload["view_ids"]
        payload_uv = payload["uv_coords"]
        payload_rays = payload["ray_features"]
        payload_grid = payload["patch_grid_hw"]
        if payload_features.shape != (length, feature_dim):
            raise RuntimeError("Dense VGGT feature payload must be [N,Dg]")
        if payload_mask.shape != (length,):
            raise RuntimeError("Dense VGGT valid_mask payload must be [N]")
        if payload_views.shape != (length,):
            raise RuntimeError("Dense VGGT view_ids payload must be [N]")
        if payload_uv.shape != (length, 2):
            raise RuntimeError("Dense VGGT uv_coords payload must be [N,2]")
        if payload_rays.shape != (length, 6):
            raise RuntimeError("Dense VGGT ray_features payload must be [N,6]")
        if payload_grid.shape != (3, 2):
            raise RuntimeError("Dense VGGT patch_grid_hw payload must be [3,2]")
        if sum(int(h) * int(w) for h, w in payload_grid.tolist()) != length:
            raise RuntimeError("Dense VGGT patch grids do not sum to N")
        if not payload_mask.any():
            raise RuntimeError("Dense VGGT payload has no valid source token")
        valid_views = payload_views[payload_mask]
        if valid_views.min() < 0 or valid_views.max() > 2:
            raise RuntimeError("Dense VGGT view IDs must be 0, 1, or 2")
        features[batch_index, :length] = payload_features.to(
            device=device, dtype=feature_dtype, non_blocking=True
        )
        valid_mask[batch_index, :length] = payload_mask.to(
            device=device, non_blocking=True
        )
        view_ids[batch_index, :length] = payload_views.to(
            device=device, dtype=torch.long, non_blocking=True
        )
        uv_coords[batch_index, :length] = payload_uv.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        ray_features[batch_index, :length] = payload_rays.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        patch_grid_hw[batch_index] = payload_grid.to(
            device=device, dtype=torch.int16, non_blocking=True
        )
    return {
        "features": features,
        "valid_mask": valid_mask,
        "view_ids": view_ids,
        "uv_coords": uv_coords,
        "ray_features": ray_features,
        "patch_grid_hw": patch_grid_hw,
    }


def apply_dense_intervention(
    dense_geometry: Mapping[str, torch.Tensor], mode: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply one causal diagnostic intervention to a complete source payload."""

    if mode not in {"real", "zero", "shuffled"}:
        raise ValueError("Dense VGGT intervention must be real, zero, or shuffled")
    result = dict(dense_geometry)
    reference = result["features"]
    skipped = torch.zeros((), device=reference.device)
    if mode == "zero":
        # Retain calibrated positions to distinguish geometry use from priors.
        result["features"] = torch.zeros_like(reference)
    elif mode == "shuffled":
        if reference.shape[0] < 2:
            skipped = torch.ones((), device=reference.device)
        else:
            for key in DENSE_PAYLOAD_KEYS:
                result[key] = result[key].roll(shifts=1, dims=0)
    return result, {"intervention_skipped": skipped.detach()}


@FRAMEWORK_REGISTRY.register("QwenOFT_VGGT_Bottleneck")
class Qwenvl_OFT_VGGT_Bottleneck(Qwenvl_OFT):
    """Inject dense final-layer VGGT only through zero-init action residuals."""

    _ACTION_ONLY_MISSING_PREFIXES = (
        "vggt_dense_bottleneck.",
        "vggt_bottleneck_aux_plan_head.",
    )

    def __init__(
        self,
        config: Optional[dict] = None,
        accelerator=None,
        infer_not_load_wan=0,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            accelerator=accelerator,
            infer_not_load_wan=infer_not_load_wan,
            **kwargs,
        )
        self.vggt_bottleneck_enabled = bool(
            OmegaConf.select(
                config, "framework.vggt_bottleneck.enabled", default=False
            )
        )
        self._vggt_intervention_mode = "real"
        if not self.vggt_bottleneck_enabled:
            return
        if (
            bool(config.datasets.video_data.load_2d_data)
            or bool(config.datasets.gs_data.load_3d_data)
            or bool(config.datasets.reward_data.load_reward_data)
            or bool(OmegaConf.select(config, "w_depth", default=False))
        ):
            raise ValueError(
                "QwenOFT_VGGT_Bottleneck is an action-only framework; disable "
                "video, GS/depth, and reward heads"
            )
        bottleneck_cfg = OmegaConf.select(
            config, "framework.vggt_bottleneck.bottleneck", default={}
        )
        teacher_cfg = OmegaConf.select(
            config, "framework.vggt_bottleneck.teacher", default={}
        )
        source_cfg = OmegaConf.select(
            config, "framework.vggt_bottleneck.source", default={}
        )
        if str(OmegaConf.select(teacher_cfg, "representation", default="")) != (
            "full_aggregated_feature"
        ):
            raise ValueError("Dense VGGT bottleneck requires full aggregated features")
        if bool(OmegaConf.select(teacher_cfg, "include_special_tokens", default=False)):
            raise ValueError("Dense VGGT bottleneck excludes camera/register tokens")
        if str(OmegaConf.select(teacher_cfg, "preprocess_mode", default="crop")) != "crop":
            raise ValueError("Dense VGGT bottleneck requires mode='crop' cache features")
        for embedding_name in (
            "use_view_embedding",
            "use_uv_embedding",
            "use_ray_embedding",
        ):
            if not bool(OmegaConf.select(source_cfg, embedding_name, default=True)):
                raise ValueError(f"Dense VGGT bottleneck requires {embedding_name}=true")
        planning_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        expected_horizons = int(
            OmegaConf.select(bottleneck_cfg, "expected_horizons", default=8)
        )
        if expected_horizons != len(self.act_query_tokens):
            raise ValueError(
                "Dense VGGT horizon count must equal the eight baseline action tokens"
            )
        self.vggt_dense_bottleneck = PlanningConditionedDenseVGGTBottleneck(
            planning_dim=planning_dim,
            source_dim=int(
                OmegaConf.select(teacher_cfg, "feature_dim", default=2048)
            ),
            bottleneck_dim=int(
                OmegaConf.select(bottleneck_cfg, "hidden_dim", default=512)
            ),
            expected_horizons=expected_horizons,
            slots_per_horizon=int(
                OmegaConf.select(bottleneck_cfg, "slots_per_horizon", default=4)
            ),
            num_heads=int(
                OmegaConf.select(bottleneck_cfg, "num_heads", default=8)
            ),
            ffn_expansion=int(
                OmegaConf.select(bottleneck_cfg, "ffn_expansion", default=2)
            ),
            detach_planning_queries=bool(
                OmegaConf.select(
                    bottleneck_cfg, "detach_planning_queries", default=True
                )
            ),
            attention_dropout=float(
                OmegaConf.select(bottleneck_cfg, "attention_dropout", default=0.0)
            ),
            return_attention_diagnostics=bool(
                OmegaConf.select(
                    bottleneck_cfg,
                    "return_attention_diagnostics",
                    default=False,
                )
            ),
            view_count=int(OmegaConf.select(source_cfg, "view_count", default=3)),
        )
        aux_cfg = OmegaConf.select(
            config, "framework.vggt_bottleneck.aux_plan_head", default={}
        )
        self.vggt_bottleneck_aux_enabled = bool(
            OmegaConf.select(aux_cfg, "enabled", default=True)
        )
        self.vggt_bottleneck_aux_loss_name = str(
            OmegaConf.select(
                aux_cfg, "loss_name", default="vggt_bottleneck_aux_plan"
            )
        )
        bottleneck_dim = self.vggt_dense_bottleneck.bottleneck_dim
        self.vggt_bottleneck_aux_plan_head = AuxiliaryTrajectoryHead(
            input_dim=bottleneck_dim,
            hidden_dim=int(
                OmegaConf.select(aux_cfg, "hidden_dim", default=bottleneck_dim)
            ),
            action_dim=4,
        )
        self._use_named_loss_contract = True

    @staticmethod
    def _payloads_from_examples(examples) -> list[Mapping[str, torch.Tensor]]:
        if examples is None:
            raise RuntimeError("Dense VGGT inference requires examples with cache payloads")
        payloads = [example.get("vggt_dense_feature_cache") for example in examples]
        if not all(payload is not None for payload in payloads):
            missing = [index for index, payload in enumerate(payloads) if payload is None]
            raise RuntimeError(
                "Dense VGGT conditioning requires a cache payload for every sample; "
                f"missing batch indices: {missing}"
            )
        return payloads

    def _dense_batch_from_examples(
        self, examples, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return pad_dense_geometry_payloads(
            self._payloads_from_examples(examples), device=device
        )

    def _compute_query_extension(
        self,
        last_hidden,
        token_positions,
        examples,
        *,
        input_ids=None,
        image_grid_thw=None,
    ):
        if not self.vggt_bottleneck_enabled:
            return super()._compute_query_extension(
                last_hidden,
                token_positions,
                examples,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
            )
        dense_geometry = self._dense_batch_from_examples(
            examples, device=last_hidden.device
        )
        return {
            "dense_geometry": dense_geometry,
            "action_targets": [example["action"] for example in examples],
            "losses": {},
            "metrics": {},
        }

    def _run_dense_bottleneck(self, action_queries, dense_geometry):
        intervened, intervention_metrics = apply_dense_intervention(
            dense_geometry, self._vggt_intervention_mode
        )
        enhanced, readout, task_tokens, diagnostics = self.vggt_dense_bottleneck(
            planning_tokens=action_queries,
            source_features=intervened["features"],
            source_valid_mask=intervened["valid_mask"],
            view_ids=intervened["view_ids"],
            uv_coords=intervened["uv_coords"],
            ray_features=intervened["ray_features"],
        )
        diagnostics.update(intervention_metrics)
        return enhanced, readout, task_tokens, diagnostics

    def _condition_action_queries(self, action_queries, extension):
        if not self.vggt_bottleneck_enabled:
            return super()._condition_action_queries(action_queries, extension)
        enhanced, readout, _, diagnostics = self._run_dense_bottleneck(
            action_queries, extension["dense_geometry"]
        )
        if self.training and self.vggt_bottleneck_aux_enabled:
            target_action = torch.stack(
                [
                    torch.as_tensor(target, dtype=torch.float32)
                    for target in extension["action_targets"]
                ]
            ).to(
                device=readout.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if target_action.shape != (*readout.shape[:2], 4):
                raise ValueError(
                    "Dense VGGT auxiliary target must be normalized [B,8,4]"
                )
            auxiliary = self.vggt_bottleneck_aux_plan_head(readout, target_action)
            extension["losses"][self.vggt_bottleneck_aux_loss_name] = auxiliary.loss
            extension["metrics"].update(auxiliary.metrics)
        extension["metrics"].update(diagnostics)
        # The unchanged FlowmatchingActionHead sees exactly eight tokens.
        return enhanced, None, diagnostics

    def _condition_inference_action_queries(
        self,
        last_hidden,
        input_ids,
        action_queries,
        *,
        image_grid_thw=None,
        examples=None,
    ):
        del last_hidden, input_ids, image_grid_thw
        if not self.vggt_bottleneck_enabled:
            return action_queries, None, {}
        dense_geometry = self._dense_batch_from_examples(
            examples, device=action_queries.device
        )
        enhanced, _, _, diagnostics = self._run_dense_bottleneck(
            action_queries, dense_geometry
        )
        return enhanced, None, diagnostics

    def set_vggt_intervention(self, mode: str) -> None:
        if mode not in {"real", "zero", "shuffled"}:
            raise ValueError(
                "Dense VGGT intervention must be one of real, zero, shuffled"
            )
        self._vggt_intervention_mode = mode

    def get_planning_usage_metrics(self) -> dict[str, torch.Tensor]:
        if not self.vggt_bottleneck_enabled:
            return {}
        parameters = {
            "up_projection": self.vggt_dense_bottleneck.up_projection.weight,
            "cross_attention": self.vggt_dense_bottleneck.cross_attention.in_proj_weight,
            "source_projection": self.vggt_dense_bottleneck.source_projection.weight,
            "readout_projection": self.vggt_dense_bottleneck.readout_projection.weight,
            "aux_plan_head": self.vggt_bottleneck_aux_plan_head.head[-1].weight,
        }
        return {
            f"vggt_bottleneck/{name}_grad_norm": parameter.grad.detach().float().norm()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load action-only checkpoints with a narrow missing-key whitelist."""

        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self._ACTION_ONLY_MISSING_PREFIXES)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if invalid_missing or unexpected:
            raise RuntimeError(
                "QwenOFT_VGGT_Bottleneck checkpoint mismatch: "
                f"missing={invalid_missing[:20]} unexpected={unexpected[:20]}"
            )
        return incompatible
