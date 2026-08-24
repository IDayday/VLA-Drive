"""Scene-query-conditioned 3D-Mix for the Qwen action-only flow planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vggt_query.scene_query_compressor import (
    SceneQueryCompressor,
)
from starVLA.model.modules.vggt_query.sq_3d_mix import (
    SceneConditionedGatedFusion,
)
from starVLA.model.modules.vggt_query.vggt_patch_pool import (
    pool_dense_vggt_per_view,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


FUSION_MODES = {"scene_only", "projected_concat", "gated"}
INTERVENTION_MODES = {"real", "zero", "gaussian", "shuffled"}


def apply_sq3dmix_intervention(
    vggt_tokens: Tensor,
    mode: str,
    seed: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply one explicitly configured intervention after pooling."""

    if mode not in INTERVENTION_MODES:
        raise ValueError(
            f"Unknown SQ-3D-Mix intervention {mode!r}; expected {sorted(INTERVENTION_MODES)}"
        )
    if vggt_tokens.ndim != 3:
        raise ValueError("pooled VGGT tokens must be [B,N,Dv]")
    metrics: dict[str, Tensor] = {}
    if mode == "real":
        return vggt_tokens, metrics
    if mode == "zero":
        return torch.zeros_like(vggt_tokens), metrics
    if mode == "gaussian":
        generator = torch.Generator(device=vggt_tokens.device)
        generator.manual_seed(int(seed))
        return (
            torch.randn(
                vggt_tokens.shape,
                generator=generator,
                device=vggt_tokens.device,
                dtype=vggt_tokens.dtype,
            ),
            metrics,
        )
    if vggt_tokens.shape[0] < 2:
        metrics["sq3dmix/intervention_skipped"] = torch.ones(
            (), device=vggt_tokens.device
        )
        return vggt_tokens, metrics
    return vggt_tokens.roll(shifts=1, dims=0), metrics


@FRAMEWORK_REGISTRY.register("QwenOFT_SQ3DMix")
class Qwenvl_OFT_SQ3DMix(Qwenvl_OFT):
    """Add scene queries and scene-conditioned dense VGGT context to QwenOFT."""

    _ACTION_ONLY_MISSING_PREFIXES = (
        "scene_query_compressor.",
        "gated_fusion.",
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
        self._validate_action_only_contract(config)
        sq_cfg = OmegaConf.select(config, "framework.sq_3d_mix", default=None)
        if sq_cfg is None:
            raise ValueError("QwenOFT_SQ3DMix requires framework.sq_3d_mix")

        self.fusion_mode = str(
            OmegaConf.select(sq_cfg, "fusion_mode", default="gated")
        ).strip().lower()
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {sorted(FUSION_MODES)}, "
                f"found {self.fusion_mode!r}"
            )

        scene_cfg = OmegaConf.select(sq_cfg, "scene_compressor", default={})
        if not bool(
            OmegaConf.select(scene_cfg, "exclude_action_tokens", default=True)
        ):
            raise ValueError("SQ-3D-Mix always excludes action tokens from scene memory")
        qwen_hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.scene_query_compressor = SceneQueryCompressor(
            input_dim=qwen_hidden_dim,
            hidden_dim=int(OmegaConf.select(scene_cfg, "hidden_dim", default=256)),
            num_queries=int(OmegaConf.select(scene_cfg, "num_queries", default=16)),
            num_layers=int(OmegaConf.select(scene_cfg, "num_layers", default=4)),
            num_heads=int(OmegaConf.select(scene_cfg, "num_heads", default=8)),
            mlp_ratio=float(OmegaConf.select(scene_cfg, "mlp_ratio", default=4.0)),
            dropout=float(OmegaConf.select(scene_cfg, "dropout", default=0.0)),
            query_init_std=float(
                OmegaConf.select(scene_cfg, "query_init_std", default=1e-6)
            ),
        )
        if self.scene_query_compressor.num_queries != 16:
            raise ValueError("SQ-3D-Mix requires exactly 16 scene queries")

        vggt_cfg = OmegaConf.select(sq_cfg, "vggt", default={})
        self.vggt_feature_dim = int(
            OmegaConf.select(vggt_cfg, "feature_dim", default=2048)
        )
        self.vggt_view_count = int(
            OmegaConf.select(vggt_cfg, "view_count", default=3)
        )
        self.vggt_view_order = list(
            OmegaConf.select(
                vggt_cfg,
                "view_order",
                default=["cam_f0", "cam_l0", "cam_r0"],
            )
        )
        self.vggt_output_hw = (
            int(OmegaConf.select(vggt_cfg, "pooled_rows", default=6)),
            int(OmegaConf.select(vggt_cfg, "pooled_cols", default=10)),
        )
        if self.vggt_feature_dim != 2048:
            raise ValueError("SQ-3D-Mix requires 2048-dimensional dense VGGT features")
        if self.vggt_view_count != 3 or self.vggt_view_order != [
            "cam_f0",
            "cam_l0",
            "cam_r0",
        ]:
            raise ValueError("SQ-3D-Mix requires front, left, right VGGT view order")
        if self.vggt_output_hw != (6, 10):
            raise ValueError("SQ-3D-Mix requires per-view VGGT pooling to 6x10")

        self.gated_fusion = SceneConditionedGatedFusion(
            scene_dim=qwen_hidden_dim,
            vggt_dim=self.vggt_feature_dim,
        )
        self._configure_fusion_trainability()

        cache_enabled = bool(
            OmegaConf.select(sq_cfg, "cache.enabled", default=False)
        )
        cache_component = str(
            OmegaConf.select(sq_cfg, "cache.component", default="vggt_dense")
        )
        if cache_component != "vggt_dense":
            raise ValueError("SQ-3D-Mix cache component must be 'vggt_dense'")
        if self.fusion_mode == "scene_only" and cache_enabled:
            raise ValueError("scene_only requires framework.sq_3d_mix.cache.enabled=false")
        if self.fusion_mode != "scene_only" and not cache_enabled:
            raise ValueError(
                f"{self.fusion_mode} requires framework.sq_3d_mix.cache.enabled=true"
            )

        intervention_cfg = OmegaConf.select(sq_cfg, "intervention", default={})
        self._sq3dmix_intervention_mode = str(
            OmegaConf.select(intervention_cfg, "mode", default="real")
        ).strip().lower()
        if self._sq3dmix_intervention_mode not in INTERVENTION_MODES:
            raise ValueError(
                "SQ-3D-Mix intervention must be real, zero, gaussian, or shuffled"
            )
        self._sq3dmix_intervention_seed = int(
            OmegaConf.select(intervention_cfg, "seed", default=20260821)
        )
        self._use_named_loss_contract = True

    def _validate_action_only_contract(self, config) -> None:
        if self.action_prompt_mode != "minimal":
            raise ValueError("QwenOFT_SQ3DMix requires action_prompt_mode='minimal'")
        if self.mlp_head != 0:
            raise ValueError("QwenOFT_SQ3DMix requires FlowmatchingActionHead")
        if int(OmegaConf.select(config, "datasets.vla_data.load_act_data", default=0)) != 1:
            raise ValueError("QwenOFT_SQ3DMix requires action data")
        disabled_paths = (
            "datasets.video_data.load_2d_data",
            "datasets.gs_data.load_3d_data",
            "datasets.reward_data.load_reward_data",
            "w_depth",
        )
        enabled = [
            path
            for path in disabled_paths
            if bool(OmegaConf.select(config, path, default=False))
        ]
        if enabled:
            raise ValueError(
                "QwenOFT_SQ3DMix is action-only; disable: " + ", ".join(enabled)
            )
        if len(self.act_query_tokens) != 8 or int(self.act_tok) != 8:
            raise ValueError("QwenOFT_SQ3DMix requires exactly eight action queries")

    def _configure_fusion_trainability(self) -> None:
        trainable_names = {
            "scene_only": set(),
            "projected_concat": {"vggt_projection"},
            "gated": {
                "vggt_projection",
                "gate_projection",
                "semantic_projection",
                "geometry_projection",
            },
        }[self.fusion_mode]
        for name in (
            "vggt_projection",
            "gate_projection",
            "semantic_projection",
            "geometry_projection",
        ):
            module = getattr(self.gated_fusion, name)
            module.requires_grad_(name in trainable_names)

    @staticmethod
    def _build_scene_memory_mask(
        attention_mask: Tensor,
        action_positions: Tensor,
    ) -> Tensor:
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [B,L]")
        if action_positions.ndim != 2 or action_positions.shape != (
            attention_mask.shape[0],
            8,
        ):
            raise ValueError("action_positions must be [B,8]")
        if action_positions.device != attention_mask.device:
            raise ValueError("action_positions and attention_mask must share a device")
        if action_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("action_positions must use an integer dtype")
        if (action_positions < 0).any() or (
            action_positions >= attention_mask.shape[1]
        ).any():
            raise ValueError("action_positions contains an out-of-range index")
        if any(
            torch.unique(sample_positions).numel() != 8
            for sample_positions in action_positions
        ):
            raise ValueError("each sample must contain eight distinct action positions")

        scene_mask = attention_mask.bool().clone()
        if not scene_mask.gather(1, action_positions.long()).all():
            raise ValueError("all action tokens must be valid in attention_mask")
        scene_mask.scatter_(
            dim=1,
            index=action_positions.long(),
            value=False,
        )
        if not scene_mask.any(dim=1).all():
            raise ValueError("scene memory is empty after excluding action tokens")
        return scene_mask

    @staticmethod
    def _payloads_from_examples(
        examples: Sequence[Mapping[str, object]] | None,
    ) -> list[Mapping[str, Tensor]]:
        if examples is None:
            raise RuntimeError("SQ-3D-Mix geometry modes require cache-backed examples")
        payloads = [example.get("vggt_dense_feature_cache") for example in examples]
        if not all(payload is not None for payload in payloads):
            missing = [index for index, payload in enumerate(payloads) if payload is None]
            raise RuntimeError(
                "SQ-3D-Mix requires vggt_dense_feature_cache for every sample; "
                f"missing batch indices: {missing}"
            )
        return payloads

    def _build_sq3dmix_context(
        self,
        last_hidden: Tensor,
        attention_mask: Tensor,
        action_positions: Tensor,
        examples,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        scene_mask = self._build_scene_memory_mask(attention_mask, action_positions)
        scene_tokens, metrics = self.scene_query_compressor(last_hidden, scene_mask)
        if scene_tokens.shape[1] != 16:
            raise RuntimeError("scene compressor did not produce 16 tokens")
        if self.fusion_mode == "scene_only":
            return scene_tokens, metrics

        vggt_tokens = pool_dense_vggt_per_view(
            self._payloads_from_examples(examples),
            output_hw=self.vggt_output_hw,
            device=last_hidden.device,
            dtype=scene_tokens.dtype,
        )
        if vggt_tokens.shape != (
            last_hidden.shape[0],
            180,
            self.vggt_feature_dim,
        ):
            raise RuntimeError(
                f"pooled VGGT shape must be [B,180,{self.vggt_feature_dim}], "
                f"found {tuple(vggt_tokens.shape)}"
            )
        pre_shuffled = bool(
            self._sq3dmix_intervention_mode == "shuffled"
            and examples
            and all(example.get("vggt_dense_pre_shuffled", False) for example in examples)
        )
        vggt_tokens, intervention_metrics = apply_sq3dmix_intervention(
            vggt_tokens,
            "real" if pre_shuffled else self._sq3dmix_intervention_mode,
            self._sq3dmix_intervention_seed,
        )
        if pre_shuffled:
            intervention_metrics["sq3dmix/topology_independent_shuffle"] = torch.ones(
                (), device=vggt_tokens.device
            )
        metrics.update(intervention_metrics)
        if self.fusion_mode == "projected_concat":
            geometry_context = self.gated_fusion.project_geometry(vggt_tokens)
            metrics["sq3dmix/projected_geometry_norm"] = (
                geometry_context.detach().float().norm(dim=-1).mean()
            )
        else:
            geometry_context, fusion_metrics = self.gated_fusion(
                scene_tokens,
                vggt_tokens,
            )
            metrics.update(fusion_metrics)
        action_context = torch.cat([scene_tokens, geometry_context], dim=1)
        if action_context.shape != (last_hidden.shape[0], 196, last_hidden.shape[-1]):
            raise RuntimeError(
                "SQ-3D-Mix extra_context must be [B,196,Dq], found "
                f"{tuple(action_context.shape)}"
            )
        return action_context, metrics

    def _compute_query_extension(
        self,
        last_hidden,
        token_positions,
        examples,
        *,
        input_ids=None,
        attention_mask=None,
        image_grid_thw=None,
    ):
        del input_ids, image_grid_thw
        action_context, metrics = self._build_sq3dmix_context(
            last_hidden=last_hidden,
            attention_mask=attention_mask,
            action_positions=token_positions["action"],
            examples=examples,
        )
        return {
            "action_context": action_context,
            "context_mask": None,
            "losses": {},
            "metrics": metrics,
        }

    def _condition_action_queries(self, action_queries, extension):
        return (
            action_queries,
            extension["action_context"],
            extension["metrics"],
        )

    def _condition_inference_action_queries(
        self,
        last_hidden,
        input_ids,
        action_queries,
        *,
        attention_mask=None,
        image_grid_thw=None,
        examples=None,
    ):
        del image_grid_thw
        token_ids = self._special_token_ids["action"]
        ids = torch.as_tensor(
            token_ids,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        present = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1)).any(dim=1)
        if not present.all():
            raise RuntimeError("Inference prompt is missing one or more action tokens")
        action_positions = self._find_token_positions(input_ids, token_ids)
        action_context, metrics = self._build_sq3dmix_context(
            last_hidden=last_hidden,
            attention_mask=attention_mask,
            action_positions=action_positions,
            examples=examples,
        )
        return action_queries, action_context, metrics

    def get_planning_usage_metrics(self) -> dict[str, Tensor]:
        parameters = {
            "sq3dmix/scene_input_projection_grad_norm": self.scene_query_compressor.input_projection.weight,
            "sq3dmix/scene_output_projection_grad_norm": self.scene_query_compressor.output_projection.weight,
            "sq3dmix/scene_query_grad_norm": self.scene_query_compressor.scene_queries,
            "sq3dmix/vggt_projection_grad_norm": self.gated_fusion.vggt_projection.weight,
            "sq3dmix/gate_projection_grad_norm": self.gated_fusion.gate_projection.weight,
            "sq3dmix/semantic_projection_grad_norm": self.gated_fusion.semantic_projection.weight,
            "sq3dmix/geometry_projection_grad_norm": self.gated_fusion.geometry_projection.weight,
        }
        return {
            name: parameter.grad.detach().float().norm()
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Allow only SQ-3D-Mix modules to be absent in action-only checkpoints."""

        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self._ACTION_ONLY_MISSING_PREFIXES)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if invalid_missing or unexpected:
            raise RuntimeError(
                "QwenOFT_SQ3DMix checkpoint mismatch: "
                f"missing={invalid_missing[:20]} unexpected={unexpected[:20]}"
            )
        return incompatible


# Public spelling retained for checkpoints/configs and external tooling.
QwenOFT_SQ3DMix = Qwenvl_OFT_SQ3DMix
