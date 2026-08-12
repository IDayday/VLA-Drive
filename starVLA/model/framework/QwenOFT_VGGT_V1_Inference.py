"""Inference-only compatibility wrapper for the original 63-query VGGT run.

The production ``QwenOFT_VGGT`` class implements the newer V2 memory contract.
This wrapper preserves the prompt and planner bridge used by checkpoints whose
saved config contains ``framework.vggt.expected_query_count``.  It never loads
the offline VGGT teacher or its cache.
"""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.modules.vggt_query.alignment import VGGTQueryAligner
from starVLA.model.modules.vggt_query.planner_bridge import PlanningQueryBridge
from starVLA.model.modules.vggt_query.types import (
    VGGTQueryLayout,
    build_vggt_query_tokens,
)


class Qwenvl_OFT_VGGT_V1_Inference(Qwenvl_OFT):
    """Load and run a legacy ``[B,63,2048]`` VGGT-query checkpoint."""

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
        self.vggt_enabled = bool(
            OmegaConf.select(config, "framework.vggt.enabled", default=False)
        )
        if not self.vggt_enabled:
            return

        layout_cfg = OmegaConf.select(config, "framework.vggt.layout", default={})
        self.vggt_layout = VGGTQueryLayout(
            view_count=int(OmegaConf.select(layout_cfg, "view_count", default=3)),
            special_per_view=int(
                OmegaConf.select(layout_cfg, "special_per_view", default=5)
            ),
            spatial_rows=int(OmegaConf.select(layout_cfg, "spatial_rows", default=4)),
            spatial_cols=int(OmegaConf.select(layout_cfg, "spatial_cols", default=4)),
            teacher_dim=int(OmegaConf.select(layout_cfg, "teacher_dim", default=2048)),
        )
        expected_count = int(
            OmegaConf.select(
                config,
                "framework.vggt.expected_query_count",
                default=self.vggt_layout.query_count,
            )
        )
        if self.vggt_layout.query_count != expected_count:
            raise ValueError(
                "Legacy VGGT query count does not match its saved layout: "
                f"layout={self.vggt_layout.query_count}, expected={expected_count}"
            )
        if self.vggt_layout.spatial_rows != self.vggt_layout.spatial_cols:
            raise ValueError("Legacy compact VGGT spatial layout must be square")

        self.vggt_query_tokens = list(build_vggt_query_tokens(self.vggt_layout))
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        vocabulary = tokenizer.get_vocab()
        missing_tokens = [
            token for token in self.vggt_query_tokens if token not in vocabulary
        ]
        if missing_tokens:
            raise RuntimeError(
                "The selected V1 VGGT VLM is missing legacy query tokens. "
                f"First missing token: {missing_tokens[0]}"
            )
        vggt_token_ids = tuple(tokenizer.convert_tokens_to_ids(self.vggt_query_tokens))
        if len(set(vggt_token_ids)) != expected_count:
            raise RuntimeError("Legacy VGGT query tokens do not map to unique IDs")
        self._special_token_ids["vggt"] = vggt_token_ids

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        configured_hidden = int(
            OmegaConf.select(
                config, "framework.qwenvl.vl_hidden_dim", default=hidden_dim
            )
        )
        if configured_hidden != hidden_dim:
            raise ValueError(
                "Qwen hidden contract mismatch: "
                f"config={configured_hidden}, model={hidden_dim}"
            )
        align_cfg = OmegaConf.select(config, "framework.vggt.alignment", default={})
        self.vggt_aligner = VGGTQueryAligner(
            student_dim=hidden_dim,
            teacher_dim=self.vggt_layout.teacher_dim,
            special_query_count=self.vggt_layout.special_query_count,
            cosine_weight=float(
                OmegaConf.select(align_cfg, "cosine_weight", default=1.0)
            ),
            smooth_l1_weight=float(
                OmegaConf.select(align_cfg, "smooth_l1_weight", default=0.1)
            ),
            relational_weight=float(
                OmegaConf.select(align_cfg, "relational_weight", default=0.05)
            ),
        )
        bridge_cfg = OmegaConf.select(
            config, "framework.vggt.planner_bridge", default={}
        )
        self.vggt_planner_bridge = PlanningQueryBridge(
            hidden_dim=hidden_dim,
            num_heads=int(OmegaConf.select(bridge_cfg, "num_heads", default=16)),
            initial_gate=float(
                OmegaConf.select(bridge_cfg, "initial_gate", default=0.5)
            ),
        )
        self.vggt_access_enabled = bool(
            OmegaConf.select(config, "framework.vggt.access_enabled", default=True)
        )

    def _build_action_prompt_suffix(self) -> str:
        if not getattr(self, "vggt_enabled", False):
            return super()._build_action_prompt_suffix()
        return (
            f" {self.robot_history_token}"
            f"{''.join(self.vggt_query_tokens)}"
            f"{''.join(self.act_query_tokens)}"
        )

    @staticmethod
    def _gather_queries(last_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Gather query states from ``[B,L,H]`` into ``[B,Q,H]``."""

        assert last_hidden.ndim == 3, "last_hidden must be [B,L,H]"
        assert positions.ndim == 2, "query positions must be [B,Q]"
        index = positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        return last_hidden.gather(dim=1, index=index)

    def _condition_inference_action_queries(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_queries: torch.Tensor,
        *,
        image_grid_thw=None,
    ):
        """Read all 63 V1 query states and return eight conditioned queries."""

        del image_grid_thw
        if not self.vggt_enabled or not self.vggt_access_enabled:
            return action_queries, None, {}
        token_ids = self._special_token_ids["vggt"]
        ids = torch.as_tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)
        present = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1)).any(dim=1)
        if not present.all():
            raise RuntimeError("Inference prompt is missing one or more V1 VGGT query tokens")
        positions = self._find_token_positions(input_ids, token_ids)
        context = self._gather_queries(last_hidden, positions)
        assert context.shape[1:] == (
            self.vggt_layout.query_count,
            last_hidden.shape[-1],
        )
        valid_mask = torch.ones(
            context.shape[:2], dtype=torch.bool, device=context.device
        )
        # DeepSpeed saved the trained V1 bridge in BF16.  The optimized Qwen
        # inference path can expose FP32 final hidden states on PPU, while PPU
        # LayerNorm requires its input and affine parameters to have one dtype.
        # Match the exact trained bridge dtype locally; no parameter is cast or
        # recreated in forward.
        bridge_dtype = self.vggt_planner_bridge.action_norm.weight.dtype
        action_queries = action_queries.to(dtype=bridge_dtype)
        context = context.to(dtype=bridge_dtype)
        enhanced, diagnostics = self.vggt_planner_bridge(
            action_queries, context, valid_mask
        )
        return enhanced, context, diagnostics
