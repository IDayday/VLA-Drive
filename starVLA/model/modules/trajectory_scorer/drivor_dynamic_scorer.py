# SPDX-License-Identifier: Apache-2.0
# Adapted from valeoai/DrivoR commit
# f02665403df799c1b4ddd8b0d34e073f0555c13a, files
# navsim/agents/drivoR/transformer_decoder.py,
# navsim/agents/drivoR/score_module/scorer.py, and drivor_model.py.
# Project adaptations: accept external proposal pools, detach proposal geometry
# at the scorer boundary, and use a shared 256-dimensional planning space.

"""DrivoR dynamic-proposal pre-scorer without its trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .attention import TransformerDecoder
from .losses import DRIVOR_METRICS, aggregate_drivor_score


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class DynamicScorerOutput:
    """DrivoR scores and Top-M tensors aligned to original dynamic IDs."""

    metric_logits: Dict[str, Tensor]
    aggregate_score: Tensor
    topm_indices: Tensor
    topm_trajectories_8: Tensor
    topm_candidate_states: Tensor
    # Optional PDMS-direct value head used by the CLOVER/DriveVLA-M0 route.
    # Legacy checkpoints leave these fields disabled and continue selecting
    # with the donor composed sub-score formula.
    aggregate_logit: Tensor | None = None
    formula_score: Tensor | None = None


class DrivoRDynamicScorer(nn.Module):
    """Score detached physical ``[B,K,8,3]`` proposal pools."""

    def __init__(
        self,
        *,
        scene_dim: int = 256,
        ego_state_dim: int = 4,
        model_dim: int = 256,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        decoder_style: str = "legacy",
        proj_drop: float = 0.1,
        drop_path: float = 0.2,
        layer_scale_init: float = 0.0,
        noc: float = 1.0,
        dac: float = 1.0,
        ddc: float = 0.0,
        ttc: float = 5.0,
        ep: float = 5.0,
        comfort: float = 2.0,
        aggregate_head: bool = False,
        selection_mode: str = "formula",
        aggregate_temperature: float = 1.0,
        selection_alpha: float = 0.0,
        debug_validate_finite: bool = False,
    ) -> None:
        super().__init__()
        if ego_state_dim <= 0 or model_dim <= 0:
            raise ValueError("ego_state_dim and model_dim must be positive")
        self.scene_dim = scene_dim
        self.ego_state_dim = ego_state_dim
        self.model_dim = model_dim
        self.debug_validate_finite = debug_validate_finite
        if selection_mode not in {
            "formula",
            "learned_aggregate",
            "calibrated_hybrid",
        }:
            raise ValueError(
                "selection_mode must be formula, learned_aggregate, or "
                "calibrated_hybrid"
            )
        if selection_mode in {"learned_aggregate", "calibrated_hybrid"} and not aggregate_head:
            raise ValueError(f"{selection_mode} selection requires aggregate_head=true")
        if aggregate_temperature <= 0:
            raise ValueError("aggregate_temperature must be positive")
        if selection_mode == "calibrated_hybrid" and aggregate_temperature != 1.0:
            raise ValueError(
                "calibrated_hybrid standardizes each scene, so aggregate_temperature "
                "must remain the non-tunable compatibility value 1.0"
            )
        if not 0.0 <= selection_alpha <= 1.0:
            raise ValueError("selection_alpha must lie in [0,1]")
        self.aggregate_head_enabled = bool(aggregate_head)
        self.selection_mode = str(selection_mode)
        self.aggregate_temperature = float(aggregate_temperature)
        self.register_buffer(
            "selection_alpha",
            torch.tensor(float(selection_alpha), dtype=torch.float32),
            # Legacy formula/direct checkpoints must retain their exact state
            # dict. The calibrated route persists alpha as part of its scorer
            # contract and therefore restores it automatically at inference.
            persistent=selection_mode == "calibrated_hybrid",
        )
        self.aggregate_weights = {
            "noc": noc,
            "dac": dac,
            "ddc": ddc,
            "ttc": ttc,
            "ep": ep,
            "comfort": comfort,
        }
        self.trajectory_embedding = _mlp(8 * 3, ffn_dim, model_dim)
        self.ego_encoder = _mlp(ego_state_dim, ffn_dim, model_dim)
        if scene_dim != model_dim:
            raise ValueError("DrivoR query and scene memory must share model_dim")
        if decoder_style == "legacy":
            # Preserve the existing Flow-DiT experiment and its checkpoints.
            self.scorer_decoder = TransformerDecoder(
                num_layers=num_layers,
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                return_intermediate=False,
            )
        elif decoder_style == "donor_register":
            from starVLA.model.modules.register_planner.decoder import (
                RegisterTrajectoryDecoder,
            )

            self.scorer_decoder = RegisterTrajectoryDecoder(
                num_layers=num_layers,
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                proj_drop=proj_drop,
                drop_path=drop_path,
                layer_scale_init=layer_scale_init,
                return_intermediate=False,
            )
        else:
            raise ValueError(
                "decoder_style must be 'legacy' or 'donor_register'"
            )
        self.decoder_style = decoder_style
        self.metric_heads = nn.ModuleDict(
            {name: _mlp(model_dim, ffn_dim, 1) for name in DRIVOR_METRICS}
        )
        self.aggregate_head = (
            _mlp(model_dim, ffn_dim, 1) if self.aggregate_head_enabled else None
        )

    @staticmethod
    def _scene_standardize(score: Tensor) -> Tensor:
        score = score.float()
        centered = score - score.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        return centered / scale

    def calibrated_hybrid_score(
        self,
        aggregate_logit: Tensor,
        formula_score: Tensor,
        *,
        alpha: float | Tensor | None = None,
    ) -> Tensor:
        """Fuse direct-PDMS and structured DrivoR ranks on a common scale."""

        if alpha is None:
            alpha = self.selection_alpha
        alpha_tensor = torch.as_tensor(
            alpha, device=aggregate_logit.device, dtype=torch.float32
        )
        if bool(((alpha_tensor < 0) | (alpha_tensor > 1)).any()):
            raise ValueError("calibrated selector alpha must lie in [0,1]")
        direct = self._scene_standardize(aggregate_logit.float())
        structured = self._scene_standardize(formula_score)
        return (1.0 - alpha_tensor) * direct + alpha_tensor * structured

    @torch.no_grad()
    def set_selection_alpha(self, alpha: float) -> None:
        if self.selection_mode != "calibrated_hybrid":
            raise RuntimeError("selection alpha is only valid for calibrated_hybrid")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("selection alpha must lie in [0,1]")
        self.selection_alpha.fill_(float(alpha))

    def _encode_ego(self, ego_state: Tensor, batch_size: int, reference: Tensor) -> Tensor:
        if ego_state is None:
            raise ValueError("DrivoR scorer requires ego_state")
        if ego_state.ndim == 3 and ego_state.shape[1] == 1:
            ego_state = ego_state[:, 0]
        if ego_state.ndim != 2 or tuple(ego_state.shape) != (
            batch_size,
            self.ego_state_dim,
        ):
            raise ValueError(
                f"ego_state must have shape [B,{self.ego_state_dim}] or "
                f"[B,1,{self.ego_state_dim}], got {tuple(ego_state.shape)}"
            )
        return self.ego_encoder(
            ego_state.to(device=reference.device, dtype=reference.dtype)
        )[:, None]

    def forward(
        self,
        proposals_navsim: Tensor,
        global_scene_tokens: Tensor,
        ego_state: Tensor,
        *,
        topm: int = 32,
    ) -> DynamicScorerOutput:
        """Score proposals; gradients never propagate to proposal geometry."""

        if proposals_navsim.ndim != 4 or tuple(proposals_navsim.shape[-2:]) != (8, 3):
            raise ValueError("proposals_navsim must have shape [B,K,8,3]")
        batch_size, candidate_count = proposals_navsim.shape[:2]
        if topm <= 0 or topm > candidate_count:
            raise ValueError(
                f"topm must be in [1,{candidate_count}], got {topm}"
            )
        if global_scene_tokens.ndim != 3 or (
            global_scene_tokens.shape[0] != batch_size
            or global_scene_tokens.shape[-1] != self.scene_dim
        ):
            raise ValueError(
                f"global_scene_tokens must have shape [B,S,{self.scene_dim}]"
            )
        if proposals_navsim.device != global_scene_tokens.device:
            raise ValueError("proposals and global scene tokens must share a device")

        detached_proposals = proposals_navsim.detach().to(
            dtype=global_scene_tokens.dtype
        )
        candidate_states = self.trajectory_embedding(
            detached_proposals.reshape(batch_size, candidate_count, 8 * 3)
        )
        candidate_states = self.scorer_decoder(
            candidate_states, global_scene_tokens
        )
        candidate_states = candidate_states + self._encode_ego(
            ego_state, batch_size, candidate_states
        )
        metric_logits = {
            name: head(candidate_states).squeeze(-1)
            for name, head in self.metric_heads.items()
        }
        formula_score = aggregate_drivor_score(
            metric_logits, **self.aggregate_weights
        )
        aggregate_logit = (
            self.aggregate_head(candidate_states).squeeze(-1)
            if self.aggregate_head is not None
            else None
        )
        if self.selection_mode == "learned_aggregate":
            aggregate_score = torch.sigmoid(
                aggregate_logit.float() / self.aggregate_temperature
            )
        elif self.selection_mode == "calibrated_hybrid":
            aggregate_score = self.calibrated_hybrid_score(
                aggregate_logit, formula_score
            )
        else:
            aggregate_score = formula_score
        if self.debug_validate_finite and (
            not torch.isfinite(candidate_states).all()
            or not torch.isfinite(aggregate_score).all()
        ):
            raise ValueError("DrivoR scorer produced NaN or Inf")
        _, topm_indices = torch.topk(aggregate_score, k=topm, dim=1)
        trajectory_index = topm_indices[..., None, None].expand(-1, -1, 8, 3)
        state_index = topm_indices[..., None].expand(-1, -1, self.model_dim)
        return DynamicScorerOutput(
            metric_logits=metric_logits,
            aggregate_score=aggregate_score,
            topm_indices=topm_indices,
            topm_trajectories_8=torch.gather(
                detached_proposals, 1, trajectory_index
            ),
            topm_candidate_states=torch.gather(
                candidate_states, 1, state_index
            ),
            aggregate_logit=aggregate_logit,
            formula_score=formula_score,
        )
