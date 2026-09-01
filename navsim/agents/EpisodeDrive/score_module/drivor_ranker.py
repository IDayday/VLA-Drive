"""DrivOR-initialized scorer for an immutable external proposal bank.

Only current-observation DrivOR registers, current ego status and proposal
geometry enter ``forward``.  The public EpisodeDrive score, PDM labels and
future files are intentionally absent from the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Literal, Mapping, Tuple

import torch
import torch.nn as nn

from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    ConservativeReferenceConfig,
    ConservativeReferenceHead,
    conservative_reference_selection_scores,
    pdms_factor_log_utility,
)
from navsim.agents.EpisodeDrive.score_module.scorer import Scorer
from navsim.agents.EpisodeDrive.transformer_decoder import TransformerDecoderScorer


@dataclass(frozen=True)
class DrivORRankerConfig:
    model_dim: int = 256
    feedforward_dim: int = 1024
    status_dim: int = 11
    num_poses: int = 8
    scorer_layers: int = 4
    attention_heads: int = 1
    projection_dropout: float = 0.1
    drop_path: float = 0.2


class DrivORInitializedProposalRanker(nn.Module):
    """Exact DrivOR scoring path plus an optional direct-ranking head."""

    _CHECKPOINT_PREFIX = "agent._drivor_model."
    _PRETRAINED_MODULES: Tuple[str, ...] = (
        "hist_encoding",
        "pos_embed",
        "scorer_attention",
        "scorer",
    )

    def __init__(self, config: DrivORRankerConfig = DrivORRankerConfig()):
        super().__init__()
        self.config = config
        external_config = SimpleNamespace(
            b2d=False,
            proposal_num=64,
            num_poses=config.num_poses,
            tf_d_model=config.model_dim,
            tf_d_ffn=config.feedforward_dim,
            scorer_ref_num=config.scorer_layers,
            refiner_num_heads=config.attention_heads,
            refiner_ls_values=0.0,
            one_token_per_traj=True,
            double_score=False,
            agent_pred=False,
            area_pred=False,
            bev_map=False,
            bev_agent=False,
        )
        self.hist_encoding = nn.Linear(config.status_dim, config.model_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(config.num_poses * 3, config.feedforward_dim),
            nn.ReLU(),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_layers,
            d_model=config.model_dim,
            proj_drop=config.projection_dropout,
            drop_path=config.drop_path,
            config=external_config,
        )
        self.scorer = Scorer(external_config)
        self.direct_utility_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )
        nn.init.zeros_(self.direct_utility_head[-1].weight)
        nn.init.zeros_(self.direct_utility_head[-1].bias)

    @staticmethod
    def _unwrap_state(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
        state = payload.get("state_dict", payload)
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint does not contain a state dictionary")
        return state  # type: ignore[return-value]

    def load_drivor_checkpoint(self, checkpoint: Path) -> Dict[str, object]:
        """Load only scorer-path parameters from the released DrivOR model."""

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_state = self._unwrap_state(payload)
        own_state = self.state_dict()
        selected: Dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if not str(key).startswith(self._CHECKPOINT_PREFIX):
                continue
            stripped = str(key)[len(self._CHECKPOINT_PREFIX) :]
            if stripped.split(".", 1)[0] in self._PRETRAINED_MODULES:
                selected[stripped] = value
        expected = {
            key
            for key in own_state
            if key.split(".", 1)[0] in self._PRETRAINED_MODULES
        }
        missing = sorted(expected.difference(selected))
        unexpected = sorted(set(selected).difference(expected))
        mismatched = sorted(
            key
            for key in expected.intersection(selected)
            if tuple(own_state[key].shape) != tuple(selected[key].shape)
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "DrivOR scorer checkpoint mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"mismatched={mismatched[:5]}"
            )
        merged = dict(own_state)
        merged.update(selected)
        self.load_state_dict(merged, strict=True)
        return {
            "loaded_tensor_count": len(selected),
            "loaded_value_count": sum(value.numel() for value in selected.values()),
            "direct_head_zero_initialized": True,
        }

    def forward(
        self,
        scene_registers: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if scene_registers.ndim != 3 or scene_registers.shape[-1] != self.config.model_dim:
            raise ValueError("scene_registers must have shape [B,S,256]")
        if status_feature.shape != (scene_registers.shape[0], self.config.status_dim):
            raise ValueError("status_feature shape mismatch")
        if proposals.ndim != 4 or proposals.shape[0] != scene_registers.shape[0]:
            raise ValueError("proposals must have shape [B,K,T,3]")
        if tuple(proposals.shape[-2:]) != (self.config.num_poses, 3):
            raise ValueError("proposal pose shape mismatch")
        if scene_valid_mask is not None:
            if scene_valid_mask.shape != scene_registers.shape[:2]:
                raise ValueError("scene_valid_mask shape mismatch")
            if not bool(scene_valid_mask.all()):
                raise ValueError("DrivOR register cache must not contain padded tokens")

        batch_size, candidate_count = proposals.shape[:2]
        ego_token = self.hist_encoding(status_feature).unsqueeze(1)
        embedded = self.pos_embed(
            proposals.reshape(batch_size, candidate_count, -1).detach()
        )
        candidate_features = (
            self.scorer_attention(embedded, scene_registers) + ego_token
        )
        predicted = self.scorer(proposals, candidate_features)[0]
        factor_logits = torch.stack(
            [predicted[key] for key in FACTOR_KEYS], dim=-1
        )
        direct_utility = self.direct_utility_head(candidate_features).squeeze(-1)
        return {
            "factor_logits": factor_logits,
            "direct_utility": direct_utility,
            "candidate_features": candidate_features,
        }


class DrivORReferenceGateRanker(nn.Module):
    """Independent DrivOR scorer plus a conservative reference ranker.

    ``factor`` and ``direct`` compare one independently selected proposal with
    a deployable reference proposal. ``all`` instead ranks every proposal by
    its predicted reference-relative gain.  The reference's numeric score is
    absent in every mode, and the exact reference proposal remains the fallback.
    """

    _ALTERNATIVE_MODES = frozenset(("factor", "direct", "all"))

    def __init__(
        self,
        ranker_config: DrivORRankerConfig = DrivORRankerConfig(),
        reference_config: ConservativeReferenceConfig | None = None,
        alternative_mode: Literal["factor", "direct", "all"] = "factor",
        alternative_count: int = 1,
    ) -> None:
        super().__init__()
        if alternative_mode not in self._ALTERNATIVE_MODES:
            raise ValueError(f"unsupported alternative mode: {alternative_mode}")
        if not 1 <= alternative_count <= 64:
            raise ValueError("alternative_count must lie in [1, 64]")
        self.ranker_config = ranker_config
        self.reference_config = reference_config or ConservativeReferenceConfig(
            model_dim=ranker_config.model_dim,
            num_heads=8,
            dropout=ranker_config.projection_dropout,
        )
        if self.reference_config.model_dim != ranker_config.model_dim:
            raise ValueError("reference and DrivOR ranker widths must match")
        self.alternative_mode = alternative_mode
        self.alternative_count = alternative_count
        self.ranker = DrivORInitializedProposalRanker(ranker_config)
        self.reference_head = ConservativeReferenceHead(self.reference_config)

    def _independent_utility(self, output: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.alternative_mode in {"factor", "all"}:
            return pdms_factor_log_utility(output["factor_logits"])
        return output["direct_utility"]

    def forward(
        self,
        scene_registers: torch.Tensor,
        status_feature: torch.Tensor,
        proposals: torch.Tensor,
        reference_indices: torch.Tensor,
        scene_valid_mask: torch.Tensor | None = None,
        *,
        gain_quantile_index: int = 0,
        minimum_lcb_gain: float = 0.0,
        maximum_safety_worse_probability: float = 0.25,
        minimum_safe_improvement_probability: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        output = self.ranker(
            scene_registers,
            status_feature,
            proposals,
            scene_valid_mask=scene_valid_mask,
        )
        independent_utility = self._independent_utility(output)
        keep_count = min(self.alternative_count, independent_utility.shape[1])
        alternative_candidate_indices = independent_utility.topk(
            keep_count, dim=1
        ).indices
        alternative_indices = alternative_candidate_indices[:, 0]
        relative = self.reference_head(
            output["candidate_features"], reference_indices
        )
        if self.alternative_mode == "all":
            allowed = torch.ones_like(independent_utility, dtype=torch.bool)
        else:
            allowed = torch.zeros_like(independent_utility, dtype=torch.bool)
            allowed.scatter_(1, reference_indices[:, None], True)
            allowed.scatter_(1, alternative_candidate_indices, True)
        selection_scores = conservative_reference_selection_scores(
            relative["gain_quantiles"],
            relative["safety_worse_logits"],
            relative["safe_improvement_logit"],
            reference_indices,
            gain_quantile_index=gain_quantile_index,
            minimum_lcb_gain=minimum_lcb_gain,
            maximum_safety_worse_probability=(
                maximum_safety_worse_probability
            ),
            minimum_safe_improvement_probability=(
                minimum_safe_improvement_probability
            ),
            allowed_candidate_mask=allowed,
        )
        return output | relative | {
            "independent_utility": independent_utility,
            "alternative_indices": alternative_indices,
            "alternative_candidate_indices": alternative_candidate_indices,
            "allowed_candidate_mask": allowed,
            "reference_selection_scores": selection_scores,
            "selected_indices": selection_scores.argmax(dim=1),
        }
