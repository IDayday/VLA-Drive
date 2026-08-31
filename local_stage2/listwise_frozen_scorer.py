"""Permutation-equivariant listwise ranker for a frozen EpisodeDrive proposal bank.

This module is an audit branch, not a change to the running Stage-2 model.  It
loads the unmodified epoch checkpoint first, then wraps the released scorer
with a zero-initialized candidate-set transformer.  At initialization the
factor logits, direct ranking score, and selected trajectory are therefore
exactly the released scorer outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from local_stage2.scorer_only_training import ScorerOnlyEpisodeDriveAgent


FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


def aggregate_episode_drive_factor_logits(
    factor_logits: Dict[str, torch.Tensor], config
) -> torch.Tensor:
    """Reproduce ActionDecoder's differentiable factor aggregation."""

    return (
        config.noc
        * factor_logits["no_at_fault_collisions"].sigmoid().log()
        + config.dac
        * factor_logits["drivable_area_compliance"].sigmoid().log()
        + config.ddc
        * factor_logits["driving_direction_compliance"].sigmoid().log()
        + (
            config.ttc
            * factor_logits["time_to_collision_within_bound"].sigmoid()
            + config.ep * factor_logits["ego_progress"].sigmoid()
            + config.comfort * factor_logits["comfort"].sigmoid()
        ).log()
    )


class _ZeroResidualHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class ListwiseCandidateScorer(nn.Module):
    """Released factor scorer plus a candidate-set contextual residual."""

    def __init__(
        self,
        base_scorer: nn.Module,
        config,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(config.tf_d_model)
        if hidden_dim % num_heads:
            raise ValueError("tf_d_model must be divisible by listwise num_heads")
        self.base_scorer = base_scorer
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(config.tf_d_ffn),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_context = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.factor_residual = nn.ModuleDict(
            {key: _ZeroResidualHead(hidden_dim) for key in FACTOR_KEYS}
        )
        self.direct_residual = _ZeroResidualHead(hidden_dim)

    def forward(self, proposals, bev_feature):
        outputs = self.base_scorer(proposals, bev_feature)
        pred_logit = dict(outputs[0])
        context = self.candidate_context(bev_feature)
        for key in FACTOR_KEYS:
            pred_logit[key] = pred_logit[key] + self.factor_residual[key](context)
        pred_logit["direct_utility"] = (
            aggregate_episode_drive_factor_logits(pred_logit, self.config)
            + self.direct_residual(context)
        )
        return (pred_logit, *outputs[1:])


class _CandidateLocalResidualBlock(nn.Module):
    """Candidate-wise capacity without a private scene representation.

    The block never mixes the candidate dimension.  It is therefore a clean
    capacity control for the dedicated-feature scorer rather than another
    candidate-set model.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expansion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_dim, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class CapacityMatchedCandidateScorer(nn.Module):
    """Released scorer plus candidate-local, parameter-matched capacity.

    S1 adds a private Q-Former and has 10,703,878 trainable parameters versus
    S0's 6,088,966.  With the released ``hidden_dim=256``, four blocks using
    ``expansion_dim=2053`` plus six residual factor heads add 4,615,194
    parameters.  The resulting 10,704,160 total differs from S1 by only 282
    values (0.0026%).

    All residual factor heads are zero initialized.  Consequently this control
    is exactly equivalent to the released scorer at installation, despite the
    additional internal capacity.
    """

    def __init__(
        self,
        base_scorer: nn.Module,
        config,
        *,
        num_blocks: int = 4,
        expansion_dim: int = 2053,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(config.tf_d_model)
        self.base_scorer = base_scorer
        self.capacity_blocks = nn.ModuleList(
            [
                _CandidateLocalResidualBlock(
                    hidden_dim=hidden_dim,
                    expansion_dim=expansion_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.factor_residual = nn.ModuleDict(
            {key: _ZeroResidualHead(hidden_dim) for key in FACTOR_KEYS}
        )

    def forward(self, proposals, bev_feature):
        outputs = self.base_scorer(proposals, bev_feature)
        pred_logit = dict(outputs[0])
        candidate_features = bev_feature
        for block in self.capacity_blocks:
            candidate_features = block(candidate_features)
        for key in FACTOR_KEYS:
            pred_logit[key] = (
                pred_logit[key] + self.factor_residual[key](candidate_features)
            )
        return (pred_logit, *outputs[1:])


@dataclass(frozen=True)
class ListwiseLossConfig:
    factor_weight: float = 1.0
    pairwise_weight: float = 1.0
    listwise_weight: float = 0.5
    target_temperature: float = 0.05
    prediction_temperature: float = 1.0
    minimum_pair_delta: float = 0.02


def pairwise_logistic_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_delta: float = 0.02,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Weighted all-pairs RankNet loss, excluding indistinguishable ties."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must both have shape [B, K]")
    candidate_count = prediction.shape[1]
    left, right = torch.triu_indices(
        candidate_count, candidate_count, offset=1, device=prediction.device
    )
    target_delta = target[:, left] - target[:, right]
    valid = target_delta.abs() >= minimum_delta
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    prediction_delta = (prediction[:, left] - prediction[:, right]) / temperature
    signs = target_delta.sign()
    weights = target_delta.abs()
    losses = F.softplus(-signs * prediction_delta)
    return (losses[valid] * weights[valid]).sum() / weights[valid].sum()


def listwise_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_temperature: float = 0.05,
    prediction_temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet-style cross entropy over candidates within each scene."""

    target_distribution = F.softmax(target / target_temperature, dim=-1)
    prediction_log_distribution = F.log_softmax(
        prediction / prediction_temperature, dim=-1
    )
    return -(target_distribution * prediction_log_distribution).sum(dim=-1).mean()


class ListwiseFrozenProposalEpisodeDriveAgent(ScorerOnlyEpisodeDriveAgent):
    """Frozen epoch proposals with a zero-initialized contextual ranker."""

    def __init__(self, *args, listwise_config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        values = dict(listwise_config or {})
        architecture_keys = {"num_layers", "num_heads", "dropout"}
        self._listwise_architecture = {
            key: values.pop(key) for key in tuple(values) if key in architecture_keys
        }
        self._listwise_loss_config = ListwiseLossConfig(**values)
        self._listwise_installed = False

    @staticmethod
    def _checkpoint_state(path: str) -> Mapping[str, torch.Tensor]:
        payload = torch.load(path, map_location="cpu")
        return payload.get("state_dict", payload)

    @staticmethod
    def _is_listwise_checkpoint(
        state: Mapping[str, torch.Tensor],
    ) -> bool:
        return any(
            ".action_head.scorer.candidate_context." in key
            or key.startswith("action_head.scorer.candidate_context.")
            for key in state
        )

    def _install_listwise_scorer(self) -> None:
        if self._listwise_installed:
            return
        self.action_head.scorer = ListwiseCandidateScorer(
            self.action_head.scorer,
            self.action_head_config,
            **self._listwise_architecture,
        )
        self._listwise_installed = True

    def _load_listwise_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        normalized = {
            (key[len("agent.") :] if key.startswith("agent.") else key): value
            for key, value in state.items()
        }
        model_state = self.state_dict()
        unexpected = sorted(set(normalized) - set(model_state))
        missing = sorted(set(model_state) - set(normalized))
        shape_errors = sorted(
            key
            for key in set(normalized).intersection(model_state)
            if normalized[key].shape != model_state[key].shape
        )
        if unexpected or missing or shape_errors:
            raise RuntimeError(
                "Invalid trained listwise scorer checkpoint; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"shape_mismatches={shape_errors[:10]}"
            )
        self.load_state_dict(normalized, strict=True)

    def initialize(self) -> None:
        if self._listwise_installed:
            return

        checkpoint_path = self.checkpoint_path
        checkpoint_state = None
        if checkpoint_path and Path(checkpoint_path).is_file():
            candidate_state = self._checkpoint_state(checkpoint_path)
            if self._is_listwise_checkpoint(candidate_state):
                checkpoint_state = candidate_state
            else:
                # The legacy loader below will reopen the checkpoint. Drop the
                # 4+ GB probe mapping first rather than holding two copies.
                del candidate_state

        if checkpoint_state is None:
            # Legacy epoch checkpoint: restore the exact released tree first,
            # then add a zero-initialized contextual residual.
            super().initialize()
            self._install_listwise_scorer()
        else:
            # A trained R3 checkpoint already contains the wrapped scorer.
            # Construct that tree before strict loading so no contextual
            # parameter is silently ignored by the legacy checkpoint filter.
            self.checkpoint_path = None
            try:
                super().initialize()
            finally:
                self.checkpoint_path = checkpoint_path
            self._install_listwise_scorer()
            self._load_listwise_state(checkpoint_state)
            print(f"✅ Trained listwise scorer checkpoint loaded: {checkpoint_path}")

        if os.environ.get("DRIVEVLA_HARD_FREEZE_SCORER_ONLY", "0") == "1":
            from local_stage2.scorer_only_training import (
                hard_freeze_non_scorer_parameters,
            )

            hard_freeze_non_scorer_parameters(self)

    def forward(self, features, targets=None, tokens_list=None):
        prediction = super().forward(features, targets, tokens_list)
        direct_utility = prediction["pred_logit"]["direct_utility"]
        prediction["factor_pdm_score"] = prediction["pdm_score"]
        prediction["pdm_score"] = direct_utility
        selected = direct_utility.argmax(dim=1)
        prediction["trajectory"] = prediction["proposals"][
            torch.arange(len(selected), device=selected.device), selected
        ]
        return prediction

    def compute_loss(self, features, targets, prediction):
        return compute_listwise_training_loss(
            self, targets, prediction, self._listwise_loss_config
        )


class CapacityMatchedSharedFeatureScorerAgent(ScorerOnlyEpisodeDriveAgent):
    """C0: S1-sized scorer that still uses the released shared features.

    This is an attribution control, not a proposed production architecture.
    It distinguishes a benefit from scorer-private scene compression from a
    generic benefit of adding roughly 4.6M trainable parameters.
    """

    def __init__(self, *args, capacity_config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._capacity_config = dict(capacity_config or {})
        self._capacity_installed = False

    @staticmethod
    def _checkpoint_state(path: str) -> Mapping[str, torch.Tensor]:
        payload = torch.load(path, map_location="cpu")
        return payload.get("state_dict", payload)

    @staticmethod
    def _is_capacity_checkpoint(state: Mapping[str, torch.Tensor]) -> bool:
        return any(
            ".action_head.scorer.capacity_blocks." in key
            or key.startswith("action_head.scorer.capacity_blocks.")
            for key in state
        )

    def _install_capacity_scorer(self) -> None:
        if self._capacity_installed:
            return
        self.action_head.scorer = CapacityMatchedCandidateScorer(
            self.action_head.scorer,
            self.action_head_config,
            **self._capacity_config,
        )
        self._capacity_installed = True

    def _load_capacity_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        normalized = {
            (key[len("agent.") :] if key.startswith("agent.") else key): value
            for key, value in state.items()
        }
        model_state = self.state_dict()
        unexpected = sorted(set(normalized) - set(model_state))
        missing = sorted(set(model_state) - set(normalized))
        shape_errors = sorted(
            key
            for key in set(normalized).intersection(model_state)
            if normalized[key].shape != model_state[key].shape
        )
        if unexpected or missing or shape_errors:
            raise RuntimeError(
                "Invalid trained capacity-control checkpoint; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"shape_mismatches={shape_errors[:10]}"
            )
        self.load_state_dict(normalized, strict=True)

    def initialize(self) -> None:
        if self._capacity_installed:
            return

        checkpoint_path = self.checkpoint_path
        checkpoint_state = None
        if checkpoint_path and Path(checkpoint_path).is_file():
            candidate_state = self._checkpoint_state(checkpoint_path)
            if self._is_capacity_checkpoint(candidate_state):
                checkpoint_state = candidate_state
            else:
                del candidate_state

        if checkpoint_state is None:
            super().initialize()
            self._install_capacity_scorer()
        else:
            self.checkpoint_path = None
            try:
                super().initialize()
            finally:
                self.checkpoint_path = checkpoint_path
            self._install_capacity_scorer()
            self._load_capacity_state(checkpoint_state)
            print(f"✅ Capacity-control scorer checkpoint loaded: {checkpoint_path}")

        if os.environ.get("DRIVEVLA_HARD_FREEZE_SCORER_ONLY", "0") == "1":
            from local_stage2.scorer_only_training import (
                hard_freeze_non_scorer_parameters,
            )

            hard_freeze_non_scorer_parameters(self)


class RankingObjectiveScorerOnlyEpisodeDriveAgent(ScorerOnlyEpisodeDriveAgent):
    """Released scorer architecture trained with direct same-scene ranking.

    Unlike :class:`ListwiseFrozenProposalEpisodeDriveAgent`, this control adds
    no candidate-set Transformer and no parameters.  It isolates the effect
    of replacing factor-BCE-only training with RankNet/ListNet supervision.
    """

    def __init__(self, *args, listwise_config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._listwise_loss_config = ListwiseLossConfig(
            **dict(listwise_config or {})
        )

    def compute_loss(self, features, targets, prediction):
        return compute_listwise_training_loss(
            self, targets, prediction, self._listwise_loss_config
        )


def compute_listwise_training_loss(agent, targets, prediction, config):
    """Shared source-factor + pairwise + listwise objective for scorer variants."""

    proposals = prediction["proposals"]
    (
        final_scores,
        best_scores,
        target_scores,
        gt_states,
        gt_valid,
        gt_ego_areas,
    ) = agent.compute_score(targets, proposals, test=False)

    target_trajectory = targets["trajectory"]
    l2_distance = -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5
    sub_score_loss, factor_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = (
        agent.loss.score_loss(
            prediction["pred_logit"],
            prediction["pred_logit2"],
            prediction["pred_agents_states"],
            prediction["pred_area_logit"],
            target_scores,
            gt_states,
            gt_valid,
            gt_ego_areas,
            l2_distance.detach(),
        )
    )

    # Architecture controls may expose an explicit residual utility, while
    # the released and dedicated factor scorers expose only their exact PDM
    # factor aggregation.  Both are differentiable scorer outputs.
    direct_utility = prediction["pred_logit"].get(
        "direct_utility", prediction["pdm_score"]
    )
    pairwise_loss = pairwise_logistic_ranking_loss(
        direct_utility,
        final_scores,
        minimum_delta=config.minimum_pair_delta,
        temperature=config.prediction_temperature,
    )
    listwise_loss = listwise_cross_entropy(
        direct_utility,
        final_scores,
        target_temperature=config.target_temperature,
        prediction_temperature=config.prediction_temperature,
    )
    loss = (
        config.factor_weight * factor_loss
        + config.pairwise_weight * pairwise_loss
        + config.listwise_weight * listwise_loss
    )

    selected = direct_utility.detach().argmax(dim=1)
    selected_score = final_scores.gather(1, selected[:, None]).mean()
    oracle_score = best_scores.mean()
    regret = oracle_score - selected_score
    da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss = (
        sub_score_loss
    )
    return {
        "loss": loss,
        "trajectory_loss": direct_utility.sum() * 0.0,
        "da_loss": da_loss,
        "ttc_loss": ttc_loss,
        "noc_loss": noc_loss,
        "progress_loss": progress_loss,
        "ddc_loss": ddc_loss,
        "comfort_loss": comfort_loss,
        "final_score_loss": factor_loss,
        "pairwise_rank_loss": pairwise_loss,
        "listwise_rank_loss": listwise_loss,
        "pred_ce_loss": pred_ce_loss,
        "pred_l1_loss": pred_l1_loss,
        "pred_area_loss": pred_area_loss,
        "inter_loss0": direct_utility.sum() * 0.0,
        "inter_loss": direct_utility.sum() * 0.0,
        "min_loss0": direct_utility.sum() * 0.0,
        "min_loss": direct_utility.sum() * 0.0,
        "score": selected_score,
        "best_score": oracle_score,
        "scorer_regret": regret,
    }


__all__ = [
    "CapacityMatchedCandidateScorer",
    "CapacityMatchedSharedFeatureScorerAgent",
    "FACTOR_KEYS",
    "ListwiseCandidateScorer",
    "ListwiseFrozenProposalEpisodeDriveAgent",
    "RankingObjectiveScorerOnlyEpisodeDriveAgent",
    "ListwiseLossConfig",
    "aggregate_episode_drive_factor_logits",
    "compute_listwise_training_loss",
    "listwise_cross_entropy",
    "pairwise_logistic_ranking_loss",
]
