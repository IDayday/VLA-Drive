"""Online EpisodeDrive adapter for a scorer-private shortlist reranker.

The released EpisodeDrive scorer supplies only a coarse candidate shortlist.
Its numeric scores and hidden candidate features are never inputs to the
independent ranker.  The fine ranker consumes current scene tokens, current ego
features and proposal geometry, matching the immutable replay training path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from navsim.agents.EpisodeDrive.episodedrive_agent import EpisodeDriveAgent
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class IndependentShortlistScorerAgent(EpisodeDriveAgent):
    """Released generator + Base coarse gate + independent fine scorer."""

    ARTIFACT_TYPE = "episode_drive_independent_shortlist_scorer_v1"

    def __init__(self, *args, **kwargs) -> None:
        action_config = kwargs.get("action_head_config")
        if action_config is None:
            raise ValueError("action_head_config is required")
        try:
            from omegaconf import OmegaConf

            OmegaConf.update(
                action_config, "return_memory_fields", True, force_add=True
            )
        except Exception:
            setattr(action_config, "return_memory_fields", True)
        super().__init__(*args, **kwargs)
        self.independent_ranker: Optional[IndependentProposalRanker] = None
        self._independent_artifact: Optional[Dict[str, object]] = None

    def initialize(self) -> None:
        if self._initialized and self.independent_ranker is not None:
            return
        if not self.checkpoint_path:
            raise RuntimeError("Independent scorer deployment artifact is required")
        requested = Path(self.checkpoint_path)
        payload = torch.load(requested, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != self.ARTIFACT_TYPE:
            raise RuntimeError(
                f"Expected {self.ARTIFACT_TYPE}, got {payload.get('artifact_type')!r}"
            )
        base_checkpoint = Path(str(payload["base_checkpoint_path"]))
        if not base_checkpoint.is_file():
            raise FileNotFoundError(base_checkpoint)
        actual_base_sha = _sha256(base_checkpoint)
        if actual_base_sha != payload["base_checkpoint_sha256"]:
            raise RuntimeError("Released Base checkpoint SHA256 mismatch")

        self.checkpoint_path = str(base_checkpoint)
        try:
            super().initialize()
        finally:
            self.checkpoint_path = str(requested)

        model = IndependentProposalRanker(
            IndependentRankerConfig(**dict(payload["model_config"]))
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        self.independent_ranker = model
        self._independent_artifact = dict(payload)
        print(f"✅ Independent shortlist scorer loaded: {requested}")

    def forward(self, features, targets=None, tokens_list=None):
        prediction = super().forward(features, targets, tokens_list)
        if self.independent_ranker is None or self._independent_artifact is None:
            raise RuntimeError("Independent scorer agent was not initialized")

        scene_features = prediction["language_feature"]
        ego_feature = prediction["ego_feature"]
        if ego_feature.ndim == 3 and ego_feature.shape[1] == 1:
            ego_feature = ego_feature[:, 0]
        proposals = prediction["proposals"]
        ranker_output = self.independent_ranker(
            scene_features,
            ego_feature,
            proposals,
        )
        score_mode = str(self._independent_artifact["score_mode"])
        if score_mode == "coarse":
            reranker_utility = ranker_output["coarse_utility"]
        elif score_mode == "factor":
            reranker_utility = pdms_factor_log_utility(
                ranker_output["factor_logits"]
            )
        elif score_mode == "direct":
            reranker_utility = ranker_output["utility"]
        else:
            raise RuntimeError(f"Unsupported independent score mode: {score_mode}")

        base_scores = prediction["pdm_score"]
        shortlist_size = int(self._independent_artifact["shortlist_size"])
        shortlist_size = min(shortlist_size, base_scores.shape[1])
        shortlist = base_scores.topk(shortlist_size, dim=1).indices
        shortlist_utility = reranker_utility.gather(1, shortlist)
        selected = shortlist.gather(
            1, shortlist_utility.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        selection_scores = torch.full_like(reranker_utility, -1.0e4)
        selection_scores.scatter_(1, shortlist, shortlist_utility)

        prediction["base_pdm_score"] = base_scores
        prediction["independent_utility"] = reranker_utility
        prediction["independent_shortlist_indices"] = shortlist
        prediction["pdm_score"] = selection_scores
        prediction["trajectory"] = proposals[
            torch.arange(len(selected), device=selected.device), selected
        ]
        return prediction


def build_independent_shortlist_artifact(
    ranker_artifact: Mapping[str, object],
    *,
    ranker_artifact_path: Path,
    base_checkpoint_path: Path,
    shortlist_size: int,
    score_mode: str,
) -> Dict[str, object]:
    """Package a replay-trained ranker with immutable deployment lineage."""

    if ranker_artifact.get("architecture") != "IndependentProposalRanker":
        raise ValueError("ranker artifact has the wrong architecture")
    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    if score_mode not in {"coarse", "factor", "direct"}:
        raise ValueError("score_mode must be coarse, factor or direct")
    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(base_checkpoint_path)
    return {
        "artifact_type": IndependentShortlistScorerAgent.ARTIFACT_TYPE,
        "artifact_version": 1,
        "base_checkpoint_path": str(base_checkpoint_path.resolve()),
        "base_checkpoint_sha256": _sha256(base_checkpoint_path),
        "source_ranker_artifact_path": str(ranker_artifact_path.resolve()),
        "source_ranker_artifact_sha256": _sha256(ranker_artifact_path),
        "source_ranker_epoch": int(ranker_artifact["epoch"]),
        "model_config": dict(ranker_artifact["model_config"]),
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in ranker_artifact["state_dict"].items()
        },
        "shortlist_size": int(shortlist_size),
        "score_mode": score_mode,
        "base_numeric_score_used_by_independent_ranker": False,
        "base_rank_used_for_shortlist": True,
        "inference_input_schema": (
            "current_scene_tokens",
            "current_ego_feature",
            "proposals",
            "base_topk_membership",
        ),
        "forbidden_inputs": (
            "future_annotations",
            "future_images",
            "official_pdm_score",
            "metric_cache",
        ),
    }


__all__ = [
    "IndependentShortlistScorerAgent",
    "build_independent_shortlist_artifact",
]
