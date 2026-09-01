#!/usr/bin/env python3
"""Compare a real M0-native private Agent with cached scorer inference."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from local_stage2.m0_native_private_scorer_agent import (
    M0NativePrivateScorerAgent,
)
from local_stage2.train_independent_scorer import load_private_observation_table
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)


def _maximum_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(
        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-predictions", type=Path, required=True)
    parser.add_argument("--public-feature-cache", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    args = parser.parse_args()

    with args.online_predictions.open("rb") as file:
        online = pickle.load(file)
    with args.public_feature_cache.open("rb") as file:
        public = pickle.load(file)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != M0NativePrivateScorerAgent.ARTIFACT_TYPE:
        raise RuntimeError("deployment artifact type mismatch")
    tokens = sorted(online)
    if len(tokens) < 4 or not set(tokens).issubset(public):
        raise RuntimeError("at least four online tokens must exist in public cache")
    private = load_private_observation_table(args.private_observation_root)
    private_row = {token: index for index, token in enumerate(private.tokens)}
    if not set(tokens).issubset(private_row):
        raise RuntimeError("online tokens are absent from private observation cache")

    device = torch.device(args.device)
    architecture = str(artifact["scorer_architecture"])
    if architecture == "IndependentProposalRanker":
        model = IndependentProposalRanker(
            IndependentRankerConfig(**dict(artifact["private_config"]))
        )
    elif architecture == "M0PrivateResidualRanker":
        model = M0PrivateResidualRanker(
            IndependentRankerConfig(**dict(artifact["private_config"])),
            M0PrivateResidualConfig(**dict(artifact["residual_config"])),
        )
    else:
        raise RuntimeError(f"unsupported scorer architecture: {architecture}")
    model.load_state_dict(artifact["scorer_state_dict"], strict=True)
    model.to(device).eval()

    rows = torch.tensor([private_row[token] for token in tokens], dtype=torch.long)
    observation = private.observation_tokens[rows].to(device).float()
    observation_mask = private.observation_valid_masks[rows].to(device)
    status = private.status_features[rows].to(device).float()
    proposals = torch.as_tensor(
        np.stack([public[token]["proposals"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    base_scores = torch.as_tensor(
        np.stack([public[token]["predicted_scores"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        if architecture == "IndependentProposalRanker":
            output = model(
                observation,
                status,
                proposals,
                observation_valid_mask=observation_mask,
            )
            mode = str(artifact["score_mode"])
            if mode == "direct":
                utility = output["utility"]
            elif mode == "coarse":
                utility = output["coarse_utility"]
            elif mode == "factor":
                utility = pdms_factor_log_utility(output["factor_logits"])
            else:
                raise RuntimeError(f"unsupported independent score mode: {mode}")
            shortlist = base_scores.topk(
                int(artifact["shortlist_size"]), dim=1
            ).indices
            cached_scores = torch.full_like(utility, -1.0e4)
            cached_scores.scatter_(1, shortlist, utility.gather(1, shortlist))
        else:
            base_factor_logits = torch.as_tensor(
                np.stack([public[token]["base_factor_logits"] for token in tokens]),
                dtype=torch.float32,
                device=device,
            )
            output = model(
                observation,
                status,
                proposals,
                base_factor_logits,
                base_scores,
                observation_valid_mask=observation_mask,
            )
            cached_scores = output["selection_scores"]
    cached_scores_np = cached_scores.float().cpu().numpy()

    proposal_max_abs = 0.0
    score_max_abs = 0.0
    online_selected: List[int] = []
    for row_index, token in enumerate(tokens):
        proposal_max_abs = max(
            proposal_max_abs,
            _maximum_abs(
                np.asarray(online[token]["proposals"]),
                np.asarray(public[token]["proposals"]),
            ),
        )
        online_scores = np.asarray(online[token]["predicted_scores"])
        score_max_abs = max(
            score_max_abs,
            _maximum_abs(online_scores, cached_scores_np[row_index]),
        )
        online_selected.append(int(online_scores.argmax()))
    cached_selected = cached_scores_np.argmax(axis=1)
    selected_match = bool(
        np.array_equal(np.asarray(online_selected), cached_selected)
    )
    passed = bool(
        proposal_max_abs <= args.atol
        and score_max_abs <= args.atol
        and selected_match
    )
    result = {
        "artifact": str(args.artifact.resolve()),
        "agent_class": (
            "local_stage2.m0_native_private_scorer_agent."
            "M0NativePrivateScorerAgent"
        ),
        "architecture": architecture,
        "online_predictions": str(args.online_predictions.resolve()),
        "public_feature_cache": str(args.public_feature_cache.resolve()),
        "private_observation_root": str(
            args.private_observation_root.resolve()
        ),
        "scene_count": len(tokens),
        "tokens": tokens,
        "proposal_max_abs": proposal_max_abs,
        "score_max_abs": score_max_abs,
        "selected_indices_match": selected_match,
        "atol": args.atol,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
