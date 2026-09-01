#!/usr/bin/env python3
"""Validate online independent-shortlist inference against locked FP32 cache."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from local_stage2.independent_scorer_agent import IndependentShortlistScorerAgent
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


def _maximum_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-predictions", type=Path, required=True)
    parser.add_argument("--public-feature-cache", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    with args.online_predictions.open("rb") as file:
        online = pickle.load(file)
    with args.public_feature_cache.open("rb") as file:
        cached = pickle.load(file)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != IndependentShortlistScorerAgent.ARTIFACT_TYPE:
        raise RuntimeError("deployment artifact type mismatch")
    tokens = sorted(online)
    if len(tokens) < 4 or not set(tokens).issubset(cached):
        raise RuntimeError("at least four online tokens must exist in the locked cache")

    device = torch.device(args.device)
    model = IndependentProposalRanker(
        IndependentRankerConfig(**dict(artifact["model_config"]))
    ).to(device)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.eval()
    proposals = torch.as_tensor(
        np.stack([cached[token]["proposals"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    scene = torch.as_tensor(
        np.stack([cached[token]["scene_features"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    ego = torch.as_tensor(
        np.stack([cached[token]["ego_features"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    if ego.ndim == 3 and ego.shape[1] == 1:
        ego = ego[:, 0]
    base_scores = torch.as_tensor(
        np.stack([cached[token]["predicted_scores"] for token in tokens]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        output = model(scene, ego, proposals)
        mode = str(artifact["score_mode"])
        if mode == "coarse":
            utility = output["coarse_utility"]
        elif mode == "factor":
            utility = pdms_factor_log_utility(output["factor_logits"])
        elif mode == "direct":
            utility = output["utility"]
        else:
            raise RuntimeError(f"unsupported score mode: {mode}")
        shortlist = base_scores.topk(int(artifact["shortlist_size"]), dim=1).indices
        cached_selection_scores = torch.full_like(utility, -1.0e4)
        cached_selection_scores.scatter_(1, shortlist, utility.gather(1, shortlist))
    cached_scores = cached_selection_scores.cpu().numpy()

    maxima = {
        "proposals": 0.0,
        "scene_features": 0.0,
        "ego_features": 0.0,
        "selection_scores": 0.0,
    }
    online_selected = []
    cached_selected = cached_scores.argmax(axis=1)
    for row, token in enumerate(tokens):
        maxima["proposals"] = max(
            maxima["proposals"],
            _maximum_abs(
                np.asarray(online[token]["proposals"]),
                np.asarray(cached[token]["proposals"]),
            ),
        )
        maxima["scene_features"] = max(
            maxima["scene_features"],
            _maximum_abs(
                np.asarray(online[token]["scene_features"]),
                np.asarray(cached[token]["scene_features"]),
            ),
        )
        maxima["ego_features"] = max(
            maxima["ego_features"],
            _maximum_abs(
                np.asarray(online[token]["ego_features"]),
                np.asarray(cached[token]["ego_features"]),
            ),
        )
        online_scores = np.asarray(online[token]["predicted_scores"])
        maxima["selection_scores"] = max(
            maxima["selection_scores"],
            _maximum_abs(online_scores, cached_scores[row]),
        )
        online_selected.append(int(online_scores.argmax()))
    selected_match = bool(
        np.array_equal(np.asarray(online_selected), cached_selected)
    )
    passed = bool(max(maxima.values()) <= args.atol and selected_match)
    result = {
        "scene_count": len(tokens),
        "tokens": tokens,
        "artifact": str(args.artifact.resolve()),
        "online_predictions": str(args.online_predictions.resolve()),
        "public_feature_cache": str(args.public_feature_cache.resolve()),
        "max_abs": maxima,
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
