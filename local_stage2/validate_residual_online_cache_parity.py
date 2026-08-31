"""Compare a residual scorer's real online agent output with cached inference."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from local_stage2.evaluate_cached_navtest_scorers import (
    _load_feature_cache,
    _score_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.online_predictions.open("rb") as file:
        online = pickle.load(file)
    cache = _load_feature_cache(args.feature_cache)
    tokens = sorted(online)
    if not tokens or not set(tokens).issubset(cache):
        raise RuntimeError("Online tokens are empty or absent from the feature cache")
    device = torch.device(args.device)
    selected, cached_scores, _payload = _score_artifact(
        args.artifact,
        cache,
        tokens,
        device=device,
        batch_size=len(tokens),
    )
    proposal_max_abs = 0.0
    score_max_abs = 0.0
    online_selected = []
    for index, token in enumerate(tokens):
        online_proposals = np.asarray(online[token]["proposals"], dtype=np.float32)
        cached_proposals = np.asarray(cache[token]["proposals"], dtype=np.float32)
        online_scores = np.asarray(online[token]["predicted_scores"], dtype=np.float32)
        proposal_max_abs = max(
            proposal_max_abs,
            float(np.max(np.abs(online_proposals.astype(np.float64) - cached_proposals))),
        )
        score_max_abs = max(
            score_max_abs,
            float(np.max(np.abs(online_scores.astype(np.float64) - cached_scores[index]))),
        )
        online_selected.append(int(np.argmax(online_scores)))
    selected_match = bool(np.array_equal(selected, np.asarray(online_selected)))
    passed = bool(
        proposal_max_abs <= args.atol and score_max_abs <= args.atol and selected_match
    )
    result = {
        "artifact": str(args.artifact.resolve()),
        "online_predictions": str(args.online_predictions.resolve()),
        "feature_cache": str(args.feature_cache.resolve()),
        "scene_count": len(tokens),
        "proposal_max_abs": proposal_max_abs,
        "score_max_abs": score_max_abs,
        "selected_indices_match": selected_match,
        "atol": args.atol,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
