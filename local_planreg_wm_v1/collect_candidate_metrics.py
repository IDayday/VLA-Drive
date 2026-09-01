#!/usr/bin/env python3
"""Summarize an already-scored immutable candidate bank without rescoring it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_array(payload, names):
    for name in names:
        if name in payload:
            return np.asarray(payload[name]), name
    return None, None


def collect(path: Path) -> dict:
    payload = np.load(path, allow_pickle=False)
    scores, score_key = _first_array(
        payload,
        ("pdm_scores", "candidate_pdm", "candidate_scores", "scores"),
    )
    if scores is None:
        raise KeyError(
            f"No candidate PDM array in {path}; available keys: {sorted(payload.files)}"
        )
    if scores.ndim == 3:
        if scores.shape[-1] < 1:
            raise ValueError("Empty candidate factor dimension")
        scores = scores[..., -1]
    if scores.ndim != 2:
        raise ValueError(f"Candidate scores must be [scenes,K], got {scores.shape}")

    selected_indices, selected_key = _first_array(
        payload,
        ("selected_indices", "selected_index", "predicted_indices"),
    )
    predicted_scores, predicted_key = _first_array(
        payload,
        ("predicted_scores", "scorer_values", "pred_pdm"),
    )
    if selected_indices is None:
        if predicted_scores is None:
            raise KeyError(
                "Candidate cache needs selected_indices or predicted scorer values"
            )
        selected_indices = np.asarray(predicted_scores).argmax(axis=1)
        selected_key = f"argmax({predicted_key})"
    selected_indices = np.asarray(selected_indices).reshape(-1).astype(np.int64)
    if selected_indices.shape[0] != scores.shape[0]:
        raise ValueError("Selected-index count does not match scene count")
    if np.any(selected_indices < 0) or np.any(selected_indices >= scores.shape[1]):
        raise ValueError("Selected index is outside candidate bank")

    selected = scores[np.arange(scores.shape[0]), selected_indices]
    oracle = scores.max(axis=1)
    report = {
        "source": str(path.resolve()),
        "source_sha256": _sha256(path),
        "score_key": score_key,
        "selected_index_key": selected_key,
        "scene_count": int(scores.shape[0]),
        "candidate_count": int(scores.shape[1]),
        "selected_pdms": float(selected.mean()),
        "best_of_k_pdms": float(oracle.mean()),
        "scorer_regret": float((oracle - selected).mean()),
        "candidate_mean_pdms": float(scores.mean()),
        "candidate_median_pdms": float(np.median(scores)),
        "fraction_candidates_above_0p8": float((scores > 0.8).mean()),
        "fraction_candidates_above_0p9": float((scores > 0.9).mean()),
        "top5_oracle_mean": float(
            np.sort(scores, axis=1)[:, -min(5, scores.shape[1]):].mean()
        ),
    }
    trajectories, trajectory_key = _first_array(
        payload,
        ("trajectories", "proposals", "candidate_trajectories"),
    )
    if trajectories is not None and trajectories.shape[:2] == scores.shape:
        flattened = trajectories.reshape(*scores.shape, -1)
        rounded = np.round(flattened, decimals=4)
        report["trajectory_key"] = trajectory_key
        report["mean_unique_candidates"] = float(
            np.mean([len(np.unique(scene, axis=0)) for scene in rounded])
        )
        pairwise = np.linalg.norm(
            flattened[:, :, None] - flattened[:, None, :], axis=-1
        )
        upper = np.triu_indices(scores.shape[1], k=1)
        report["mean_pairwise_geometry_l2"] = float(pairwise[:, upper[0], upper[1]].mean())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.candidate_npz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
