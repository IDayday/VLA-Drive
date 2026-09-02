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
    raw_scores, score_key = _first_array(
        payload,
        ("pdm_scores", "candidate_pdm", "candidate_scores", "scores"),
    )
    if raw_scores is None:
        raise KeyError(
            f"No candidate PDM array in {path}; available keys: {sorted(payload.files)}"
        )
    component_scores = None
    if raw_scores.ndim == 3:
        if raw_scores.shape[-1] < 1:
            raise ValueError("Empty candidate factor dimension")
        component_scores = raw_scores
        scores = raw_scores[..., -1]
    else:
        scores = raw_scores
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
        "oracle_at_64_pdms": float(oracle.mean()),
        "scorer_regret": float((oracle - selected).mean()),
        "candidate_mean_pdms": float(scores.mean()),
        "candidate_p10_pdms": float(np.percentile(scores, 10)),
        "candidate_p25_pdms": float(np.percentile(scores, 25)),
        "candidate_median_pdms": float(np.median(scores)),
        "fraction_candidates_above_0p8": float((scores > 0.8).mean()),
        "fraction_candidates_above_0p9": float((scores > 0.9).mean()),
        "top5_oracle_mean": float(
            np.sort(scores, axis=1)[:, -min(5, scores.shape[1]):].mean()
        ),
    }
    if component_scores is not None and component_scores.shape[-1] >= 7:
        component_names = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
            "pdm_score",
        )
        selected_components = component_scores[
            np.arange(component_scores.shape[0]), selected_indices
        ]
        report["selected_components"] = {
            name: float(selected_components[:, index].mean())
            for index, name in enumerate(component_names)
        }
        report["candidate_components"] = {
            name: float(component_scores[..., index].mean())
            for index, name in enumerate(component_names)
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
        report["candidate_duplicate_rate"] = float(
            1.0 - report["mean_unique_candidates"] / scores.shape[1]
        )
        upper = np.triu_indices(scores.shape[1], k=1)
        # Greedy connected components under trajectory RMS distance. This is a
        # fixed diagnostic, not a training target or candidate-ranking loss.
        # Compute one scene at a time: materializing [S,K,K,24] for full
        # Navtest would consume several gigabytes.
        pairwise_sum = 0.0
        pairwise_count = 0
        effective_clusters = []
        for scene in flattened:
            scene_distances = np.linalg.norm(
                scene[:, None] - scene[None, :], axis=-1
            )
            upper_values = scene_distances[upper]
            pairwise_sum += float(upper_values.sum())
            pairwise_count += int(upper_values.size)
            scene_distances = scene_distances / np.sqrt(flattened.shape[-1])
            remaining = set(range(scores.shape[1]))
            count = 0
            while remaining:
                seed = min(remaining)
                connected = {seed}
                frontier = [seed]
                while frontier:
                    current = frontier.pop()
                    neighbors = {
                        index
                        for index in remaining
                        if scene_distances[current, index] <= 0.5
                    }
                    new = neighbors - connected
                    connected.update(new)
                    frontier.extend(new)
                remaining.difference_update(connected)
                count += 1
            effective_clusters.append(count)
        report["mean_pairwise_geometry_l2"] = pairwise_sum / max(
            1, pairwise_count
        )
        report["effective_clusters_rms_0p5"] = float(
            np.mean(effective_clusters)
        )

    if "planning_registers" in payload:
        registers = np.asarray(payload["planning_registers"], dtype=np.float64)
        centered = registers - registers.mean(axis=1, keepdims=True)
        covariance = centered @ np.swapaxes(centered, -1, -2)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        probabilities = eigenvalues / np.maximum(
            eigenvalues.sum(axis=-1, keepdims=True), 1e-12
        )
        effective_rank = np.exp(
            -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=-1)
        )
        report["register_effective_rank"] = float(effective_rank.mean())
    if "tile_gate" in payload:
        report["tile_gate_mean"] = float(np.asarray(payload["tile_gate"]).mean())
    if "semantic_gate" in payload:
        report["semantic_gate_probability_mean"] = float(
            np.asarray(payload["semantic_gate"]).mean()
        )
    if "inference_latency_seconds" in payload:
        latency = np.asarray(payload["inference_latency_seconds"], dtype=np.float64)
        report["inference_latency_median_seconds"] = float(np.median(latency))
        report["inference_latency_p90_seconds"] = float(np.percentile(latency, 90))
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
