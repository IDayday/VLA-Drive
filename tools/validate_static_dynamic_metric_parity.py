#!/usr/bin/env python3
"""Validate NAVSIM label parity before Register64 hybrid DriveSuprim training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.candidate_bank import CandidateBankReader
from starVLA.model.modules.register_planner.checkpoint import sha256_file
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS
from starVLA.model.modules.trajectory_scorer.static_score_store import (
    StaticVocabScoreStore,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.navsim_metric_supervisor import DynamicMetricSupervisor
from starVLA.training.register_stage_utils import atomic_json


DISCRETE_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
    "traffic_light_compliance",
)
CONTINUOUS_METRICS = tuple(
    name for name in SUPRIM_METRICS if name not in DISCRETE_METRICS
)


def _load_vocab(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        value = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = next(
                loaded[name]
                for name in ("vocab", "trajectory_vocab", "static_vocab")
                if name in loaded
            )
        value = torch.as_tensor(loaded)
    if tuple(value.shape) != (8192, 40, 3):
        raise ValueError("hybrid parity tool requires static vocabulary [8192,40,3]")
    if not torch.isfinite(value).all():
        raise ValueError("static vocabulary contains NaN or Inf")
    return value.float()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    if str(config.stage_mode) != "hybrid":
        raise ValueError("parity validation requires the hybrid config")
    gate = config.parity_gate
    vocabulary_path = Path(str(config.model.static_vocab_path))
    vocabulary_sha = sha256_file(vocabulary_path)
    cache_vocabulary_sha = str(gate.static_score_cache_vocabulary_sha256)
    vocabulary_order_consistent = vocabulary_sha == cache_vocabulary_sha
    vocabulary = _load_vocab(vocabulary_path)
    reader = CandidateBankReader(str(config.candidate_bank.train_root))
    rng = np.random.default_rng(int(gate.get("seed", 42)))
    scene_count = int(gate.get("num_scenes", 100))
    candidate_count = int(gate.get("candidates_per_scene", 128))
    if len(reader) < scene_count:
        raise RuntimeError(
            f"parity gate requested {scene_count} scenes but bank has {len(reader)}"
        )
    token_indices = rng.choice(len(reader), size=scene_count, replace=False)
    tokens = [reader.manifest.records[int(index)].token for index in token_indices]
    reader.close()
    sampled_indices = np.stack(
        [rng.choice(8192, size=candidate_count, replace=False) for _ in tokens]
    )
    store = StaticVocabScoreStore(
        str(config.static_score_store.cache_root),
        split="train",
        vocab_size=8192,
        cache_size=int(config.static_score_store.get("cache_size", 64)),
        include_aggregate_score=True,
    )
    supervisor = DynamicMetricSupervisor(
        {
            "metric_cache_root": str(gate.metric_cache_root),
            "backend": str(gate.get("backend", "process")),
            "workers_per_rank": int(gate.get("workers_per_rank", 8)),
            "score_interval": 1,
        }
    )
    actual = {name: [] for name in (*SUPRIM_METRICS, "aggregate_score")}
    expected = {name: [] for name in (*SUPRIM_METRICS, "aggregate_score")}
    batch_size = int(gate.get("batch_size", 4))
    try:
        for start in range(0, scene_count, batch_size):
            batch_tokens = tokens[start : start + batch_size]
            batch_indices = sampled_indices[start : start + batch_size]
            trajectories_40 = torch.stack(
                [vocabulary[torch.as_tensor(indices)] for indices in batch_indices]
            )
            rescored = supervisor.score_40(batch_tokens, trajectories_40)
            cached = store.get(
                batch_tokens, device=torch.device("cpu"), dtype=torch.float32
            )
            if "aggregate_score" not in cached:
                raise RuntimeError(
                    "official static cache must include pdm_score/aggregate_score"
                )
            for row, indices in enumerate(batch_indices):
                index = torch.as_tensor(indices, dtype=torch.long)
                for name in (*SUPRIM_METRICS, "aggregate_score"):
                    actual[name].append(rescored[name][row].float().cpu().numpy())
                    expected[name].append(cached[name][row, index].cpu().numpy())
    finally:
        supervisor.close()
    actual = {name: np.concatenate(rows) for name, rows in actual.items()}
    expected = {name: np.concatenate(rows) for name, rows in expected.items()}
    discrete = {
        name: float(np.isclose(actual[name], expected[name], atol=1e-6).mean())
        for name in DISCRETE_METRICS
    }
    continuous = {
        name: float(np.abs(actual[name] - expected[name]).mean())
        for name in CONTINUOUS_METRICS
    }
    aggregate_mae = float(
        np.abs(actual["aggregate_score"] - expected["aggregate_score"]).mean()
    )
    correlations = []
    for row in range(scene_count):
        start, end = row * candidate_count, (row + 1) * candidate_count
        correlations.append(
            _spearman(
                actual["aggregate_score"][start:end],
                expected["aggregate_score"][start:end],
            )
        )
    mean_spearman = float(np.mean(correlations))
    passed = (
        vocabulary_order_consistent
        and min(discrete.values()) >= float(gate.discrete_agreement_min)
        and max(continuous.values()) <= float(gate.continuous_metric_mae_max)
        and aggregate_mae <= float(gate.aggregate_score_mae_max)
        and mean_spearman >= float(gate.spearman_min)
    )
    report = {
        "passed": passed,
        "num_scenes": scene_count,
        "candidates_per_scene": candidate_count,
        "static_vocabulary_sha256": vocabulary_sha,
        "static_score_cache_vocabulary_sha256": cache_vocabulary_sha,
        "trajectory_order_consistent": vocabulary_order_consistent,
        "discrete_metric_agreement": discrete,
        "continuous_metric_mae": continuous,
        "aggregate_score_mae": aggregate_mae,
        "spearman_ranking_correlation": mean_spearman,
        "thresholds": {
            "discrete_agreement_min": float(gate.discrete_agreement_min),
            "continuous_metric_mae_max": float(gate.continuous_metric_mae_max),
            "aggregate_score_mae_max": float(gate.aggregate_score_mae_max),
            "spearman_min": float(gate.spearman_min),
        },
    }
    atomic_json(gate.report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("hybrid static/dynamic metric parity gate failed")


if __name__ == "__main__":
    main()
