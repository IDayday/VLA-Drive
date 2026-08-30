#!/usr/bin/env python3
"""Compare two paired Stage-2 proposal exports with log-level bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion


def _load(path: Path) -> dict:
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "tokens",
        "logs",
        "selected_index",
        "selected_trajectory",
        "proposals",
        "selected_factors",
        "best_pdm_score",
        "selected_l2_2s",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    return payload


def _grouped_bootstrap_ci(
    differences: np.ndarray,
    logs: list[str],
    *,
    seed: int,
    samples: int,
) -> list[float]:
    if differences.ndim != 1 or len(differences) != len(logs):
        raise ValueError("differences and logs must be paired one-dimensional data")
    grouped: dict[str, list[float]] = {}
    for log_name, difference in zip(logs, differences):
        grouped.setdefault(log_name, []).append(float(difference))
    group_means = np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=np.float64
    )
    if not len(group_means):
        raise ValueError("cannot bootstrap an empty artifact")
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    # Chunk the index matrix so this remains bounded for full validation sets.
    for start in range(0, samples, 4096):
        stop = min(samples, start + 4096)
        indices = generator.integers(
            0, len(group_means), size=(stop - start, len(group_means))
        )
        bootstrap_means[start:stop] = group_means[indices].mean(axis=1)
    return [
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    ]


def _paired_metric(
    left: torch.Tensor,
    right: torch.Tensor,
    logs: list[str],
    *,
    seed: int,
    samples: int,
) -> dict[str, float | list[float]]:
    left = torch.as_tensor(left).float().reshape(-1)
    right = torch.as_tensor(right).float().reshape(-1)
    if left.shape != right.shape or left.numel() != len(logs):
        raise ValueError("paired metric shape mismatch")
    difference = (right - left).numpy()
    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "right_minus_left": float(np.mean(difference)),
        "log_bootstrap_95_ci": _grouped_bootstrap_ci(
            difference, logs, seed=seed, samples=samples
        ),
        "right_better_fraction": float(np.mean(difference > 0)),
        "equal_fraction": float(np.mean(difference == 0)),
    }


def compare(
    left_path: Path,
    right_path: Path,
    *,
    seed: int = 20260830,
    bootstrap_samples: int = 20_000,
) -> dict:
    left = _load(left_path)
    right = _load(right_path)
    if left["tokens"] != right["tokens"] or left["logs"] != right["logs"]:
        raise ValueError("artifacts do not contain the same ordered scenes")
    logs = [str(value) for value in left["logs"]]
    factor_names = tuple(left.get("factor_names", ()))
    if factor_names != tuple(right.get("factor_names", ())):
        raise ValueError("factor-name mismatch")

    left_factors = torch.as_tensor(left["selected_factors"]).float()
    right_factors = torch.as_tensor(right["selected_factors"]).float()
    metrics = {
        "selected_pdm": _paired_metric(
            left_factors[:, -1], right_factors[:, -1], logs,
            seed=seed, samples=bootstrap_samples,
        ),
        "best_of_64_pdm": _paired_metric(
            left["best_pdm_score"], right["best_pdm_score"], logs,
            seed=seed + 1, samples=bootstrap_samples,
        ),
        "selection_regret": _paired_metric(
            torch.as_tensor(left["best_pdm_score"]) - left_factors[:, -1],
            torch.as_tensor(right["best_pdm_score"]) - right_factors[:, -1],
            logs, seed=seed + 2, samples=bootstrap_samples,
        ),
    }
    left_l2 = torch.as_tensor(left["selected_l2_2s"]).float().reshape(-1)
    right_l2 = torch.as_tensor(right["selected_l2_2s"]).float().reshape(-1)
    if left_l2.shape != right_l2.shape:
        raise ValueError("selected L2 shape mismatch")
    if left_l2.numel() == len(logs):
        metrics["selected_l2_2s"] = _paired_metric(
            left_l2, right_l2, logs,
            seed=seed + 3, samples=bootstrap_samples,
        )
    else:
        # The runtime exporter records the evaluator's batch-mean L2 scalar.
        # Keep the paired aggregate, but do not mislabel batches as logs.
        metrics["selected_l2_2s_batch_mean"] = {
            "left_mean": float(left_l2.mean()),
            "right_mean": float(right_l2.mean()),
            "right_minus_left": float((right_l2 - left_l2).mean()),
            "pair_count": left_l2.numel(),
            "pair_unit": "evaluation_batch",
        }
    for index, name in enumerate(factor_names):
        metrics[f"selected_factor/{name}"] = _paired_metric(
            left_factors[:, index], right_factors[:, index], logs,
            seed=seed + 10 + index, samples=bootstrap_samples,
        )

    proposal_difference = (
        torch.as_tensor(right["proposals"]).float()
        - torch.as_tensor(left["proposals"]).float()
    )
    selected_difference = (
        torch.as_tensor(right["selected_trajectory"]).float()
        - torch.as_tensor(left["selected_trajectory"]).float()
    )
    return {
        "left_artifact": str(left_path.resolve()),
        "right_artifact": str(right_path.resolve()),
        "sample_count": len(logs),
        "log_count": len(set(logs)),
        "ordered_scene_identity": True,
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "metrics": metrics,
        "proposal_coordinate_rmse": float(
            proposal_difference.square().mean().sqrt()
        ),
        "selected_trajectory_coordinate_rmse": float(
            selected_difference.square().mean().sqrt()
        ),
        "selected_index_changed_fraction": float(
            (
                torch.as_tensor(left["selected_index"])
                != torch.as_tensor(right["selected_index"])
            ).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    report = compare(
        args.left,
        args.right,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    report["left_name"] = args.left_name
    report["right_name"] = args.right_name
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
