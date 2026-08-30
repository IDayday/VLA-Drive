#!/usr/bin/env python3
"""Audit proposal-mode specialization induced by the Stage-2 long target."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_target(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def candidate_l1(
    proposals: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return released loss's per-candidate mean-pose L1(x,y,heading)."""

    if proposals.ndim != 4 or target.ndim != 3:
        raise ValueError("expected proposals [B,K,H,3] and target [B,H,3]")
    if (
        proposals.shape[0] != target.shape[0]
        or proposals.shape[2:] != target.shape[1:]
    ):
        raise ValueError("proposal and target batch/horizon dimensions differ")
    return torch.linalg.vector_norm(
        proposals.float() - target.float()[:, None], ord=1, dim=-1
    ).mean(-1)


def specialization_vectors(
    proposals: torch.Tensor,
    standard_target: torch.Tensor,
    long_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-scene short/long proposal-mode specialization metrics."""

    standard_loss = candidate_l1(proposals, standard_target)
    long_loss = candidate_l1(proposals, long_target)
    standard_index = standard_loss.argmin(1)
    long_index = long_loss.argmin(1)
    rows = torch.arange(proposals.shape[0])
    proposals = proposals.float()

    independent = (
        standard_loss[rows, standard_index] + long_loss[rows, long_index]
    )
    compromise = (standard_loss + long_loss).min(1).values
    standard_mode = proposals[rows, standard_index]
    long_mode = proposals[rows, long_index]
    standard_endpoint = torch.linalg.vector_norm(
        standard_mode[:, -1, :2], dim=-1
    )
    long_endpoint = torch.linalg.vector_norm(long_mode[:, -1, :2], dim=-1)

    return {
        "best_standard_l1": standard_loss[rows, standard_index],
        "best_long_l1": long_loss[rows, long_index],
        "independent_two_min_loss": independent,
        "single_candidate_compromise_loss": compromise,
        "specialization_advantage": compromise - independent,
        "distinct_argmin": (standard_index != long_index).float(),
        "mode_mean_position_distance_m": torch.linalg.vector_norm(
            standard_mode[..., :2] - long_mode[..., :2], dim=-1
        ).mean(-1),
        "mode_endpoint_distance_m": torch.linalg.vector_norm(
            standard_mode[:, -1, :2] - long_mode[:, -1, :2], dim=-1
        ),
        "standard_mode_endpoint_radius_m": standard_endpoint,
        "long_mode_endpoint_radius_m": long_endpoint,
        "long_minus_standard_endpoint_radius_m": long_endpoint
        - standard_endpoint,
    }


def _summarize(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().double().numpy()
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _cluster_bootstrap_delta(
    delta: np.ndarray,
    groups: list[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    group_array = np.asarray(groups, dtype=object)
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    group_indices = {
        group: np.flatnonzero(group_array == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_values = np.concatenate(
            [delta[group_indices[group]] for group in sampled]
        )
        bootstrap_means[index] = sampled_values.mean()
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "fraction_positive": float((delta > 0).mean()),
    }


def audit(
    standard_artifact_path: Path,
    long_artifact_path: Path,
    long_cache: Path,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    artifacts = {
        "standard": torch.load(
            standard_artifact_path, map_location="cpu", weights_only=False
        ),
        "long2": torch.load(
            long_artifact_path, map_location="cpu", weights_only=False
        ),
    }
    if artifacts["standard"]["tokens"] != artifacts["long2"]["tokens"]:
        raise ValueError("artifacts do not contain the same ordered scene tokens")
    if artifacts["standard"]["logs"] != artifacts["long2"]["logs"]:
        raise ValueError("artifacts do not contain the same ordered logs")

    tokens = artifacts["standard"]["tokens"]
    logs = artifacts["standard"]["logs"]
    standard_targets: list[torch.Tensor] = []
    long_targets: list[torch.Tensor] = []
    for log_name, token in zip(logs, tokens):
        target_path = long_cache / log_name / token / "trajectory_target.gz"
        target = _load_target(target_path)
        if target.get("token") != token or "trajectory_long" not in target:
            raise ValueError(f"invalid long-target cache entry: {target_path}")
        standard_targets.append(target["trajectory"].float())
        long_targets.append(target["trajectory_long"].float())
    standard_target = torch.stack(standard_targets)
    long_target = torch.stack(long_targets)

    vectors = {
        name: specialization_vectors(
            artifact["proposals"], standard_target, long_target
        )
        for name, artifact in artifacts.items()
    }
    paired_delta = {}
    for metric in vectors["standard"]:
        delta = (
            vectors["long2"][metric] - vectors["standard"][metric]
        ).detach().cpu().double().numpy()
        paired_delta[metric] = _cluster_bootstrap_delta(
            delta,
            logs,
            seed=seed,
            replicates=bootstrap_replicates,
        )

    displacement = torch.linalg.vector_norm(
        long_target[..., :2] - standard_target[..., :2], dim=-1
    )
    return {
        "audit": "matched_step1000_long_target_candidate_specialization",
        "semantics": {
            "distance": "mean over 8 poses of L1(x,y,heading), matching released loss",
            "independent_two_min_loss": "min_i L1(proposal_i,standard) + min_j L1(proposal_j,long)",
            "single_candidate_compromise_loss": "min_i [L1(proposal_i,standard) + L1(proposal_i,long)]",
            "paired_delta_direction": "long2-trained artifact minus standard-trained artifact",
            "bootstrap_unit": "log_name",
        },
        "inputs": {
            "standard_artifact": str(standard_artifact_path),
            "standard_artifact_sha256": _sha256(standard_artifact_path),
            "long2_artifact": str(long_artifact_path),
            "long2_artifact_sha256": _sha256(long_artifact_path),
            "long_target_cache": str(long_cache),
            "sample_count": len(tokens),
            "log_count": len(set(logs)),
            "ordered_tokens_sha256": hashlib.sha256(
                "\n".join(tokens).encode("utf-8")
            ).hexdigest(),
            "bootstrap_seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "target_separation": {
            "mean_position_displacement_m": float(displacement.mean()),
            "mean_endpoint_displacement_m": float(displacement[:, -1].mean()),
        },
        "models": {
            name: {
                metric: _summarize(value) for metric, value in result.items()
            }
            for name, result in vectors.items()
        },
        "paired_long2_minus_standard": paired_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-artifact", type=Path, required=True)
    parser.add_argument("--long2-artifact", type=Path, required=True)
    parser.add_argument("--long-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()

    result = audit(
        args.standard_artifact,
        args.long2_artifact,
        args.long_cache,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["paired_long2_minus_standard"], indent=2))


if __name__ == "__main__":
    main()
