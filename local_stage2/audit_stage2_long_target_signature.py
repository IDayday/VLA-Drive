#!/usr/bin/env python3
"""Measure whether a proposal bank covers both four- and five-second targets."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.torch_version import TorchVersion

from local_stage2.build_stage2_long_target_cache import (
    _relative_future,
    build_long_trajectory,
)


DEFAULT_PROPOSALS = Path(
    "/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/numerics/"
    "public_tf4483_subset128.pt"
)
DEFAULT_LONG_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_long2"
)
DEFAULT_RAW_LOGS = Path("/mnt/navsim/trainval_navsim_logs/trainval")


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
    }


def _raw_long_target_variant(
    payload: dict, raw_logs: Path, additional_poses: int
) -> torch.Tensor:
    targets = []
    for log_name, token in zip(payload["logs"], payload["tokens"]):
        with (raw_logs / f"{log_name}.pkl").open("rb") as stream:
            frames = pickle.load(stream)
        token_to_index = {
            frame["token"]: index for index, frame in enumerate(frames)
        }
        relative_future = _relative_future(
            frames, token_to_index[token], 8 + additional_poses
        )
        targets.append(
            torch.as_tensor(
                build_long_trajectory(
                    relative_future,
                    num_poses=8,
                    additional_poses=additional_poses,
                )
            ).float()
        )
    return torch.stack(targets)


def _target_fit_summary(
    proposals: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    loss = (proposals - target[:, None]).abs().sum(-1).mean(-1).min(dim=1).values
    endpoint = torch.linalg.vector_norm(
        proposals[:, :, -1, :2] - target[:, None, -1, :2], dim=-1
    ).min(dim=1).values
    return {
        "mean_min_official_l1": float(loss.mean()),
        "min_official_l1_lt_1_fraction": float((loss < 1).float().mean()),
        "mean_min_endpoint_error_m": float(endpoint.mean()),
    }


def audit(
    proposal_path: Path,
    long_cache: Path,
    raw_logs: Path | None = None,
    alternate_additional_poses: tuple[int, ...] = (),
) -> dict:
    # Runtime-audit artifacts are created locally by
    # ``audit_stage2_public_runtime.py``.  Torch 2.5 represents
    # ``torch.__version__`` as this harmless string subclass, which is not in
    # the default weights-only allowlist.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(proposal_path, map_location="cpu", weights_only=True)
    required = {"tokens", "logs", "proposals"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"proposal artifact missing keys: {sorted(missing)}")

    standard_targets = []
    long_targets = []
    for log_name, token in zip(payload["logs"], payload["tokens"]):
        target_path = long_cache / log_name / token / "trajectory_target.gz"
        with gzip.open(target_path, "rb") as stream:
            target = pickle.load(stream)
        standard_targets.append(torch.as_tensor(target["trajectory"]).float())
        long_targets.append(torch.as_tensor(target["trajectory_long"]).float())

    standard = torch.stack(standard_targets)
    long = torch.stack(long_targets)
    proposals = torch.as_tensor(payload["proposals"]).float()
    expected_shape = (len(standard), 64, 8, 3)
    if tuple(proposals.shape) != expected_shape:
        raise ValueError(f"expected proposal shape {expected_shape}, got {proposals.shape}")

    # This is the exact min-over-proposals L1 convention in the released loss.
    standard_loss = (proposals - standard[:, None]).abs().sum(-1).mean(-1)
    long_loss = (proposals - long[:, None]).abs().sum(-1).mean(-1)
    standard_min, standard_index = standard_loss.min(dim=1)
    long_min, long_index = long_loss.min(dim=1)
    standard_endpoint = torch.linalg.vector_norm(
        proposals[:, :, -1, :2] - standard[:, None, -1, :2], dim=-1
    ).min(dim=1).values
    long_endpoint = torch.linalg.vector_norm(
        proposals[:, :, -1, :2] - long[:, None, -1, :2], dim=-1
    ).min(dim=1).values
    target_endpoint_separation = torch.linalg.vector_norm(
        standard[:, -1, :2] - long[:, -1, :2], dim=-1
    )
    target_l1_separation = (standard - long).abs().sum(-1).mean(-1)
    proposal_endpoints = proposals[:, :, -1, :2]
    endpoint_radius = torch.linalg.vector_norm(proposal_endpoints, dim=-1)
    endpoint_pairwise = torch.linalg.vector_norm(
        proposal_endpoints[:, :, None] - proposal_endpoints[:, None, :], dim=-1
    )
    off_diagonal = ~torch.eye(
        proposals.shape[1], dtype=torch.bool, device=proposals.device
    )
    endpoint_pairwise_per_scene = endpoint_pairwise[:, off_diagonal].mean(dim=1)

    report = {
        "proposal_artifact": str(proposal_path.resolve()),
        "long_target_cache": str(long_cache.resolve()),
        "checkpoint": payload.get("summary", {}).get("checkpoint"),
        "sample_count": len(standard),
        "log_count": len(set(payload["logs"])),
        "mean_min_official_l1_standard": float(standard_min.mean()),
        "mean_min_official_l1_long": float(long_min.mean()),
        "min_official_l1_standard_quantiles": _quantiles(standard_min),
        "min_official_l1_long_quantiles": _quantiles(long_min),
        "mean_target_l1_separation": float(target_l1_separation.mean()),
        "mean_target_endpoint_separation_m": float(
            target_endpoint_separation.mean()
        ),
        "mean_min_endpoint_error_standard_m": float(standard_endpoint.mean()),
        "mean_min_endpoint_error_long_m": float(long_endpoint.mean()),
        "mean_max_proposal_endpoint_radius_m": float(
            endpoint_radius.max(dim=1).values.mean()
        ),
        "max_proposal_endpoint_radius_quantiles_m": _quantiles(
            endpoint_radius.max(dim=1).values
        ),
        "mean_proposal_endpoint_pairwise_distance_m": float(
            endpoint_pairwise_per_scene.mean()
        ),
        "proposal_endpoint_pairwise_distance_quantiles_m": _quantiles(
            endpoint_pairwise_per_scene
        ),
        "mean_proposal_endpoint_x_span_m": float(
            (proposal_endpoints[:, :, 0].max(dim=1).values
             - proposal_endpoints[:, :, 0].min(dim=1).values).mean()
        ),
        "mean_proposal_endpoint_y_span_m": float(
            (proposal_endpoints[:, :, 1].max(dim=1).values
             - proposal_endpoints[:, :, 1].min(dim=1).values).mean()
        ),
        "same_nearest_candidate_fraction": float(
            (standard_index == long_index).float().mean()
        ),
        "standard_min_l1_lt_0_5_fraction": float(
            (standard_min < 0.5).float().mean()
        ),
        "standard_min_l1_lt_1_fraction": float(
            (standard_min < 1.0).float().mean()
        ),
        "long_min_l1_lt_0_5_fraction": float((long_min < 0.5).float().mean()),
        "long_min_l1_lt_1_fraction": float((long_min < 1.0).float().mean()),
        "nearest_candidate_pair_sha256": __import__("hashlib").sha256(
            np.stack([standard_index.numpy(), long_index.numpy()], axis=1).tobytes()
        ).hexdigest(),
    }
    if alternate_additional_poses:
        if raw_logs is None:
            raise ValueError("raw_logs is required for alternate long targets")
        report["alternate_long_target_fit"] = {}
        for additional_poses in alternate_additional_poses:
            target = _raw_long_target_variant(
                payload, raw_logs, additional_poses
            )
            report["alternate_long_target_fit"][str(additional_poses)] = {
                "horizon_seconds": (8 + additional_poses) * 0.5,
                **_target_fit_summary(proposals, target),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--long-cache", type=Path, default=DEFAULT_LONG_CACHE)
    parser.add_argument("--raw-logs", type=Path, default=DEFAULT_RAW_LOGS)
    parser.add_argument(
        "--alternate-additional-poses", type=int, nargs="*", default=[]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(
        args.proposals,
        args.long_cache,
        args.raw_logs,
        tuple(args.alternate_additional_poses),
    )
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
