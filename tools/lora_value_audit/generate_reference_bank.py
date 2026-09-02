#!/usr/bin/env python3
"""Create the explicitly non-deployable metric-cache PDM reference bank."""

from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array

from .utils import atomic_json, sha256_file


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(), end="")
        return
    with np.load(args.base_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        logs = archive["log_names"].astype(str)
    loader = MetricCacheLoader(args.metric_cache)
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    payload = {}
    failures = []
    for index, (token, log_name) in enumerate(zip(tokens, logs)):
        try:
            with lzma.open(loader.metric_cache_paths[token], "rb") as stream:
                cache = pickle.load(stream)
            states = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
            absolute = [StateSE2(*state[:3]) for state in states]
            relative = absolute_to_relative_poses(absolute)[1:]
            poses = np.asarray([pose.serialize() for pose in relative], dtype=np.float32)
            if poses.shape != (8, 3) or not np.isfinite(poses).all():
                raise RuntimeError(f"invalid reference shape {poses.shape}")
            payload[token] = {"log_name": log_name, "proposals": poses[None]}
        except Exception as error:
            failures.append({"token": token, "error": repr(error)})
        if (index + 1) % 1000 == 0:
            print(json.dumps({"processed": index + 1, "total": len(tokens)}), flush=True)
    if failures:
        raise RuntimeError(f"Reference bank failed for {len(failures)} tokens; first={failures[0]}")
    output = args.output_dir / "reference_upper_bound1.pkl"
    _atomic_pickle(output, payload)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bank": str(output.resolve()),
        "bank_sha256": sha256_file(output),
        "scene_count": len(payload),
        "candidate_count": 1,
        "source": "metric_cache PDM reference trajectory sampled at 8 poses / 0.5 seconds",
        "deployable": False,
        "future_or_evaluator_information": True,
        "allowed_use": "diagnostic learnable-target upper bound only",
        "base_matrix_sha256": sha256_file(args.base_matrix),
        "metric_cache": str(args.metric_cache.resolve()),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
