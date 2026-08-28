#!/usr/bin/env python3
"""Hash and independently reproduce 32 scenes from the prior feasibility audit."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.build_candidate_relative_targets import build_scene_targets
from tools.navsim_candidate_relative_audit.candidate_generator import generate_candidates, resample_gt
from tools.navsim_candidate_relative_audit.common import load_scenes_for_tokens, metric_cache_loader
from tools.navsim_candidate_relative_audit.score_candidates import score_pose_batch

from .common import (
    BASE_COMMIT,
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    REPO_ROOT,
    append_command,
    assert_feature_names_safe,
    ensure_dir,
    git_output,
    hash_files,
    navsim_paths,
    sha256_file,
    update_gate,
    write_json,
    write_markdown,
)


OLD_REPORT = REPO_ROOT / "reports/navsim_candidate_relative_audit"
SCORE_FIELDS = (
    "no_at_fault_collision",
    "dac",
    "ddc",
    "progress",
    "raw_progress_m",
    "ttc",
    "comfort",
    "aggregate_score",
)


def _package(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "version": getattr(module, "__version__", None),
            "path": getattr(module, "__file__", None),
        }
    except Exception as exc:  # environment inventory must continue
        return {"error": f"{type(exc).__name__}: {exc}"}


def _balanced_old_tokens(metrics: pd.DataFrame, count: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    pools: dict[str, list[str]] = {}
    for log_name, group in metrics.groupby("log_name", sort=True):
        values = group.scene_token.drop_duplicates().to_numpy(dtype=object)
        rng.shuffle(values)
        pools[log_name] = values.tolist()
    logs = list(pools)
    rng.shuffle(logs)
    result: list[str] = []
    while len(result) < count:
        changed = False
        for log_name in logs:
            if pools[log_name] and len(result) < count:
                result.append(pools[log_name].pop())
                changed = True
        if not changed:
            break
    return result


def _poses(group: pd.DataFrame) -> np.ndarray:
    ordered = group.sort_values("candidate_index")
    return np.stack(
        [
            np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad])
            for row in ordered.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float64)


def _score_error(original: pd.DataFrame, reproduced: dict[str, Any]) -> dict[str, float]:
    ordered = original.sort_values("candidate_index")
    errors: dict[str, float] = {}
    mapping = {
        "no_at_fault_collision": "no_at_fault_collision",
        "dac": "dac",
        "ddc": "ddc",
        "progress": "progress",
        "raw_progress_m": "raw_progress_m",
        "ttc": "ttc",
        "comfort": "comfort",
        "aggregate_score": "score",
    }
    for old_name, new_name in mapping.items():
        left = ordered[old_name].to_numpy(dtype=np.float64)
        right = np.asarray(reproduced[new_name], dtype=np.float64)
        errors[old_name] = float(np.nanmax(np.abs(left - right)))
    old_states = np.stack(
        [
            np.column_stack(
                [
                    row.sim_x_m,
                    row.sim_y_m,
                    row.sim_heading_rad,
                    row.sim_velocity_x_mps,
                    row.sim_velocity_y_mps,
                    row.sim_acceleration_x_mps2,
                    row.sim_acceleration_y_mps2,
                    row.sim_steering_angle_rad,
                    row.sim_steering_rate_radps,
                    row.sim_angular_velocity_radps,
                    row.sim_angular_acceleration_radps2,
                ]
            )
            for row in ordered.itertuples(index=False)
        ],
        axis=0,
    )
    # The committed parquet intentionally stores simulated state arrays as
    # float32 lists. Compare at that persisted precision; global UTM y around
    # 4e6 has a float32 ULP of 0.25 m even though rerun scoring is float64.
    persisted = np.asarray(reproduced["simulated_states"], dtype=np.float32).astype(np.float64)
    errors["simulated_states"] = float(np.max(np.abs(old_states - persisted)))
    return errors


def _environment(args: argparse.Namespace, paths: Any) -> dict[str, Any]:
    import torch

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "project_path": str(REPO_ROOT),
        "git": {
            "branch": git_output(["branch", "--show-current"]),
            "head": git_output(["rev-parse", "HEAD"]),
            "base_commit_required": BASE_COMMIT,
            "base_is_ancestor": git_output(["merge-base", BASE_COMMIT, "HEAD"]) == BASE_COMMIT,
            "status_short": git_output(["status", "--short"]),
            "remotes": git_output(["remote", "-v"]),
        },
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": {name: _package(name) for name in ("navsim", "nuplan", "numpy", "pandas", "torch")},
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
        "paths": paths.to_json(),
        "split": args.split,
        "privacy_note": "Only allowlisted environment metadata is recorded; credentials are not inspected.",
    }


def reproduce(args: argparse.Namespace) -> dict[str, Any]:
    paths = navsim_paths(args.split)
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir / "reproduction")
    manifest_path = OLD_REPORT / "candidate_manifest.parquet"
    metrics_path = OLD_REPORT / "candidate_metrics.parquet"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Prior audit manifest/metrics are missing")
    manifest = pd.read_parquet(manifest_path)
    metrics = pd.read_parquet(metrics_path)
    tokens = _balanced_old_tokens(metrics, args.num_scenes, args.seed)
    manifest = manifest[manifest.scene_token.isin(tokens)].copy()
    metrics = metrics[metrics.scene_token.isin(tokens)].copy()
    loader = load_scenes_for_tokens(paths, tokens)
    caches = metric_cache_loader(paths)

    candidate_errors: list[float] = []
    score_errors: dict[str, list[float]] = {name: [] for name in (*SCORE_FIELDS, "simulated_states")}
    target_errors: dict[str, list[float]] = {}
    order_matches: list[bool] = []
    for token in tokens:
        scene = loader.get_scene_from_token(token)
        original_manifest = manifest[manifest.scene_token == token].sort_values("candidate_index")
        original_metrics = metrics[metrics.scene_token == token].sort_values("candidate_index")
        expected_poses = _poses(original_manifest)
        base = resample_gt(scene)
        seed = int(original_manifest.seed.iloc[0])
        regenerated, _ = generate_candidates(base, len(original_manifest), seed)
        candidate_errors.append(float(np.max(np.abs(expected_poses - regenerated))))
        order_matches.append(original_manifest.candidate_index.tolist() == list(range(len(original_manifest))))

        cache = caches.get_from_token(token)
        score = score_pose_batch(cache, expected_poses)
        for field, error in _score_error(original_metrics, score).items():
            score_errors[field].append(error)
        rebuilt = build_scene_targets(scene, cache, original_metrics, paths, max_actors=16, max_shared_actors=64)
        old_target_path = OLD_REPORT / "targets" / f"{token}.npz"
        with np.load(old_target_path, allow_pickle=False) as old:
            for field in old.files:
                if field not in rebuilt:
                    target_errors.setdefault(f"missing:{field}", []).append(float("inf"))
                    continue
                left = old[field]
                right = rebuilt[field]
                if left.dtype.kind in "biu":
                    error = 0.0 if np.array_equal(left, right) else 1.0
                else:
                    error = float(np.nanmax(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                target_errors.setdefault(field, []).append(error)

    code_inputs = list((REPO_ROOT / "tools/navsim_candidate_relative_audit").glob("*.py"))
    report_inputs = [
        manifest_path,
        metrics_path,
        OLD_REPORT / "target_schema.json",
        OLD_REPORT / "oracle_probe_results.json",
        OLD_REPORT / "gate_status.json",
    ]
    selected_targets = [OLD_REPORT / "targets" / f"{token}.npz" for token in tokens]
    hashes = {
        "base_commit": BASE_COMMIT,
        "files": hash_files([*code_inputs, *report_inputs, *selected_targets], REPO_ROOT),
        "sampled_scene_tokens": tokens,
        "sample_seed": args.seed,
    }
    write_json(report_dir / "input_hashes.json", hashes)
    write_json(report_dir / "environment.json", _environment(args, paths))

    max_candidate_error = max(candidate_errors, default=float("inf"))
    max_score_errors = {key: max(values, default=float("inf")) for key, values in score_errors.items()}
    max_target_errors = {key: max(values, default=float("inf")) for key, values in target_errors.items()}
    # These are the only model features in the old probes. Target/evaluation
    # columns are deliberately not passed to this assertion.
    assert_feature_names_safe(["trajectory", "current_scene", "candidate_relative_environment"])
    passed = (
        len(tokens) == args.num_scenes
        and max_candidate_error <= args.atol
        and max(max_score_errors.values(), default=float("inf")) <= args.atol
        and max(max_target_errors.values(), default=float("inf")) <= args.atol
        and all(order_matches)
        and git_output(["merge-base", BASE_COMMIT, "HEAD"]) == BASE_COMMIT
    )
    result = {
        "gate": "C0",
        "passed": passed,
        "scene_count": len(tokens),
        "log_count": int(metrics.log_name.nunique()),
        "candidate_count": int(len(manifest)),
        "candidate_max_abs_error": max_candidate_error,
        "score_max_abs_errors": max_score_errors,
        "target_max_abs_errors": max_target_errors,
        "candidate_order_preserved": all(order_matches),
        "target_schema_sha256": sha256_file(OLD_REPORT / "target_schema.json"),
        "official_score_in_model_inputs": False,
        "cache_output": str(cache_dir),
    }
    write_json(report_dir / "reproduction_results.json", result)
    update_gate(report_dir, "gate_c0", {"passed": passed, "result": result})
    write_markdown(
        report_dir / "REPRODUCTION_REPORT.md",
        f"""# Gate C0 Reproduction Report

## Gate C0: {'PASS' if passed else 'FAIL'}

- Base commit required: `{BASE_COMMIT}`
- Sampled scenes/logs/candidates: {len(tokens)} / {metrics.log_name.nunique()} / {len(manifest)}
- Candidate trajectory max absolute error: {max_candidate_error:.3g}
- Official score/simulation maximum absolute error: {max(max_score_errors.values()):.3g}
- Structured target maximum absolute error: {max(max_target_errors.values()):.3g}
- Candidate order preserved: {all(order_matches)}
- Official score present in model inputs: false
- Input hashes: `input_hashes.json`

The check regenerates prior deterministic candidates, reruns the deployed
PDMSimulator/PDMScorer and rebuilds every structured target for the sampled
scenes. Existing cache files are read-only and are never overwritten.
""",
    )
    if not passed:
        raise SystemExit("Gate C0 failed; model development must stop")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    reproduce(args)
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.reproduce_gate_c0 " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
