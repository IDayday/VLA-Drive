#!/usr/bin/env python3
"""Consolidate per-scene v3 targets into a resumable memory-mapped oracle store."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    require_gate,
    sha256_file,
    write_json,
    write_markdown,
)
from .run_oracle_decomposition import _completed_prefix_length, _risk_from_actor


FAMILY_NAMES = (
    "gt",
    "lateral_offset",
    "speed_change",
    "brake_timing",
    "same_prefix_different_tail",
    "same_endpoint_mid_curve",
    "different_prefix_similar_endpoint",
    "progress_shape",
)


ARRAY_SPECS = {
    "candidate_indices": (np.int16, ("N", "K")),
    "candidate_family_ids": (np.uint8, ("N", "K")),
    "is_gt": (np.bool_, ("N", "K")),
    "trajectory": (np.float32, ("N", "K", 24)),
    "current": (np.float32, ("N", 150)),
    "k_exact": (np.float32, ("N", "K", 48)),
    "static": (np.float32, ("N", "K", 48)),
    "d_state": (np.float32, ("N", "K", 1448)),
    "d_risk": (np.float32, ("N", "K", 40)),
    "d_signal": (np.float32, ("N", "K", 24)),
    "recomputed_risk": (np.float32, ("N", "K", 40)),
    "score": (np.float32, ("N", "K")),
    "factors": (np.float32, ("N", "K", 6)),
    "completed": (np.bool_, ("N",)),
}


def _shape(spec: tuple[Any, tuple[Any, ...]], scenes: int, candidates: int) -> tuple[int, ...]:
    return tuple(scenes if value == "N" else candidates if value == "K" else int(value) for value in spec[1])


def _open_arrays(store_dir: Path, scenes: int, candidates: int, create: bool) -> dict[str, np.ndarray]:
    arrays = {}
    for name, spec in ARRAY_SPECS.items():
        path = store_dir / f"{name}.npy"
        shape = _shape(spec, scenes, candidates)
        if create:
            arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=spec[0], shape=shape)
            arrays[name][...] = False if spec[0] == np.bool_ else 0
        else:
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.dtype(spec[0]):
                raise RuntimeError(
                    f"Oracle-store array mismatch for {name}: {array.shape}/{array.dtype}, "
                    f"expected {shape}/{np.dtype(spec[0])}"
                )
            arrays[name] = array
    return arrays


def _target_path(cache_dir: Path, log_name: str, token: str) -> Path:
    nested = cache_dir / "targets_v3" / log_name.replace("/", "_") / f"{token}.npz"
    if nested.is_file():
        return nested
    flat = cache_dir / "targets_v3" / f"{token}.npz"
    return flat


def _build_scene(
    arrays: dict[str, np.ndarray],
    scene_index: int,
    target_path: Path,
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    family_to_id: dict[str, int],
) -> None:
    with np.load(target_path, allow_pickle=False) as target:
        indices = target["candidate_index"].astype(np.int64)
        if len(indices) != len(candidates):
            raise ValueError(f"candidate count mismatch: target={len(indices)} manifest={len(candidates)}")
        candidates = candidates.set_index("candidate_index").loc[indices]
        metrics = metrics.set_index("candidate_index").loc[indices]
        poses = np.stack(
            [
                np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]).reshape(-1)
                for row in candidates.itertuples(index=False)
            ],
            axis=0,
        ).astype(np.float32)
        actor = target["D_state_actor"].astype(np.float32)
        actor_mask = target["D_state_actor_mask"].astype(bool)
        actor_masked = actor * actor_mask[..., None]
        d_state = np.concatenate(
            [
                target["D_state_summary"].reshape(len(indices), -1),
                actor_masked.reshape(len(indices), -1),
                actor_mask.astype(np.float32).reshape(len(indices), -1),
            ],
            axis=-1,
        ).astype(np.float32)
        recomputed = _risk_from_actor(
            target["D_state_summary"][None], actor[None], actor_mask[None]
        )[0].reshape(len(indices), -1)
        factor = np.column_stack(
            [
                1.0 - metrics.no_at_fault_collision.to_numpy(float),
                1.0 - metrics.ttc.to_numpy(float),
                1.0 - metrics.dac.to_numpy(float),
                1.0 - metrics.ddc.to_numpy(float),
                1.0 - metrics.comfort.to_numpy(float),
                metrics.progress.to_numpy(float),
            ]
        ).astype(np.float32)
        family_ids = np.asarray(
            [family_to_id[str(value)] for value in candidates.candidate_family], dtype=np.uint8
        )
        values = {
            "candidate_indices": indices.astype(np.int16),
            "candidate_family_ids": family_ids,
            "is_gt": target["is_gt"].astype(bool),
            "trajectory": poses,
            "current": np.concatenate(
                [
                    target["current_scene_features"].astype(np.float32),
                    (
                        target["current_actor_state"].astype(np.float32)
                        * target["current_actor_mask"].astype(np.float32)[:, None]
                    ).reshape(-1),
                    target["current_actor_mask"].astype(np.float32),
                ]
            ),
            "k_exact": target["K_exact"].reshape(len(indices), -1).astype(np.float32),
            "static": target["S_static"].reshape(len(indices), -1).astype(np.float32),
            "d_state": d_state,
            "d_risk": target["D_risk"].reshape(len(indices), -1).astype(np.float32),
            "d_signal": target["D_signal"].reshape(len(indices), -1).astype(np.float32),
            "recomputed_risk": recomputed.astype(np.float32),
            "score": metrics.aggregate_score.to_numpy(dtype=np.float32),
            "factors": factor,
        }
        for name, value in values.items():
            if value.shape != arrays[name][scene_index].shape:
                raise ValueError(
                    f"{name} shape mismatch at scene {scene_index}: "
                    f"{value.shape} != {arrays[name][scene_index].shape}"
                )
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"non-finite {name} at scene {scene_index}")
            arrays[name][scene_index] = value
        arrays["completed"][scene_index] = True


def build(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "target_v3")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    store_dir = ensure_dir(args.store_dir or cache_dir / "oracle_store")
    manifest_path = report_dir / "balanced_scene_manifest.parquet"
    scene_manifest = pd.read_parquet(manifest_path).sort_values("selection_index").reset_index(drop=True)
    # Preserve every formally selected scene in metadata, including audited
    # failures, while placing preflight-missing targets at the physical tail of
    # the mmap.  The loader may trim a *contiguous trailing* failure suffix but
    # will still reject holes in the middle, which avoids both silent deletion
    # and a multi-GiB advanced-index copy in every distributed probe process.
    scene_manifest["target_preflight_available"] = [
        _target_path(cache_dir, str(row.log_name), str(row.scene_token)).is_file()
        for row in scene_manifest.itertuples(index=False)
    ]
    scene_manifest = scene_manifest.sort_values(
        ["target_preflight_available", "selection_index"],
        ascending=[False, True],
    ).reset_index(drop=True)
    scenes = len(scene_manifest)
    candidates_per_scene = args.num_candidates
    manifest_hash = sha256_file(manifest_path)
    config_path = store_dir / "store_config.json"
    expected_config = {
        "schema_version": "1.0.0",
        "scene_count": scenes,
        "candidates_per_scene": candidates_per_scene,
        "scene_manifest_sha256": manifest_hash,
        "family_names": list(FAMILY_NAMES),
        "target_preflight_missing": int((~scene_manifest.target_preflight_available).sum()),
        "incomplete_scene_policy": "only a contiguous trailing suffix may be trimmed by the loader",
        "array_specs": {
            name: {"dtype": np.dtype(spec[0]).name, "shape": list(_shape(spec, scenes, candidates_per_scene))}
            for name, spec in ARRAY_SPECS.items()
        },
    }
    create = args.force or not config_path.is_file()
    if config_path.is_file() and not args.force:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != expected_config:
            raise RuntimeError("Existing oracle store does not match current manifest/schema; use a new store path")
    arrays = _open_arrays(store_dir, scenes, candidates_per_scene, create=create)
    if create:
        write_json(config_path, expected_config)
    family_to_id = {name: index for index, name in enumerate(FAMILY_NAMES)}
    metadata_path = store_dir / "scene_metadata.parquet"
    metadata = scene_manifest[
        ["scene_token", "log_name", "fold", "selection_index", "target_preflight_available"]
    ].copy()
    metadata["scene_index"] = np.arange(scenes, dtype=np.int64)
    metadata.to_parquet(metadata_path, index=False, engine="pyarrow", compression="zstd")

    failures: list[dict[str, str]] = []
    started = time.time()
    processed = 0
    combined_candidates = None
    combined_metrics = None
    combined_candidate_path = cache_dir / "controlled_candidate_manifest.parquet"
    combined_metric_path = cache_dir / "candidate_metrics.parquet"
    if combined_candidate_path.is_file() and combined_metric_path.is_file():
        combined_candidates = pd.read_parquet(combined_candidate_path)
        combined_metrics = pd.read_parquet(combined_metric_path)
    for log_position, (log_name, log_scenes) in enumerate(metadata.groupby("log_name", sort=True)):
        indices = log_scenes.scene_index.to_numpy(dtype=np.int64)
        if arrays["completed"][indices].all() and not args.force:
            continue
        safe_name = str(log_name).replace("/", "_")
        candidate_path = cache_dir / "candidate_shards" / f"{safe_name}.parquet"
        metric_path = cache_dir / "metric_shards" / f"{safe_name}.parquet"
        if candidate_path.is_file() and metric_path.is_file():
            candidate_frame = pd.read_parquet(candidate_path)
            metric_frame = pd.read_parquet(metric_path)
        elif combined_candidates is not None and combined_metrics is not None:
            candidate_frame = combined_candidates[combined_candidates.log_name == log_name].copy()
            metric_frame = combined_metrics[combined_metrics.log_name == log_name].copy()
        else:
            failures.append({"log_name": str(log_name), "error": "missing candidate or metric shard"})
            continue
        for row in log_scenes.itertuples(index=False):
            scene_index = int(row.scene_index)
            if arrays["completed"][scene_index] and not args.force:
                continue
            token = str(row.scene_token)
            try:
                _build_scene(
                    arrays,
                    scene_index,
                    _target_path(cache_dir, str(log_name), token),
                    candidate_frame[candidate_frame.scene_token == token].copy(),
                    metric_frame[metric_frame.scene_token == token].copy(),
                    family_to_id,
                )
                processed += 1
            except Exception as exc:
                failures.append(
                    {"scene_token": token, "log_name": str(log_name), "error": f"{type(exc).__name__}: {exc}"}
                )
        if (log_position + 1) % 25 == 0:
            for array in arrays.values():
                if hasattr(array, "flush"):
                    array.flush()
            print(
                f"logs={log_position + 1}/{metadata.log_name.nunique()} "
                f"complete={int(arrays['completed'].sum())}/{scenes} failures={len(failures)}",
                flush=True,
            )
    for array in arrays.values():
        if hasattr(array, "flush"):
            array.flush()
    completed = int(arrays["completed"].sum())
    completed_prefix = _completed_prefix_length(np.asarray(arrays["completed"]))
    success_rate = completed / scenes if scenes else 0.0
    byte_count = sum((store_dir / f"{name}.npy").stat().st_size for name in ARRAY_SPECS)
    valid_score = np.asarray(arrays["score"][:completed_prefix])
    valid_factors = np.asarray(arrays["factors"][:completed_prefix])
    store_statistics = {
        "scene_count": completed_prefix,
        "candidate_count": int(valid_score.size),
        "aggregate_score_mean": float(valid_score.mean()),
        "aggregate_score_std": float(valid_score.std()),
        "nonconstant_score_scene_rate": float((np.ptp(valid_score, axis=1) > 1e-7).mean()),
        "mean_within_scene_score_range": float(np.ptp(valid_score, axis=1).mean()),
        "collision_bad_candidate_rate": float(valid_factors[..., 0].mean()),
        "ttc_bad_candidate_rate": float(valid_factors[..., 1].mean()),
        "dac_bad_candidate_rate": float(valid_factors[..., 2].mean()),
        "ddc_bad_candidate_rate": float(valid_factors[..., 3].mean()),
        "comfort_bad_candidate_rate": float(valid_factors[..., 4].mean()),
        "collision_varies_scene_rate": float((np.ptp(valid_factors[..., 0], axis=1) > 0).mean()),
        "ttc_varies_scene_rate": float((np.ptp(valid_factors[..., 1], axis=1) > 0).mean()),
        "dac_varies_scene_rate": float((np.ptp(valid_factors[..., 2], axis=1) > 0).mean()),
        "ddc_varies_scene_rate": float((np.ptp(valid_factors[..., 3], axis=1) > 0).mean()),
    }
    write_json(report_dir / "oracle_store_statistics.json", store_statistics)
    result = {
        "scene_count": scenes,
        "completed_scene_count": completed,
        "completed_prefix_scene_count": completed_prefix,
        "trailing_incomplete_scene_count": scenes - completed_prefix,
        "success_rate": success_rate,
        "processed_this_run": processed,
        "failure_count": len(failures),
        "failure_examples": failures[:50],
        "store_bytes": byte_count,
        "store_dir": str(store_dir),
        "elapsed_seconds": time.time() - started,
        "official_scores_are_targets_only": True,
    }
    write_json(report_dir / "oracle_store_summary.json", result)
    write_markdown(
        report_dir / "ORACLE_STORE_REPORT.md",
        f"""# Oracle Feature Store

- Scenes: {completed:,}/{scenes:,} ({success_rate:.3%})
- Mmap-safe completed prefix: {completed_prefix:,}; trailing audited failures: {scenes - completed_prefix:,}
- Fixed candidates per scene: {candidates_per_scene}
- Store size: {byte_count / 2**30:.2f} GiB (not committed)
- Store: `{store_dir}`
- Failures: {len(failures)}

The memory-mapped store separates model features from offline labels. Official
aggregate/factor values are retained only as ranking/evaluation targets; they are
never concatenated into O0–O13 inputs. The store is resumable at scene granularity.
""",
    )
    if success_rate <= 0.98:
        raise SystemExit("Oracle store coverage is below 98%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    parser.add_argument("--store-dir", type=Path)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.build_oracle_store " + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
