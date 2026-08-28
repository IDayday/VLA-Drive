#!/usr/bin/env python3
"""Export frozen EpisodeDrive proposals and current-scene embeddings on legal train data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
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
    sha256_file,
    stable_scene_seed,
    validate_training_split,
    write_json,
    write_markdown,
    write_parquet,
)


KNOWN_CHECKPOINTS = (
    Path("/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt"),
)
KNOWN_VLM_CONFIGS = (
    Path("/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope"),
    Path("/mnt/project/DriveVLA-M0-models/InternVL3-2B"),
)
KNOWN_FEATURE_CACHES = (
    Path("/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"),
)
EXPECTED_BASE_SHA256 = "7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d"
EXPECTED_BASE_BYTES = 4_271_779_662


def _resolve_file(explicit: Path | None, environment: str, known: tuple[Path, ...]) -> Path:
    candidates = [explicit, Path(os.environ[environment]) if os.environ.get(environment) else None, *known]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {environment}; checked {[str(value) for value in candidates if value]}")


def _resolve_dir(explicit: Path | None, environment: str, known: tuple[Path, ...]) -> Path:
    candidates = [explicit, Path(os.environ[environment]) if os.environ.get(environment) else None, *known]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {environment}; checked {[str(value) for value in candidates if value]}")


def discover_assets(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_file(args.checkpoint, "DRIVEVLA_BASE_CHECKPOINT", KNOWN_CHECKPOINTS)
    vlm_config = _resolve_dir(args.vlm_config, "DRIVEVLA_VLM_CONFIG", KNOWN_VLM_CONFIGS)
    feature_cache = _resolve_dir(args.feature_cache, "DRIVEVLA_NAVTRAIN_FEATURE_CACHE", KNOWN_FEATURE_CACHES)
    config = (args.config or Path("configs/base_model_navtest.yaml")).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint_mtime_ns = checkpoint.stat().st_mtime_ns
    inventory_path = Path(args.output_dir) / "episode_drive_asset_inventory.json"
    cached = None
    if inventory_path.is_file() and not args.rehash_assets:
        try:
            candidate = json.loads(inventory_path.read_text(encoding="utf-8"))
            if (
                candidate.get("checkpoint") == str(checkpoint)
                and candidate.get("checkpoint_bytes") == checkpoint_bytes
                and candidate.get("checkpoint_mtime_ns") == checkpoint_mtime_ns
                and candidate.get("checkpoint_sha256") == EXPECTED_BASE_SHA256
            ):
                cached = candidate
        except (json.JSONDecodeError, OSError):
            pass
    checkpoint_sha = cached["checkpoint_sha256"] if cached else sha256_file(checkpoint)
    if checkpoint_bytes != EXPECTED_BASE_BYTES or checkpoint_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"EpisodeDrive Base checkpoint identity mismatch: bytes={checkpoint_bytes}, sha256={checkpoint_sha}"
        )
    feature_tokens = (
        int(cached["feature_cache_complete_token_count"])
        if cached and cached.get("feature_cache") == str(feature_cache)
        else sum(
            1
            for path in feature_cache.glob("*/*/internvl_feature.gz")
            if (path.parent / "trajectory_target.gz").is_file()
        )
    )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mtime_ns": checkpoint_mtime_ns,
        "checkpoint_sha256": checkpoint_sha,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "vlm_config": str(vlm_config),
        "feature_cache": str(feature_cache),
        "feature_cache_complete_token_count": feature_tokens,
    }


def select_scenes(
    scene_manifest: pd.DataFrame,
    scenes_per_log: int,
    seed: int,
    num_shards: int,
    shard_index: int,
    max_scenes: int,
) -> pd.DataFrame:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard-index is out of range")
    rows = []
    for log_name, group in scene_manifest.groupby("log_name", sort=True):
        ranked = group.copy()
        ranked["model_candidate_selection_key"] = [
            stable_scene_seed(str(token), seed ^ 0xE91D) for token in ranked.scene_token
        ]
        ranked = ranked.sort_values(["model_candidate_selection_key", "scene_token"])
        rows.append(ranked.head(scenes_per_log))
    selected = pd.concat(rows, ignore_index=True)
    logs = sorted(selected.log_name.unique())
    shard_logs = {log_name for position, log_name in enumerate(logs) if position % num_shards == shard_index}
    selected = selected[selected.log_name.isin(shard_logs)].copy()
    if max_scenes > 0:
        selected = selected.head(max_scenes).copy()
    return selected.sort_values(["log_name", "model_candidate_selection_key"]).reset_index(drop=True)


def _read_cached_feature(feature_cache: Path, log_name: str, token: str) -> dict[str, Any]:
    path = feature_cache / log_name / token / "internvl_feature.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as stream:
        feature = pickle.load(stream)
    required = {"history_trajectory", "high_command_one_hot", "status_feature", "image_path_tensor"}
    missing = required - set(feature)
    if missing:
        raise ValueError(f"Cached feature missing {sorted(missing)}: {path}")
    image_path = Path("".join(chr(int(value)) for value in feature["image_path_tensor"].tolist()))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    feature["image_path"] = str(image_path)
    return feature


def _build_agent(assets: dict[str, Any]):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # The frozen proposal/scorer forward never calls training-time oracle
    # scoring. Starting Ray here would waste shared memory and can rewrite
    # CUDA visibility, so disable that optional constructor path explicitly.
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"
    os.environ["DRIVEVLA_SCORE_PROCESSES"] = "0"
    config = OmegaConf.load(assets["config"])
    config.checkpoint_path = assets["checkpoint"]
    config.stage1_checkpoint_path = None
    config.vlm_config.vlm_path = assets["vlm_config"]
    config.vlm_config.cache_hidden_state = False
    config.vlm_config.cache_mode = False
    config.vlm_config.initialize_from_config = True
    # This bypass is proven bit-identical for hidden states by the repository's
    # verify_lm_head_bypass.py and avoids unused 151k-way logits.
    config.vlm_config.skip_lm_head = True
    config.vlm_config.use_flash_attn = False
    config.action_head_config.return_memory_fields = True
    config.cache_data = False
    config.progress_bar = False
    agent = instantiate(config)
    agent.initialize()
    agent = agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in agent.parameters()):
        raise AssertionError("Frozen EpisodeDrive export has trainable parameters")
    torch.set_grad_enabled(False)
    return agent


def _factor_arrays(prediction: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    factor_dict = prediction["pred_logit"]
    names = sorted(factor_dict)
    values = np.stack(
        [factor_dict[name].detach().float().cpu().numpy()[0] for name in names], axis=-1
    ).astype(np.float32)
    return names, values


def _feature_payload(cached: dict[str, Any]) -> dict[str, Any]:
    return {
        "history_trajectory": cached["history_trajectory"].unsqueeze(0),
        "high_command_one_hot": cached["high_command_one_hot"].unsqueeze(0),
        "status_feature": cached["status_feature"].unsqueeze(0),
        "image_path_tensor": cached["image_path_tensor"].unsqueeze(0),
    }


def _logical_array_sha(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def export(args: argparse.Namespace) -> dict[str, Any]:
    validate_training_split(args.split)
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    assets = discover_assets(args)
    write_json(report_dir / "episode_drive_asset_inventory.json", assets)
    if args.asset_audit_only:
        return assets
    scene_manifest = pd.read_parquet(report_dir / "balanced_scene_manifest.parquet")
    selected = select_scenes(
        scene_manifest,
        args.scenes_per_log,
        args.seed,
        args.num_shards,
        args.shard_index,
        args.max_scenes,
    )
    feature_cache = Path(assets["feature_cache"])
    raw_dir = ensure_dir(cache_dir / "model_candidates" / "raw")
    agent = _build_agent(assets)
    import torch

    rows = []
    failures = []
    deterministic_errors: list[float] = []
    determinism_checked = 0
    started = time.time()
    for position, row in enumerate(selected.itertuples(index=False)):
        fold_dir = ensure_dir(raw_dir / f"fold_{int(row.fold)}")
        output_path = fold_dir / f"{row.scene_token}.npz"
        if output_path.is_file() and not args.force:
            try:
                with np.load(output_path, allow_pickle=False) as existing:
                    cached_arrays = {
                        name: existing[name]
                        for name in (
                            "proposals", "baseline_scorer_score", "baseline_factor_logits",
                            "scene_features", "ego_token",
                        )
                    }
                    cached_logical_sha = _logical_array_sha(cached_arrays)
                    if (
                        str(existing["checkpoint_sha256"].item()) == assets["checkpoint_sha256"]
                        and str(existing["config_sha256"].item()) == assets["config_sha256"]
                        and existing["proposals"].shape == (64, 8, 3)
                        and cached_logical_sha == str(existing["logical_sha256"].item())
                    ):
                        rows.append(
                            {
                                "scene_token": row.scene_token,
                                "log_name": row.log_name,
                                "fold": int(row.fold),
                                "path": str(output_path),
                                "baseline_selected_index": int(existing["baseline_selected_index"]),
                                "logical_sha256": cached_logical_sha,
                                "reused": True,
                            }
                        )
                        continue
            except Exception:
                pass
        try:
            cached = _read_cached_feature(feature_cache, str(row.log_name), str(row.scene_token))
            features = _feature_payload(cached)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = agent.forward(features)
            proposals = prediction["proposals"].detach().float().cpu().numpy()[0].astype(np.float32)
            baseline_score = prediction["pdm_score"].detach().float().cpu().numpy()[0].astype(np.float32)
            scene_features = prediction["language_feature"].detach().float().cpu().numpy()[0].astype(np.float16)
            ego_token = prediction["ego_feature"].detach().float().cpu().numpy()[0].astype(np.float16)
            factor_names, factor_logits = _factor_arrays(prediction)
            arrays = {
                "proposals": proposals,
                "baseline_scorer_score": baseline_score,
                "baseline_factor_logits": factor_logits,
                "scene_features": scene_features,
                "ego_token": ego_token,
            }
            if determinism_checked < args.determinism_scenes:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    repeated = agent.forward(_feature_payload(cached))
                repeated_names, repeated_factors = _factor_arrays(repeated)
                repeated_arrays = {
                    "proposals": repeated["proposals"].detach().float().cpu().numpy()[0],
                    "baseline_scorer_score": repeated["pdm_score"].detach().float().cpu().numpy()[0],
                    "baseline_factor_logits": repeated_factors,
                    "scene_features": repeated["language_feature"].detach().float().cpu().numpy()[0],
                    "ego_token": repeated["ego_feature"].detach().float().cpu().numpy()[0],
                }
                if repeated_names != factor_names:
                    raise AssertionError("Factor ordering changed on repeated forward")
                for name, value in arrays.items():
                    reference = value.astype(np.float32)
                    repeated_value = repeated_arrays[name].astype(value.dtype).astype(np.float32)
                    error = float(np.max(np.abs(reference - repeated_value)))
                    deterministic_errors.append(error)
                    if error != 0.0:
                        raise AssertionError(f"Frozen forward is not deterministic for {name}: {error}")
                determinism_checked += 1
            logical_sha = _logical_array_sha(arrays)
            np.savez_compressed(
                output_path,
                **arrays,
                factor_names=np.asarray(factor_names, dtype="U64"),
                baseline_selected_index=np.asarray(int(np.argmax(baseline_score)), dtype=np.int16),
                checkpoint_sha256=np.asarray(assets["checkpoint_sha256"]),
                config_sha256=np.asarray(assets["config_sha256"]),
                image_path_sha256=np.asarray(sha256_file(Path(cached["image_path"]))),
                logical_sha256=np.asarray(logical_sha),
            )
            rows.append(
                {
                    "scene_token": row.scene_token,
                    "log_name": row.log_name,
                    "fold": int(row.fold),
                    "path": str(output_path),
                    "baseline_selected_index": int(np.argmax(baseline_score)),
                    "logical_sha256": logical_sha,
                    "reused": False,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "scene_token": str(row.scene_token),
                    "log_name": str(row.log_name),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (position + 1) % 25 == 0:
            print(
                f"shard={args.shard_index} scenes={position + 1}/{len(selected)} failures={len(failures)}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    shard_dir = ensure_dir(cache_dir / "model_candidates" / "export_shards")
    write_parquet(frame, shard_dir / f"shard_{args.shard_index:04d}.parquet")
    result = {
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_scene_count": int(len(selected)),
        "selected_log_count": int(selected.log_name.nunique()),
        "exported_scene_count": int(len(frame)),
        "success_rate": float(len(frame) / len(selected)) if len(selected) else 0.0,
        "failure_count": len(failures),
        "failure_examples": failures[:20],
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "proposal_count": 64,
        "pose_count": 8,
        "base_model_frozen": True,
        "lm_head_bypassed_bit_identically": True,
        "deterministic_max_abs_error": max(deterministic_errors, default=None),
        "determinism_scene_count": determinism_checked,
        "elapsed_seconds": time.time() - started,
    }
    write_json(shard_dir / f"shard_{args.shard_index:04d}_summary.json", result)
    return result


def aggregate_exports(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    shard_dir = cache_dir / "model_candidates" / "export_shards"
    frames = [pd.read_parquet(path) for path in sorted(shard_dir.glob("shard_*.parquet"))]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        raise FileNotFoundError(f"No exported proposal shards under {shard_dir}")
    frame = pd.concat(frames, ignore_index=True)
    if frame.scene_token.duplicated().any():
        raise RuntimeError("Duplicate scene tokens across proposal export shards")
    scene_manifest_path = (
        report_dir / "all_scene_inventory.parquet"
        if (report_dir / "all_scene_inventory.parquet").is_file()
        else report_dir / "balanced_scene_manifest.parquet"
    )
    scene_manifest = pd.read_parquet(scene_manifest_path)
    # The controlled split can cap scenes per log after export starts. Folds are
    # log-level, so remap them from the current official fold manifest.
    fold_manifest = pd.read_parquet(report_dir / "balanced_scene_manifest.parquet")
    fold_by_log = fold_manifest.groupby("log_name").fold.first().to_dict()
    frame["fold"] = frame.log_name.map(fold_by_log).astype(int)
    write_parquet(frame, report_dir / "candidates/episode_drive_raw_exports.parquet")
    expected = select_scenes(
        scene_manifest, args.scenes_per_log, args.seed, 1, 0, args.max_scenes
    )
    missing = sorted(set(expected.scene_token) - set(frame.scene_token))
    summary_paths = sorted(shard_dir.glob("shard_*_summary.json"))
    shard_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in summary_paths
    ]
    determinism_errors = [
        float(item["deterministic_max_abs_error"])
        for item in shard_summaries
        if item.get("deterministic_max_abs_error") is not None
    ]
    result = {
        "expected_scene_count": int(len(expected)),
        "exported_scene_count": int(len(frame)),
        "expected_log_count": int(expected.log_name.nunique()),
        "exported_log_count": int(frame.log_name.nunique()),
        "coverage": float(len(frame) / len(expected)),
        "missing_scene_examples": missing[:20],
        "checkpoint_sha256": EXPECTED_BASE_SHA256,
        "proposal_count": 64,
        "gt_forced_into_inference_candidates": False,
        "shard_count": len(shard_summaries),
        "deterministic_max_abs_error": max(determinism_errors, default=None),
        "all_shards_frozen": all(item.get("base_model_frozen") for item in shard_summaries),
        "selection_inventory": str(scene_manifest_path),
    }
    write_json(report_dir / "candidates/episode_drive_export_summary.json", result)
    write_markdown(
        report_dir / "MODEL_CANDIDATE_AUDIT.md",
        f"""# EpisodeDrive Model Candidate Audit

- Frozen Base checkpoint SHA256: `{EXPECTED_BASE_SHA256}`
- Legal trainval scenes/logs exported: {len(frame):,} / {frame.log_name.nunique():,}
- Coverage against deterministic all-log selection: {result['coverage']:.3%}
- Raw proposals per scene: 64; poses per proposal: 8
- Ground truth inserted into inference candidate set: no
- Cached fields: pre-scorer proposals, original factor logits/score, original selection,
  Q-Former scene features, ego token, current-input image hash

The exporter reads only current EpisodeDrive inputs from the existing navtrain
feature cache. It never opens logged-future annotations, future images or
official score files. The Base model, Q-Former, generator and scorer remain frozen.
""",
    )
    if (
        result["coverage"] <= 0.98
        or result["deterministic_max_abs_error"] not in (None, 0.0)
        or not result["all_shards_frozen"]
    ):
        raise RuntimeError(f"EpisodeDrive proposal export audit failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "aggregate", "asset-audit"), default="export")
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--scenes-per-log", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--determinism-scenes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-config", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rehash-assets", action="store_true")
    parser.add_argument("--asset-audit-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    if args.mode == "aggregate":
        result = aggregate_exports(args)
    else:
        args.asset_audit_only = args.asset_audit_only or args.mode == "asset-audit"
        result = export(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.export_episode_drive_candidates "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
