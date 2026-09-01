#!/usr/bin/env python3
"""Cache frozen DrivOR current-observation registers for scorer retraining.

The cache deliberately contains no proposal, target, future annotation,
metric-cache value, or evaluator score.  It is schema-compatible with the
private-observation loader used by ``train_independent_scorer.py`` so that a
new ranker can be trained on Base proposals while retaining DrivOR's
scorer-specific visual representation.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import torch

from local_stage2.export_drivor_external_proposal_scores import (
    ProposalEntry,
    _atomic_json_dump,
    _atomic_torch_save,
    _batch_features,
    _build_pickle_source,
    _build_chunk_source,
    _encode_context,
    _git_commit,
    _load_log_mapping,
    _load_model,
    _sha256_file,
    _source_manifests,
)

import navsim
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drivor-repo", type=Path, default=Path("/mnt/project/external/DrivoR")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/project/external/DrivoR/weights/releases/drivor_Nav1_25epochs.pth"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/mnt/project/external/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml"
        ),
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        default=Path(
            "/mnt/project/external/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposal-pickle", type=Path)
    source.add_argument("--proposal-root", type=Path)
    parser.add_argument(
        "--candidate-matrix",
        type=Path,
        help=(
            "NPZ containing token/log-name alignment. Required when a proposal "
            "pickle does not store log_name per token."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("navtrain", "navtest"),
        default="navtrain",
        help="Dataset lineage; defaults to the legacy Navtrain behavior.",
    )
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--cache-dtype",
        choices=("auto", "float32", "float16"),
        default="auto",
        help=(
            "auto uses FP16 for Navtrain throughput and FP32 for Navtest "
            "benchmark parity, while preserving the dtype of a resumable "
            "partial cache."
        ),
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--token",
        action="append",
        default=[],
        help="Optional exact scene token; repeat to export a parity subset.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def _flush(
    buffer: Dict[str, List[object]],
    shard_dir: Path,
    chunk_index: int,
    cache_dtype: torch.dtype,
) -> int:
    if not buffer["tokens"]:
        return chunk_index
    payload = {
        "schema_version": 1,
        "tokens": list(buffer["tokens"]),
        "log_names": list(buffer["log_names"]),
        "visual_tokens": torch.cat(buffer["visual_tokens"]).to(cache_dtype),
        "visual_valid_mask": torch.cat(buffer["visual_valid_mask"]).bool(),
        "status_feature": torch.cat(buffer["status_feature"]).float(),
        # The generic private-cache loader concatenates these current-context
        # tensors. DrivOR already packs pose/velocity/acceleration/command in
        # its 11-D current ego status, so the two extra tensors are empty.
        "history_trajectory": torch.empty(len(buffer["tokens"]), 0),
        "high_command_one_hot": torch.empty(len(buffer["tokens"]), 0),
    }
    _atomic_torch_save(payload, shard_dir / f"chunk_{chunk_index:06d}.pt")
    for values in buffer.values():
        values.clear()
    return chunk_index + 1


def _load_completed_tokens(shard_dir: Path) -> set[str]:
    completed: set[str] = set()
    for path in sorted(shard_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        chunk_tokens = {str(value) for value in payload["tokens"]}
        if len(chunk_tokens) != len(payload["tokens"]):
            raise RuntimeError(f"Duplicate token within {path}")
        overlap = completed.intersection(chunk_tokens)
        if overlap:
            raise RuntimeError(f"Duplicate completed token: {sorted(overlap)[:3]}")
        completed.update(chunk_tokens)
    return completed


def _lineage(args: argparse.Namespace, source_manifests: List[Path]) -> Mapping[str, object]:
    if args.proposal_pickle is not None:
        proposal_source: Mapping[str, object] = {
            "kind": "pickle",
            "path": str(args.proposal_pickle.resolve()),
            "sha256": _sha256_file(args.proposal_pickle),
        }
    else:
        assert args.proposal_root is not None
        proposal_source = {
            "kind": "chunk_root",
            "path": str(args.proposal_root.resolve()),
            "manifest_sha256": {
                str(path.relative_to(args.proposal_root)): _sha256_file(path)
                for path in source_manifests
            },
        }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "DrivORSceneReplayExporter",
        "producer_sha256": _sha256_file(Path(__file__).resolve()),
        "drivor_repo": str(args.drivor_repo.resolve()),
        "drivor_commit": _git_commit(args.drivor_repo),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256_file(args.config),
        "dino_weights": str(args.dino_weights.resolve()),
        "dino_weights_sha256": _sha256_file(args.dino_weights),
        "proposal_index_source": proposal_source,
        "candidate_matrix": (
            {
                "path": str(args.candidate_matrix.resolve()),
                "sha256": _sha256_file(args.candidate_matrix),
            }
            if args.candidate_matrix is not None
            else None
        ),
        "split": args.split,
        "precision": f"fp32_compute_{args.cache_dtype}_cache",
        "inference_inputs": ["current_camera_images", "current_ego_status"],
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "proposal_or_score_payload": False,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "explicit_tokens": sorted(str(value) for value in args.token),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.chunk_size <= 0 or args.batch_size <= 0:
        raise ValueError("batch-size and chunk-size must be positive")
    for path in (
        args.drivor_repo,
        args.checkpoint,
        args.config,
        args.dino_weights,
        args.log_path,
        args.sensor_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    proposal_path = args.proposal_pickle or args.proposal_root
    assert proposal_path is not None
    if not proposal_path.exists():
        raise FileNotFoundError(proposal_path)
    if args.candidate_matrix is not None and not args.candidate_matrix.exists():
        raise FileNotFoundError(args.candidate_matrix)
    imported_navsim = Path(navsim.__file__).resolve()
    if args.drivor_repo.resolve() not in imported_navsim.parents:
        raise RuntimeError(
            f"Imported navsim from {imported_navsim}, not {args.drivor_repo}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    source_manifests: List[Path] = []
    if args.proposal_pickle is not None:
        source = _build_pickle_source(
            args.proposal_pickle,
            _load_log_mapping(args.candidate_matrix),
            allowed_physical_logs=None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )
    else:
        assert args.proposal_root is not None
        source_manifests = _source_manifests(args.proposal_root)
        source = _build_chunk_source(
            args.proposal_root,
            allowed_physical_logs=None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            max_scenes=args.max_scenes,
        )
    entries = source.entries
    if args.token:
        if args.max_scenes:
            raise ValueError("--token and --max-scenes are mutually exclusive")
        wanted_tokens = {str(value) for value in args.token}
        if len(wanted_tokens) != len(args.token):
            raise ValueError("--token values must be unique")
        entries = [entry for entry in entries if entry.token in wanted_tokens]
        missing_explicit = sorted(
            wanted_tokens.difference(entry.token for entry in entries)
        )
        if missing_explicit:
            raise RuntimeError(
                f"Proposal source lacks explicit tokens: {missing_explicit}"
            )
    if not entries:
        raise RuntimeError("No current-observation entries selected")

    shard_dir = args.output_dir / (
        f"all_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = shard_dir / "lineage.json"
    if args.cache_dtype == "auto":
        if lineage_path.exists():
            persisted_precision = str(
                json.loads(lineage_path.read_text()).get("precision", "")
            )
            if persisted_precision.endswith("float16_cache"):
                args.cache_dtype = "float16"
            elif persisted_precision.endswith("float32_cache"):
                args.cache_dtype = "float32"
            else:
                raise RuntimeError(
                    "Cannot infer resumable register-cache dtype from "
                    f"{lineage_path}: {persisted_precision}"
                )
        else:
            args.cache_dtype = "float32" if args.split == "navtest" else "float16"
    lineage = _lineage(args, source_manifests)
    if lineage_path.exists():
        existing = json.loads(lineage_path.read_text())
        comparable_existing = dict(existing)
        comparable_current = dict(lineage)
        comparable_existing.pop("created_utc", None)
        comparable_current.pop("created_utc", None)
        if comparable_existing != comparable_current:
            raise RuntimeError(f"Partial output lineage mismatch: {lineage_path}")
        lineage = existing
    else:
        _atomic_json_dump(lineage, lineage_path)

    completed = _load_completed_tokens(shard_dir)
    entry_tokens = {entry.token for entry in entries}
    if not completed.issubset(entry_tokens):
        raise RuntimeError("Partial cache has tokens outside selected shard")
    pending = [entry for entry in entries if entry.token not in completed]

    unique_logs = sorted({entry.log_name for entry in entries})
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        log_names=unique_logs,
        tokens=[entry.token for entry in entries],
    )
    sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[3],
        lidar_pc=[],
    )
    loader = SceneLoader(args.log_path, args.sensor_root, scene_filter, sensor_config)
    missing = sorted(entry_tokens.difference(loader.tokens))
    if missing:
        raise RuntimeError(f"SceneLoader is missing {len(missing)} tokens")

    model, config = _load_model(
        args.config, args.checkpoint, args.dino_weights, device
    )
    cache_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
    }[args.cache_dtype]
    builder = DrivoRFeatureBuilder(config)
    buffer: Dict[str, List[object]] = {
        "tokens": [],
        "log_names": [],
        "visual_tokens": [],
        "visual_valid_mask": [],
        "status_feature": [],
    }
    chunk_index = len(list(shard_dir.glob("chunk_*.pt")))
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_entries: List[ProposalEntry] = pending[start : start + args.batch_size]
            features = _batch_features(batch_entries, loader, builder, device)
            scene_features, _ego_token = _encode_context(model, features)
            if scene_features.ndim != 3 or scene_features.shape[-1] != 256:
                raise RuntimeError(
                    f"Unexpected DrivOR register shape {tuple(scene_features.shape)}"
                )
            buffer["tokens"].extend(entry.token for entry in batch_entries)
            buffer["log_names"].extend(entry.log_name for entry in batch_entries)
            buffer["visual_tokens"].append(scene_features.float().cpu())
            buffer["visual_valid_mask"].append(
                torch.ones(scene_features.shape[:2], dtype=torch.bool)
            )
            buffer["status_feature"].append(features["ego_status"][:, -1].float().cpu())
            if len(buffer["tokens"]) >= args.chunk_size:
                chunk_index = _flush(
                    buffer, shard_dir, chunk_index, cache_dtype
                )
            print(
                json.dumps(
                    {
                        "processed": min(start + len(batch_entries), len(pending)),
                        "pending_total": len(pending),
                        "already_complete": len(completed),
                        "shard": f"{args.shard_index}/{args.shard_count}",
                    }
                ),
                flush=True,
            )
    chunk_index = _flush(buffer, shard_dir, chunk_index, cache_dtype)

    completed_after = _load_completed_tokens(shard_dir)
    if completed_after != entry_tokens:
        raise RuntimeError(
            f"Incomplete cache: {len(completed_after)} / {len(entry_tokens)}"
        )
    manifest = dict(lineage) | {
        "scene_count": len(entry_tokens),
        "log_count": len(unique_logs),
        "chunk_count": chunk_index,
        "visual_width": 256,
        "visual_token_count": 64,
        "status_width": 11,
        "current_context_fields": ["ego_status"],
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
