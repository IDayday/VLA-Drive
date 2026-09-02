#!/usr/bin/env python3
"""Verify No-VQA scorer feature/label cache identity and row alignment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
TARGET_FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _source_manifests(root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    paths = sorted(root.glob("all_shard_*-of-*/manifest.json"))
    if not paths:
        raise RuntimeError(f"No completed source manifests under {root}")
    return [(path, json.loads(path.read_text())) for path in paths]


def _validate_manifest_set(
    manifests: List[Tuple[Path, Dict[str, object]]],
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> Dict[str, object]:
    shard_counts = {int(payload["shard_count"]) for _, payload in manifests}
    if len(shard_counts) != 1:
        raise RuntimeError(f"Source manifests disagree on shard count: {shard_counts}")
    shard_count = shard_counts.pop()
    indices = {int(payload["shard_index"]) for _, payload in manifests}
    if indices != set(range(shard_count)):
        raise RuntimeError(f"Incomplete source shards: {sorted(indices)} / {shard_count}")
    for path, payload in manifests:
        if payload.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise RuntimeError(f"Checkpoint SHA mismatch in {path}")
        if payload.get("resolved_config_sha256") != expected_config_sha256:
            raise RuntimeError(f"Resolved-config SHA mismatch in {path}")
        if payload.get("split") != "all":
            raise RuntimeError(f"Source shard is not split=all: {path}")
        if not bool(payload.get("inference_inputs_only")):
            raise RuntimeError(f"Inference-only declaration missing: {path}")
        if bool(payload.get("official_score_present")):
            raise RuntimeError(f"Official score leaked into source cache: {path}")
        if bool(payload.get("future_target_present")):
            raise RuntimeError(f"Future target leaked into source cache: {path}")
        if tuple(payload.get("factor_keys", ())) != FACTOR_KEYS:
            raise RuntimeError(f"Factor schema mismatch: {path}")
        config = payload.get("config", {})
        if config.get("lora_config", {}).get("use_lora") is not False:
            raise RuntimeError(f"No-VQA source unexpectedly enables LoRA: {path}")
        action = config.get("action_head_config", {})
        if action.get("return_scorer_features") is not True:
            raise RuntimeError(
                f"No-VQA source does not export scorer candidate features: {path}"
            )
        if action.get("return_memory_fields") is not True:
            raise RuntimeError(
                f"No-VQA source does not export current scene/ego features: {path}"
            )
        vlm = config.get("vlm_config", {})
        expected = {
            "freeze_backbone": True,
            "use_flash_attn": False,
            "gradient_checkpointing": False,
            "frozen_backbone_mode": "eval",
        }
        for key, value in expected.items():
            if vlm.get(key) != value:
                raise RuntimeError(f"Unexpected vlm_config.{key} in {path}")
    return {
        "shard_count": shard_count,
        "declared_scene_count": sum(
            int(payload["scene_count"]) for _, payload in manifests
        ),
        "declared_log_counts_by_shard": [
            int(payload["log_count"]) for _, payload in manifests
        ],
    }


def _iter_chunks(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("all_shard_*-of-*/chunk_*.pt"))


def verify(args: argparse.Namespace) -> Dict[str, object]:
    manifests = _source_manifests(args.source_root)
    manifest_summary = _validate_manifest_set(
        manifests,
        args.expected_checkpoint_sha256,
        args.expected_config_sha256,
    )
    source_paths = list(_iter_chunks(args.source_root))
    if not source_paths:
        raise RuntimeError("No source chunks found")
    label_relatives = {
        str(path.relative_to(args.label_root))
        for path in args.label_root.glob("all_shard_*-of-*/chunk_*.pt")
    }
    source_relatives = {
        str(path.relative_to(args.source_root)) for path in source_paths
    }
    if label_relatives != source_relatives:
        missing = sorted(source_relatives - label_relatives)
        extra = sorted(label_relatives - source_relatives)
        raise RuntimeError(
            f"Feature/label chunk mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    worker_manifests = sorted(args.label_root.glob("worker_manifest_*-of-*.json"))
    if not worker_manifests:
        raise RuntimeError("No label worker manifests found")
    for path in worker_manifests:
        payload = json.loads(path.read_text())
        if not payload.get("source_complete") or not payload.get("worker_complete"):
            raise RuntimeError(f"Incomplete label worker: {path}")
        if int(payload.get("failed_chunk_count", -1)) != 0:
            raise RuntimeError(f"Failed label chunks recorded by {path}")
        if not bool(payload.get("offline_training_labels_only")):
            raise RuntimeError(f"Label-only declaration missing: {path}")

    seen: set[str] = set()
    log_counts: Counter[str] = Counter()
    scene_count = 0
    candidate_count = None
    for source_path in source_paths:
        relative = source_path.relative_to(args.source_root)
        label_path = args.label_root / relative
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        labels = torch.load(label_path, map_location="cpu", weights_only=False)
        tokens = [str(value) for value in source["tokens"]]
        if tokens != [str(value) for value in labels["tokens"]]:
            raise RuntimeError(f"Token order mismatch: {relative}")
        if source["log_names"] != labels["log_names"]:
            raise RuntimeError(f"Log order mismatch: {relative}")
        duplicate = seen.intersection(tokens)
        if duplicate:
            raise RuntimeError(f"Duplicate scene tokens: {sorted(duplicate)[:5]}")
        seen.update(tokens)
        count = len(tokens)
        proposals = source["proposals"]
        base_scores = source["base_scores"]
        factor_logits = source["factor_logits"]
        scene_features = source["scene_features"]
        ego_features = source["ego_features"]
        selected_indices = source["selected_indices"]
        target_factors = labels["target_factors"]
        valid_mask = labels["valid_mask"].bool()
        if proposals.ndim != 4 or proposals.shape[0] != count:
            raise RuntimeError(f"Invalid proposal shape: {relative} {proposals.shape}")
        if tuple(proposals.shape[2:]) != (8, 3):
            raise RuntimeError(f"Invalid proposal horizon: {relative}")
        candidates = int(proposals.shape[1])
        candidate_count = candidates if candidate_count is None else candidate_count
        if candidates != candidate_count or candidates != 64:
            raise RuntimeError(f"Expected exactly 64 candidates: {relative}")
        expected_shapes = {
            "base_scores": (count, candidates),
            "factor_logits": (count, candidates, len(FACTOR_KEYS)),
            "candidate_features": (count, candidates, 256),
            "scene_features": (count, 16, 256),
            "ego_features": (count, 1, 256),
            "selected_indices": (count,),
            "target_factors": (count, candidates, len(TARGET_FACTOR_KEYS)),
            "valid_mask": (count,),
        }
        actual = {
            "base_scores": tuple(base_scores.shape),
            "factor_logits": tuple(factor_logits.shape),
            "candidate_features": tuple(source["candidate_features"].shape),
            "scene_features": tuple(scene_features.shape),
            "ego_features": tuple(ego_features.shape),
            "selected_indices": tuple(selected_indices.shape),
            "target_factors": tuple(target_factors.shape),
            "valid_mask": tuple(valid_mask.shape),
        }
        for key, shape in expected_shapes.items():
            if actual[key] != shape:
                raise RuntimeError(
                    f"Invalid {key} shape in {relative}: {actual[key]} != {shape}"
                )
        if tuple(source["factor_keys"]) != FACTOR_KEYS:
            raise RuntimeError(f"Source factor order mismatch: {relative}")
        if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_KEYS:
            raise RuntimeError(f"Target factor order mismatch: {relative}")
        tensors = (
            proposals,
            base_scores,
            factor_logits,
            source["candidate_features"],
            scene_features,
            ego_features,
            target_factors,
        )
        if not all(bool(torch.isfinite(value.float()).all()) for value in tensors):
            raise RuntimeError(f"Non-finite cache tensor: {relative}")
        if not bool(valid_mask.all()):
            raise RuntimeError(f"Invalid PDM rows remain: {relative}")
        if not torch.equal(selected_indices.long(), base_scores.argmax(dim=1)):
            raise RuntimeError(f"Cached Base selection mismatch: {relative}")
        for log_name in source["log_names"]:
            log_counts[str(log_name)] += 1
        scene_count += count

    if scene_count != int(manifest_summary["declared_scene_count"]):
        raise RuntimeError("Loaded scene count differs from source manifests")
    if args.expected_scenes and scene_count != args.expected_scenes:
        raise RuntimeError(
            f"Expected {args.expected_scenes} scenes, verified {scene_count}"
        )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_root": str(args.source_root.resolve()),
        "label_root": str(args.label_root.resolve()),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "resolved_config_sha256": args.expected_config_sha256,
        "scene_count": scene_count,
        "unique_scene_count": len(seen),
        "source_log_name_count": len(log_counts),
        "candidate_count": candidate_count,
        "source_chunk_count": len(source_paths),
        "label_worker_count": len(worker_manifests),
        "invalid_scene_count": 0,
        "future_or_official_input_present": False,
        **manifest_summary,
    }


def _markdown(result: Dict[str, object]) -> str:
    return "\n".join(
        (
            "# No-VQA scorer cache verification",
            "",
            f"Status: **{result['status']}**",
            "",
            f"- Scenes: `{result['scene_count']}` unique `{result['unique_scene_count']}`",
            f"- Candidate count: `{result['candidate_count']}`",
            f"- Source shards/chunks: `{result['shard_count']}` / `{result['source_chunk_count']}`",
            f"- Label workers: `{result['label_worker_count']}`",
            f"- Invalid scenes: `{result['invalid_scene_count']}`",
            f"- Future/evaluator fields in inference cache: `{result['future_or_official_input_present']}`",
            f"- Checkpoint SHA256: `{result['checkpoint_sha256']}`",
            f"- Resolved-config SHA256: `{result['resolved_config_sha256']}`",
            "",
            "The source and label trees have identical relative chunk sets, token/log",
            "order, 64-candidate geometry, and row counts. All tensors are finite, all",
            "offline PDM rows are valid, and cached Base indices equal `argmax(base_scores)`.",
            "",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-scenes", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args)
    _atomic_text(args.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.output_md, _markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
