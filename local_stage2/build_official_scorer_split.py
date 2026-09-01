#!/usr/bin/env python3
"""Build an auditable official train/validation physical-log split manifest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import torch

from local_stage2.export_public_base_scorer_cache import _compose_agent_config, _sha256
from local_stage2.train_independent_scorer import physical_log_name


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--available-fold-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.repo_root,
        args.checkpoint,
        args.available_fold_manifest,
        args.feature_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    fold = json.loads(args.available_fold_manifest.read_text())
    available = {
        str(value)
        for key in ("train_physical_logs", "validation_physical_logs")
        for value in fold[key]
    }
    cfg = _compose_agent_config(args.repo_root.resolve(), args.checkpoint.resolve())
    official_train = {physical_log_name(str(value)) for value in cfg.train_logs}
    official_validation = {physical_log_name(str(value)) for value in cfg.val_logs}
    overlap = official_train.intersection(official_validation)
    if overlap:
        raise RuntimeError(f"official train/validation physical-log overlap: {sorted(overlap)[:5]}")
    unknown = available.difference(official_train.union(official_validation))
    if unknown:
        raise RuntimeError(f"{len(unknown)} replay physical logs are outside official split")
    train = sorted(available.intersection(official_train))
    validation = sorted(available.intersection(official_validation))
    if not train or not validation:
        raise RuntimeError("official replay train or validation split is empty")
    train_scene_count = 0
    validation_scene_count = 0
    seen_tokens: set[str] = set()
    for chunk_path in sorted(args.feature_root.glob("*_shard_*-of-*/chunk_*.pt")):
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        tokens = [str(value) for value in chunk["tokens"]]
        logs = [physical_log_name(str(value)) for value in chunk["log_names"]]
        if len(tokens) != len(logs):
            raise RuntimeError(f"token/log count mismatch in {chunk_path}")
        duplicate = seen_tokens.intersection(tokens)
        if duplicate:
            raise RuntimeError(f"duplicate replay tokens: {sorted(duplicate)[:5]}")
        seen_tokens.update(tokens)
        train_scene_count += sum(value in official_train for value in logs)
        validation_scene_count += sum(value in official_validation for value in logs)
    if train_scene_count + validation_scene_count != len(seen_tokens):
        raise RuntimeError("official split does not cover every replay scene")
    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "official_config_physical_log_boundary",
        "repo_root": str(args.repo_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "available_fold_manifest": str(args.available_fold_manifest.resolve()),
        "available_fold_manifest_sha256": _sha256(args.available_fold_manifest),
        "feature_root": str(args.feature_root.resolve()),
        "available_physical_log_count": len(available),
        "train_physical_log_count": len(train),
        "validation_physical_log_count": len(validation),
        "train_scene_count": train_scene_count,
        "validation_scene_count": validation_scene_count,
        "train_physical_logs": train,
        "validation_physical_logs": validation,
        "physical_log_overlap": [],
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
