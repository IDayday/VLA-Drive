#!/usr/bin/env python3
"""Build deterministic risk-stratified physical-log folds from Navtrain only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from local_stage2.train_independent_scorer import physical_log_name


_STAT_KEYS = (
    "scene_count",
    "risk_scene_count",
    "unsafe_candidate_count",
    "score_span_sum",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assign_risk_stratified_log_folds(
    log_stats: Mapping[str, Mapping[str, float]],
    num_folds: int,
    seed: int,
) -> Dict[str, int]:
    """Greedily balance scene volume and safety difficulty without log leaks."""

    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    if len(log_stats) < num_folds:
        raise ValueError("fewer physical logs than folds")
    vectors = {
        str(name): np.asarray(
            [float(values[key]) for key in _STAT_KEYS], dtype=np.float64
        )
        for name, values in log_stats.items()
    }
    if any(vector[0] <= 0 for vector in vectors.values()):
        raise ValueError("every physical log must contain at least one scene")
    total = np.stack(list(vectors.values())).sum(axis=0)
    target = np.maximum(total / float(num_folds), 1.0e-9)
    rng = random.Random(seed)
    tie_break = {name: rng.random() for name in sorted(vectors)}
    ordered = sorted(
        vectors,
        key=lambda name: (
            -float(np.linalg.norm(vectors[name] / target)),
            -vectors[name][0],
            tie_break[name],
            name,
        ),
    )
    fold_vectors = np.zeros((num_folds, len(_STAT_KEYS)), dtype=np.float64)
    fold_log_counts = np.zeros(num_folds, dtype=np.float64)
    target_logs = len(vectors) / float(num_folds)
    assignment: Dict[str, int] = {}
    for name in ordered:
        objectives = []
        for fold in range(num_folds):
            candidate_vectors = fold_vectors.copy()
            candidate_vectors[fold] += vectors[name]
            candidate_counts = fold_log_counts.copy()
            candidate_counts[fold] += 1.0
            vector_error = np.square(
                (candidate_vectors - target[None, :]) / target[None, :]
            ).sum()
            log_error = np.square(
                (candidate_counts - target_logs) / max(target_logs, 1.0)
            ).sum()
            objectives.append((float(vector_error + 0.1 * log_error), fold))
        _, selected_fold = min(objectives)
        assignment[name] = selected_fold
        fold_vectors[selected_fold] += vectors[name]
        fold_log_counts[selected_fold] += 1.0
    if set(assignment) != set(vectors):
        raise RuntimeError("fold assignment lost physical logs")
    if len(set(assignment.values())) != num_folds:
        raise RuntimeError("fold assignment produced an empty fold")
    return assignment


def _combined_manifest_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def scan_label_stats(label_root: Path) -> tuple[Dict[str, Dict[str, float]], int]:
    chunks = sorted(label_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not chunks:
        raise RuntimeError(f"no label chunks found under {label_root}")
    mutable: MutableMapping[str, MutableMapping[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    seen_tokens: set[str] = set()
    valid_scene_count = 0
    for path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tokens = [str(value) for value in payload["tokens"]]
        logs = [physical_log_name(str(value)) for value in payload["log_names"]]
        factors = payload["target_factors"].float()
        valid = payload["valid_mask"].bool()
        if factors.shape != (len(tokens), 64, 7):
            raise RuntimeError(f"unexpected target shape in {path}: {factors.shape}")
        if len(logs) != len(tokens) or valid.shape != (len(tokens),):
            raise RuntimeError(f"label metadata shape mismatch in {path}")
        duplicate = seen_tokens.intersection(tokens)
        if duplicate:
            raise RuntimeError(f"duplicate label tokens: {sorted(duplicate)[:5]}")
        seen_tokens.update(tokens)
        unsafe = (factors[..., (0, 1, 3)] < 1.0 - 1.0e-6).any(dim=-1)
        score_span = factors[..., -1].amax(dim=1) - factors[..., -1].amin(dim=1)
        for row, (log_name, is_valid) in enumerate(zip(logs, valid.tolist())):
            if not is_valid:
                continue
            stats = mutable[log_name]
            stats["scene_count"] += 1.0
            stats["risk_scene_count"] += float(unsafe[row].any())
            stats["unsafe_candidate_count"] += float(unsafe[row].sum())
            stats["score_span_sum"] += float(score_span[row])
            valid_scene_count += 1
    result = {
        name: {key: float(values[key]) for key in _STAT_KEYS}
        for name, values in mutable.items()
    }
    if sum(int(value["scene_count"]) for value in result.values()) != valid_scene_count:
        raise RuntimeError("per-log scene counts do not reconstruct scanned labels")
    return result, valid_scene_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--available-split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.label_root.is_dir():
        raise FileNotFoundError(args.label_root)
    if not args.available_split_manifest.is_file():
        raise FileNotFoundError(args.available_split_manifest)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    available_payload = json.loads(args.available_split_manifest.read_text())
    available_logs = {
        str(value)
        for key in ("train_physical_logs", "validation_physical_logs")
        for value in available_payload[key]
    }
    log_stats, scene_count = scan_label_stats(args.label_root)
    if set(log_stats) != available_logs:
        missing = available_logs.difference(log_stats)
        extra = set(log_stats).difference(available_logs)
        raise RuntimeError(
            f"label/split physical-log mismatch: missing={len(missing)} extra={len(extra)}"
        )
    assignment = assign_risk_stratified_log_folds(
        log_stats, args.num_folds, args.seed
    )
    manifest_paths = sorted(args.label_root.glob("worker_manifest_*.json"))
    if not manifest_paths:
        raise RuntimeError("label cache has no worker manifests")
    common = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "risk_stratified_disjoint_physical_log_fivefold",
        "seed": args.seed,
        "num_folds": args.num_folds,
        "label_root": str(args.label_root.resolve()),
        "label_worker_manifest_sha256": _combined_manifest_sha256(manifest_paths),
        "available_split_manifest": str(args.available_split_manifest.resolve()),
        "available_split_manifest_sha256": _sha256(args.available_split_manifest),
        "scene_count": scene_count,
        "physical_log_count": len(log_stats),
        "navtest_used": False,
        "future_or_evaluator_input_to_model": False,
    }
    fold_summaries = []
    all_logs = set(log_stats)
    for fold in range(args.num_folds):
        validation = sorted(name for name, value in assignment.items() if value == fold)
        train = sorted(all_logs.difference(validation))
        validation_stats = {
            key: float(sum(log_stats[name][key] for name in validation))
            for key in _STAT_KEYS
        }
        train_stats = {
            key: float(sum(log_stats[name][key] for name in train))
            for key in _STAT_KEYS
        }
        payload = common | {
            "fold_index": fold,
            "train_physical_logs": train,
            "validation_physical_logs": validation,
            "physical_log_overlap": [],
            "train_stats": train_stats,
            "validation_stats": validation_stats,
        }
        output = args.output_dir / f"fold_{fold}.json"
        _atomic_json(output, payload)
        fold_summaries.append(
            {
                "fold_index": fold,
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "train_log_count": len(train),
                "validation_log_count": len(validation),
                "train_stats": train_stats,
                "validation_stats": validation_stats,
            }
        )
    validation_sets = [
        {name for name, value in assignment.items() if value == fold}
        for fold in range(args.num_folds)
    ]
    if set().union(*validation_sets) != all_logs:
        raise RuntimeError("validation folds do not cover every physical log")
    for left in range(args.num_folds):
        for right in range(left + 1, args.num_folds):
            if validation_sets[left].intersection(validation_sets[right]):
                raise RuntimeError("validation folds overlap")
    index = common | {
        "folds": fold_summaries,
        "validation_logs_cover_all_available_logs_once": True,
        "per_log_stats": log_stats,
    }
    _atomic_json(args.output_dir / "index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
