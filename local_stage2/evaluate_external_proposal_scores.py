#!/usr/bin/env python3
"""Evaluate an inference-only external scorer on immutable replay proposals.

The scorer output is loaded first and candidate selections are fixed before
offline PDM labels are joined.  This preserves the inference boundary: future
state and official evaluator outputs are never available to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


EXPECTED_CANDIDATES = 64
EXTERNAL_FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
TARGET_FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
TARGET_TO_EXTERNAL_ORDER = (0, 1, 5, 3, 2, 4)
_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")


def _physical_log_name(log_name: str) -> str:
    return _SEGMENT_SUFFIX.sub("", str(log_name))


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _comparable_lineage(payload: Mapping[str, object]) -> Dict[str, object]:
    value = dict(payload)
    value.pop("created_utc", None)
    value.pop("shard_index", None)
    return value


def _pairwise_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    minimum_delta: float,
    chunk_size: int = 512,
) -> float:
    left, right = np.triu_indices(predictions.shape[1], k=1)
    correct = 0
    total = 0
    for start in range(0, len(predictions), chunk_size):
        pred_delta = (
            predictions[start : start + chunk_size, left]
            - predictions[start : start + chunk_size, right]
        )
        target_delta = (
            targets[start : start + chunk_size, left]
            - targets[start : start + chunk_size, right]
        )
        valid = np.abs(target_delta) >= minimum_delta
        correct += int(np.sum((np.sign(pred_delta) == np.sign(target_delta)) & valid))
        total += int(valid.sum())
    return float(correct / max(total, 1))


def _cluster_bootstrap(
    delta: np.ndarray,
    physical_logs: Sequence[str],
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    names = np.asarray(physical_logs, dtype=str)
    unique, inverse = np.unique(names, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=delta.astype(np.float64))
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = generator.integers(0, len(unique), size=len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _load_score_outputs(
    root: Path,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    manifests = sorted(root.glob("shard_*-of-*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed scorer manifests under {root}")
    manifest_payloads = [json.loads(path.read_text()) for path in manifests]
    declared_shard_counts = {
        int(payload["lineage"]["shard_count"]) for payload in manifest_payloads
    }
    if len(declared_shard_counts) != 1:
        raise RuntimeError("External scorer manifests disagree on shard count")
    shard_count = next(iter(declared_shard_counts))
    shard_indices = {
        int(payload["lineage"]["shard_index"]) for payload in manifest_payloads
    }
    if len(manifests) != shard_count or shard_indices != set(range(shard_count)):
        raise RuntimeError(
            f"Incomplete scorer shards: manifests={len(manifests)}, declared={shard_count}"
        )
    reference_lineage = _comparable_lineage(manifest_payloads[0]["lineage"])
    for payload in manifest_payloads:
        if _comparable_lineage(payload["lineage"]) != reference_lineage:
            raise RuntimeError("External scorer shard lineage mismatch")
        if int(payload.get("invalid_scene_count", -1)) != 0:
            raise RuntimeError("External scorer manifest contains invalid scenes")
        parity = payload.get("self_parity", {})
        if not parity.get("all_passed") or int(parity.get("evaluated_scene_count", 0)) < 1:
            raise RuntimeError("External scorer self-parity gate did not pass")

    tokens: List[str] = []
    log_names: List[str] = []
    scores: List[np.ndarray] = []
    factors: List[np.ndarray] = []
    selected: List[np.ndarray] = []
    for chunk_path in sorted(root.glob("shard_*-of-*/chunk_*.pt")):
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if tuple(payload["factor_keys"]) != EXTERNAL_FACTOR_KEYS:
            raise RuntimeError(f"Unexpected factor order in {chunk_path}")
        chunk_tokens = [str(value) for value in payload["tokens"]]
        chunk_logs = [str(value) for value in payload["log_names"]]
        chunk_scores = payload["scores"].float().numpy()
        chunk_factors = payload["factor_logits"].float().numpy()
        chunk_selected = payload["selected_indices"].long().numpy()
        expected_scores = (len(chunk_tokens), EXPECTED_CANDIDATES)
        expected_factors = expected_scores + (len(EXTERNAL_FACTOR_KEYS),)
        if chunk_scores.shape != expected_scores or chunk_factors.shape != expected_factors:
            raise RuntimeError(f"Malformed external scorer chunk: {chunk_path}")
        if len(chunk_logs) != len(chunk_tokens) or chunk_selected.shape != (len(chunk_tokens),):
            raise RuntimeError(f"External scorer row mismatch: {chunk_path}")
        if not np.isfinite(chunk_scores).all() or not np.isfinite(chunk_factors).all():
            raise RuntimeError(f"Non-finite external scorer output: {chunk_path}")
        if not np.array_equal(chunk_selected, chunk_scores.argmax(axis=1)):
            raise RuntimeError(f"Saved external selections disagree with scores: {chunk_path}")
        tokens.extend(chunk_tokens)
        log_names.extend(chunk_logs)
        scores.append(chunk_scores)
        factors.append(chunk_factors)
        selected.append(chunk_selected)

    if len(tokens) != len(set(tokens)):
        raise RuntimeError("Duplicate token in external scorer output")
    declared_scenes = sum(int(payload["scene_count"]) for payload in manifest_payloads)
    if declared_scenes != len(tokens):
        raise RuntimeError(f"Manifest/output scene mismatch: {declared_scenes} != {len(tokens)}")
    order = np.argsort(np.asarray(tokens, dtype=str))
    return (
        np.asarray(tokens, dtype=str)[order].tolist(),
        np.asarray(log_names, dtype=str)[order].tolist(),
        np.concatenate(scores, axis=0)[order],
        np.concatenate(factors, axis=0)[order],
        np.concatenate(selected, axis=0)[order],
        {
            "root": str(root.resolve()),
            "manifest_sha256": {
                str(path.relative_to(root)): _sha256(path) for path in manifests
            },
            "lineage": manifest_payloads[0]["lineage"],
        },
    )


def _joined_replay_chunks(
    feature_root: Path, label_root: Path
) -> Iterable[Tuple[Path, Path]]:
    feature_paths = sorted(feature_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not feature_paths:
        raise FileNotFoundError(f"No replay chunks under {feature_root}")
    for feature_path in feature_paths:
        label_path = label_root / feature_path.relative_to(feature_root)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        yield feature_path, label_path


def _load_offline_targets(
    feature_root: Path,
    label_root: Path,
    requested_tokens: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    row_for_token = {token: index for index, token in enumerate(requested_tokens)}
    base_scores = np.empty((len(requested_tokens), EXPECTED_CANDIDATES), dtype=np.float32)
    target_factors = np.empty(
        (len(requested_tokens), EXPECTED_CANDIDATES, len(TARGET_FACTOR_KEYS)),
        dtype=np.float32,
    )
    log_names: List[str | None] = [None] * len(requested_tokens)
    found: set[str] = set()
    for feature_path, label_path in _joined_replay_chunks(feature_root, label_root):
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        labels = torch.load(label_path, map_location="cpu", weights_only=False)
        feature_tokens = [str(value) for value in features["tokens"]]
        if feature_tokens != [str(value) for value in labels["tokens"]]:
            raise RuntimeError(f"Feature/label token mismatch: {feature_path}")
        if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_KEYS:
            raise RuntimeError(f"Unexpected target factor order: {label_path}")
        feature_logs = [str(value) for value in features["log_names"]]
        if feature_logs != [str(value) for value in labels["log_names"]]:
            raise RuntimeError(f"Feature/label log mismatch: {feature_path}")
        valid = labels["valid_mask"].bool()
        for source_row, token in enumerate(feature_tokens):
            target_row = row_for_token.get(token)
            if target_row is None:
                continue
            if token in found:
                raise RuntimeError(f"Duplicate replay token: {token}")
            if not bool(valid[source_row]):
                raise RuntimeError(f"Requested replay token has invalid PDM labels: {token}")
            found.add(token)
            base_scores[target_row] = features["base_scores"][source_row].float().numpy()
            target_factors[target_row] = labels["target_factors"][source_row].float().numpy()
            log_names[target_row] = feature_logs[source_row]
    missing = sorted(set(requested_tokens).difference(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} offline target rows, e.g. {missing[:5]}")
    if not np.isfinite(base_scores).all() or not np.isfinite(target_factors).all():
        raise RuntimeError("Non-finite replay targets")
    return (
        base_scores,
        target_factors,
        [str(value) for value in log_names],
        {
            "feature_root": str(feature_root.resolve()),
            "label_root": str(label_root.resolve()),
            "feature_manifest_sha256": {
                str(path.relative_to(feature_root)): _sha256(path)
                for path in sorted(feature_root.glob("*_shard_*-of-*/manifest.json"))
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.score_root, args.feature_root, args.label_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    tokens, scorer_logs, scores, factor_logits, selected, score_lineage = (
        _load_score_outputs(args.score_root)
    )
    base_scores, target_factors, replay_logs, replay_lineage = _load_offline_targets(
        args.feature_root, args.label_root, tokens
    )
    if scorer_logs != replay_logs:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(scorer_logs, replay_logs))
            if left != right
        )
        raise RuntimeError(
            f"Scorer/replay log mismatch for {tokens[mismatch]}: "
            f"{scorer_logs[mismatch]} != {replay_logs[mismatch]}"
        )
    physical_logs = [_physical_log_name(value) for value in replay_logs]
    if args.fold_manifest is not None:
        fold = json.loads(args.fold_manifest.read_text())
        expected_logs = {str(value) for value in fold["validation_physical_logs"]}
        actual_logs = set(physical_logs)
        if actual_logs != expected_logs:
            raise RuntimeError(
                "Held-out physical-log mismatch: "
                f"missing={sorted(expected_logs - actual_logs)[:5]}, "
                f"extra={sorted(actual_logs - expected_logs)[:5]}"
            )
        expected_scenes = int(fold["validation_scene_count"])
        if len(tokens) != expected_scenes:
            raise RuntimeError(f"Held-out scene mismatch: {len(tokens)} != {expected_scenes}")

    rows = np.arange(len(tokens))
    target_pdms = target_factors[..., -1]
    base_selected = base_scores.argmax(axis=1)
    oracle = target_pdms.argmax(axis=1)
    selected_pdms = target_pdms[rows, selected]
    base_pdms = target_pdms[rows, base_selected]
    oracle_pdms = target_pdms[rows, oracle]
    delta = selected_pdms - base_pdms
    bootstrap = _cluster_bootstrap(
        delta,
        physical_logs,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    target_six = target_factors[..., list(TARGET_TO_EXTERNAL_ORDER)]
    selected_factors = target_six[rows, selected]
    base_factors = target_six[rows, base_selected]
    sorted_indices = np.argsort(-scores, axis=1)
    wins = int(np.sum(delta > 1e-9))
    losses = int(np.sum(delta < -1e-9))
    metrics: Dict[str, object] = {
        "selected_pdms": float(selected_pdms.mean()),
        "base_selected_pdms": float(base_pdms.mean()),
        "offline_oracle_candidate_bank_upper_bound": float(oracle_pdms.mean()),
        "selected_delta": float(delta.mean()),
        "selected_regret": float((oracle_pdms - selected_pdms).mean()),
        "base_regret": float((oracle_pdms - base_pdms).mean()),
        "switch_rate": float(np.mean(selected != base_selected)),
        "wins": wins,
        "losses": losses,
        "ties": int(len(tokens) - wins - losses),
        "pairwise_accuracy_all_non_ties": _pairwise_accuracy(scores, target_pdms, 1e-9),
        "pairwise_accuracy_delta_002": _pairwise_accuracy(scores, target_pdms, 0.02),
        "pairwise_accuracy_delta_005": _pairwise_accuracy(scores, target_pdms, 0.05),
        "pairwise_accuracy_delta_010": _pairwise_accuracy(scores, target_pdms, 0.10),
        "oracle_recall_at_1": float(np.mean(sorted_indices[:, :1] == oracle[:, None])),
        "oracle_recall_at_2": float(
            np.mean(np.any(sorted_indices[:, :2] == oracle[:, None], axis=1))
        ),
        "oracle_recall_at_4": float(
            np.mean(np.any(sorted_indices[:, :4] == oracle[:, None], axis=1))
        ),
        "selected_factors": {
            key: float(selected_factors[:, index].mean())
            for index, key in enumerate(EXTERNAL_FACTOR_KEYS)
        },
        "base_selected_factors": {
            key: float(base_factors[:, index].mean())
            for index, key in enumerate(EXTERNAL_FACTOR_KEYS)
        },
        "safe_to_unsafe_switches": {
            key: int(
                np.sum(
                    (base_factors[:, index] >= 1.0 - 1e-9)
                    & (selected_factors[:, index] < 1.0 - 1e-9)
                )
            )
            for index, key in enumerate(EXTERNAL_FACTOR_KEYS)
        },
    }
    promotion_pass = bool(delta.mean() > 0 and bootstrap["ci95_low"] > 0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "token": tokens,
            "log_name": replay_logs,
            "physical_log_name": physical_logs,
            "base_index": base_selected,
            "selected_index": selected,
            "oracle_index": oracle,
            "base_pdms": base_pdms,
            "selected_pdms": selected_pdms,
            "offline_oracle_best_of_64": oracle_pdms,
            "pdms_delta": delta,
        }
    ).to_csv(args.output_dir / "per_scene.csv", index=False)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(tokens),
        "segment_log_count": len(set(replay_logs)),
        "physical_log_count": len(set(physical_logs)),
        "candidate_count": int(scores.shape[1]),
        "invalid_scene_count": 0,
        "inference_inputs_only": True,
        "future_or_evaluator_input": False,
        "official_pdm_joined_only_after_selection": True,
        "score_lineage": score_lineage,
        "replay_lineage": replay_lineage,
        "fold_manifest": (
            {
                "path": str(args.fold_manifest.resolve()),
                "sha256": _sha256(args.fold_manifest),
            }
            if args.fold_manifest is not None
            else None
        ),
        "metrics": metrics,
        "pdms_delta_log_cluster_bootstrap": bootstrap,
        "promotion_gate": {
            "rule": "mean_delta>0 and physical-log bootstrap ci95_low>0",
            "passed": promotion_pass,
        },
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
