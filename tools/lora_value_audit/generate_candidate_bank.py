#!/usr/bin/env python3
"""Build deterministic stage, structured, jitter, and diagnostic banks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .utils import atomic_json, load_proposal_pickle, sha256_file, token_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced-export-root", type=Path, required=True)
    parser.add_argument("--base-proposals", type=Path, required=True)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dedup-ade", type=float, default=0.05)
    parser.add_argument("--max-jump", type=float, default=20.0)
    parser.add_argument(
        "--banks",
        default="all",
        help="Comma-separated bank names, or 'all'. This permits a fast critical-path build.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _heading(xy: np.ndarray) -> np.ndarray:
    previous = np.vstack([np.zeros((1, 2), dtype=np.float64), xy[:-1]])
    delta = xy - previous
    values = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    small = np.linalg.norm(delta, axis=1) < 1e-5
    for index in np.flatnonzero(small):
        values[index] = values[index - 1] if index else 0.0
    return values


def _with_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.column_stack([xy, _heading(xy)]).astype(np.float32)


def _scale(trajectory: np.ndarray, scale: float) -> np.ndarray:
    return _with_xy(np.asarray(trajectory)[:, :2] * scale)


def _lateral(trajectory: np.ndarray, offset: float) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    alpha = np.linspace(1 / len(trajectory), 1.0, len(trajectory))
    normal = np.column_stack([-np.sin(trajectory[:, 2]), np.cos(trajectory[:, 2])])
    return _with_xy(trajectory[:, :2] + normal * (alpha * offset)[:, None])


def _tangent(trajectory: np.ndarray, offset: float) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    alpha = np.linspace(1 / len(trajectory), 1.0, len(trajectory))
    tangent = np.column_stack([np.cos(trajectory[:, 2]), np.sin(trajectory[:, 2])])
    return _with_xy(trajectory[:, :2] + tangent * (alpha * offset)[:, None])


def _interpolate(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    return _with_xy((1.0 - weight) * left[:, :2] + weight * right[:, :2])


def _hold(trajectory: np.ndarray, pose_number: int) -> np.ndarray:
    xy = np.asarray(trajectory, dtype=np.float64)[:, :2].copy()
    keep = max(1, min(len(xy), pose_number))
    xy[keep:] = xy[keep - 1]
    return _with_xy(xy)


def _smooth(trajectory: np.ndarray, strength: float = 1.0) -> np.ndarray:
    xy = np.asarray(trajectory, dtype=np.float64)[:, :2].copy()
    smooth = xy.copy()
    smooth[1:-1] = (xy[:-2] + 2.0 * xy[1:-1] + xy[2:]) / 4.0
    return _with_xy((1.0 - strength) * xy + strength * smooth)


def _valid(trajectory: np.ndarray, max_jump: float) -> bool:
    if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
        return False
    xy = trajectory[:, :2]
    previous = np.vstack([np.zeros((1, 2)), xy[:-1]])
    return bool(np.max(np.linalg.norm(xy - previous, axis=1)) <= max_jump and np.max(np.abs(xy)) <= 100.0)


def _append_unique(
    result: List[np.ndarray],
    candidate: np.ndarray,
    *,
    dedup_ade: float,
    max_jump: float,
) -> None:
    candidate = np.asarray(candidate, dtype=np.float32)
    if not _valid(candidate, max_jump):
        return
    if result:
        existing = np.stack(result)
        ade = np.linalg.norm(existing[:, :, :2] - candidate[None, :, :2], axis=-1).mean(axis=-1)
        if np.any(ade <= dedup_ade):
            return
    result.append(candidate)


def _fps(indices: Sequence[int], proposals: np.ndarray, count: int) -> List[int]:
    indices = list(map(int, indices))
    if not indices or count <= 0:
        return []
    selected = [indices[0]]
    xy = proposals[:, :, :2]
    while len(selected) < min(count, len(indices)):
        remaining = [index for index in indices if index not in selected]
        distances = []
        for index in remaining:
            nearest = min(
                float(np.linalg.norm(xy[index] - xy[chosen], axis=-1).mean())
                for chosen in selected
            )
            distances.append(nearest)
        selected.append(remaining[int(np.argmax(distances))])
    return selected


def _ensure_count(
    values: List[np.ndarray],
    anchor: np.ndarray,
    count: int,
    dedup_ade: float,
    max_jump: float,
) -> np.ndarray:
    # Deterministic, progressively larger sinusoidal normal offsets are only a
    # fallback when scene geometry collapses nominal transformations together.
    attempt = 1
    while len(values) < count and attempt <= 4096:
        phase = np.linspace(0.0, np.pi, len(anchor))
        amplitude = dedup_ade * (1.1 + attempt * 0.13)
        sign = -1.0 if attempt % 2 else 1.0
        normal = np.column_stack([-np.sin(anchor[:, 2]), np.cos(anchor[:, 2])])
        candidate = _with_xy(anchor[:, :2] + sign * amplitude * np.sin(phase)[:, None] * normal)
        _append_unique(values, candidate, dedup_ade=dedup_ade, max_jump=max_jump)
        attempt += 1
    if len(values) < count:
        raise RuntimeError(f"Could only construct {len(values)} / {count} unique candidates")
    return np.stack(values[:count]).astype(np.float32)


def _structured16(base: np.ndarray, predicted: np.ndarray, dedup: float, jump: float) -> np.ndarray:
    order = np.argsort(-predicted, kind="stable")
    selected, second, third = base[order[0]], base[order[1]], base[order[2]]
    candidates: List[np.ndarray] = []
    for scale in (0.65, 0.80, 0.90, 1.10, 1.20):
        _append_unique(candidates, _scale(selected, scale), dedup_ade=dedup, max_jump=jump)
    for offset in (-1.0, -0.5, 0.5, 1.0):
        _append_unique(candidates, _lateral(selected, offset), dedup_ade=dedup, max_jump=jump)
    for weight in (0.25, 0.50, 0.75):
        _append_unique(candidates, _interpolate(selected, second, weight), dedup_ade=dedup, max_jump=jump)
    _append_unique(candidates, _interpolate(selected, third, 0.5), dedup_ade=dedup, max_jump=jump)
    for pose in (4, 6):
        _append_unique(candidates, _hold(selected, pose), dedup_ade=dedup, max_jump=jump)
    _append_unique(candidates, _smooth(selected), dedup_ade=dedup, max_jump=jump)
    return _ensure_count(candidates, selected, 16, dedup, jump)


def _jitter8(base: np.ndarray, predicted: np.ndarray, dedup: float, jump: float) -> np.ndarray:
    selected = base[int(np.argmax(predicted))]
    values: List[np.ndarray] = []
    for offset in (-0.20, -0.10, -0.05, -0.025, 0.025, 0.05, 0.10, 0.20):
        _append_unique(values, _lateral(selected, offset), dedup_ade=min(dedup, 0.01), max_jump=jump)
    return _ensure_count(values, selected, 8, min(dedup, 0.01), jump)


def _large256(base: np.ndarray, predicted: np.ndarray, dedup: float, jump: float) -> np.ndarray:
    order = np.argsort(-predicted, kind="stable")
    anchors = list(order[:8]) + _fps(order[:32], base, 8)
    anchors = list(dict.fromkeys(map(int, anchors)))
    values: List[np.ndarray] = []
    for index in anchors:
        anchor = base[index]
        for scale in (0.65, 0.80, 0.90, 1.10, 1.20):
            _append_unique(values, _scale(anchor, scale), dedup_ade=dedup, max_jump=jump)
        for offset in (-1.0, -0.5, 0.5, 1.0):
            _append_unique(values, _lateral(anchor, offset), dedup_ade=dedup, max_jump=jump)
        for offset in (-1.5, -0.75, 0.75, 1.5):
            _append_unique(values, _tangent(anchor, offset), dedup_ade=dedup, max_jump=jump)
        for pose in (4, 6):
            _append_unique(values, _hold(anchor, pose), dedup_ade=dedup, max_jump=jump)
        _append_unique(values, _smooth(anchor), dedup_ade=dedup, max_jump=jump)
    selected = base[order[0]]
    for target_index in list(order[1:9]) + _fps(order[:32], base, 8):
        for weight in (0.25, 0.50, 0.75):
            _append_unique(values, _interpolate(selected, base[target_index], weight), dedup_ade=dedup, max_jump=jump)
    return _ensure_count(values, selected, 256, dedup, jump)


def _oracle_neighborhood(
    base: np.ndarray,
    oracle_index: int,
    selected_index: int,
    count: int,
    dedup: float,
    jump: float,
) -> np.ndarray:
    oracle = base[oracle_index]
    values: List[np.ndarray] = []
    for scale in (0.80, 0.90, 0.95, 0.975, 1.025, 1.05, 1.10, 1.20):
        _append_unique(values, _scale(oracle, scale), dedup_ade=dedup, max_jump=jump)
    for offset in (-1.0, -0.5, -0.25, -0.10, 0.10, 0.25, 0.5, 1.0):
        _append_unique(values, _lateral(oracle, offset), dedup_ade=dedup, max_jump=jump)
    for offset in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        _append_unique(values, _tangent(oracle, offset), dedup_ade=dedup, max_jump=jump)
    for pose in (4, 6):
        _append_unique(values, _hold(oracle, pose), dedup_ade=dedup, max_jump=jump)
    for strength in (0.5, 1.0):
        _append_unique(values, _smooth(oracle, strength), dedup_ade=dedup, max_jump=jump)
    for weight in (0.25, 0.50, 0.75):
        _append_unique(values, _interpolate(oracle, base[selected_index], weight), dedup_ade=dedup, max_jump=jump)
    return _ensure_count(values, oracle, count, dedup, jump)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "bank_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(), end="")
        return
    shards = sorted(args.enhanced_export_root.glob("shard_*/candidate_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No enhanced export shards under {args.enhanced_export_root}")
    base = load_proposal_pickle(args.base_proposals)
    with np.load(args.base_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str)
        true_scores = archive["candidate_scores"].astype(np.float64)
        predicted = archive["predicted_scores"].astype(np.float64)
        oracle_indices = archive["oracle_indices"].astype(np.int64)
        selected_indices = archive["selected_indices"].astype(np.int64)
    row_for_token = token_index(tokens)
    bottom_count = int(np.ceil(0.10 * len(tokens)))
    selected_true = true_scores[np.arange(len(tokens)), selected_indices]
    bottom_tokens = set(tokens[np.argsort(selected_true, kind="stable")[:bottom_count]])
    low_tokens = set(tokens[selected_true < 0.90]) | bottom_tokens
    inventory = {
        **{f"stage{index}_64": {} for index in range(5)},
        "structured16": {},
        "jitter8": {},
        "oracle_neighborhood16_low": {},
        "oracle_neighborhood64_low": {},
        "structured256_low": {},
        "all_intermediate256_low": {},
        "all_intermediate256": {},
    }
    requested = set(inventory) if args.banks == "all" else {value.strip() for value in args.banks.split(",") if value.strip()}
    unknown = requested.difference(inventory)
    if unknown:
        raise ValueError(f"Unknown banks: {sorted(unknown)}; choices={sorted(inventory)}")
    outputs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        name: payload for name, payload in inventory.items() if name in requested
    }
    seen: set[str] = set()
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as archive:
            shard_tokens = archive["tokens"].astype(str)
            shard_logs = archive["log_names"].astype(str)
            stages = archive["proposal_stages"].astype(np.float32)
        for local_index, token in enumerate(shard_tokens):
            if token in seen:
                raise RuntimeError(f"Duplicate enhanced token: {token}")
            seen.add(token)
            matrix_row = row_for_token[token]
            final = np.asarray(base[token]["proposals"], dtype=np.float32)
            if np.max(np.abs(stages[local_index, -1] - final)) > 1e-6:
                raise RuntimeError(f"Enhanced final stage differs for {token}")
            common = {"log_name": str(shard_logs[local_index])}
            for stage_index in range(5):
                name = f"stage{stage_index}_64"
                if name in outputs:
                    outputs[name][token] = {**common, "proposals": stages[local_index, stage_index]}
            if "structured16" in outputs:
                outputs["structured16"][token] = {
                    **common,
                    "proposals": _structured16(final, predicted[matrix_row], args.dedup_ade, args.max_jump),
                }
            if "jitter8" in outputs:
                outputs["jitter8"][token] = {
                    **common,
                    "proposals": _jitter8(final, predicted[matrix_row], args.dedup_ade, args.max_jump),
                }
            if "all_intermediate256" in outputs:
                outputs["all_intermediate256"][token] = {
                    **common,
                    "proposals": stages[local_index, :4].reshape(256, 8, 3),
                }
            if token in low_tokens:
                if "oracle_neighborhood16_low" in outputs:
                    outputs["oracle_neighborhood16_low"][token] = {
                        **common,
                        "proposals": _oracle_neighborhood(final, int(oracle_indices[matrix_row]), int(selected_indices[matrix_row]), 16, args.dedup_ade, args.max_jump),
                    }
                if "oracle_neighborhood64_low" in outputs:
                    outputs["oracle_neighborhood64_low"][token] = {
                        **common,
                        "proposals": _oracle_neighborhood(final, int(oracle_indices[matrix_row]), int(selected_indices[matrix_row]), 64, args.dedup_ade, args.max_jump),
                    }
                if "structured256_low" in outputs:
                    outputs["structured256_low"][token] = {
                        **common,
                        "proposals": _large256(final, predicted[matrix_row], args.dedup_ade, args.max_jump),
                    }
                if "all_intermediate256_low" in outputs:
                    outputs["all_intermediate256_low"][token] = {
                        **common,
                        "proposals": stages[local_index, :4].reshape(256, 8, 3),
                    }
    if seen != set(tokens):
        raise RuntimeError(f"Enhanced export token mismatch missing={len(set(tokens)-seen)} extra={len(seen-set(tokens))}")

    bank_meta = {}
    for name, payload in outputs.items():
        path = args.output_dir / f"{name}.pkl"
        _atomic_pickle(path, payload)
        counts = sorted({len(value["proposals"]) for value in payload.values()})
        bank_meta[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "scene_count": len(payload),
            "candidate_counts": counts,
            "uses_true_oracle_anchor": name.startswith("oracle_neighborhood"),
            "deployable": not name.startswith("oracle_neighborhood"),
        }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "enhanced_export_root": str(args.enhanced_export_root.resolve()),
        "base_proposals": str(args.base_proposals.resolve()),
        "base_proposals_sha256": sha256_file(args.base_proposals),
        "base_matrix": str(args.base_matrix.resolve()),
        "base_matrix_sha256": sha256_file(args.base_matrix),
        "parameters": {
            "time_progress_scaling": [0.65, 0.80, 0.90, 1.10, 1.20],
            "lateral_endpoint_offset_m": [-1.0, -0.5, 0.5, 1.0],
            "interpolation_lambda": [0.25, 0.50, 0.75],
            "hold_after_pose": [4, 6],
            "dedup_mean_ade_m": args.dedup_ade,
            "max_step_jump_m": args.max_jump,
            "heading": "finite differences from implicit origin, numpy.unwrap",
            "anchors": "selected, predicted top ranks, top-32 geometric FPS; no true oracle for deployable banks",
        },
        "large_bank_subset": "V_B < 0.90 union bottom 10% by V_B",
        "large_bank_scene_count": len(low_tokens),
        "requested_banks": sorted(requested),
        "banks": bank_meta,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
