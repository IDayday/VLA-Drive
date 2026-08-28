#!/usr/bin/env python3
"""Build deterministic log-balanced, legal-training-only Gate C folds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    navsim_paths,
    stable_scene_seed,
    validate_training_split,
    write_json,
    write_markdown,
    write_parquet,
)


MODE_DEFAULTS = {
    "smoke": {"num_scenes": 32, "min_logs": 4, "per_log_cap": 8},
    "pilot": {"num_scenes": 500, "min_logs": 20, "per_log_cap": 30},
    "full": {"num_scenes": 2000, "min_logs": 40, "per_log_cap": 50},
    "all_logs": {"num_scenes": 0, "min_logs": 0, "per_log_cap": 50},
    "all": {"num_scenes": 0, "min_logs": 0, "per_log_cap": 0},
}


def parse_cache_metadata(metric_cache_path: Path) -> pd.DataFrame:
    metadata_files = sorted((metric_cache_path / "metadata").glob("*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No MetricCache metadata CSV under {metric_cache_path / 'metadata'}")
    frames = []
    for path in metadata_files:
        frame = pd.read_csv(path, usecols=["file_name"])
        frame["metadata_file"] = str(path)
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True).drop_duplicates("file_name")
    paths = metadata["file_name"].map(Path)
    metadata["log_name"] = paths.map(lambda value: value.parts[-4])
    metadata["scene_token"] = paths.map(lambda value: value.parts[-2])
    metadata["cache_exists"] = metadata["file_name"].map(lambda value: Path(value).is_file())
    return metadata


def _load_selected_log_records(
    log_path: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Inspect selected samples without loading sensor blobs."""

    import pickle

    requested = defaultdict(set)
    for row in selected.itertuples(index=False):
        requested[str(row.log_name)].add(str(row.scene_token))
    rows: list[dict[str, Any]] = []
    for log_name, tokens in requested.items():
        pickle_path = log_path / f"{log_name}.pkl"
        if not pickle_path.is_file():
            for token in tokens:
                rows.append({"log_name": log_name, "scene_token": token, "log_pickle_exists": False})
            continue
        with pickle_path.open("rb") as stream:
            records = pickle.load(stream)
        by_token = {str(record.get("token")): record for record in records}
        for token in sorted(tokens):
            record = by_token.get(token)
            camera = (record or {}).get("cams", {}).get("CAM_F0", {})
            annotation = (record or {}).get("anns")
            traffic_lights = (record or {}).get("traffic_lights")
            rows.append(
                {
                    "log_name": log_name,
                    "scene_token": token,
                    "log_pickle_exists": True,
                    "record_found": record is not None,
                    "current_camera_declared": bool(camera.get("data_path")),
                    "current_annotations_declared": annotation is not None,
                    "current_traffic_lights_declared": traffic_lights is not None,
                    "map_name": (record or {}).get("map_location"),
                    "scene_metadata_token": (record or {}).get("scene_token"),
                    "frame_idx": (record or {}).get("frame_idx"),
                }
            )
    return pd.DataFrame(rows)


def balanced_select(
    metadata: pd.DataFrame,
    num_scenes: int,
    min_logs: int,
    per_log_cap: int,
    seed: int,
) -> pd.DataFrame:
    available = metadata[metadata["cache_exists"]].copy()
    grouped = {
        log_name: group.drop_duplicates("scene_token").copy()
        for log_name, group in available.groupby("log_name", sort=True)
    }
    eligible = [name for name, group in grouped.items() if len(group) > 0]
    if len(eligible) < min_logs:
        min_logs = len(eligible)
    required_logs = min(len(eligible), max(min_logs, int(np.ceil(num_scenes / per_log_cap))))
    rng = np.random.default_rng(seed)
    shuffled_logs = np.asarray(eligible, dtype=object)
    rng.shuffle(shuffled_logs)
    # Prefer logs capable of contributing the target cap while keeping the
    # choice random and reproducible.
    ranked = sorted(
        shuffled_logs.tolist(),
        key=lambda name: min(len(grouped[name]), per_log_cap),
        reverse=True,
    )
    chosen_logs = ranked[:required_logs]
    pools: dict[str, list[int]] = {}
    for log_name in chosen_logs:
        indices = grouped[log_name].index.to_numpy().copy()
        rng.shuffle(indices)
        pools[log_name] = indices[:per_log_cap].tolist()
    selected_indices: list[int] = []
    # Round-robin prevents early logs from monopolizing the selection.
    while len(selected_indices) < num_scenes:
        changed = False
        for log_name in chosen_logs:
            if pools[log_name] and len(selected_indices) < num_scenes:
                selected_indices.append(pools[log_name].pop())
                changed = True
        if not changed:
            break
    if len(selected_indices) < num_scenes:
        raise RuntimeError(
            f"Only {len(selected_indices)} scenes available under {per_log_cap=} from "
            f"{len(chosen_logs)} logs; requested {num_scenes}"
        )
    selected = available.loc[selected_indices].copy().reset_index(drop=True)
    selected["selection_index"] = np.arange(len(selected), dtype=np.int64)
    selected["scene_subseed"] = [stable_scene_seed(token, seed) for token in selected.scene_token]
    return selected


def assign_folds(selected: pd.DataFrame, num_folds: int, seed: int) -> dict[int, list[str]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    counts = selected.groupby("log_name").size().to_dict()
    logs = list(counts)
    rng = np.random.default_rng(seed)
    rng.shuffle(logs)
    logs.sort(key=lambda name: counts[name], reverse=True)
    folds: dict[int, list[str]] = {index: [] for index in range(num_folds)}
    totals = [0] * num_folds
    for log_name in logs:
        fold_index = int(np.argmin(totals))
        folds[fold_index].append(log_name)
        totals[fold_index] += int(counts[log_name])
    return folds


def build(args: argparse.Namespace) -> dict[str, Any]:
    split = validate_training_split(args.split)
    paths = navsim_paths(split)
    report_dir = ensure_dir(args.output_dir)
    fold_dir = ensure_dir(report_dir / "folds")
    defaults = MODE_DEFAULTS[args.mode]
    num_scenes = args.num_scenes or defaults["num_scenes"]
    min_logs = args.min_logs or defaults["min_logs"]
    per_log_cap = args.per_log_cap or defaults["per_log_cap"]

    metadata = parse_cache_metadata(paths.metric_cache_path)
    # Preserve the complete cache-backed inventory separately from the
    # log-balanced experiment manifest.  ``all_logs`` deliberately caps
    # adjacent scenes per log; that selection must not be mistaken for the
    # larger inventory that was actually scanned.
    if args.mode in {"full", "all_logs", "all"}:
        all_available = (
            metadata[metadata.cache_exists]
            .drop_duplicates("scene_token")
            .copy()
            .reset_index(drop=True)
        )
        all_available["selection_index"] = np.arange(len(all_available), dtype=np.int64)
        all_available["scene_subseed"] = [
            stable_scene_seed(token, args.seed) for token in all_available.scene_token
        ]
        all_inventory = _load_selected_log_records(paths.log_path, all_available)
        all_available = all_available.merge(
            all_inventory, on=["log_name", "scene_token"], how="left"
        )
        write_parquet(all_available, report_dir / "all_scene_inventory.parquet")
    if args.mode == "all":
        selected = metadata[metadata.cache_exists].drop_duplicates("scene_token").copy().reset_index(drop=True)
        selected["selection_index"] = np.arange(len(selected), dtype=np.int64)
        selected["scene_subseed"] = [stable_scene_seed(token, args.seed) for token in selected.scene_token]
    else:
        if args.mode == "all_logs":
            available_counts = (
                metadata[metadata.cache_exists]
                .drop_duplicates("scene_token")
                .groupby("log_name")
                .size()
            )
            num_scenes = int(available_counts.clip(upper=per_log_cap).sum())
            min_logs = int(len(available_counts))
        selected = balanced_select(metadata, num_scenes, min_logs, per_log_cap, args.seed)
    selected_inventory = _load_selected_log_records(paths.log_path, selected)
    selected = selected.merge(selected_inventory, on=["log_name", "scene_token"], how="left")
    if not selected["record_found"].fillna(False).all():
        missing = selected.loc[~selected["record_found"].fillna(False), "scene_token"].tolist()
        raise RuntimeError(f"Selected cache tokens missing from log pickle: {missing[:10]}")
    folds = assign_folds(selected, args.num_folds, args.seed)
    selected["fold"] = -1
    for fold_index, validation_logs in folds.items():
        selected.loc[selected.log_name.isin(validation_logs), "fold"] = fold_index
        validation = selected[selected.log_name.isin(validation_logs)]
        training = selected[~selected.log_name.isin(validation_logs)]
        payload = {
            "fold": fold_index,
            "seed": args.seed,
            "split": split,
            "train_logs": sorted(training.log_name.unique().tolist()),
            "validation_logs": sorted(validation_logs),
            "train_scene_count": int(len(training)),
            "validation_scene_count": int(len(validation)),
            "log_overlap": sorted(set(training.log_name) & set(validation.log_name)),
        }
        write_json(fold_dir / f"fold_{fold_index}.json", payload)
    assert (selected.fold >= 0).all()
    write_parquet(selected, report_dir / "balanced_scene_manifest.parquet")

    all_log_counts = metadata.groupby("log_name").size()
    result = {
        "mode": args.mode,
        "split": split,
        "seed": args.seed,
        "metric_cache_rows": int(len(metadata)),
        "metric_cache_existing_rate": float(metadata.cache_exists.mean()),
        "available_log_count": int(metadata.log_name.nunique()),
        "all_inventory_scene_count": int(
            metadata[metadata.cache_exists].scene_token.nunique()
        ),
        "log_pickle_count": len(list(paths.log_path.glob("*.pkl"))),
        "selected_scene_count": int(len(selected)),
        "selected_log_count": int(selected.log_name.nunique()),
        "per_log_max": int(selected.groupby("log_name").size().max()),
        "per_log_min": int(selected.groupby("log_name").size().min()),
        "current_camera_declared_coverage": float(selected.current_camera_declared.mean()),
        "current_annotation_declared_coverage": float(selected.current_annotations_declared.mean()),
        "current_traffic_light_declared_coverage": float(selected.current_traffic_lights_declared.mean()),
        "all_cache_log_scene_count": {
            "min": int(all_log_counts.min()),
            "median": float(all_log_counts.median()),
            "max": int(all_log_counts.max()),
        },
        "fold_scene_counts": {
            str(index): int((selected.fold == index).sum()) for index in range(args.num_folds)
        },
    }
    write_json(report_dir / "dataset_split_summary.json", result)
    report = f"""# Gate C Dataset Split Report

- Legal split: `{split}` (no navtest/navhard/private-test inputs)
- MetricCache entries/logs scanned: {len(metadata):,} / {metadata.log_name.nunique():,}
- Log pickle files available: {result['log_pickle_count']:,}
- Selected scenes/logs: {len(selected):,} / {selected.log_name.nunique():,}
- Per-log selected range: {result['per_log_min']}–{result['per_log_max']} ({'all cache-backed scenes' if args.mode == 'all' else f'cap {per_log_cap}'})
- Current CAM_F0 declaration coverage: {result['current_camera_declared_coverage']:.3%}
- Current annotation declaration coverage: {result['current_annotation_declared_coverage']:.3%}
- Five-fold assignment seed: {args.seed}

Selection is randomized and log-balanced. It does not sort tokens and truncate the
first scenes. Each complete log is assigned to exactly one validation fold; train
and validation log overlap is asserted empty in every fold JSON.
"""
    write_markdown(report_dir / "DATASET_SPLIT_REPORT.md", report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_DEFAULTS), default="smoke")
    parser.add_argument("--split", choices=("train", "trainval"), default="trainval")
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--min-logs", type=int, default=0)
    parser.add_argument("--per-log-cap", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    build(args)
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.build_balanced_split " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
