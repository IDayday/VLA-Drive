"""Create only the NAVSIM navtrain metric caches required by fixed Gate tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


def _tokens(paths: Sequence[Path], limit: int | None) -> list[str]:
    """Read one or more non-overlapping fixed split files in argument order."""
    values: list[str] = []
    seen: set[str] = set()
    for path in paths:
        file_values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not file_values or len(file_values) != len(set(file_values)):
            raise ValueError(f"token file must be non-empty and unique: {path}")
        overlap = seen.intersection(file_values)
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                f"token files must be mutually disjoint; duplicate {example} in {path}"
            )
        values.extend(file_values)
        seen.update(file_values)
    return values if limit is None else values[:limit]


def cache_subset(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing metric cache root: {args.output}")
    values = _tokens(args.tokens, args.limit)
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.map_root)
    sys.path.insert(0, str(args.wote_root))
    from nuplan.planning.training.experiments.cache_metadata_entry import (
        save_cache_metadata,
    )
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    loader = SceneLoader(
        data_path=args.data_root / "navsim_logs/trainval",
        sensor_blobs_path=None,
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            tokens=values,
        ),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if set(loader.tokens) != set(values):
        missing = sorted(set(values) - set(loader.tokens))
        extra = sorted(set(loader.tokens) - set(values))
        raise ValueError(f"metric subset scene mismatch: missing={missing[:5]} extra={extra[:5]}")
    processor = MetricCacheProcessor(
        cache_path=args.output,
        force_feature_computation=False,
    )
    metadata = []
    for index, token in enumerate(values):
        frame_list = loader.scene_frames_dicts[token]
        scene = Scene.from_scene_dict_list(
            frame_list,
            None,
            num_history_frames=4,
            num_future_frames=10,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        scenario = NavSimScenario(
            scene,
            map_root=str(args.map_root),
            map_version="nuplan-maps-v1.0",
        )
        entry = processor.compute_metric_cache(scenario)
        if entry is None:
            raise RuntimeError(f"metric caching returned no metadata for {token}")
        metadata.append(entry)
        print(f"[metric-subset] {index + 1}/{len(values)} {token}", flush=True)
    save_cache_metadata(metadata, args.output, node_id=0)
    from navsim.common.dataloader import MetricCacheLoader

    cached = MetricCacheLoader(args.output)
    if set(cached.tokens) != set(values):
        raise RuntimeError("final metric cache metadata does not match requested tokens")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wote-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=Path,
        nargs="+",
        required=True,
        help="One or more mutually disjoint fixed split token files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    resolved = {
        "wote_root": str(args.wote_root),
        "data_root": str(args.data_root),
        "map_root": str(args.map_root),
        "tokens": [str(path) for path in args.tokens],
        "output": str(args.output),
        "limit": args.limit,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if not args.dry_run:
        cache_subset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
