"""Create only the NAVSIM navtrain metric caches required by fixed Gate tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .direct_rehab_contracts import AccessAuditLog, AccessPolicy


def _tokens(
    paths: Sequence[Path],
    limit: int | None,
    *,
    access_policy: AccessPolicy | None = None,
    phase: str = "legacy",
) -> list[str]:
    """Read one or more non-overlapping fixed split files in argument order."""
    values: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if access_policy is None:
            file_values = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            file_values = list(access_policy.read_token_file(path, phase))
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
    policy = AccessPolicy.load(args.access_policy) if args.access_policy else None
    values = _tokens(
        args.tokens,
        args.limit,
        access_policy=policy,
        phase=args.access_phase,
    )
    if policy is not None:
        if args.access_log is None:
            raise ValueError("--access-log is required with --access-policy")
        audit = AccessAuditLog(args.access_log, policy, args.access_phase)
        # SceneLoader opens the requested log records during construction, so
        # authorize and audit the complete token set before instantiating it.
        for token in values:
            audit.record(token, "metric_cache_generation")
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
    parser.add_argument("--access-policy", type=Path)
    parser.add_argument("--access-log", type=Path)
    parser.add_argument("--access-phase", default="legacy")
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
        "access_policy": str(args.access_policy) if args.access_policy else None,
        "access_log": str(args.access_log) if args.access_log else None,
        "access_phase": args.access_phase,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if not args.dry_run:
        cache_subset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
