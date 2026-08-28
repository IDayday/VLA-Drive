#!/usr/bin/env python3
"""Shared utilities for the candidate-relative NAVSIM audit.

Imports from :mod:`navsim` are intentionally delayed until after path discovery
sets the environment variables consumed at import time by this local NAVSIM 1.1
checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/navsim_candidate_relative_audit"
ALLOWED_SPLITS = {"mini", "trainval", "navtrain"}
DEFAULT_HORIZONS = (0.5, 1.0, 2.0, 4.0)
AUDIT_HISTORY_FRAMES = 4
AUDIT_FUTURE_FRAMES = 10


@dataclass(frozen=True)
class AuditPaths:
    repo_root: Path
    devkit_root: Path
    log_path: Path
    sensor_blobs_path: Path
    map_path: Path
    metric_cache_path: Path
    navsim_exp_root: Path
    v2_devkit_root: Optional[Path]
    synthetic_scene_path: Optional[Path]
    synthetic_sensor_path: Optional[Path]
    split: str

    def to_json(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def _first_path(values: Iterable[Optional[str | Path]], *, allow_missing: bool = False) -> Optional[Path]:
    fallback: Optional[Path] = None
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        path = Path(value).expanduser().resolve()
        if fallback is None:
            fallback = path
        if path.exists():
            return path
    return fallback if allow_missing else None


def validate_split(split: str) -> str:
    split = split.strip().lower()
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Audit split {split!r} is not allowed. Use one of {sorted(ALLOWED_SPLITS)}; "
            "test/navtest/navhard/private-test are intentionally rejected."
        )
    return split


def discover_paths(split: str = "trainval") -> AuditPaths:
    """Resolve local paths without assuming that shell environment was configured."""

    split = validate_split(split)
    dataset_split = "mini" if split == "mini" else "trainval"
    log_default = {
        "mini": Path("/mnt/navsim/mini_navsim_logs/mini"),
        "trainval": Path("/mnt/navsim/trainval_navsim_logs/trainval"),
    }[dataset_split]
    sensor_default = {
        "mini": Path("/mnt/navsim/mini_sensor_blobs/mini"),
        "trainval": Path("/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval"),
    }[dataset_split]

    openscene_root = _first_path([os.environ.get("OPENSCENE_DATA_ROOT")])
    env_log = os.environ.get("NAVSIM_LOG_PATH")
    env_sensor = os.environ.get("NAVSIM_SENSOR_BLOBS_PATH")
    if openscene_root is not None:
        root_log_candidates = [
            openscene_root / "meta_datas" / dataset_split,
            openscene_root / "openscene_meta_datas" / dataset_split,
            openscene_root / "navsim_logs" / dataset_split,
        ]
        root_sensor_candidates = [
            openscene_root / "sensor_blobs" / dataset_split,
            openscene_root / dataset_split,
        ]
    else:
        root_log_candidates = []
        root_sensor_candidates = []

    log_path = _first_path([env_log, *root_log_candidates, log_default], allow_missing=True)
    sensor_path = _first_path([env_sensor, *root_sensor_candidates, sensor_default], allow_missing=True)
    map_path = _first_path(
        [os.environ.get("NUPLAN_MAPS_ROOT"), Path("/mnt/navsim/maps")], allow_missing=True
    )
    metric_cache = _first_path(
        [
            os.environ.get("NAVSIM_METRIC_CACHE_PATH"),
            os.environ.get("METRIC_CACHE_PATH"),
            os.environ.get("PDMS_METRIC_CACHE_PATH"),
            Path("/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full"),
            Path("/mnt/project/DriveDreamer-Policy/navsim_exp/metric_cache"),
        ],
        allow_missing=True,
    )
    devkit = _first_path(
        [os.environ.get("NAVSIM_DEVKIT_ROOT"), os.environ.get("NAVSIM_ROOT"), REPO_ROOT],
        allow_missing=True,
    )
    exp_root = _first_path(
        [os.environ.get("NAVSIM_EXP_ROOT"), REPO_ROOT / "outputs"], allow_missing=True
    )
    v2_root = _first_path(
        [
            os.environ.get("DRIVEVLA_NAVSIM_V2_ROOT"),
            os.environ.get("NAVSIM_V2_ROOT"),
            Path("/mnt/project/DriveDreamer-Policy/navsim"),
        ]
    )
    synthetic_scene = _first_path(
        [
            os.environ.get("NAVSIM_SYNTHETIC_SCENES"),
            Path("/mnt/navsim/warmup_two_stage/synthetic_scene_pickles"),
        ]
    )
    synthetic_sensor = _first_path(
        [
            os.environ.get("NAVSIM_SYNTHETIC_SENSOR_PATH"),
            Path("/mnt/navsim/warmup_two_stage/sensor_blobs"),
        ]
    )
    assert log_path and sensor_path and map_path and metric_cache and devkit and exp_root
    return AuditPaths(
        repo_root=REPO_ROOT,
        devkit_root=devkit,
        log_path=log_path,
        sensor_blobs_path=sensor_path,
        map_path=map_path,
        metric_cache_path=metric_cache,
        navsim_exp_root=exp_root,
        v2_devkit_root=v2_root,
        synthetic_scene_path=synthetic_scene,
        synthetic_sensor_path=synthetic_sensor,
        split=split,
    )


def configure_navsim_environment(paths: AuditPaths) -> None:
    """Set only path variables required by the checked-out devkit."""

    os.environ["NAVSIM_ROOT"] = str(paths.devkit_root)
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(paths.devkit_root)
    os.environ["NAVSIM_EXP_ROOT"] = str(paths.navsim_exp_root)
    os.environ["OPENSCENE_DATA_ROOT"] = str(paths.sensor_blobs_path.parent)
    os.environ["NUPLAN_MAPS_ROOT"] = str(paths.map_path)
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ["NAVSIM_LOG_PATH"] = str(paths.log_path)
    os.environ["NAVSIM_SENSOR_BLOBS_PATH"] = str(paths.sensor_blobs_path)
    os.environ["NAVSIM_METRIC_CACHE_PATH"] = str(paths.metric_cache_path)


def add_common_arguments(parser: argparse.ArgumentParser, *, include_candidates: bool = False) -> None:
    parser.add_argument("--split", default="trainval", choices=sorted(ALLOWED_SPLITS))
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("smoke", "audit", "statistics", "full"), default="smoke")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--sensor-blobs-path", type=Path)
    parser.add_argument("--map-path", type=Path)
    parser.add_argument("--metric-cache-path", type=Path)
    if include_candidates:
        parser.add_argument("--num-candidates", type=int, default=12)
        parser.add_argument("--seed", type=int, default=20260828)


def paths_from_args(args: argparse.Namespace) -> AuditPaths:
    paths = discover_paths(args.split)
    replacements = {}
    for arg_name, field_name in (
        ("log_path", "log_path"),
        ("sensor_blobs_path", "sensor_blobs_path"),
        ("map_path", "map_path"),
        ("metric_cache_path", "metric_cache_path"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            replacements[field_name] = value.expanduser().resolve()
    if replacements:
        values = asdict(paths)
        values.update(replacements)
        paths = AuditPaths(**values)
    configure_navsim_environment(paths)
    return paths


def effective_max_scenes(mode: str, requested: int) -> int:
    limits = {"smoke": 8, "audit": 64, "statistics": 500, "full": requested}
    return min(requested, limits.get(mode, requested)) if requested > 0 else limits.get(mode, 8)


def ensure_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def run_text(command: Sequence[str], cwd: Optional[Path] = None, timeout: int = 15) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def directory_size(path: Optional[Path], timeout: int = 8) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "bytes": None, "method": "not_configured"}
    if not path.exists():
        return {"exists": False, "bytes": None, "method": "missing"}
    value = run_text(["du", "-sB1", str(path)], timeout=timeout)
    if value:
        try:
            return {"exists": True, "bytes": int(value.split()[0]), "method": "du_-sB1"}
        except (ValueError, IndexError):
            pass
    try:
        entries = sum(1 for _ in path.iterdir()) if path.is_dir() else 1
    except OSError:
        entries = None
    return {"exists": True, "bytes": None, "method": "timeout_top_level_only", "entries": entries}


def stable_hash(value: str, bits: int = 63) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << bits) - 1)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def local_to_global(origin: Sequence[float], poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    c, s = np.cos(origin[2]), np.sin(origin[2])
    result = np.empty_like(poses, dtype=np.float64)
    result[..., 0] = origin[0] + c * poses[..., 0] - s * poses[..., 1]
    result[..., 1] = origin[1] + s * poses[..., 0] + c * poses[..., 1]
    result[..., 2] = wrap_angle(origin[2] + poses[..., 2])
    return result


def global_to_local(origin: Sequence[float], poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    dx = poses[..., 0] - origin[0]
    dy = poses[..., 1] - origin[1]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    result = np.empty_like(poses, dtype=np.float64)
    result[..., 0] = c * dx + s * dy
    result[..., 1] = -s * dx + c * dy
    result[..., 2] = wrap_angle(poses[..., 2] - origin[2])
    return result


def resolve_horizon_index(
    timestamps: Sequence[int | float], target_seconds: float, *, origin_index: int = 0
) -> int:
    """Return the closest timestamp index to ``origin + target_seconds``.

    Integer timestamps with magnitudes above 1e9 are interpreted as microseconds;
    otherwise timestamps are interpreted as seconds.  Ties resolve to the earlier
    frame, making the function deterministic.
    """

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional sequence")
    if not 0 <= origin_index < len(values):
        raise IndexError(f"origin_index {origin_index} out of range for {len(values)} timestamps")
    scale = 1e6 if np.nanmax(np.abs(values)) > 1e9 else 1.0
    target = values[origin_index] + target_seconds * scale
    distance = np.abs(values - target)
    return int(np.flatnonzero(distance == np.nanmin(distance))[0])


def metric_cache_loader(paths: AuditPaths):
    from navsim.common.dataloader import MetricCacheLoader

    return MetricCacheLoader(paths.metric_cache_path)


def scene_loader(
    paths: AuditPaths,
    *,
    max_scenes: int,
    frame_interval: int = 1,
    tokens: Optional[Sequence[str]] = None,
    sensor_config: Any = None,
    load_image_path: bool = False,
):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    if sensor_config is None:
        sensor_config = SensorConfig.build_no_sensors()
    scene_filter = SceneFilter(
        num_history_frames=AUDIT_HISTORY_FRAMES,
        num_future_frames=AUDIT_FUTURE_FRAMES,
        frame_interval=frame_interval,
        has_route=True,
        max_scenes=max_scenes,
        tokens=list(tokens) if tokens is not None else None,
    )
    return SceneLoader(
        data_path=paths.log_path,
        sensor_blobs_path=paths.sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=load_image_path,
    )


def load_scenes_for_tokens(paths: AuditPaths, tokens: Sequence[str]):
    unique = list(dict.fromkeys(str(token) for token in tokens))
    return scene_loader(paths, max_scenes=len(unique), frame_interval=1, tokens=unique)


def append_command(output_dir: Path, command: str) -> None:
    command_path = output_dir / "COMMANDS.sh"
    if not command_path.exists():
        command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
        command_path.chmod(0o755)
    with command_path.open("a", encoding="utf-8") as stream:
        stream.write(command.rstrip() + "\n")


def percentile_summary(values: Sequence[float]) -> dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }
