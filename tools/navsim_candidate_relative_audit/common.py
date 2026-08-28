"""Shared, version-adaptive helpers for the NAVSIM feasibility audit.

The module deliberately imports NAVSIM only inside functions.  The local server
contains both NAVSIM 1.1 and 2.0 source trees, so callers must first resolve and
bootstrap the exact source tree they intend to audit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import lzma
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


AUDIT_VERSION = "navsim_candidate_relative_audit_v1"
HORIZONS_S = (0.5, 1.0, 2.0, 4.0)
PRIVATE_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_readonly_git(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=str(cwd or repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def primary_checkout() -> Path:
    common_dir = run_readonly_git("rev-parse", "--git-common-dir")
    if common_dir:
        path = Path(common_dir)
        if not path.is_absolute():
            path = (repo_root() / path).resolve()
        if path.name == ".git":
            return path.parent
    return repo_root()


def _first_existing(candidates: Iterable[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and (candidate.exists() or candidate.is_symlink()):
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class AuditPaths:
    repository: Path
    primary_checkout: Path
    navsim_devkit: Path | None
    navsim_v1_devkit: Path | None
    public_root: Path | None
    logs_root: Path | None
    sensors_root: Path | None
    maps_root: Path | None
    experiment_root: Path | None
    processed_root: Path | None
    datalist: Path | None
    action_effect_root: Path | None
    metric_cache: Path | None
    candidate_cache: Path | None
    consequence_cache: Path | None
    effect_tube_cache: Path | None
    synthetic_scenes: Path | None
    synthetic_sensors: Path | None

    def to_json(self) -> dict[str, str | None]:
        return {
            key: None if value is None else str(value)
            for key, value in asdict(self).items()
        }


def discover_paths(
    overrides: argparse.Namespace | None = None, split: str = "trainval"
) -> AuditPaths:
    """Resolve machine paths with CLI > environment > checkout defaults."""

    current = repo_root()
    primary = primary_checkout()

    def override(name: str) -> Path | None:
        value = getattr(overrides, name, None) if overrides is not None else None
        return Path(value).expanduser() if value else None

    def env_path(name: str) -> Path | None:
        value = os.environ.get(name)
        return Path(value).expanduser() if value else None

    navsim_devkit = _first_existing(
        [
            override("navsim_root"),
            env_path("NAVSIM_DEVKIT_ROOT"),
            current / "navsim",
            primary / "navsim",
        ]
    )
    navsim_v1 = _first_existing(
        [
            override("navsim_v1_root"),
            env_path("NAVSIM_V1_DEVKIT_ROOT"),
            current / "navsim_v1.1" / "navsim",
            primary / "navsim_v1.1" / "navsim",
        ]
    )
    public_root = _first_existing(
        [
            override("data_root"),
            env_path("NAVSIM_PUBLIC_ROOT"),
            current / "navsim_dataset_raw",
            primary / "navsim_dataset_raw",
        ]
    )
    logs_root = _first_existing(
        [
            override("log_root"),
            public_root / "navsim_logs" / split if public_root else None,
        ]
    )
    sensors_root = _first_existing(
        [
            override("sensor_root"),
            public_root / "sensor_blobs" / split if public_root else None,
        ]
    )
    maps_root = _first_existing(
        [
            override("map_root"),
            env_path("NUPLAN_MAPS_ROOT"),
            public_root / "maps" if public_root else None,
        ]
    )
    experiment_root = _first_existing(
        [
            override("experiment_root"),
            env_path("NAVSIM_EXP_ROOT"),
            current / "navsim_exp",
            primary / "navsim_exp",
        ]
    )
    processed_root = _first_existing(
        [
            override("processed_root"),
            env_path("DATA_ROOT"),
            current / "navsim_dataset",
            primary / "navsim_dataset",
        ]
    )
    datalist = _first_existing(
        [
            override("datalist"),
            env_path("NAVSIM_DATALIST_PATH"),
            current / "train_meta.json",
            primary / "train_meta.json",
        ]
    )
    action_effect = _first_existing(
        [
            override("action_effect_root"),
            env_path("ACTION_EFFECT_CACHE_ROOT"),
            current / "action_effect_cache",
            primary / "action_effect_cache",
        ]
    )
    metric_cache = _first_existing(
        [
            override("metric_cache"),
            action_effect / "metric_cache" / "pilot_small" / "train_phase6_v1"
            if action_effect
            else None,
            action_effect / "metric_cache" / "pilot_tiny" / "train"
            if action_effect
            else None,
            env_path("NAVSIM_V1_METRIC_CACHE_PATH"),
        ]
    )
    candidate_cache = _first_existing(
        [
            override("candidate_cache"),
            action_effect / "candidates" / "pilot_small" / "expert_phase6_v1"
            if action_effect
            else None,
            action_effect / "candidates" / "pilot_tiny" / "expert"
            if action_effect
            else None,
        ]
    )
    consequence_cache = _first_existing(
        [
            override("consequence_cache"),
            action_effect / "consequences" / "pilot_small" / "expert_phase6_v1"
            if action_effect
            else None,
            action_effect / "consequences" / "pilot_tiny" / "expert"
            if action_effect
            else None,
        ]
    )
    effect_tube = _first_existing(
        [
            override("effect_tube_cache"),
            action_effect
            / "effect_tube"
            / "pilot_small"
            / "expert_log_replay_32_phase6_v1"
            if action_effect
            else None,
            action_effect / "effect_tube" / "pilot_tiny" / "expert_log_replay_32_v1"
            if action_effect
            else None,
        ]
    )
    synthetic_scenes = _first_existing(
        [
            override("synthetic_scenes"),
            public_root / "navhard_two_stage" / "synthetic_scene_pickles"
            if public_root
            else None,
        ]
    )
    synthetic_sensors = _first_existing(
        [
            override("synthetic_sensors"),
            public_root / "navhard_two_stage" / "sensor_blobs" if public_root else None,
        ]
    )
    return AuditPaths(
        repository=current,
        primary_checkout=primary,
        navsim_devkit=navsim_devkit,
        navsim_v1_devkit=navsim_v1,
        public_root=public_root,
        logs_root=logs_root,
        sensors_root=sensors_root,
        maps_root=maps_root,
        experiment_root=experiment_root,
        processed_root=processed_root,
        datalist=datalist,
        action_effect_root=action_effect,
        metric_cache=metric_cache,
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        effect_tube_cache=effect_tube,
        synthetic_scenes=synthetic_scenes,
        synthetic_sensors=synthetic_sensors,
    )


PATH_ARGUMENTS = (
    ("--navsim-root", "navsim_root"),
    ("--navsim-v1-root", "navsim_v1_root"),
    ("--data-root", "data_root"),
    ("--log-root", "log_root"),
    ("--sensor-root", "sensor_root"),
    ("--map-root", "map_root"),
    ("--experiment-root", "experiment_root"),
    ("--processed-root", "processed_root"),
    ("--datalist", "datalist"),
    ("--action-effect-root", "action_effect_root"),
    ("--metric-cache", "metric_cache"),
    ("--candidate-cache", "candidate_cache"),
    ("--consequence-cache", "consequence_cache"),
    ("--effect-tube-cache", "effect_tube_cache"),
    ("--synthetic-scenes", "synthetic_scenes"),
    ("--synthetic-sensors", "synthetic_sensors"),
)


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    for flag, destination in PATH_ARGUMENTS:
        parser.add_argument(flag, dest=destination, default=None)


def add_common_arguments(
    parser: argparse.ArgumentParser, *, max_scenes: int = 8
) -> None:
    parser.add_argument(
        "--split", default="trainval", choices=("mini", "trainval", "train")
    )
    parser.add_argument("--max-scenes", type=int, default=max_scenes)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "reports" / "navsim_candidate_relative_audit",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    add_path_arguments(parser)


def bootstrap_navsim(paths: AuditPaths, *, v1: bool = False) -> dict[str, Any]:
    """Import exactly one local NAVSIM source tree and return provenance."""

    root = paths.navsim_v1_devkit if v1 else paths.navsim_devkit
    if root is None:
        raise FileNotFoundError("no local NAVSIM source tree was resolved")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    if paths.maps_root is not None:
        os.environ["NUPLAN_MAPS_ROOT"] = str(paths.maps_root)
    if paths.public_root is not None:
        os.environ["OPENSCENE_DATA_ROOT"] = str(paths.public_root)
    module = importlib.import_module("navsim")
    setup_path = root / "setup.py"
    setup_version = None
    if setup_path.is_file():
        for line in setup_path.read_text(encoding="utf-8").splitlines():
            if "version=" in line:
                setup_version = line.split("version=", 1)[1].strip(" ,\"'")
                break
    return {
        "requested_root": root_string,
        "import_path": str(Path(module.__file__).resolve()),
        "setup_version": setup_version,
        "is_v1": v1,
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    )


def write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected object at {path}:{line_number}")
                yield value


def load_metric_cache_index(root: Path) -> dict[str, Path]:
    relative_index = root / "metric_cache_index.json"
    if relative_index.is_file():
        raw = json.loads(relative_index.read_text(encoding="utf-8"))
        return {str(token): root / str(relative) for token, relative in raw.items()}
    metadata = root / "metadata"
    csv_files = sorted(metadata.glob("*.csv")) if metadata.is_dir() else []
    if not csv_files:
        raise FileNotFoundError(
            f"no relative index or official metadata CSV under {root}"
        )
    paths: dict[str, Path] = {}
    with csv_files[0].open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_path = next(
                (value for value in row.values() if value and "metric_cache" in value),
                None,
            )
            if raw_path:
                path = Path(raw_path)
                paths[path.parent.name] = path
    return paths


def load_metric_cache(path: Path) -> Any:
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)


def metric_log_name(path: Path) -> str:
    parts = path.parts
    marker = "unknown"
    if marker in parts:
        marker_index = parts.index(marker)
        if marker_index:
            return parts[marker_index - 1]
    return path.parents[2].name


def raw_log_path(log_root: Path, log_name: str) -> Path:
    candidates = [log_root / f"{log_name}.pkl", log_root / log_name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"raw log missing for {log_name} below {log_root}")


def load_raw_window(
    log_root: Path,
    log_name: str,
    token: str,
    *,
    history_frames: int = 4,
    future_frames: int = 8,
) -> list[dict[str, Any]]:
    path = raw_log_path(log_root, log_name)
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    current_index = next(
        (index for index, frame in enumerate(frames) if frame.get("token") == token),
        None,
    )
    if current_index is None:
        raise KeyError(f"token {token} not in {path}")
    start = current_index - history_frames + 1
    stop = current_index + future_frames + 1
    if start < 0 or stop > len(frames):
        raise IndexError(
            f"token {token} lacks {history_frames}/{future_frames} frame context"
        )
    return list(frames[start:stop])


def build_scene_from_raw(
    paths: AuditPaths,
    token: str,
    metric_path: Path,
    *,
    sensor_indices: Sequence[int] = (),
) -> Any:
    if paths.logs_root is None or paths.sensors_root is None:
        raise FileNotFoundError("raw log/sensor roots are unavailable")
    bootstrap_navsim(paths)
    from navsim.common.dataclasses import Scene, SensorConfig

    frames = load_raw_window(paths.logs_root, metric_log_name(metric_path), token)
    sensors = SensorConfig.build_no_sensors()
    if sensor_indices:
        sensors.cam_f0 = list(sensor_indices)
    return Scene.from_scene_dict_list(
        frames,
        paths.sensors_root,
        num_history_frames=4,
        num_future_frames=8,
        sensor_config=sensors,
    )


def wrap_heading(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def se2_local_to_global(local: np.ndarray, origin: Sequence[float]) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    original_shape = values.shape
    values = np.atleast_2d(values)
    x0, y0, heading0 = map(float, origin[:3])
    cosine, sine = math.cos(heading0), math.sin(heading0)
    result = values.copy()
    result[:, 0] = x0 + cosine * values[:, 0] - sine * values[:, 1]
    result[:, 1] = y0 + sine * values[:, 0] + cosine * values[:, 1]
    if result.shape[1] >= 3:
        result[:, 2] = wrap_heading(values[:, 2] + heading0)
    return result.reshape(original_shape)


def se2_global_to_local(
    global_values: np.ndarray, origin: Sequence[float]
) -> np.ndarray:
    values = np.asarray(global_values, dtype=np.float64)
    original_shape = values.shape
    values = np.atleast_2d(values)
    x0, y0, heading0 = map(float, origin[:3])
    cosine, sine = math.cos(heading0), math.sin(heading0)
    dx, dy = values[:, 0] - x0, values[:, 1] - y0
    result = values.copy()
    result[:, 0] = cosine * dx + sine * dy
    result[:, 1] = -sine * dx + cosine * dy
    if result.shape[1] >= 3:
        result[:, 2] = wrap_heading(values[:, 2] - heading0)
    return result.reshape(original_shape)


def rotate_global_vector_to_local(vectors: np.ndarray, heading: float) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    cosine, sine = math.cos(heading), math.sin(heading)
    result = values.copy()
    result[..., 0] = cosine * values[..., 0] + sine * values[..., 1]
    result[..., 1] = -sine * values[..., 0] + cosine * values[..., 1]
    return result


def resolve_horizon_index(
    timestamps: Sequence[int | float], target_seconds: float
) -> int:
    """Return the closest timestamp index relative to the first entry."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional sequence")
    # Infer units from spacing, not epoch magnitude: Unix seconds are also
    # around 1e9, while NAVSIM microsecond timestamps have ~5e5 increments.
    positive_deltas = np.abs(np.diff(values))
    positive_deltas = positive_deltas[positive_deltas > 0]
    scale = 1e6 if len(positive_deltas) and np.nanmedian(positive_deltas) > 1e3 else 1.0
    relative = (values - values[0]) / scale
    return int(np.argmin(np.abs(relative - float(target_seconds))))


def uniform_horizon_index(
    target_seconds: float,
    interval_seconds: float,
    num_steps: int,
    *,
    includes_current: bool,
) -> int:
    if interval_seconds <= 0 or num_steps <= 0:
        raise ValueError("interval_seconds and num_steps must be positive")
    raw = int(round(float(target_seconds) / interval_seconds))
    index = raw if includes_current else raw - 1
    return int(np.clip(index, 0, num_steps - 1))


def stable_token_hash(token: str) -> int:
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8], "big", signed=False
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def candidate_sources(paths: AuditPaths) -> tuple[Path, Path]:
    if paths.candidate_cache is None or paths.consequence_cache is None:
        raise FileNotFoundError(
            "existing candidate/consequence caches were not resolved"
        )
    return paths.candidate_cache, paths.consequence_cache


def load_candidate_source(
    paths: AuditPaths,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    candidate_root, _ = candidate_sources(paths)
    with np.load(candidate_root / "candidates.npz") as payload:
        trajectories = np.asarray(payload["trajectories"], dtype=np.float32)
    metadata = list(iter_jsonl(candidate_root / "metadata.jsonl"))
    scene_index = json.loads(
        (candidate_root / "scene_index.json").read_text(encoding="utf-8")
    )
    if len(metadata) != len(trajectories):
        raise ValueError("candidate metadata/trajectory length mismatch")
    return trajectories, metadata, scene_index


def consequence_rows_by_scene(paths: AuditPaths) -> dict[str, list[dict[str, Any]]]:
    _, consequence_root = candidate_sources(paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(consequence_root / "consequences.jsonl"):
        grouped.setdefault(str(row["scene_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["scene_candidate_index"]))
    return grouped


def select_eligible_scenes(
    paths: AuditPaths,
    *,
    max_scenes: int,
    num_candidates: int,
    require_reactive: bool = False,
    preferred_tokens: Sequence[str] | None = None,
) -> list[str]:
    grouped = consequence_rows_by_scene(paths)
    eligible = []
    preferred = list(dict.fromkeys(preferred_tokens or ()))
    order = preferred + [
        token for token in sorted(grouped) if token not in set(preferred)
    ]
    for token in order:
        rows = grouped.get(token, [])
        accepted = [
            row
            for row in rows
            if row.get("candidate_accepted")
            and row.get("log_replay", {}).get("available")
            and (not require_reactive or row.get("reactive_model", {}).get("available"))
        ]
        anchor_ok = any(row.get("perturbation_type") == "anchor" for row in accepted)
        if anchor_ok and len(accepted) >= num_candidates:
            eligible.append(token)
            if len(eligible) >= max_scenes:
                break
    return eligible


def output_tokens(output_dir: Path) -> list[str]:
    coverage = output_dir / "scene_coverage.csv"
    if not coverage.is_file():
        return []
    frame = pd.read_csv(coverage)
    return [str(value) for value in frame.get("scene_token", [])]


def trajectory_kinematics(
    trajectory: np.ndarray, interval_s: float = 0.5
) -> dict[str, np.ndarray | float]:
    poses = np.asarray(trajectory, dtype=np.float64)
    points = np.concatenate([np.zeros((1, 2)), poses[:, :2]], axis=0)
    headings = np.unwrap(np.concatenate([[0.0], poses[:, 2]]))
    displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = displacement / interval_s
    acceleration = np.diff(np.concatenate([[speed[0]], speed])) / interval_s
    yaw_rate = np.diff(headings) / interval_s
    curvature = np.divide(yaw_rate, speed, out=np.zeros_like(speed), where=speed > 0.05)
    jerk = np.diff(np.concatenate([[acceleration[0]], acceleration])) / interval_s
    return {
        "speed": speed,
        "acceleration": acceleration,
        "yaw_rate": yaw_rate,
        "curvature": curvature,
        "jerk": jerk,
        "terminal_displacement": float(np.linalg.norm(poses[-1, :2])),
    }


def robust_standardize(
    values: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, list[float]]]:
    data = np.asarray(values, dtype=np.float64)
    mask = (
        np.isfinite(data)
        if valid is None
        else (np.isfinite(data) & np.asarray(valid, dtype=bool))
    )
    flattened = data.reshape(-1, data.shape[-1])
    flat_mask = mask.reshape(-1, data.shape[-1])
    medians = np.zeros(data.shape[-1], dtype=np.float64)
    scales = np.ones(data.shape[-1], dtype=np.float64)
    for index in range(data.shape[-1]):
        column = flattened[flat_mask[:, index], index]
        if len(column):
            medians[index] = np.median(column)
            q25, q75 = np.quantile(column, [0.25, 0.75])
            scales[index] = max(float(q75 - q25), 1e-3)
    normalized = np.where(mask, (data - medians) / scales, 0.0)
    return normalized.astype(np.float32), {
        "median": medians.tolist(),
        "scale": scales.tolist(),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
