"""Common, inference-safe utilities for the Gate C experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from tools.navsim_candidate_relative_audit.common import (
    configure_navsim_environment,
    discover_paths,
    json_safe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/shared_future_candidate_consequence_gate_c"
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SHARED_FUTURE_GATE_C_CACHE",
        str(REPO_ROOT / "outputs/shared_future_candidate_consequence_gate_c"),
    )
)
ALLOWED_SPLITS = {"train", "trainval"}
FORBIDDEN_SPLIT_PARTS = {"test", "navtest", "navhard", "private"}
BASE_COMMIT = "6e96cf7321b134c42c2cf0fbbc315cd61c925b11"

# These names may be targets or offline evaluation columns.  They may never be
# accepted by a deployable inference model/dataloader.
FORBIDDEN_INFERENCE_KEYS = {
    "future_image",
    "future_images",
    "future_annotation",
    "future_annotations",
    "future_gt_trajectory",
    "gt_future_trajectory",
    "logged_future",
    "shared_logged_future",
    "official_score",
    "aggregate_score",
    "pdm_score",
    "candidate_metrics",
}


def validate_training_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in ALLOWED_SPLITS or any(part in normalized for part in FORBIDDEN_SPLIT_PARTS):
        raise ValueError(
            f"Gate C split {split!r} is not legal training data; use train or trainval"
        )
    # This deployment exposes the legal training material under trainval.
    return normalized


def navsim_paths(split: str = "trainval"):
    validate_training_split(split)
    physical_split = "trainval"
    paths = discover_paths(physical_split)
    configure_navsim_environment(paths)
    return paths


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown(path: Path | str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_parquet(frame: pd.DataFrame, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, engine="pyarrow", compression="zstd")


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path], root: Path | None = None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(set(Path(value).resolve() for value in paths)):
        if not path.is_file():
            continue
        key = str(path.relative_to(root.resolve())) if root and path.is_relative_to(root.resolve()) else str(path)
        records[key] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def stable_scene_seed(scene_token: str, global_seed: int) -> int:
    payload = f"{global_seed}:{scene_token}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_output(args: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return result.stdout.strip()


def append_command(report_dir: Path | str, command: str) -> None:
    path = ensure_dir(report_dir) / "COMMANDS.sh"
    if not path.exists():
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
        path.chmod(0o755)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(command.rstrip() + "\n")


def assert_inference_batch_safe(batch: Mapping[str, Any]) -> None:
    offending = sorted(
        key for key in batch
        if key.lower() in FORBIDDEN_INFERENCE_KEYS
        or key.lower().startswith("future_")
        or key.lower().startswith("official_")
    )
    if offending:
        raise AssertionError(f"Future/offline-only fields reached inference: {offending}")


def assert_feature_names_safe(names: Sequence[str]) -> None:
    lowered = {str(name).lower() for name in names}
    offending = sorted(lowered & FORBIDDEN_INFERENCE_KEYS)
    offending.extend(sorted(name for name in lowered if name.startswith("official_")))
    if offending:
        raise AssertionError(f"Forbidden model input features: {sorted(set(offending))}")


def log_bootstrap_ci(
    values: pd.DataFrame,
    metric_column: str,
    log_column: str = "log_name",
    seed: int = 20260828,
    samples: int = 2000,
) -> tuple[float, float]:
    grouped = values.groupby(log_column, sort=True)[metric_column].mean().dropna()
    if grouped.empty:
        return float("nan"), float("nan")
    data = grouped.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(data, size=len(data), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def require_gate(report_dir: Path | str, gate: str) -> None:
    path = Path(report_dir) / "gate_status.json"
    if not path.is_file():
        raise RuntimeError(f"Missing gate status: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get(gate, {}).get("passed", False):
        raise RuntimeError(f"{gate} has not passed: {payload.get(gate)}")


def update_gate(report_dir: Path | str, gate: str, payload: Mapping[str, Any]) -> None:
    path = Path(report_dir) / "gate_status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current[gate] = dict(payload)
    write_json(path, current)
