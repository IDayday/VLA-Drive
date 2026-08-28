#!/usr/bin/env python3
"""Phase 0: record the exact local NAVSIM/runtime/data deployment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .common import (
    AUDIT_VERSION,
    add_common_arguments,
    bootstrap_navsim,
    discover_paths,
    package_version,
    repo_root,
    run_readonly_git,
    write_json,
    write_text,
)


def bounded_tree_stats(
    path: Path | None, maximum_entries: int = 1_000
) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
        "entry_count_scanned": 0,
        "regular_file_count_scanned": 0,
        "bytes_scanned": 0,
        "scan_truncated": False,
    }
    if not resolved.exists():
        return result
    if resolved.is_file():
        result.update(
            entry_count_scanned=1,
            regular_file_count_scanned=1,
            bytes_scanned=resolved.stat().st_size,
        )
        return result
    stack = [resolved]
    while stack and result["entry_count_scanned"] < maximum_entries:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except (OSError, PermissionError):
            continue
        for entry in entries:
            result["entry_count_scanned"] += 1
            if result["entry_count_scanned"] > maximum_entries:
                result["scan_truncated"] = True
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    result["regular_file_count_scanned"] += 1
                    result["bytes_scanned"] += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    result["scan_truncated"] = bool(stack) or result["scan_truncated"]
    result["size_is_lower_bound"] = result["scan_truncated"]
    return result


def command_output(arguments: list[str]) -> str | None:
    result = subprocess.run(arguments, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_inventory() -> dict[str, Any]:
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": count,
            "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        }
    except Exception as error:  # pragma: no cover - hardware/runtime specific
        return {"error": f"{type(error).__name__}: {error}"}


def discover_splits(paths: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if paths.public_root is None:
        return result
    for split in ("mini", "trainval", "test"):
        log_path = paths.public_root / "navsim_logs" / split
        sensor_path = paths.public_root / "sensor_blobs" / split
        result[split] = {
            "log_path": str(log_path),
            "log_exists": log_path.exists(),
            "log_pickle_count": len(list(log_path.glob("*.pkl")))
            if log_path.is_dir()
            else 0,
            "sensor_path": str(sensor_path),
            "sensor_exists": sensor_path.exists(),
        }
    return result


def existing_project_assets(repository: Path, paths: Any) -> dict[str, Any]:
    patterns = (
        "candidate",
        "trajectory",
        "pdm_score",
        "metric_cache",
        "prediction",
        "rollout",
    )
    result: dict[str, Any] = {}
    for pattern in patterns:
        output = command_output(
            ["rg", "-l", "-i", pattern, "research", "tools", "tests", "infer.py"]
        )
        result[pattern] = [] if not output else output.splitlines()[:80]
    for name in (
        "metric_cache",
        "candidate_cache",
        "consequence_cache",
        "effect_tube_cache",
    ):
        root = getattr(paths, name)
        result[f"resolved_{name}"] = bounded_tree_stats(root, maximum_entries=500)
    return result


def build_environment(args: argparse.Namespace) -> dict[str, Any]:
    paths = discover_paths(
        args, split="trainval" if args.split == "train" else args.split
    )
    navsim_runtime: dict[str, Any]
    try:
        navsim_runtime = bootstrap_navsim(paths)
        from navsim.common.dataclasses import (
            Annotations,
            Frame,
            Scene,
            SceneFilter,
            SensorConfig,
        )
        from navsim.common.dataloader import SceneLoader
        from navsim.planning.metric_caching.metric_cache import MetricCache
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
            PDMScorer,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
            PDMSimulator,
        )

        navsim_runtime["interfaces"] = {
            name: f"{value.__module__}.{value.__qualname__}"
            for name, value in {
                "Scene": Scene,
                "Frame": Frame,
                "Annotations": Annotations,
                "SceneLoader": SceneLoader,
                "SceneFilter": SceneFilter,
                "SensorConfig": SensorConfig,
                "MetricCache": MetricCache,
                "PDMSimulator": PDMSimulator,
                "PDMScorer": PDMScorer,
            }.items()
        }
    except Exception as error:
        navsim_runtime = {"error": f"{type(error).__name__}: {error}"}
    directory_names = (
        "navsim_devkit",
        "navsim_v1_devkit",
        "public_root",
        "logs_root",
        "sensors_root",
        "maps_root",
        "experiment_root",
        "processed_root",
        "metric_cache",
        "candidate_cache",
        "consequence_cache",
        "effect_tube_cache",
        "synthetic_scenes",
        "synthetic_sensors",
    )
    return {
        "audit_version": AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "path": str(repo_root()),
            "primary_checkout": str(paths.primary_checkout),
            "branch": run_readonly_git("branch", "--show-current"),
            "commit": run_readonly_git("rev-parse", "HEAD"),
            "dirty_short": run_readonly_git("status", "--short").splitlines(),
            "worktrees": run_readonly_git(
                "worktree", "list", "--porcelain"
            ).splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "numpy",
                    "nuplan-devkit",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "shapely",
                )
            },
            "gpu": gpu_inventory(),
        },
        "navsim_runtime": navsim_runtime,
        "paths": paths.to_json(),
        "path_inventory": {
            name: bounded_tree_stats(getattr(paths, name)) for name in directory_names
        },
        "available_splits": discover_splits(paths),
        "existing_project_assets": existing_project_assets(repo_root(), paths),
        "credential_redaction": {
            "environment_values_recorded": False,
            "policy": "Only named filesystem paths are recorded; credentials and tokens are omitted.",
        },
    }


def render_markdown(data: dict[str, Any]) -> str:
    repository = data["repository"]
    runtime = data["runtime"]
    navsim = data["navsim_runtime"]
    rows = []
    for name, item in data["path_inventory"].items():
        bytes_scanned = item.get("bytes_scanned")
        size = (
            "n/a" if bytes_scanned is None else f"{bytes_scanned / (1024**3):.3f} GiB"
        )
        suffix = " (lower bound)" if item.get("size_is_lower_bound") else ""
        rows.append(
            f"| `{name}` | {item.get('exists', False)} | `{item.get('resolved_path')}` | {size}{suffix} |"
        )
    interface_lines = [
        f"- `{name}`: `{path}`" for name, path in navsim.get("interfaces", {}).items()
    ]
    return "\n".join(
        [
            "# NAVSIM Candidate-Relative Audit: Environment",
            "",
            f"- Repository: `{repository['path']}`",
            f"- Branch: `{repository['branch']}`",
            f"- Commit: `{repository['commit']}`",
            f"- Dirty at capture: `{bool(repository['dirty_short'])}` (audit outputs themselves count as dirty before commit)",
            f"- Python: `{runtime['python'].splitlines()[0]}`",
            f"- NAVSIM requested root: `{navsim.get('requested_root')}`",
            f"- NAVSIM actual import: `{navsim.get('import_path')}`",
            f"- NAVSIM setup version: `{navsim.get('setup_version')}`",
            "",
            "## Audited local interfaces",
            "",
            *(interface_lines or [f"- Import error: `{navsim.get('error')}`"]),
            "",
            "## Filesystem deployment",
            "",
            "| Resource | Exists | Resolved path | Bounded size scan |",
            "|---|---:|---|---:|",
            *rows,
            "",
            "Large trees use a bounded scan and are explicitly marked as lower bounds. No credential values were captured.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = build_environment(args)
    write_json(args.output_dir / "environment.json", environment)
    write_text(args.output_dir / "ENVIRONMENT.md", render_markdown(environment))
    print(
        json.dumps(
            {
                "environment": str(args.output_dir / "environment.json"),
                "navsim": environment["navsim_runtime"],
                "split": args.split,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
