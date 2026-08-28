#!/usr/bin/env python3
"""Audit the exact local repository, runtime, data and cache deployment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .common import (
    REPO_ROOT,
    add_common_arguments,
    append_command,
    directory_size,
    ensure_output_dir,
    paths_from_args,
    run_text,
    write_json,
    write_markdown,
)


def package_info(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # audit records failures instead of hiding them
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return {
        "available": True,
        "version": version,
        "import_path": getattr(module, "__file__", None),
    }


def find_navsim_installations() -> list[str]:
    results = []
    for entry in sys.path:
        if not entry:
            entry = os.getcwd()
        candidate = Path(entry) / "navsim/__init__.py"
        if candidate.is_file():
            results.append(str(candidate.resolve()))
    for candidate in (Path("/mnt/navsim/__init__.py"), Path("/mnt/project/DriveDreamer-Policy/navsim/navsim/__init__.py")):
        if candidate.is_file() and str(candidate.resolve()) not in results:
            results.append(str(candidate.resolve()))
    return sorted(results)


def inventory_candidates(repo_root: Path, max_results: int = 200) -> list[dict[str, Any]]:
    terms = ("candidate", "proposal", "trajectory", "prediction", "rollout", "pdm_score")
    extensions = {".pkl", ".pickle", ".pt", ".pth", ".npz", ".parquet", ".csv", ".json"}
    results: list[dict[str, Any]] = []
    skip = {".git", ".cache", "nuplan-devkit", "reports", "__pycache__"}
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [name for name in dirs if name not in skip]
        for name in files:
            lower = name.lower()
            path = Path(root) / name
            if path.suffix.lower() in extensions and any(term in lower for term in terms):
                try:
                    size = path.stat().st_size
                except OSError:
                    size = None
                results.append({"path": str(path), "bytes": size})
                if len(results) >= max_results:
                    return results
    return results


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_from_args(args)
    output_dir = ensure_output_dir(args.output_dir)
    packages = {
        name: package_info(name)
        for name in ("navsim", "torch", "numpy", "nuplan", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "shapely")
    }
    cuda: dict[str, Any] = {}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        cuda = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    split_paths = {
        "mini": "/mnt/navsim/mini_navsim_logs/mini",
        "trainval": "/mnt/navsim/trainval_navsim_logs/trainval",
    }
    available_splits = {
        name: {"path": path, "exists": Path(path).is_dir(), "log_pickles": len(list(Path(path).glob("*.pkl"))) if Path(path).is_dir() else 0}
        for name, path in split_paths.items()
    }
    path_inventory = {
        "log_path": directory_size(paths.log_path),
        "sensor_blobs_path": directory_size(paths.sensor_blobs_path),
        "map_path": directory_size(paths.map_path),
        "metric_cache_path": directory_size(paths.metric_cache_path),
        "navsim_exp_root": directory_size(paths.navsim_exp_root),
        "synthetic_scene_path": directory_size(paths.synthetic_scene_path),
        "synthetic_sensor_path": directory_size(paths.synthetic_sensor_path),
        "v2_devkit_root": directory_size(paths.v2_devkit_root),
    }
    cache_metadata = list((paths.metric_cache_path / "metadata").glob("*.csv")) if (paths.metric_cache_path / "metadata").is_dir() else []
    cache_rows = None
    if cache_metadata:
        try:
            with cache_metadata[0].open("r", encoding="utf-8") as stream:
                cache_rows = max(sum(1 for _ in stream) - 1, 0)
        except OSError:
            pass
    git = {
        "branch": run_text(["git", "branch", "--show-current"], REPO_ROOT),
        "commit": run_text(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "status_short": run_text(["git", "status", "--short"], REPO_ROOT) or "",
        "describe": run_text(["git", "describe", "--tags", "--always", "--dirty"], REPO_ROOT),
    }
    v2_git = run_text(["git", "rev-parse", "HEAD"], paths.v2_devkit_root) if paths.v2_devkit_root else None
    result = {
        "audit_scope": "training/public validation only",
        "project_path": str(REPO_ROOT),
        "git": git,
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": packages,
        "cuda": cuda,
        "nvidia_smi": run_text(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            timeout=10,
        ),
        "paths": paths.to_json(),
        "path_inventory": path_inventory,
        "available_splits": available_splits,
        "metric_cache_metadata_files": [str(path) for path in cache_metadata],
        "metric_cache_metadata_rows": cache_rows,
        "navsim_installations": find_navsim_installations(),
        "runtime_navsim_import": packages["navsim"].get("import_path"),
        "v2_git_commit": v2_git,
        "existing_candidate_or_score_artifacts": inventory_candidates(REPO_ROOT),
        "privacy_note": "Only allowlisted path/configuration variables were recorded; credentials and tokens were not enumerated.",
    }
    write_json(output_dir / "environment.json", result)
    runtime = result["runtime_navsim_import"]
    navsim_version = packages["navsim"].get("version")
    md = f"""# NAVSIM Candidate-relative Audit: Environment

## Runtime identity

- Repository: `{REPO_ROOT}`
- Branch: `{git['branch']}`
- Commit: `{git['commit']}`
- Dirty state at inspection: `{git['status_short'] or 'clean'}`
- Python: `{sys.version.splitlines()[0]}` (`{sys.executable}`)
- NAVSIM version: `{navsim_version}`
- Runtime NAVSIM import: `{runtime}`
- Other discovered NAVSIM package roots: {len(result['navsim_installations'])}
- CUDA: `{cuda.get('available')}`, devices: `{cuda.get('device_count', 0)}`

The runtime package is the code in this checkout.  The separately deployed NAVSIM v2 tree is recorded but is not imported into the v1 audit process.

## Read-only data inputs

- Split: `{paths.split}` (test/private-test splits are rejected by the CLI)
- Logs: `{paths.log_path}`
- Sensor blobs: `{paths.sensor_blobs_path}`
- Maps: `{paths.map_path}`
- Metric cache: `{paths.metric_cache_path}` ({cache_rows} metadata rows)
- Synthetic scenes discovered: `{paths.synthetic_scene_path}`
- Synthetic sensors discovered: `{paths.synthetic_sensor_path}`
- NAVSIM v2 devkit discovered: `{paths.v2_devkit_root}` (commit `{v2_git}`)

Directory byte counts in `environment.json` are best-effort `du` estimates with a timeout; `null` means the mount was too large to traverse within that bound, not that it is empty.

## Existing project artifacts

The scan found {len(result['existing_candidate_or_score_artifacts'])} filename-matched candidate/trajectory/PDM artifacts in the repository.  Metric-cache entries themselves are inventoried separately and never modified.
"""
    write_markdown(output_dir / "ENVIRONMENT.md", md)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    inspect(args)
    append_command(args.output_dir.resolve(), "python -m tools.navsim_candidate_relative_audit.inspect_environment " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
