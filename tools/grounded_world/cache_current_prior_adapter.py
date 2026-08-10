#!/usr/bin/env python3
"""Run a user-supplied local Driving-JEPA adapter into a strict prior cache.

The adapter is lazy-imported from ``module:function``. No public API is guessed,
no package is installed, and no checkpoint is downloaded. Import or protocol
errors are explicit before cache generation starts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.field2plan_cache import (
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    sha256_file,
)
from starVLA.dataloader.grounded_world_cache import PriorCacheReader
from starVLA.model.modules.grounded_world.teacher_protocols import (
    CurrentPriorTeacherOutput,
)
from tools.field2plan.cache_dynamics_vjepa import (
    DEFAULT_DYNAMICS_VIEWS,
    _load_tokens,
    load_navsim_vjepa_inputs,
)


HISTORY = (0, 1, 2, 3)


def _factory(spec: str):
    if ":" not in spec:
        raise ValueError("adapter factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f"cannot import local Driving-JEPA adapter module {module_name!r}; "
            "install/provide it explicitly, automatic installation is disabled"
        ) from error
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise RuntimeError(f"Driving-JEPA adapter factory is not callable: {spec}")
    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--teacher-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--runtime-raw-root", type=Path, required=True)
    parser.add_argument("--trainval-sensor-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--output-height", type=int, default=16)
    parser.add_argument("--output-width", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.teacher_repo.is_dir():
        raise FileNotFoundError(f"Driving-JEPA local repository not found: {args.teacher_repo}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Driving-JEPA local checkpoint not found: {args.checkpoint}")
    tokens = _load_tokens(args.datalist, args.max_samples)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    if args.validate_only:
        reader = PriorCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        for token in tokens[rank::world_size]:
            reader.load(token)
        if world_size > 1:
            dist.barrier()
        return
    device = f"cuda:{local_rank}" if args.device == "cuda" and world_size > 1 else args.device
    adapter = _factory(args.adapter_factory)(
        local_repo=args.teacher_repo,
        checkpoint=args.checkpoint,
        device=device,
    )
    metadata = dict(getattr(adapter, "metadata", {}))
    if metadata.get("domain") != "driving" or not metadata.get("name"):
        raise ValueError("Driving-JEPA adapter metadata requires name and domain=driving")
    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    expected = (
        len(HISTORY),
        len(DEFAULT_DYNAMICS_VIEWS),
        args.feature_channels,
        args.output_height,
        args.output_width,
    )
    for token in tokens[rank::world_size]:
        path = output_split / f"{token}.npz"
        if path.is_file() and not args.overwrite:
            continue
        inputs = load_navsim_vjepa_inputs(
            token=token,
            meta_path=args.meta_root / f"{token}.pkl",
            view_names=DEFAULT_DYNAMICS_VIEWS,
            input_frame_indices=HISTORY,
            runtime_raw_root=args.runtime_raw_root,
            trainval_sensor_root=args.trainval_sensor_root,
        )
        result = adapter.infer(
            inputs.load_rgb(),
            history_frame_indices=HISTORY,
            output_hw=(args.output_height, args.output_width),
            output_channels=args.feature_channels,
        )
        if not isinstance(result, CurrentPriorTeacherOutput):
            raise TypeError("Driving-JEPA adapter must return CurrentPriorTeacherOutput")
        result.validate(len(HISTORY), len(DEFAULT_DYNAMICS_VIEWS))
        if result.features.shape != expected:
            raise ValueError(f"Driving-JEPA feature shape {result.features.shape} != {expected}")
        atomic_write_npz(
            path,
            token=np.asarray(token),
            features=result.features.astype(np.float16),
            confidence=result.confidence.astype(np.float32),
        )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = [token for token in tokens if not (output_split / f"{token}.npz").is_file()]
        if missing:
            raise RuntimeError(f"Driving-JEPA prior cache incomplete: {missing[:10]}")
        repo_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.teacher_repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        project_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "cache_type": "grounded_world_prior",
            "status": "complete",
            "teacher": {
                **metadata,
                "repo": str(args.teacher_repo.resolve()),
                "repo_commit": repo_commit,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint),
            },
            "generator": {
                "git_commit": project_commit,
                "tool": "tools/grounded_world/cache_current_prior_adapter.py",
                "adapter_factory": args.adapter_factory,
            },
            "splits": {
                args.split: {
                    "entry_count": len(tokens),
                    "tokens_sha256": hash_tokens(tokens),
                    "datalist_sha256": sha256_file(args.datalist),
                }
            },
            "temporal": {
                "current_frame_index": 3,
                "history_frame_indices": list(HISTORY),
                "frame_interval_s": 0.5,
            },
            "tensor_schema": {
                "features": {"shape": list(expected), "dtype": "float16"},
                "confidence": {
                    "shape": [4, 3, args.output_height, args.output_width],
                    "dtype": "float32",
                },
            },
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)
        reader = PriorCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        print(f"[driving-jepa-prior] complete entries={len(tokens)}", flush=True)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
