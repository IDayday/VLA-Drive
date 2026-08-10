#!/usr/bin/env python3
"""Cache a generic V-JEPA current/history prior control.

This is the generic-temporal-prior control, not the Driving-JEPA main teacher.
It uses only explicit local repository/checkpoint paths, never downloads, and
stores frames 0..3 only. The separate cache type prevents accidental use of
the old future-V-JEPA cache as GroundedWorld prior supervision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
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
from starVLA.model.modules.field2plan.dynamics_teachers import (
    OfficialVJEPA2Adapter,
    seeded_orthogonal_projection,
)
from tools.field2plan.cache_dynamics_vjepa import (
    DEFAULT_DYNAMICS_VIEWS,
    _load_tokens,
    load_navsim_vjepa_inputs,
    project_and_resize_features,
    resample_vjepa_tokens,
)


HISTORY = (0, 1, 2, 3)
# The generic control must obey the same current/history-only information
# boundary as Driving-JEPA. Encoding frames 4..11 here would leak privileged
# future pixels through temporal self-attention even if only early tokens were
# serialized.
INPUT_FRAMES = HISTORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--runtime-raw-root", type=Path, required=True)
    parser.add_argument("--trainval-sensor-root", type=Path)
    parser.add_argument("--vjepa-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-variant", default="vit_large")
    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--output-height", type=int, default=16)
    parser.add_argument("--output-width", type=int, default=16)
    parser.add_argument("--projection-seed", type=int, default=20260810)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _rank_world() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def main() -> None:
    args = parse_args()
    if not args.vjepa_repo.is_dir():
        raise FileNotFoundError(f"local V-JEPA repository not found: {args.vjepa_repo}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"local V-JEPA checkpoint not found: {args.checkpoint}")
    tokens = _load_tokens(args.datalist, args.max_samples)
    rank, world_size, local_rank = _rank_world()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    if args.validate_only:
        reader = PriorCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        for token in tokens[rank::world_size]:
            reader.load(token)
        if world_size > 1:
            dist.barrier()
        print(f"[generic-vjepa-prior] validation OK rank={rank}", flush=True)
        return

    device_name = (
        f"cuda:{local_rank}" if args.device == "cuda" and world_size > 1 else args.device
    )
    adapter = OfficialVJEPA2Adapter(
        local_repo=args.vjepa_repo,
        checkpoint=args.checkpoint,
        model_variant=args.model_variant,
        device=device_name,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        num_frames=len(INPUT_FRAMES),
        image_size=args.image_size,
    )
    projection = seeded_orthogonal_projection(
        1024,
        args.feature_channels,
        seed=args.projection_seed,
        device=device_name,
    )
    token_centers = torch.arange(
        0.5, len(INPUT_FRAMES), 2.0, device=device_name, dtype=torch.float32
    )
    target_frames = torch.tensor(HISTORY, device=device_name, dtype=torch.float32)
    expected_shape = (
        len(HISTORY),
        len(DEFAULT_DYNAMICS_VIEWS),
        args.feature_channels,
        args.output_height,
        args.output_width,
    )
    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    generated = resumed = 0
    for index, token in enumerate(tokens[rank::world_size], start=1):
        output_path = output_split / f"{token}.npz"
        if output_path.is_file() and not args.overwrite:
            try:
                with np.load(output_path, allow_pickle=False) as payload:
                    valid_existing = (
                        str(payload["token"].item()) == token
                        and payload["features"].shape == expected_shape
                        and "input_frame_indices" in payload.files
                        and np.array_equal(
                            payload["input_frame_indices"],
                            np.asarray(INPUT_FRAMES, dtype=np.int64),
                        )
                    )
            except (OSError, ValueError, KeyError):
                valid_existing = False
            if valid_existing:
                resumed += 1
                continue
        inputs = load_navsim_vjepa_inputs(
            token=token,
            meta_path=args.meta_root / f"{token}.pkl",
            view_names=DEFAULT_DYNAMICS_VIEWS,
            input_frame_indices=INPUT_FRAMES,
            runtime_raw_root=args.runtime_raw_root,
            trainval_sensor_root=args.trainval_sensor_root,
        )
        preprocessed = adapter.preprocess_video(inputs.load_rgb())
        encoded = adapter.encode_video(preprocessed)
        aligned = resample_vjepa_tokens(
            encoded,
            token_center_indices=token_centers,
            target_frame_indices=target_frames,
        )
        features = project_and_resize_features(
            aligned,
            projection,
            (args.output_height, args.output_width),
        )
        finite = torch.isfinite(features).all(dim=2)
        safe = torch.where(finite[:, :, None], features, torch.zeros_like(features))
        atomic_write_npz(
            output_path,
            token=np.asarray(token),
            features=safe.cpu().numpy().astype(np.float16),
            confidence=finite.float().cpu().numpy().astype(np.float32),
            input_frame_indices=np.asarray(INPUT_FRAMES, dtype=np.int64),
        )
        generated += 1
        if index % 20 == 0:
            print(
                f"[generic-vjepa-prior rank={rank}] {index} generated={generated} resumed={resumed}",
                flush=True,
            )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = [token for token in tokens if not (output_split / f"{token}.npz").is_file()]
        if missing:
            raise RuntimeError(f"generic V-JEPA prior cache incomplete: {missing[:10]}")
        repo_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.vjepa_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        project_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "cache_type": "grounded_world_prior",
            "status": "complete",
            "teacher": {
                "name": "generic_vjepa2_1_current_only_v2",
                "domain": "generic_video",
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "repo": str(args.vjepa_repo.resolve()),
                "repo_commit": repo_commit,
            },
            "generator": {
                "git_commit": project_commit,
                "tool": "tools/grounded_world/cache_current_prior_vjepa.py",
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
                "input_frame_indices": list(INPUT_FRAMES),
                "frame_interval_s": 0.5,
            },
            "tensor_schema": {
                "features": {"shape": list(expected_shape), "dtype": "float16"},
                "confidence": {
                    "shape": [
                        len(HISTORY),
                        len(DEFAULT_DYNAMICS_VIEWS),
                        args.output_height,
                        args.output_width,
                    ],
                    "dtype": "float32",
                },
            },
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)
        reader = PriorCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        print(f"[generic-vjepa-prior] complete entries={len(tokens)}", flush=True)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
