#!/usr/bin/env python3
"""Cache shared future targets from a frozen Stage-I EMA student.

This tool runs no external teacher and performs no download. For every token it
encodes frames 0..11 once, then constructs eight ego-motion-aligned Stage-I EMA
memories for target frames 4..11. Entries are written atomically and bound to
the Stage-I checkpoint, config, datalist, and repository commit.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.field2plan_cache import (
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    sha256_file,
)
from starVLA.dataloader.grounded_world_cache import FutureTargetCacheReader
from starVLA.model.framework.QwenOFT_GroundedWorld import Qwenvl_OFT_GroundedWorld
from starVLA.model.modules.field2plan.camera_geometry import (
    center_crop_xywh,
    scale_intrinsics_for_crop_resize,
    sensor_to_lidar_to_ego_to_camera,
)
from starVLA.model.modules.field2plan.temporal_alignment import se2_poses_to_transforms
from starVLA.model.modules.field2plan.types import CameraBatch
from tools.field2plan.cache_dynamics_vjepa import _resolve_navsim_image_path


VIEWS = ("cam_f0", "cam_l0", "cam_r0")
ALL_FRAMES = tuple(range(12))
FUTURE_FRAMES = tuple(range(4, 12))


def _tokens(path: Path, max_samples: int) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid datalist: {path}") from error
    if not isinstance(payload, list) or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise ValueError("datalist must contain non-empty token strings")
    selected = payload[:max_samples] if max_samples > 0 else payload
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected datalist is empty or contains duplicates")
    return selected


def _metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"NAVSIM metadata not found: {path}")
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"corrupt NAVSIM metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"NAVSIM metadata must be a mapping: {path}")
    return value


def _frame_images(
    metadata: dict,
    frame: int,
    runtime_raw_root: Path,
    trainval_sensor_root: Path | None,
) -> tuple[list[Image.Image], np.ndarray]:
    images = []
    sizes = []
    for view in VIEWS:
        try:
            embedded = metadata["glo_images"][view]["image_paths"][frame]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(f"missing image metadata view={view} frame={frame}") from error
        path = _resolve_navsim_image_path(
            embedded,
            runtime_raw_root=runtime_raw_root,
            trainval_sensor_root=trainval_sensor_root,
        )
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(rgb.copy())
                sizes.append((rgb.height, rgb.width))
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"invalid NAVSIM image: {path}") from error
    return images, np.asarray(sizes, dtype=np.float32)


def _camera(
    metadata: dict,
    frame: int,
    raw_hw: np.ndarray,
    output_hw: tuple[int, int],
    lidar_to_ego: torch.Tensor,
    device: torch.device,
) -> CameraBatch:
    try:
        intrinsics = torch.as_tensor(
            np.stack(
                [metadata["glo_images"][view]["intrinsics"][frame] for view in VIEWS]
            ),
            dtype=torch.float32,
        )
        rotations = torch.as_tensor(
            np.stack(
                [
                    metadata["glo_images"][view]["sensor2lidar_rotations"][frame]
                    for view in VIEWS
                ]
            ),
            dtype=torch.float32,
        )
        translations = torch.as_tensor(
            np.stack(
                [
                    metadata["glo_images"][view]["sensor2lidar_translations"][frame]
                    for view in VIEWS
                ]
            ),
            dtype=torch.float32,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid camera calibration frame={frame}") from error
    sensor_to_lidar = torch.eye(4).repeat(len(VIEWS), 1, 1)
    sensor_to_lidar[:, :3, :3] = rotations
    sensor_to_lidar[:, :3, 3] = translations
    raw = torch.as_tensor(raw_hw, dtype=torch.float32)
    output = torch.tensor(output_hw, dtype=torch.float32).repeat(len(VIEWS), 1)
    crop = center_crop_xywh(raw, output)
    scaled = scale_intrinsics_for_crop_resize(intrinsics, crop, output)
    ego_to_camera = sensor_to_lidar_to_ego_to_camera(
        sensor_to_lidar[None], lidar_to_ego[None]
    )
    return CameraBatch(
        intrinsics=scaled[None].to(device),
        ego_to_camera=ego_to_camera.to(device),
        image_hw=output[None].to(device),
        view_names=VIEWS,
        frame_index=int(frame),
    ).validate()


def _rank_world() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--runtime-raw-root", type=Path, required=True)
    parser.add_argument("--trainval-sensor-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = _rank_world()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    tokens = _tokens(args.datalist, args.max_samples)
    checkpoint = Qwenvl_OFT_GroundedWorld._checkpoint_file(args.stage1_checkpoint)
    if args.validate_only:
        reader = FutureTargetCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        for token in tokens[rank::world_size]:
            reader.load(token)
        if world_size > 1:
            dist.barrier()
        print(f"[future-ema-cache] validation OK rank={rank}", flush=True)
        return

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "grounded_world.training.stage", "prior")
    OmegaConf.update(cfg, "grounded_world.training.init_checkpoint", None)
    OmegaConf.update(cfg, "grounded_world.future.enabled", False)
    OmegaConf.update(cfg, "grounded_world.planner.enabled", False)
    OmegaConf.update(cfg, "grounded_world.prior.source", "none")
    OmegaConf.update(cfg, "grounded_world.prior.teacher_mode", "none")
    device = torch.device(
        f"cuda:{local_rank}" if args.device == "cuda" and world_size > 1 else args.device
    )
    model = Qwenvl_OFT_GroundedWorld(cfg, load_checkpoints=False)
    model._load_declared_checkpoint(model, checkpoint)
    if model.ema_geometry_grounder is None or model.ema_world_core is None:
        raise RuntimeError("Stage-I checkpoint/config has no EMA world path")
    model.to(device).eval()
    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    output_hw = tuple(int(value) for value in cfg.grounded_world.camera.output_image_hw)
    lidar_to_ego = torch.as_tensor(
        cfg.grounded_world.camera.lidar_to_planning_ego, dtype=torch.float32
    )
    generated = resumed = 0
    with torch.inference_mode():
        for index, token in enumerate(tokens[rank::world_size], start=1):
            output_path = output_split / f"{token}.npz"
            if output_path.is_file() and not args.overwrite:
                resumed += 1
                continue
            metadata = _metadata(args.meta_root / f"{token}.pkl")
            geometry_fields = []
            global_poses = torch.as_tensor(
                metadata["glo_status"]["global_poses"][:12], dtype=torch.float32
            )
            global_from_ego = se2_poses_to_transforms(global_poses).to(device)
            for frame in ALL_FRAMES:
                images, raw_hw = _frame_images(
                    metadata,
                    frame,
                    args.runtime_raw_root,
                    args.trainval_sensor_root,
                )
                _, captured = model._run_visual_only(
                    [{"image": images, "lang": "Encode the driving scene."}]
                )
                if captured is None:
                    raise RuntimeError("Qwen visual encoder exposed no spatial features")
                camera = _camera(
                    metadata,
                    frame,
                    raw_hw,
                    output_hw,
                    lidar_to_ego,
                    device,
                )
                geometry_fields.append(
                    model.ema_geometry_grounder(captured.features, camera).field
                )
            fields = torch.stack(geometry_fields, dim=1)
            future_targets = []
            for frame in FUTURE_FRAMES:
                history = tuple(range(frame - 3, frame + 1))
                current_from_ego = (
                    torch.linalg.inv(global_from_ego[frame])[None]
                    @ global_from_ego[list(history)]
                )[None]
                current = model.ema_world_core(
                    fields[:, frame],
                    current_from_ego,
                    history_valid_mask=torch.ones(1, 4, device=device, dtype=torch.bool),
                    history_geometry=fields[:, list(history)],
                    stage="prior",
                ).current_dynamics.field
                future_targets.append(current[0])
            features = torch.stack(future_targets).float()
            valid = torch.isfinite(features).all(dim=1)
            safe = torch.where(valid[:, None], features, torch.zeros_like(features))
            atomic_write_npz(
                output_path,
                token=np.asarray(token),
                features=safe.cpu().numpy().astype(np.float16),
                valid_mask=valid.cpu().numpy().astype(np.bool_),
            )
            generated += 1
            if index % 10 == 0:
                print(
                    f"[future-ema-cache rank={rank}] {index} generated={generated} resumed={resumed}",
                    flush=True,
                )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        missing = [token for token in tokens if not (output_split / f"{token}.npz").is_file()]
        if missing:
            raise RuntimeError(f"future EMA cache incomplete; missing={missing[:10]}")
        sample = np.load(output_split / f"{tokens[0]}.npz", allow_pickle=False)
        feature_shape = list(sample["features"].shape)
        valid_shape = list(sample["valid_mask"].shape)
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = {
            "schema_version": 1,
            "cache_type": "grounded_world_future_target",
            "status": "complete",
            "producer": {
                "source": "student_ema",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "ema_decay": float(model.ema_decay),
                "shared_across_teacher_controls": True,
            },
            "generator": {
                "git_commit": git_commit,
                "tool": "tools/grounded_world/cache_future_student_ema.py",
                "config_sha256": sha256_file(args.config),
            },
            "splits": {
                args.split: {
                    "entry_count": len(tokens),
                    "tokens_sha256": hash_tokens(tokens),
                    "datalist_sha256": sha256_file(args.datalist),
                }
            },
            "temporal": {
                "future_frame_indices": list(FUTURE_FRAMES),
                "frame_interval_s": float(cfg.grounded_world.memory.frame_interval_s),
            },
            "tensor_schema": {
                "features": {"shape": feature_shape, "dtype": "float16"},
                "valid_mask": {"shape": valid_shape, "dtype": "bool"},
            },
        }
        atomic_write_json(args.output_dir / "manifest.json", manifest)
        reader = FutureTargetCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        print(f"[future-ema-cache] complete entries={len(tokens)}", flush=True)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
