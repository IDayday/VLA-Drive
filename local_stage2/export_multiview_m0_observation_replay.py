#!/usr/bin/env python3
"""Cache scorer-private multiview tokens from a specified M0 vision encoder.

Only current F0/L0/R0/B0 images and current ego/navigation state enter the
visual forward.  The selected M0 checkpoint supplies its own vision weights;
no DrivOR checkpoint, generic DINO checkpoint, proposal tensor, future field,
or evaluator value is read by this exporter.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from local_stage2.export_multiview_dino_observation_replay import (
    CAMERA_NAMES,
    _atomic_json_dump,
    _atomic_torch_save,
    _chunked,
    _current_status,
    _load_inventory,
    _load_proposal_inventory,
    _loader_log_mapping,
    _sha256,
    _stable_shard,
)
from local_stage2.export_private_visual_replay import _resolve_visual_model
from local_stage2.export_public_base_scorer_cache import _compose_agent_config
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


def pool_m0_visual_tokens(
    raw_visual_tokens: torch.Tensor,
    crop_counts: torch.Tensor,
    pool_grid: Tuple[int, int],
    max_crops_per_camera: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool InternVL crop tokens into fixed camera-specific blocks.

    Args:
        raw_visual_tokens: ``[sum(crop_counts), 256, D]`` in scene-major,
            camera-major order.
        crop_counts: ``[B, 4]`` crop counts for F0/L0/R0/B0.
        pool_grid: spatial output grid per dynamic crop.
        max_crops_per_camera: fixed block capacity including the optional
            thumbnail produced by InternVL dynamic preprocessing.

    Returns:
        Padded tokens ``[B, 4 * max_crops * gh * gw, D]`` and a validity mask.
        Each camera owns a fixed contiguous block, so variable crop counts do
        not shift the semantic location of later cameras.
    """

    if raw_visual_tokens.ndim != 3 or raw_visual_tokens.shape[1] != 256:
        raise ValueError(
            "raw_visual_tokens must have shape [sum_crops, 256, width]"
        )
    if crop_counts.ndim != 2 or crop_counts.shape[1] != len(CAMERA_NAMES):
        raise ValueError(
            f"crop_counts must have shape [B, {len(CAMERA_NAMES)}]"
        )
    if crop_counts.dtype == torch.bool:
        raise ValueError("crop_counts must be integer counts, not bool")
    if not crop_counts.dtype.is_floating_point:
        integer_counts = crop_counts.to(dtype=torch.long)
    else:
        if not torch.equal(crop_counts, crop_counts.round()):
            raise ValueError("crop_counts must contain integers")
        integer_counts = crop_counts.to(dtype=torch.long)
    grid_height, grid_width = (int(pool_grid[0]), int(pool_grid[1]))
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("pool_grid dimensions must be positive")
    if max_crops_per_camera <= 0:
        raise ValueError("max_crops_per_camera must be positive")
    if bool((integer_counts <= 0).any()) or bool(
        (integer_counts > max_crops_per_camera).any()
    ):
        raise ValueError("crop count is outside the fixed camera-block capacity")
    if int(integer_counts.sum()) != raw_visual_tokens.shape[0]:
        raise ValueError("crop_counts do not match raw_visual_tokens")

    crop_count, _patch_count, width = raw_visual_tokens.shape
    patch_map = raw_visual_tokens.reshape(crop_count, 16, 16, width).permute(
        0, 3, 1, 2
    )
    pooled = F.adaptive_avg_pool2d(
        patch_map.float(), (grid_height, grid_width)
    )
    pooled = pooled.permute(0, 2, 3, 1).reshape(
        crop_count, grid_height * grid_width, width
    )

    tokens_per_crop = grid_height * grid_width
    camera_block = max_crops_per_camera * tokens_per_crop
    output = torch.zeros(
        crop_counts.shape[0],
        len(CAMERA_NAMES) * camera_block,
        width,
        dtype=torch.float16,
    )
    valid = torch.zeros(output.shape[:2], dtype=torch.bool)
    crop_offset = 0
    for scene_index in range(crop_counts.shape[0]):
        for camera_index in range(len(CAMERA_NAMES)):
            count = int(integer_counts[scene_index, camera_index])
            token_count = count * tokens_per_crop
            block_start = camera_index * camera_block
            camera_tokens = pooled[crop_offset : crop_offset + count].reshape(
                token_count, width
            )
            output[scene_index, block_start : block_start + token_count] = (
                camera_tokens.to(dtype=torch.float16, device="cpu")
            )
            valid[scene_index, block_start : block_start + token_count] = True
            crop_offset += count
    return output, valid


def _load_camera_crops(path: Path, max_dynamic_tiles: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_image(str(path), max_num=max_dynamic_tiles)


def _load_current_batch(
    tokens: Sequence[str],
    loader: SceneLoader,
    max_dynamic_tiles: int,
    image_executor: Optional[ThreadPoolExecutor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene_paths: List[List[Path]] = []
    statuses: List[torch.Tensor] = []
    for token in tokens:
        agent_input = loader.get_agent_input_from_token(token)
        cameras = agent_input.cameras[-1]
        paths = [getattr(cameras, name).image for name in CAMERA_NAMES]
        if any(not isinstance(path, Path) or not path.is_file() for path in paths):
            raise RuntimeError(f"Missing current camera path for {token}: {paths}")
        scene_paths.append(paths)
        statuses.append(_current_status(agent_input))

    flat_paths = [path for paths in scene_paths for path in paths]
    if image_executor is None:
        crop_groups = [
            _load_camera_crops(path, max_dynamic_tiles) for path in flat_paths
        ]
    else:
        crop_groups = list(
            image_executor.map(
                lambda path: _load_camera_crops(path, max_dynamic_tiles),
                flat_paths,
            )
        )
    counts = torch.as_tensor(
        [group.shape[0] for group in crop_groups], dtype=torch.int16
    ).reshape(len(tokens), len(CAMERA_NAMES))
    return torch.cat(crop_groups), counts, torch.stack(statuses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--feature-root", type=Path)
    inventory.add_argument("--proposal-pickle", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=None,
        help=(
            "Resolved Hydra config that constructed the checkpoint. Required "
            "when its VLM/LoRA architecture differs from the repository default."
        ),
    )
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-dynamic-tiles", type=int, default=4)
    parser.add_argument("--pool-height", type=int, default=2)
    parser.add_argument("--pool-width", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.batch_size <= 0 or args.chunk_size <= 0 or args.image_workers < 0:
        raise ValueError("batch-size/chunk-size must be positive")
    if args.chunk_size % args.batch_size:
        raise ValueError("chunk-size must be divisible by batch-size")
    if args.max_dynamic_tiles <= 0:
        raise ValueError("max-dynamic-tiles must be positive")
    for path in (
        args.feature_root or args.proposal_pickle,
        args.repo_root,
        args.checkpoint,
        args.vlm_path,
        args.log_path,
        args.sensor_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.resolved_config is not None and not args.resolved_config.is_file():
        raise FileNotFoundError(args.resolved_config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M0 vision export")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(2)
    np.random.seed(2)
    os.environ["DRIVEVLA_VLM_CONFIG"] = str(args.vlm_path.resolve())
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"

    if args.feature_root is not None:
        all_tokens, inventory_log_for_token = _load_inventory(args.feature_root)
    else:
        all_tokens = _load_proposal_inventory(args.proposal_pickle)
        inventory_log_for_token = {}
    tokens = [
        token
        for token in all_tokens
        if _stable_shard(token, args.shard_count) == args.shard_index
    ]
    if args.max_scenes > 0:
        tokens = tokens[: args.max_scenes]
    requested_logs = (
        sorted({inventory_log_for_token[token] for token in tokens})
        if inventory_log_for_token
        else None
    )
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        log_names=requested_logs,
        tokens=tokens,
    )
    sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[3],
        lidar_pc=[],
    )
    loader = SceneLoader(
        args.log_path,
        args.sensor_root,
        scene_filter,
        sensor_config,
        load_image_path=True,
    )
    missing = set(tokens).difference(str(value) for value in loader.tokens)
    if missing:
        raise RuntimeError(f"SceneLoader misses {len(missing)} inventory tokens")
    log_for_token = _loader_log_mapping(loader)
    if set(tokens).difference(log_for_token):
        raise RuntimeError("SceneLoader log mapping does not cover inventory")
    if inventory_log_for_token:
        mismatched = [
            token
            for token in tokens
            if inventory_log_for_token[token] != log_for_token[token]
        ]
        if mismatched:
            raise RuntimeError(
                f"Inventory/SceneLoader log mismatch for {len(mismatched)} tokens"
            )

    cfg = _compose_agent_config(
        args.repo_root.resolve(),
        args.checkpoint.resolve(),
        resolved_config=(
            args.resolved_config.resolve()
            if args.resolved_config is not None
            else None
        ),
    )
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    visual_model, wrapper_chain = _resolve_visual_model(agent.backbone)

    shard_dir = args.output_dir / (
        f"m0_multiview_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(_chunked(tokens, args.chunk_size))
    existing = sorted(shard_dir.glob("chunk_*.pt"))
    completed_chunks = len(existing)
    for index in range(completed_chunks):
        if not (shard_dir / f"chunk_{index:06d}.pt").is_file():
            raise RuntimeError("non-contiguous resumable M0 multiview cache")
    completed_rows = sum(len(chunk) for chunk in chunks[:completed_chunks])
    pending_tokens = tokens[completed_rows:]
    chunk_index = completed_chunks
    buffer: Dict[str, List[object]] = {
        "tokens": [],
        "log_names": [],
        "visual_tokens": [],
        "visual_valid_mask": [],
        "crop_counts": [],
        "status_feature": [],
    }
    started = time.monotonic()
    max_crops_per_camera = args.max_dynamic_tiles + 1
    executor_context = (
        ThreadPoolExecutor(max_workers=args.image_workers)
        if args.image_workers
        else nullcontext(None)
    )
    print(
        f"M0_MULTIVIEW_EXPORT_READY shard={args.shard_index}/{args.shard_count} "
        f"pending={len(pending_tokens)} batch={args.batch_size}",
        flush=True,
    )
    with executor_context as image_executor, torch.inference_mode():
        for start in range(0, len(pending_tokens), args.batch_size):
            batch_tokens = pending_tokens[start : start + args.batch_size]
            pixels, crop_counts, statuses = _load_current_batch(
                batch_tokens,
                loader,
                args.max_dynamic_tiles,
                image_executor,
            )
            raw = visual_model.extract_feature(
                pixels.cuda(non_blocking=True).bfloat16()
            )
            pooled, valid = pool_m0_visual_tokens(
                raw,
                crop_counts,
                (args.pool_height, args.pool_width),
                max_crops_per_camera,
            )
            buffer["tokens"].extend(batch_tokens)
            buffer["log_names"].extend(
                log_for_token[token] for token in batch_tokens
            )
            buffer["visual_tokens"].append(pooled)
            buffer["visual_valid_mask"].append(valid)
            buffer["crop_counts"].append(crop_counts)
            buffer["status_feature"].append(statuses)
            expected = len(chunks[chunk_index])
            if len(buffer["tokens"]) == expected:
                row_count = len(buffer["tokens"])
                payload = {
                    "schema_version": 1,
                    "tokens": list(buffer["tokens"]),
                    "log_names": list(buffer["log_names"]),
                    "visual_tokens": torch.cat(buffer["visual_tokens"]),
                    "visual_valid_mask": torch.cat(buffer["visual_valid_mask"]),
                    "crop_counts": torch.cat(buffer["crop_counts"]),
                    "status_feature": torch.cat(buffer["status_feature"]),
                    "history_trajectory": torch.empty(row_count, 0),
                    "high_command_one_hot": torch.empty(row_count, 0),
                    "camera_names": CAMERA_NAMES,
                    "pool_grid": (args.pool_height, args.pool_width),
                    "max_crops_per_camera": max_crops_per_camera,
                }
                _atomic_torch_save(
                    payload, shard_dir / f"chunk_{chunk_index:06d}.pt"
                )
                chunk_index += 1
                buffer = {key: [] for key in buffer}
                print(
                    json.dumps(
                        {
                            "shard": f"{args.shard_index}/{args.shard_count}",
                            "chunks": f"{chunk_index}/{len(chunks)}",
                            "processed": start + len(batch_tokens),
                            "pending_total": len(pending_tokens),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            elif len(buffer["tokens"]) > expected:
                raise RuntimeError("batch crossed deterministic chunk boundary")
    if any(buffer.values()) or chunk_index != len(chunks):
        raise RuntimeError("M0 multiview cache did not finish cleanly")

    checkpoint_sha256 = _sha256(args.checkpoint)
    manifest: Mapping[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "M0NativeMultiviewObservationExporter",
        "inventory_type": (
            "feature_replay"
            if args.feature_root is not None
            else "m0_proposal_submission"
        ),
        "feature_inventory": (
            str(args.feature_root.resolve())
            if args.feature_root is not None
            else None
        ),
        "proposal_inventory": (
            str(args.proposal_pickle.resolve())
            if args.proposal_pickle is not None
            else None
        ),
        "proposal_inventory_sha256": (
            _sha256(args.proposal_pickle)
            if args.proposal_pickle is not None
            else None
        ),
        "m0_checkpoint": str(args.checkpoint.resolve()),
        "m0_checkpoint_sha256": checkpoint_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_agent_config": (
            str(args.resolved_config.resolve())
            if args.resolved_config is not None
            else None
        ),
        "resolved_class": (
            "navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent"
        ),
        "resolved_vlm_path": str(args.vlm_path.resolve()),
        "visual_model_wrapper_chain": wrapper_chain,
        "representation_source": "checkpoint_m0_vision_encoder",
        "additional_external_model_checkpoint_or_representation_used": False,
        "drivor_checkpoint_or_representation_used": False,
        "camera_names": CAMERA_NAMES,
        "max_dynamic_tiles": args.max_dynamic_tiles,
        "max_crops_per_camera": max_crops_per_camera,
        "pool_grid": (args.pool_height, args.pool_width),
        "visual_token_count": (
            len(CAMERA_NAMES)
            * max_crops_per_camera
            * args.pool_height
            * args.pool_width
        ),
        "visual_width": int(raw.shape[-1]),
        "scene_count": len(tokens),
        "log_count": len({log_for_token[token] for token in tokens}),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "chunk_count": len(chunks),
        "current_context_fields": [
            "ego_pose",
            "ego_velocity",
            "ego_acceleration",
            "driving_command",
        ],
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_or_factor_input": False,
        "proposal_input": False,
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
