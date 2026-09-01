#!/usr/bin/env python3
"""Cache standalone scorer-private DINO tokens from current NAVSIM cameras.

This exporter never constructs a driving model.  It applies a generic frozen
DINOv2 image backbone to the four *current* camera views and stores spatial
tokens plus current ego state.  M0 proposals and offline PDM factors stay in
separate replay artifacts and are not read by the visual forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader


CAMERA_NAMES: Tuple[str, ...] = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_shard(token: str, shard_count: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_inventory(feature_root: Path) -> Tuple[List[str], Dict[str, str]]:
    """Read immutable token/log inventory without loading proposal tensors."""

    chunk_paths = sorted(feature_root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not chunk_paths:
        raise RuntimeError(f"No replay feature chunks found in {feature_root}")
    log_for_token: Dict[str, str] = {}
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tokens = [str(value) for value in payload["tokens"]]
        logs = [str(value) for value in payload["log_names"]]
        if len(tokens) != len(logs):
            raise RuntimeError(f"Token/log row mismatch in {path}")
        for token, log_name in zip(tokens, logs):
            if token in log_for_token:
                raise RuntimeError(f"Duplicate inventory token {token}")
            log_for_token[token] = log_name
    return sorted(log_for_token), log_for_token


def _load_proposal_inventory(proposal_pickle: Path) -> List[str]:
    """Read only the scene-token inventory from an M0 proposal submission."""

    with proposal_pickle.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected token dictionary in {proposal_pickle}")
    tokens = sorted(str(value) for value in payload)
    if len(tokens) != len(set(tokens)):
        raise RuntimeError(f"Duplicate proposal tokens in {proposal_pickle}")
    return tokens


def _loader_log_mapping(loader: SceneLoader) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for log_name, tokens in loader.get_tokens_list_per_log().items():
        for token in tokens:
            token = str(token)
            if token in mapping:
                raise RuntimeError(f"Scene token {token} occurs in multiple logs")
            mapping[token] = str(log_name)
    return mapping


def _load_image(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _current_status(agent_input) -> torch.Tensor:
    status = agent_input.ego_statuses[-1]
    return torch.cat(
        (
            torch.as_tensor(status.ego_pose, dtype=torch.float32),
            torch.as_tensor(status.ego_velocity, dtype=torch.float32),
            torch.as_tensor(status.ego_acceleration, dtype=torch.float32),
            torch.as_tensor(status.driving_command, dtype=torch.float32),
        )
    )


def _load_current_batch(
    entries: Sequence[str],
    loader: SceneLoader,
    image_size: Tuple[int, int],
    image_executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scene_paths: List[List[Path]] = []
    statuses: List[torch.Tensor] = []
    for token in entries:
        agent_input = loader.get_agent_input_from_token(token)
        cameras = agent_input.cameras[-1]
        paths: List[Path] = []
        for name in CAMERA_NAMES:
            image_path = getattr(cameras, name).image
            if not isinstance(image_path, Path) or not image_path.is_file():
                raise RuntimeError(f"Missing current {name} path for {token}: {image_path}")
            paths.append(image_path)
        scene_paths.append(paths)
        statuses.append(_current_status(agent_input))
    flat_paths = [path for paths in scene_paths for path in paths]
    if image_executor is None:
        flat_images = [_load_image(path, image_size) for path in flat_paths]
    else:
        flat_images = list(
            image_executor.map(
                lambda path: _load_image(path, image_size),
                flat_paths,
            )
        )
    camera_count = len(CAMERA_NAMES)
    images = [
        torch.stack(flat_images[start : start + camera_count])
        for start in range(0, len(flat_images), camera_count)
    ]
    return torch.stack(images), torch.stack(statuses)


def _build_backbone(
    model_name: str,
    model_weights: Path,
    image_size: Tuple[int, int],
    device: torch.device,
) -> torch.nn.Module:
    width, height = image_size
    model = timm.create_model(
        model_name,
        pretrained=True,
        pretrained_cfg_overlay={"file": str(model_weights)},
        img_size=(height, width),
        num_classes=0,
        in_chans=3,
    ).to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_spatial_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    pool_grid: Tuple[int, int],
) -> torch.Tensor:
    batch_size, camera_count, channels, height, width = images.shape
    flat = images.reshape(batch_size * camera_count, channels, height, width)
    encoded = model.forward_features(flat)
    prefix_count = int(model.num_prefix_tokens)
    patch_tokens = encoded[:, prefix_count:, :]
    patch_size = int(model.patch_embed.patch_size[0])
    patch_height, patch_width = height // patch_size, width // patch_size
    if patch_tokens.shape[1] != patch_height * patch_width:
        raise RuntimeError(
            f"DINO patch count mismatch: {patch_tokens.shape[1]} != "
            f"{patch_height}x{patch_width}"
        )
    patch_map = patch_tokens.transpose(1, 2).reshape(
        batch_size * camera_count,
        patch_tokens.shape[-1],
        patch_height,
        patch_width,
    )
    pooled = F.adaptive_avg_pool2d(patch_map, pool_grid)
    pooled = pooled.flatten(2).transpose(1, 2)
    return pooled.reshape(batch_size, camera_count * pooled.shape[1], -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--feature-root", type=Path)
    inventory.add_argument("--proposal-pickle", type=Path)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="timm/vit_small_patch14_reg4_dinov2.lvd142m"
    )
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=644)
    parser.add_argument("--image-height", type=int, default=392)
    parser.add_argument("--pool-height", type=int, default=7)
    parser.add_argument("--pool-width", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=16,
        help="Thread workers for read-only JPEG decode/resize; zero is serial.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.chunk_size <= 0 or args.batch_size <= 0 or args.image_workers < 0:
        raise ValueError("batch-size and chunk-size must be positive")
    if args.chunk_size % args.batch_size:
        raise ValueError("chunk-size must be divisible by batch-size")
    for value in (
        args.feature_root or args.proposal_pickle,
        args.log_path,
        args.sensor_root,
        args.model_weights,
    ):
        if not value.exists():
            raise FileNotFoundError(value)
    if args.image_width % 14 or args.image_height % 14:
        raise ValueError("DINO image dimensions must be divisible by patch size 14")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(2)
    np.random.seed(2)
    if args.feature_root is not None:
        tokens, inventory_log_for_token = _load_inventory(args.feature_root)
    else:
        tokens = _load_proposal_inventory(args.proposal_pickle)
        inventory_log_for_token = {}
    tokens = [
        token
        for token in tokens
        if _stable_shard(token, args.shard_count) == args.shard_index
    ]
    if args.max_scenes > 0:
        tokens = tokens[: args.max_scenes]
    logs = (
        sorted({inventory_log_for_token[token] for token in tokens})
        if inventory_log_for_token
        else None
    )
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        log_names=logs,
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
    missing_logs = set(tokens).difference(log_for_token)
    if missing_logs:
        raise RuntimeError(f"SceneLoader has no log mapping for {len(missing_logs)} tokens")
    if inventory_log_for_token:
        mismatched_logs = [
            token
            for token in tokens
            if inventory_log_for_token[token] != log_for_token[token]
        ]
        if mismatched_logs:
            raise RuntimeError(
                f"Inventory/SceneLoader log mismatch for {len(mismatched_logs)} tokens"
            )
    logs = sorted({log_for_token[token] for token in tokens})

    image_size = (args.image_width, args.image_height)
    pool_grid = (args.pool_height, args.pool_width)
    model = _build_backbone(args.model_name, args.model_weights, image_size, device)
    shard_dir = args.output_dir / (
        f"dino_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(_chunked(tokens, args.chunk_size))
    existing = sorted(shard_dir.glob("chunk_*.pt"))
    completed_chunks = len(existing)
    for index in range(completed_chunks):
        if not (shard_dir / f"chunk_{index:06d}.pt").is_file():
            raise RuntimeError("non-contiguous resumable DINO cache")
    pending_tokens = tokens[sum(len(chunk) for chunk in chunks[:completed_chunks]) :]
    started = time.monotonic()
    chunk_index = completed_chunks
    buffer: Dict[str, List[object]] = {
        "tokens": [],
        "log_names": [],
        "visual_tokens": [],
        "status_feature": [],
    }
    executor_context = (
        ThreadPoolExecutor(max_workers=args.image_workers)
        if args.image_workers
        else nullcontext(None)
    )
    with executor_context as image_executor, torch.inference_mode():
        for start in range(0, len(pending_tokens), args.batch_size):
            batch_tokens = pending_tokens[start : start + args.batch_size]
            images, statuses = _load_current_batch(
                batch_tokens,
                loader,
                image_size,
                image_executor,
            )
            visual = _encode_spatial_tokens(
                model,
                images.to(device=device, dtype=torch.float32, non_blocking=True),
                pool_grid,
            )
            buffer["tokens"].extend(batch_tokens)
            buffer["log_names"].extend(log_for_token[token] for token in batch_tokens)
            buffer["visual_tokens"].append(visual.half().cpu())
            buffer["status_feature"].append(statuses)
            expected_chunk_size = len(chunks[chunk_index])
            if len(buffer["tokens"]) == expected_chunk_size:
                row_count = len(buffer["tokens"])
                visual_rows = torch.cat(buffer["visual_tokens"])
                payload = {
                    "schema_version": 1,
                    "tokens": list(buffer["tokens"]),
                    "log_names": list(buffer["log_names"]),
                    "visual_tokens": visual_rows,
                    "visual_valid_mask": torch.ones(
                        visual_rows.shape[:2], dtype=torch.bool
                    ),
                    "status_feature": torch.cat(buffer["status_feature"]),
                    "history_trajectory": torch.empty(row_count, 0),
                    "high_command_one_hot": torch.empty(row_count, 0),
                    "camera_names": CAMERA_NAMES,
                    "pool_grid": pool_grid,
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
            elif len(buffer["tokens"]) > expected_chunk_size:
                raise RuntimeError("batch crossed deterministic chunk boundary")
    if any(buffer.values()) or chunk_index != len(chunks):
        raise RuntimeError("DINO current-observation cache did not finish cleanly")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "StandaloneMultiviewDinoObservationExporter",
        "inventory_type": (
            "feature_replay" if args.feature_root is not None else "m0_proposal_submission"
        ),
        "feature_inventory": (
            str(args.feature_root.resolve()) if args.feature_root is not None else None
        ),
        "proposal_inventory": (
            str(args.proposal_pickle.resolve())
            if args.proposal_pickle is not None
            else None
        ),
        "proposal_inventory_sha256": (
            _sha256(args.proposal_pickle) if args.proposal_pickle is not None else None
        ),
        "sensor_root": str(args.sensor_root.resolve()),
        "model_name": args.model_name,
        "model_weights": str(args.model_weights.resolve()),
        "checkpoint_sha256": _sha256(args.model_weights),
        "generic_visual_pretraining_only": True,
        "driving_model_checkpoint": None,
        "drivor_checkpoint_or_representation_used": False,
        "camera_names": CAMERA_NAMES,
        "image_size": image_size,
        "pool_grid": pool_grid,
        "image_workers": args.image_workers,
        "visual_token_count": len(CAMERA_NAMES) * args.pool_height * args.pool_width,
        "visual_width": int(model.num_features),
        "scene_count": len(tokens),
        "log_count": len(logs),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "chunk_count": len(chunks),
        "current_context_fields": ["ego_pose", "ego_velocity", "ego_acceleration", "driving_command"],
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_or_factor_input": False,
        "proposal_input": False,
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
