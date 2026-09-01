"""Export pooled current-image tokens for scorer-private representation.

The cache contains no trajectory target, future annotation, metric cache or
PDM value.  It uses the exact frozen public-Base vision/LoRA weights and pools
each 16x16 InternVL crop grid to a configurable spatial grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from local_stage2.export_public_base_scorer_cache import (
    _CacheNameBuilder,
    _compose_agent_config,
)
from navsim.planning.training.dataset import CacheOnlyDataset, drivevla_cached_collate


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _partition(values: Sequence[str], count: int, index: int) -> List[str]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid shard count/index")
    return list(values[index::count])


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _first_missing_chunk(directory: Path, count: int) -> int:
    missing = next(
        (
            index
            for index in range(count)
            if not (directory / f"chunk_{index:06d}.pt").is_file()
        ),
        count,
    )
    later = [
        index
        for index in range(missing + 1, count)
        if (directory / f"chunk_{index:06d}.pt").is_file()
    ]
    if later:
        raise RuntimeError(f"Non-contiguous cache after first missing chunk: {later[:5]}")
    return missing


def _resolve_visual_model(backbone):
    chain = []
    current = backbone
    for _ in range(6):
        chain.append(f"{type(current).__module__}.{type(current).__name__}")
        if callable(getattr(current, "extract_feature", None)):
            return current, chain
        current = getattr(current, "model", None)
        if current is None:
            break
    raise RuntimeError(f"Could not resolve extract_feature through {chain}")


def pool_visual_tokens(
    raw_visual_tokens: torch.Tensor,
    crop_counts: Sequence[int],
    pool_grid: int,
    max_crops: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool `[sum(crops), 256, D]` and pad scenes to fixed token length."""

    if raw_visual_tokens.ndim != 3 or raw_visual_tokens.shape[1] != 256:
        raise ValueError(
            f"Expected raw visual tokens [C,256,D], got {raw_visual_tokens.shape}"
        )
    if sum(crop_counts) != raw_visual_tokens.shape[0]:
        raise ValueError("crop counts do not match raw token batch")
    if not 1 <= pool_grid <= 16 or 16 % pool_grid:
        raise ValueError("pool_grid must evenly divide the 16x16 patch grid")
    if any(count <= 0 or count > max_crops for count in crop_counts):
        raise ValueError(f"crop count exceeds configured max_crops={max_crops}")

    crop_count, _patch_count, width = raw_visual_tokens.shape
    grid = raw_visual_tokens.reshape(crop_count, 16, 16, width).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid.float(), (pool_grid, pool_grid))
    pooled = pooled.permute(0, 2, 3, 1).reshape(crop_count, pool_grid * pool_grid, width)
    max_tokens = max_crops * pool_grid * pool_grid
    output = torch.zeros(
        len(crop_counts),
        max_tokens,
        width,
        dtype=torch.float16,
    )
    valid = torch.zeros(len(crop_counts), max_tokens, dtype=torch.bool)
    crop_offset = 0
    for scene_index, count in enumerate(crop_counts):
        token_count = count * pool_grid * pool_grid
        scene = pooled[crop_offset : crop_offset + count].reshape(token_count, width)
        output[scene_index, :token_count] = scene.to(dtype=torch.float16, device="cpu")
        valid[scene_index, :token_count] = True
        crop_offset += count
    return output, valid


def _pixel_list(pixel_values) -> List[torch.Tensor]:
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected pixel tensor [B,C,3,H,W], got {pixel_values.shape}")
        return list(pixel_values.unbind(0))
    if not isinstance(pixel_values, list) or not pixel_values:
        raise TypeError("pixel_values must be a non-empty tensor or list")
    return pixel_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument("--max-crops", type=int, default=12)
    parser.add_argument("--limit-scenes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.repo_root, args.checkpoint, args.vlm_path, args.feature_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.chunk_size % args.batch_size:
        raise ValueError("chunk_size must be divisible by batch_size")
    os.environ["DRIVEVLA_VLM_CONFIG"] = str(args.vlm_path.resolve())
    # The released agent reads this supported environment switch during
    # construction.  ``cfg.agent`` is a structured Hydra config and therefore
    # must not be mutated with an undeclared ``ray`` key.
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"

    cfg = _compose_agent_config(args.repo_root.resolve(), args.checkpoint.resolve())
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    visual_model, wrapper_chain = _resolve_visual_model(agent.backbone)

    logs = [str(value) for value in cfg.train_logs + cfg.val_logs]
    dataset = CacheOnlyDataset(
        cache_path=str(args.feature_cache),
        feature_builders=[_CacheNameBuilder("internvl_feature")],
        target_builders=[],
        log_names=logs,
        append_token_to_batch=True,
        preprocess_images=True,
        preprocess_image_dtype="bfloat16",
        pretokenize_inputs=False,
    )
    dataset.tokens = sorted(dataset.tokens)
    tokens = _partition(dataset.tokens, args.shard_count, args.shard_index)
    if args.limit_scenes > 0:
        tokens = tokens[: args.limit_scenes]
    log_for_token = {
        token: dataset._valid_cache_paths[token].parent.name for token in tokens
    }
    token_chunks = list(_chunked(tokens, args.chunk_size))
    shard_dir = args.output_dir / (
        f"all_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    first_missing = _first_missing_chunk(shard_dir, len(token_chunks))
    completed_scenes = sum(len(chunk) for chunk in token_chunks[:first_missing])
    dataset.tokens = tokens[completed_scenes:]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=bool(args.num_workers),
        shuffle=False,
        collate_fn=drivevla_cached_collate,
    )

    chunk_index = first_missing
    buffer: Dict[str, List[object]] = {
        "tokens": [],
        "log_names": [],
        "visual_tokens": [],
        "visual_valid_mask": [],
        "crop_counts": [],
        "status_feature": [],
        "history_trajectory": [],
        "high_command_one_hot": [],
    }
    started = time.monotonic()
    print(
        f"PRIVATE_VISUAL_EXPORT_READY shard={args.shard_index}/{args.shard_count} "
        f"remaining={len(dataset)} batch={args.batch_size}",
        flush=True,
    )
    with torch.inference_mode():
        for features, _targets, batch_tokens in loader:
            pixels = _pixel_list(features.pop("pixel_values"))
            crop_counts = [int(value.shape[0]) for value in pixels]
            flattened = torch.cat(pixels, dim=0).cuda(non_blocking=True)
            raw = visual_model.extract_feature(flattened)
            pooled, valid = pool_visual_tokens(
                raw,
                crop_counts,
                args.pool_grid,
                args.max_crops,
            )
            buffer["tokens"].extend(str(value) for value in batch_tokens)
            buffer["log_names"].extend(
                log_for_token[str(value)] for value in batch_tokens
            )
            buffer["visual_tokens"].append(pooled)
            buffer["visual_valid_mask"].append(valid)
            buffer["crop_counts"].append(torch.tensor(crop_counts, dtype=torch.int16))
            for key in ("status_feature", "history_trajectory", "high_command_one_hot"):
                buffer[key].append(features[key].detach().cpu())

            if len(buffer["tokens"]) >= len(token_chunks[chunk_index]):
                expected = len(token_chunks[chunk_index])
                if len(buffer["tokens"]) != expected:
                    raise RuntimeError("batch crossed deterministic chunk boundary")
                payload = {
                    "schema_version": 1,
                    "tokens": list(buffer["tokens"]),
                    "log_names": list(buffer["log_names"]),
                    "visual_tokens": torch.cat(buffer["visual_tokens"]),
                    "visual_valid_mask": torch.cat(buffer["visual_valid_mask"]),
                    "crop_counts": torch.cat(buffer["crop_counts"]),
                    "status_feature": torch.cat(buffer["status_feature"]),
                    "history_trajectory": torch.cat(buffer["history_trajectory"]),
                    "high_command_one_hot": torch.cat(buffer["high_command_one_hot"]),
                    "pool_grid": args.pool_grid,
                    "max_crops": args.max_crops,
                }
                _atomic_torch_save(payload, shard_dir / f"chunk_{chunk_index:06d}.pt")
                chunk_index += 1
                buffer = {key: [] for key in buffer}
                print(
                    f"PRIVATE_VISUAL_EXPORT shard={args.shard_index}/{args.shard_count} "
                    f"chunk={chunk_index}/{len(token_chunks)} "
                    f"elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    if any(buffer.values()):
        raise RuntimeError("unflushed private visual cache buffer")
    if chunk_index != len(token_chunks):
        raise RuntimeError("private visual cache did not complete all chunks")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "resolved_class": "navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent",
        "resolved_vlm_path": str(args.vlm_path.resolve()),
        "visual_model_wrapper_chain": wrapper_chain,
        "feature_cache": str(args.feature_cache.resolve()),
        "scene_count": len(tokens),
        "log_count": len(set(log_for_token.values())),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "chunk_size": args.chunk_size,
        "chunk_count": len(token_chunks),
        "pool_grid": args.pool_grid,
        "max_crops": args.max_crops,
        "max_visual_tokens": args.max_crops * args.pool_grid * args.pool_grid,
        "visual_width": 1536,
        "current_context_fields": (
            "status_feature",
            "history_trajectory",
            "high_command_one_hot",
        ),
        "future_or_evaluator_input": False,
        "current_observation_only": True,
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
