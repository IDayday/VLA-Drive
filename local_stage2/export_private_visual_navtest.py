"""Export scorer-private current-image tokens for the locked Navtest bank.

Only the current CAM_F0 image and current ego/navigation features are read.
The frozen public-Base vision encoder is reused, while proposal geometry and
official candidate factors remain in separate immutable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# NAVSIM snapshots the map root into a module-level constant during import.
# Establish the safe local default before importing SceneLoader/dataclasses.
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/navsim/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from local_stage2.export_private_visual_replay import (
    _atomic_json_dump,
    _atomic_torch_save,
    _first_missing_chunk,
    _resolve_visual_model,
    pool_visual_tokens,
)
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset, _decode_image_path


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _partition(values: Sequence[str], count: int, index: int) -> List[str]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid shard count/index")
    return list(values[index::count])


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _visual_collate(batch):
    pixels: List[torch.Tensor] = []
    statuses: List[torch.Tensor] = []
    histories: List[torch.Tensor] = []
    commands: List[torch.Tensor] = []
    tokens: List[str] = []
    for features, _targets, token in batch:
        leaked = {
            "future_image",
            "future_annotations",
            "future_trajectory",
            "official_score",
            "pdm_score",
        }.intersection(features)
        if leaked:
            raise RuntimeError(f"future/evaluator input leaked: {sorted(leaked)}")
        image_path = _decode_image_path(features["image_path_tensor"])
        pixels.append(load_image(image_path, max_num=12))
        statuses.append(features["status_feature"].float())
        histories.append(features["history_trajectory"].float())
        commands.append(features["high_command_one_hot"].float())
        tokens.append(str(token))
    return {
        "pixel_values": pixels,
        "status_feature": torch.stack(statuses),
        "history_trajectory": torch.stack(histories),
        "high_command_one_hot": torch.stack(commands),
        "tokens": tokens,
    }


def _compose_config(repo_root: Path, checkpoint: Path):
    config_dir = repo_root / "navsim/planning/script/config/pdm_scoring"
    overrides = [
        "train_test_split=navtest",
        "agent=episode_drive",
        f"agent.checkpoint_path={checkpoint}",
        "agent.stage1_checkpoint_path=null",
        "agent.cache_data=false",
        "agent.vlm_config.freeze_backbone=true",
        "agent.vlm_config.cache_hidden_state=false",
        "agent.vlm_config.cache_mode=false",
        "agent.vlm_config.initialize_from_config=true",
        "agent.vlm_config.use_flash_attn=false",
        "agent.vlm_config.frozen_backbone_mode=eval",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="default_run_pdm_score_gpu", overrides=overrides)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--proposal-pickle", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, default=Path("/mnt/navsim/maps"))
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
    for path in (
        args.repo_root,
        args.checkpoint,
        args.vlm_path,
        args.proposal_pickle,
        args.log_path,
        args.sensor_root,
        args.map_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.chunk_size % args.batch_size:
        raise ValueError("chunk size must be divisible by batch size")
    os.environ["DRIVEVLA_VLM_CONFIG"] = str(args.vlm_path.resolve())
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.map_root.resolve())
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

    with args.proposal_pickle.open("rb") as file:
        proposal_bank = pickle.load(file)
    locked_tokens = sorted(str(value) for value in proposal_bank)
    if len(locked_tokens) != 12_146 or len(set(locked_tokens)) != 12_146:
        raise RuntimeError("locked Navtest proposal bank must contain 12,146 scenes")

    cfg = _compose_config(args.repo_root.resolve(), args.checkpoint.resolve())
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    visual_model, wrapper_chain = _resolve_visual_model(agent.backbone)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=args.sensor_root,
        data_path=args.log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    scene_tokens = set(str(value) for value in scene_loader.tokens)
    if scene_tokens != set(locked_tokens):
        raise RuntimeError(
            "SceneLoader/proposal token mismatch: "
            f"missing={len(set(locked_tokens) - scene_tokens)}, "
            f"extra={len(scene_tokens - set(locked_tokens))}"
        )
    log_for_token = {
        str(token): str(log_name)
        for log_name, values in scene_loader.get_tokens_list_per_log().items()
        for token in values
    }
    selected_tokens = _partition(
        locked_tokens, args.shard_count, args.shard_index
    )
    if args.limit_scenes > 0:
        selected_tokens = selected_tokens[: args.limit_scenes]
    full_dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=[],
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    chunks = list(_chunked(selected_tokens, args.chunk_size))
    shard_dir = args.output_dir / (
        f"navtest_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    first_missing = _first_missing_chunk(shard_dir, len(chunks))
    completed = sum(len(chunk) for chunk in chunks[:first_missing])
    index_for_token = {
        str(token): index for index, token in enumerate(scene_loader.tokens)
    }
    dataset = Subset(
        full_dataset,
        [index_for_token[token] for token in selected_tokens[completed:]],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=bool(args.num_workers),
        shuffle=False,
        collate_fn=_visual_collate,
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
        f"PRIVATE_NAVTEST_READY shard={args.shard_index}/{args.shard_count} "
        f"remaining={len(dataset)} batch={args.batch_size}",
        flush=True,
    )
    with torch.inference_mode():
        for batch in loader:
            pixel_list = batch.pop("pixel_values")
            crop_counts = [int(value.shape[0]) for value in pixel_list]
            pixels = torch.cat(pixel_list).to(
                device="cuda", dtype=torch.bfloat16, non_blocking=True
            )
            raw = visual_model.extract_feature(pixels)
            pooled, valid = pool_visual_tokens(
                raw, crop_counts, args.pool_grid, args.max_crops
            )
            batch_tokens = batch.pop("tokens")
            buffer["tokens"].extend(batch_tokens)
            buffer["log_names"].extend(
                log_for_token[token] for token in batch_tokens
            )
            buffer["visual_tokens"].append(pooled)
            buffer["visual_valid_mask"].append(valid)
            buffer["crop_counts"].append(
                torch.tensor(crop_counts, dtype=torch.int16)
            )
            for key in (
                "status_feature",
                "history_trajectory",
                "high_command_one_hot",
            ):
                buffer[key].append(batch[key].cpu())

            if len(buffer["tokens"]) >= len(chunks[chunk_index]):
                expected = len(chunks[chunk_index])
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
                    "high_command_one_hot": torch.cat(
                        buffer["high_command_one_hot"]
                    ),
                    "pool_grid": args.pool_grid,
                    "max_crops": args.max_crops,
                }
                _atomic_torch_save(
                    payload, shard_dir / f"chunk_{chunk_index:06d}.pt"
                )
                chunk_index += 1
                buffer = {key: [] for key in buffer}
                print(
                    f"PRIVATE_NAVTEST shard={args.shard_index}/{args.shard_count} "
                    f"chunk={chunk_index}/{len(chunks)} "
                    f"elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    if any(buffer.values()) or chunk_index != len(chunks):
        raise RuntimeError("private Navtest cache did not finish cleanly")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "proposal_pickle": str(args.proposal_pickle.resolve()),
        "proposal_pickle_sha256": _sha256(args.proposal_pickle),
        "split": "navtest",
        "scene_count": len(selected_tokens),
        "log_count": len({log_for_token[token] for token in selected_tokens}),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "pool_grid": args.pool_grid,
        "max_crops": args.max_crops,
        "max_visual_tokens": args.max_crops * args.pool_grid * args.pool_grid,
        "visual_width": 1536,
        "visual_model_wrapper_chain": wrapper_chain,
        "current_observation_only": True,
        "future_or_evaluator_input": False,
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
