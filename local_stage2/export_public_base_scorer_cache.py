"""Export the released EpisodeDrive proposal/scorer representation on Navtrain.

This is an inference-only cache builder.  It deliberately does not load PDM
metric caches or future annotations, so the exported tensors are all available
to a deployable scorer.  PDM supervision is attached later by a separate,
offline-only scoring pass.

Independent processes partition the sorted scene-token list with ``tokens[i::N]``.
Each process writes immutable, atomic chunks under its own shard directory.  A
restart resumes at the first missing chunk without touching completed output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from navsim.planning.training.dataset import CacheOnlyDataset, drivevla_cached_collate


FACTOR_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


@dataclass(frozen=True)
class _CacheNameBuilder:
    name: str

    def get_unique_name(self) -> str:
        return self.name


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
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


def _partition_tokens(tokens: Sequence[str], shard_count: int, shard_index: int) -> List[str]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return list(tokens[shard_index::shard_count])


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _first_missing_chunk(shard_dir: Path, chunk_count: int) -> int:
    missing = next(
        (
            index
            for index in range(chunk_count)
            if not (shard_dir / f"chunk_{index:06d}.pt").is_file()
        ),
        chunk_count,
    )
    later = [
        index
        for index in range(missing + 1, chunk_count)
        if (shard_dir / f"chunk_{index:06d}.pt").is_file()
    ]
    if later:
        raise RuntimeError(
            "Non-contiguous cache chunks found after first missing chunk: "
            f"{later[:10]}"
        )
    return missing


def _compose_agent_config(
    repo_root: Path,
    checkpoint: Path,
    resolved_config: Path | None = None,
):
    if resolved_config is not None:
        cfg = OmegaConf.load(resolved_config)
        if "agent" not in cfg:
            raise KeyError(f"Resolved config has no agent section: {resolved_config}")
        cfg.agent.checkpoint_path = str(checkpoint)
        cfg.agent.stage1_checkpoint_path = None
        cfg.agent.cache_data = False
        cfg.agent.vlm_config.freeze_backbone = True
        cfg.agent.vlm_config.use_flash_attn = False
        cfg.agent.vlm_config.gradient_checkpointing = False
        with open_dict(cfg.agent.vlm_config):
            cfg.agent.vlm_config.frozen_backbone_mode = "eval"
        with open_dict(cfg.agent.action_head_config):
            cfg.agent.action_head_config.return_scorer_features = True
            cfg.agent.action_head_config.return_memory_fields = True
        return cfg

    config_dir = repo_root / "navsim/planning/script/config/training"
    overrides = [
        "agent=episode_drive",
        # Hydra treats an unquoted ``=`` inside an override value as grammar.
        # Lightning checkpoint filenames commonly contain ``epoch=...`` and
        # therefore must be serialized as a quoted string.
        f"agent.checkpoint_path={json.dumps(str(checkpoint))}",
        "agent.stage1_checkpoint_path=null",
        "agent.cache_data=false",
        "agent.vlm_config.freeze_backbone=true",
        "agent.vlm_config.cache_hidden_state=false",
        "agent.vlm_config.cache_mode=false",
        "agent.vlm_config.initialize_from_config=true",
        "agent.vlm_config.use_flash_attn=false",
        "agent.vlm_config.frozen_backbone_mode=eval",
        "+agent.action_head_config.return_scorer_features=true",
        "+agent.action_head_config.return_memory_fields=true",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="default_training", overrides=overrides)


def _select_logs(cfg, split: str) -> List[str]:
    train_logs = [str(value) for value in cfg.train_logs]
    val_logs = [str(value) for value in cfg.val_logs]
    overlap = sorted(set(train_logs).intersection(val_logs))
    if overlap:
        raise RuntimeError(f"Train/validation log overlap: {overlap[:5]}")
    if split == "train":
        return train_logs
    if split == "val":
        return val_logs
    if split == "all":
        return train_logs + val_logs
    raise ValueError(f"Unsupported split: {split}")


def _move_batch_to_cuda(features: Dict[str, object]) -> Dict[str, object]:
    # DriveVLABaseAgent handles device transfer itself.  This helper only checks
    # that inference inputs contain no future supervision before the forward.
    forbidden = {
        "future_image",
        "future_annotations",
        "future_trajectory",
        "official_score",
        "pdm_score",
    }
    overlap = forbidden.intersection(features)
    if overlap:
        raise RuntimeError(f"Future/evaluator input leaked into inference: {sorted(overlap)}")
    return features


def _append_batch(buffer: Dict[str, List[object]], prediction, tokens, log_for_token) -> None:
    proposals = prediction["proposals"].detach().float().cpu()
    base_scores = prediction["pdm_score"].detach().float().cpu()
    factor_logits = torch.stack(
        [prediction["pred_logit"][key].detach().float().cpu() for key in FACTOR_KEYS],
        dim=-1,
    )
    candidate_features = prediction["scorer_candidate_features"].detach().cpu()
    scene_features = prediction["language_feature"].detach().cpu()
    ego_features = prediction["ego_feature"].detach().cpu()
    selected = base_scores.argmax(dim=1)

    if proposals.ndim != 4 or proposals.shape[:2] != base_scores.shape:
        raise RuntimeError(
            f"Invalid proposal/base-score shapes: {proposals.shape}, {base_scores.shape}"
        )
    expected_feature_shape = (*base_scores.shape, candidate_features.shape[-1])
    if tuple(candidate_features.shape) != expected_feature_shape:
        raise RuntimeError(
            "Invalid scorer feature shape: "
            f"{tuple(candidate_features.shape)} != {expected_feature_shape}"
        )

    buffer["tokens"].extend(str(token) for token in tokens)
    buffer["log_names"].extend(log_for_token[str(token)] for token in tokens)
    buffer["proposals"].append(proposals)
    buffer["base_scores"].append(base_scores)
    buffer["factor_logits"].append(factor_logits)
    buffer["candidate_features"].append(candidate_features)
    buffer["scene_features"].append(scene_features)
    buffer["ego_features"].append(ego_features)
    buffer["selected_indices"].append(selected)


def _finalize_buffer(buffer: Dict[str, List[object]]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "factor_keys": FACTOR_KEYS,
        "tokens": list(buffer["tokens"]),
        "log_names": list(buffer["log_names"]),
        "proposals": torch.cat(buffer["proposals"], dim=0),
        "base_scores": torch.cat(buffer["base_scores"], dim=0),
        "factor_logits": torch.cat(buffer["factor_logits"], dim=0).to(torch.float16),
        "candidate_features": torch.cat(buffer["candidate_features"], dim=0).to(torch.float16),
        "scene_features": torch.cat(buffer["scene_features"], dim=0).to(torch.float16),
        "ego_features": torch.cat(buffer["ego_features"], dim=0).to(torch.float16),
        "selected_indices": torch.cat(buffer["selected_indices"], dim=0),
    }


def _empty_buffer() -> Dict[str, List[object]]:
    return {
        key: []
        for key in (
            "tokens",
            "log_names",
            "proposals",
            "base_scores",
            "factor_logits",
            "candidate_features",
            "scene_features",
            "ego_features",
            "selected_indices",
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=None,
        help=(
            "Previously resolved Hydra training/evaluation config. Use this "
            "for checkpoints whose architecture differs from the released "
            "EpisodeDrive defaults, such as the no-LoRA No-VQA run."
        ),
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument(
        "--repair-chunk-index",
        type=int,
        default=-1,
        help=(
            "Recompute exactly one zero-byte chunk atomically without touching "
            "later completed chunks or the shard manifest."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for public Base cache export")
    required_paths = [args.repo_root, args.checkpoint, args.feature_cache]
    if args.resolved_config is not None:
        required_paths.append(args.resolved_config)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = _compose_agent_config(
        args.repo_root.resolve(),
        args.checkpoint.resolve(),
        args.resolved_config.resolve() if args.resolved_config is not None else None,
    )
    logs = _select_logs(cfg, args.split)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)

    dataset = CacheOnlyDataset(
        cache_path=str(args.feature_cache),
        feature_builders=[_CacheNameBuilder("internvl_feature")],
        target_builders=[],
        log_names=logs,
        append_token_to_batch=True,
        preprocess_images=True,
        preprocess_image_dtype="bfloat16",
        pretokenize_inputs=True,
        tokenizer=agent.backbone.tokenizer,
    )
    dataset.tokens = sorted(dataset.tokens)
    tokens = _partition_tokens(dataset.tokens, args.shard_count, args.shard_index)
    if args.limit_scenes > 0:
        tokens = tokens[: args.limit_scenes]
    dataset.tokens = tokens
    log_for_token = {
        token: dataset._valid_cache_paths[token].parent.name for token in tokens
    }

    token_chunks = list(_chunked(tokens, args.chunk_size))
    shard_dir = args.output_dir / (
        f"{args.split}_shard_{args.shard_index:03d}-of-{args.shard_count:03d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    repair_mode = args.repair_chunk_index >= 0
    if repair_mode:
        if args.repair_chunk_index >= len(token_chunks):
            raise ValueError(
                f"repair chunk {args.repair_chunk_index} is outside "
                f"[0, {len(token_chunks)})"
            )
        repair_path = shard_dir / f"chunk_{args.repair_chunk_index:06d}.pt"
        if repair_path.is_file() and repair_path.stat().st_size > 0:
            raise RuntimeError(
                f"Refusing to overwrite nonempty repair target: {repair_path}"
            )
        first_missing = args.repair_chunk_index
        final_chunk_exclusive = first_missing + 1
    else:
        first_missing = _first_missing_chunk(shard_dir, len(token_chunks))
        final_chunk_exclusive = len(token_chunks)
    completed_scenes = sum(len(chunk) for chunk in token_chunks[:first_missing])

    # A contiguous suffix keeps deterministic chunk boundaries on resume.
    if repair_mode:
        dataset.tokens = list(token_chunks[first_missing])
    else:
        dataset.tokens = tokens[completed_scenes:]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=bool(args.num_workers),
        shuffle=False,
        collate_fn=drivevla_cached_collate,
    )

    buffer = _empty_buffer()
    chunk_index = first_missing
    inference_started = time.monotonic()
    print(
        f"CACHE_EXPORT_READY shard={args.shard_index}/{args.shard_count} "
        f"remaining_scenes={len(dataset)} batch_size={args.batch_size}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, (features, _targets, batch_tokens) in enumerate(dataloader):
            features = _move_batch_to_cuda(features)
            prediction = agent.forward(features)
            _append_batch(buffer, prediction, batch_tokens, log_for_token)
            if len(buffer["tokens"]) >= len(token_chunks[chunk_index]):
                if len(buffer["tokens"]) != len(token_chunks[chunk_index]):
                    raise RuntimeError(
                        "Batch crossed a deterministic chunk boundary; choose a "
                        "chunk size divisible by batch size"
                    )
                path = shard_dir / f"chunk_{chunk_index:06d}.pt"
                _atomic_torch_save(_finalize_buffer(buffer), path)
                chunk_index += 1
                buffer = _empty_buffer()
                print(
                    f"CACHE_EXPORT shard={args.shard_index}/{args.shard_count} "
                    f"chunk={chunk_index}/{len(token_chunks)} "
                    f"scenes={completed_scenes + sum(len(c) for c in token_chunks[first_missing:chunk_index])} "
                    f"inference_elapsed_s={time.monotonic() - inference_started:.1f}",
                    flush=True,
                )

    if buffer["tokens"]:
        raise RuntimeError("Unflushed cache buffer remains after export")
    if chunk_index != final_chunk_exclusive:
        raise RuntimeError(
            f"Expected to stop at chunk {final_chunk_exclusive}, wrote {chunk_index}"
        )

    if repair_mode:
        repaired_path = shard_dir / f"chunk_{first_missing:06d}.pt"
        print(
            json.dumps(
                {
                    "status": "repaired",
                    "chunk_index": first_missing,
                    "scene_count": len(token_chunks[first_missing]),
                    "path": str(repaired_path.resolve()),
                    "sha256": _sha256(repaired_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "resolved_config": (
            str(args.resolved_config.resolve())
            if args.resolved_config is not None
            else None
        ),
        "resolved_config_sha256": (
            _sha256(args.resolved_config)
            if args.resolved_config is not None
            else None
        ),
        "feature_cache": str(args.feature_cache.resolve()),
        "split": args.split,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "scene_count": len(tokens),
        "log_count": len(set(log_for_token.values())),
        "chunk_size": args.chunk_size,
        "chunk_count": len(token_chunks),
        "factor_keys": FACTOR_KEYS,
        "inference_inputs_only": True,
        "official_score_present": False,
        "future_target_present": False,
        "config": OmegaConf.to_container(cfg.agent, resolve=True),
    }
    _atomic_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
