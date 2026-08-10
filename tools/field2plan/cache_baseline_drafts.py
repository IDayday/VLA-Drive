#!/usr/bin/env python3
"""Generate a deterministic, manifest-validated pure-trajectory draft cache.

Heavy model and dataset imports are intentionally confined to generation so
that schema validation and unit tests run without Qwen weights or a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from starVLA.dataloader.field2plan_cache import (
    DraftCacheReader,
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    load_draft_entry,
    sha256_file,
)
from starVLA.model.modules.field2plan.trajectory_codec import (
    TrajectoryCodec,
    TrajectoryStats,
)


def _resolve_checkpoint_and_config(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.is_dir():
        candidates = (
            checkpoint_path / "final_model" / "pytorch_model.pt",
            checkpoint_path / "pytorch_model.pt",
        )
        checkpoint_file = next((path for path in candidates if path.is_file()), None)
        if checkpoint_file is None:
            raise FileNotFoundError(
                f"no final checkpoint found below {checkpoint_path}"
            )
        inferred_config = checkpoint_path / "config.yaml"
    else:
        checkpoint_file = checkpoint_path
        inferred_config = checkpoint_file.parent.parent / "config.yaml"
        if not inferred_config.is_file():
            inferred_config = checkpoint_file.parent.parent.parent / "config.yaml"
    selected_config = (config_path or inferred_config).resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")
    if not selected_config.is_file():
        raise FileNotFoundError(f"config not found: {selected_config}")
    return checkpoint_file, selected_config


def _normalization_manifest() -> Dict[str, Any]:
    stats = TrajectoryStats()
    return {
        "version": "ver_1225_act_norm_1",
        "x_mean": stats.x_mean,
        "x_std": stats.x_std,
        "y_mean": stats.y_mean,
        "y_std": stats.y_std,
        "heading": "sin_cos",
    }


def build_draft_manifest(
    *,
    checkpoint_path: Path,
    config_path: Path,
    datalist_path: Path,
    split: str,
    tokens: Sequence[str],
    seed: int,
    inference_steps: int,
    num_candidates: int,
    world_size: int,
    batch_size: int,
    qwen_forward_mode: str,
    git_commit: str,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Build the complete schema-v2 manifest from verified local inputs."""

    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = Path(config_path).resolve()
    datalist_path = Path(datalist_path).resolve()
    for name, value in (
        ("seed", seed),
        ("inference_steps", inference_steps),
        ("num_candidates", num_candidates),
        ("world_size", world_size),
        ("batch_size", batch_size),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if qwen_forward_mode not in {"legacy", "optimized"}:
        raise ValueError("qwen_forward_mode must resolve to legacy or optimized")
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("tokens must be non-empty and unique")
    return {
        "schema_version": 2,
        "cache_type": "baseline_draft",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": os.fspath(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "config": {
            "path": os.fspath(config_path),
            "sha256": sha256_file(config_path),
        },
        "generator": {
            "git_commit": str(git_commit),
            "tool": "tools/field2plan/cache_baseline_drafts.py",
        },
        "inference": {
            "seed": seed,
            "steps": inference_steps,
            "num_candidates": num_candidates,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "qwen_forward_mode": qwen_forward_mode,
            "max_samples": int(max_samples),
        },
        "splits": {
            str(split): {
                "datalist_path": os.fspath(datalist_path),
                "datalist_sha256": sha256_file(datalist_path),
                "entry_count": len(tokens),
                "tokens_sha256": hash_tokens(tokens),
            }
        },
        "tensor_schema": {
            "draft_action": {
                "dtype": "float32",
                "shape": ["M", 8, 4],
                "horizon": 8,
                "last_dim": 4,
            },
            "physical_trajectory": {
                "dtype": "float32",
                "shape": ["M", 8, 3],
            },
        },
        "normalization": _normalization_manifest(),
    }


def validate_cache(
    cache_root: Path,
    split: str,
    tokens: Sequence[str],
) -> Dict[str, Any]:
    """Validate manifest coverage and every requested cache entry."""

    reader = DraftCacheReader(os.fspath(cache_root), split)
    split_metadata = reader.manifest["splits"][split]
    if hash_tokens(tokens) != split_metadata["tokens_sha256"]:
        raise ValueError("draft token list checksum differs from manifest")
    if len(tokens) != split_metadata["entry_count"]:
        raise ValueError("draft token count differs from manifest")
    for token in tokens:
        reader.load(str(token))
    return {"split": str(split), "validated_entries": len(tokens)}


def _read_tokens(datalist_path: Path) -> List[str]:
    try:
        payload = json.loads(datalist_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid datalist JSON: {datalist_path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("datalist must be a non-empty token list")
    tokens = [str(token) for token in payload]
    if len(set(tokens)) != len(tokens):
        raise ValueError("datalist contains duplicate tokens")
    return tokens


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _set_inference_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_context(args: argparse.Namespace):
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    rank = int(os.environ.get("RANK", args.rank))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        initialized_here = True
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, initialized_here


def generate_cache(args: argparse.Namespace) -> Dict[str, Any]:
    """Run frozen baseline inference and atomically materialize the cache."""

    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from infer import VLAAgent, to_device
    from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn

    repo_root = Path(__file__).resolve().parents[2]
    checkpoint, config = _resolve_checkpoint_and_config(
        Path(args.checkpoint), Path(args.config) if args.config else None
    )
    datalist = Path(args.datalist).resolve()
    cache_root = Path(args.output_dir).resolve()
    tokens = _read_tokens(datalist)
    if args.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if args.max_samples:
        tokens = tokens[: args.max_samples]
    rank, world_size, local_rank, initialized_here = _distributed_context(args)
    try:
        if args.validate_only:
            if rank == 0:
                return validate_cache(cache_root, args.split, tokens)
            return {"split": args.split, "validated_entries": 0}

        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        triton_root = os.environ.get("TRITON_CACHE_DIR")
        if triton_root:
            rank_cache = Path(triton_root) / f"local_rank{local_rank}"
            rank_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = os.fspath(rank_cache)
        agent = VLAAgent(
            os.fspath(checkpoint),
            device=device,
            qwen_forward_mode=args.qwen_forward_mode,
            disable_aux_models=True,
        )
        agent.model.action_model.num_inference_timesteps = int(
            args.inference_steps
        )
        cfg = copy.deepcopy(agent.model_config)
        cfg.datasets.video_data.load_2d_data = 0
        cfg.datasets.video_data.load_3d_data = 0
        cfg.datasets.vla_data.act_norm = 1
        cfg.w_depth = 0
        cfg.enable_image_aug = 0
        dataset = NavSimDataset(
            datalist_path=os.fspath(datalist),
            split=args.split,
            video_data_cfg=cfg.datasets.video_data,
            gs_data_cfg=cfg.datasets.gs_data,
            reward_data_cfg=cfg.datasets.reward_data,
            ver_1225=1,
            dataset_cfg=cfg.datasets.vla_data,
            all_cfg=cfg,
            data_root=args.data_root,
        )
        if list(dataset.raw_list[: len(tokens)]) != tokens:
            raise ValueError("dataset token order differs from the datalist")
        shard_indices = list(range(rank, len(tokens), world_size))
        loader = DataLoader(
            Subset(dataset, shard_indices),
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        codec = TrajectoryCodec()
        split_dir = cache_root / args.split
        split_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        skipped = 0
        for batch_index, batch in enumerate(
            tqdm(loader, disable=rank != 0, desc=f"draft-cache rank {rank}")
        ):
            batch_tokens = [str(sample["token"]) for sample in batch]
            valid_existing = []
            for token in batch_tokens:
                entry_path = split_dir / f"{token}.npz"
                if args.overwrite or not entry_path.is_file():
                    valid_existing.append(False)
                    continue
                try:
                    load_draft_entry(entry_path, token, args.num_candidates)
                    valid_existing.append(True)
                except ValueError:
                    valid_existing.append(False)
            if all(valid_existing):
                skipped += len(batch_tokens)
                continue
            batch_on_device = to_device(batch, agent.device)
            candidate_actions = []
            for candidate_index in range(args.num_candidates):
                call_seed = (
                    args.seed
                    + rank * 1_000_000_000
                    + batch_index * args.num_candidates
                    + candidate_index
                )
                _set_inference_seed(call_seed)
                prediction = agent.predict(batch_on_device)
                candidate_actions.append(
                    np.asarray(prediction["normalized_actions"], dtype=np.float32)
                )
            normalized = np.stack(candidate_actions, axis=1)
            physical = np.asarray(codec.decode_action(normalized), dtype=np.float32)
            for sample_index, token in enumerate(batch_tokens):
                if valid_existing[sample_index] and not args.overwrite:
                    skipped += 1
                    continue
                atomic_write_npz(
                    split_dir / f"{token}.npz",
                    token=np.asarray(token),
                    draft_action=normalized[sample_index],
                    physical_trajectory=physical[sample_index],
                )
                processed += 1
        atomic_write_json(
            cache_root / f"rank-{rank:05d}.complete.json",
            {"rank": rank, "world_size": world_size, "processed": processed, "skipped": skipped},
        )
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            for token in tokens:
                load_draft_entry(
                    split_dir / f"{token}.npz", token, args.num_candidates
                )
            manifest = build_draft_manifest(
                checkpoint_path=checkpoint,
                config_path=config,
                datalist_path=datalist,
                split=args.split,
                tokens=tokens,
                seed=args.seed,
                inference_steps=args.inference_steps,
                num_candidates=args.num_candidates,
                world_size=world_size,
                batch_size=args.batch_size,
                qwen_forward_mode=agent.qwen_forward_mode,
                git_commit=_git_commit(repo_root),
                max_samples=args.max_samples,
            )
            atomic_write_json(cache_root / "manifest.json", manifest)
            return validate_cache(cache_root, args.split, tokens)
        return {"split": args.split, "validated_entries": len(shard_indices)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cache frozen baseline trajectory drafts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug-only prefix length; 0 caches the full datalist.",
    )
    parser.add_argument(
        "--qwen-forward-mode", choices=("auto", "legacy", "optimized"), default="optimized"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = generate_cache(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
