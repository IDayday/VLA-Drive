#!/usr/bin/env python3
"""Evaluate the public Stage-2 checkpoint on a fixed stratified cache subset.

This is intentionally smaller than a full NAVTEST evaluation.  Its purpose is
to decide whether a software-runtime candidate is worth an expensive training
control: every runtime receives the same checkpoint, cache tokens, pixels, and
offline PDM evaluator inputs.  The script writes per-scene tensors so two runs
can be compared without relying only on an aggregate mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import peft
import pytorch_lightning
import torch
import tokenizers
import transformers
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from local_stage2.audit_stage2_numerics import _optimized_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
)
DEFAULT_METRIC_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)
DEFAULT_VLM = Path("/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope")
FACTOR_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "pdm_score",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratified_samples(
    cache_root: Path, log_names: Iterable[str], count: int
) -> list[Path]:
    """Round-robin complete cache samples across validation logs."""

    queues: list[deque[Path]] = []
    for log_name in log_names:
        candidates = []
        for feature_path in sorted(
            (cache_root / log_name).glob("*/internvl_feature.gz")
        ):
            sample_dir = feature_path.parent
            if (sample_dir / "trajectory_target.gz").is_file():
                candidates.append(sample_dir)
        if candidates:
            queues.append(deque(candidates))

    selected: list[Path] = []
    active = deque(queues)
    while active and len(selected) < count:
        queue = active.popleft()
        selected.append(queue.popleft())
        if queue:
            active.append(queue)
    if len(selected) != count:
        raise RuntimeError(
            f"Requested {count} validation samples, found {len(selected)}"
        )
    return selected


def _compose_config(args: argparse.Namespace):
    config_dir = REPO_ROOT / "navsim/planning/script/config/training"
    overrides = [
        "train_test_split=navtrain",
        f"agent.checkpoint_path={args.checkpoint}",
        "agent.stage1_checkpoint_path=null",
        "agent.cache_data=false",
        f"agent.vlm_config.vlm_path={args.vlm_path}",
        "agent.vlm_config.freeze_backbone=true",
        "agent.vlm_config.cache_hidden_state=false",
        "agent.vlm_config.cache_mode=false",
        "agent.vlm_config.initialize_from_config=true",
        f"agent.vlm_config.use_flash_attn={str(args.flash_attention).lower()}",
        "agent.vlm_config.frozen_backbone_mode=eval",
        "agent.vlm_config.extra_token_count=8",
        "agent.vlm_config.target_vocab_size=151682",
        "agent.lora_config.use_lora=true",
        f"agent.batch_size={args.batch_size}",
        "agent.num_gpus=1",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="default_training", overrides=overrides)


def _move_targets(targets):
    return {
        key: value.cuda(non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in targets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vlm-path", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--metric-cache", type=Path, default=DEFAULT_METRIC_CACHE
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--flash-attention", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    for path in (
        args.checkpoint,
        args.vlm_path,
        args.cache_root,
        args.metric_cache,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")

    os.environ["NAVSIM_TRAIN_METRIC_CACHE"] = str(args.metric_cache)
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"
    os.environ["DRIVEVLA_SCORE_PROCESSES"] = "0"
    _seed_everything(args.seed)
    cfg = _compose_config(args)
    sample_dirs = _stratified_samples(
        args.cache_root, cfg.val_logs, args.samples
    )

    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()

    records = {
        "tokens": [],
        "logs": [],
        "selected_index": [],
        "selected_trajectory": [],
        "proposals": [],
        "predicted_pdm_logit": [],
        "selected_factors": [],
        "best_pdm_score": [],
        "selected_l2_2s": [],
    }

    for start in range(0, len(sample_dirs), args.batch_size):
        batch_dirs = sample_dirs[start : start + args.batch_size]
        features, targets = _optimized_batch(
            batch_dirs, agent.backbone.tokenizer
        )
        targets = _move_targets(targets)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = agent.forward(features)

        selected = prediction["trajectory"][:, None]
        selected_result = agent.compute_score(targets, selected, test=True)
        proposal_result = agent.compute_score(
            targets, prediction["proposals"], test=True
        )

        selected_index = prediction["pdm_score"].argmax(dim=1)
        selected_factors = selected_result[4]
        if selected_factors.shape[-1] != len(FACTOR_NAMES):
            raise RuntimeError(
                "Unexpected PDM factor width: "
                f"{selected_factors.shape[-1]} != {len(FACTOR_NAMES)}"
            )
        records["tokens"].extend(str(token) for token in targets["token"])
        records["logs"].extend(path.parent.name for path in batch_dirs)
        records["selected_index"].append(selected_index.detach().cpu())
        records["selected_trajectory"].append(selected.detach().cpu()[:, 0])
        records["proposals"].append(prediction["proposals"].detach().cpu())
        records["predicted_pdm_logit"].append(
            prediction["pdm_score"].detach().cpu()
        )
        records["selected_factors"].append(selected_factors.detach().cpu())
        records["best_pdm_score"].append(
            proposal_result[2].amax(dim=1).detach().cpu()
        )
        records["selected_l2_2s"].append(
            torch.as_tensor(selected_result[3]).detach().cpu().reshape(-1)
        )
        print(
            f"processed={min(start + args.batch_size, len(sample_dirs))}/"
            f"{len(sample_dirs)}",
            flush=True,
        )

    tensor_keys = tuple(key for key in records if key not in {"tokens", "logs"})
    for key in tensor_keys:
        records[key] = torch.cat(records[key], dim=0)

    selected_factors = records["selected_factors"].float()
    selected_score = selected_factors[:, -1]
    best_score = records["best_pdm_score"].float()
    summary = {
        "name": args.name,
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "pytorch_lightning_version": pytorch_lightning.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "peft_version": peft.__version__,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "cache_root": str(args.cache_root),
        "metric_cache": str(args.metric_cache),
        "sample_count": len(sample_dirs),
        "log_count": len(set(records["logs"])),
        "sample_tokens_sha256": hashlib.sha256(
            "\n".join(records["tokens"]).encode("utf-8")
        ).hexdigest(),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "flash_attention": args.flash_attention,
        "selected_pdm": float(selected_score.mean()),
        "best64_pdm": float(best_score.mean()),
        "selection_regret": float((best_score - selected_score).mean()),
        "l2_2s": float(records["selected_l2_2s"].float().mean()),
        "factor_means": {
            name: float(selected_factors[:, index].mean())
            for index, name in enumerate(FACTOR_NAMES)
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": summary,
            "factor_names": FACTOR_NAMES,
            **records,
        },
        args.output,
    )
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
