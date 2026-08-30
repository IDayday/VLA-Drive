#!/usr/bin/env python3
"""Fingerprint Stage-2 VLM runtime semantics for one identical camera input.

The merged DriveVLA checkpoint stores weights but not the tokenizer, remote
model code, or non-persistent configuration.  This read-only audit restores the
same public Stage-1 backbone into a VLM architecture built from a requested
model directory, then saves the final hidden state for exact A/B comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.EpisodeDrive.drivevla_backbone import system_message
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from navsim.agents.EpisodeDrive.utils.internvl_tokenize import (
    build_internvl_model_inputs,
)
from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
)
DEFAULT_STAGE1 = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_feature(sample_dir: Path):
    with gzip.open(sample_dir / "internvl_feature.gz", "rb") as stream:
        return pickle.load(stream)


def _decode_path(tensor: torch.Tensor) -> Path:
    return Path("".join(chr(int(value)) for value in tensor.flatten().tolist()))


def _first_sample(cache_root: Path) -> Path:
    for feature_path in sorted(cache_root.glob("*/*/internvl_feature.gz")):
        return feature_path.parent
    raise RuntimeError(f"No Stage-2 feature cache below {cache_root}")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--extra-token-count", type=int, required=True)
    parser.add_argument("--target-vocab-size", type=int, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample_dir = args.sample_dir or _first_sample(args.cache_root)
    for path in (args.vlm_path, args.stage1_checkpoint, sample_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ["DRIVEVLA_SCORE_RAY"] = "0"
    os.environ["DRIVEVLA_SCORE_PROCESSES"] = "0"
    _seed(args.seed)
    config_dir = REPO_ROOT / "navsim/planning/script/config/training"
    overrides = [
        "train_test_split=navtrain",
        "agent.checkpoint_path=null",
        f"agent.stage1_checkpoint_path={args.stage1_checkpoint}",
        "agent.cache_data=false",
        f"agent.vlm_config.vlm_path={args.vlm_path}",
        "agent.vlm_config.freeze_backbone=true",
        "agent.vlm_config.cache_hidden_state=false",
        "agent.vlm_config.cache_mode=false",
        "agent.vlm_config.initialize_from_config=true",
        "agent.vlm_config.use_flash_attn=false",
        "agent.vlm_config.frozen_backbone_mode=eval",
        f"agent.vlm_config.extra_token_count={args.extra_token_count}",
        f"agent.vlm_config.target_vocab_size={args.target_vocab_size}",
        "agent.lora_config.use_lora=true",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()

    feature = _load_feature(sample_dir)
    image_path = _decode_path(feature["image_path_tensor"])
    question = build_drivevla_questions(
        feature["history_trajectory"].unsqueeze(0),
        feature["high_command_one_hot"].unsqueeze(0),
    )
    pixel_values = load_image(str(image_path)).cuda().to(torch.bfloat16)
    model_inputs = build_internvl_model_inputs(
        agent.backbone.tokenizer,
        question,
        [pixel_values.shape[0]],
        system_message,
    )
    active_ids = model_inputs["input_ids"][model_inputs["attention_mask"].bool()]
    _seed(args.seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = agent.backbone(
            pixel_values,
            question,
            num_patches_list=[pixel_values.shape[0]],
            model_inputs=model_inputs,
        )
    hidden = output.hidden_states[-1].detach().cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": args.name,
            "sample_dir": str(sample_dir),
            "image_path": str(image_path),
            "input_ids": model_inputs["input_ids"].cpu(),
            "attention_mask": model_inputs["attention_mask"].cpu(),
            "last_hidden_state": hidden,
        },
        args.output,
    )
    summary = {
        "name": args.name,
        "vlm_path": str(args.vlm_path),
        "sample_dir": str(sample_dir),
        "image_path": str(image_path),
        "extra_token_count": args.extra_token_count,
        "target_vocab_size": args.target_vocab_size,
        "tokenizer_size": len(agent.backbone.tokenizer),
        "active_token_count": int(active_ids.numel()),
        "active_token_sha256": _tensor_sha256(active_ids),
        "last_hidden_shape": list(hidden.shape),
        "last_hidden_dtype": str(hidden.dtype),
        "last_hidden_sha256": _tensor_sha256(hidden),
        "last_hidden_mean": float(hidden.float().mean()),
        "last_hidden_std": float(hidden.float().std()),
        "config_sha256": _sha256(args.vlm_path / "config.json"),
        "tokenizer_config_sha256": _sha256(
            args.vlm_path / "tokenizer_config.json"
        ),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
