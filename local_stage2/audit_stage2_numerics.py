#!/usr/bin/env python3
"""Audit Stage-2 numerical semantics from pixels through one optimizer step.

The released repository is deployment-oriented and does not provide the exact
launcher used for the public checkpoint.  This tool makes the important
ambiguous choices explicit and records comparable tensors for controlled A/B
experiments:

* FlashAttention versus the released eager-attention config;
* frozen-backbone eval mode versus standard Lightning train-mode semantics;
* worker preprocessing versus the original serial preprocessing path; and
* all-parameter AdamW decay versus the fork's norm/bias no-decay convention.

It consumes two existing NAVSIM cache samples read-only and writes a compact
``.pt`` artifact plus JSON summary.  No dataset or metric cache is modified.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data._utils.collate import default_collate

from navsim.agents.EpisodeDrive.drivevla_backbone import system_message
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from navsim.agents.EpisodeDrive.utils.internvl_tokenize import (
    build_internvl_model_inputs,
)
from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions
from navsim.planning.training.dataset import drivevla_cached_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
)
DEFAULT_METRIC_CACHE = Path(
    "/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full"
)
DEFAULT_STAGE1 = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)
DEFAULT_VLM = Path("/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope")


def _load_gzip(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _decode_image_path(path_tensor: torch.Tensor) -> str:
    return "".join(chr(int(value)) for value in path_tensor.flatten().tolist())


def _find_samples(cache_root: Path, count: int) -> List[Path]:
    samples = []
    for feature_path in sorted(cache_root.glob("*/*/internvl_feature.gz")):
        sample_dir = feature_path.parent
        if (sample_dir / "trajectory_target.gz").is_file():
            samples.append(sample_dir)
        if len(samples) == count:
            break
    if len(samples) != count:
        raise RuntimeError(
            f"Requested {count} complete cache samples under {cache_root}, "
            f"found {len(samples)}"
        )
    return samples


def _raw_batch(sample_dirs: Iterable[Path]):
    samples = []
    for sample_dir in sample_dirs:
        feature = _load_gzip(sample_dir / "internvl_feature.gz")
        target = _load_gzip(sample_dir / "trajectory_target.gz")
        samples.append((feature, target))
    return default_collate(samples)


def _optimized_batch(sample_dirs: Iterable[Path], tokenizer):
    samples = []
    for sample_dir in sample_dirs:
        feature = _load_gzip(sample_dir / "internvl_feature.gz")
        target = _load_gzip(sample_dir / "trajectory_target.gz")
        image_path = _decode_image_path(feature.pop("image_path_tensor"))
        feature["pixel_values"] = load_image(image_path).to(torch.bfloat16)
        questions = build_drivevla_questions(
            feature["history_trajectory"], feature["high_command_one_hot"]
        )
        model_inputs = build_internvl_model_inputs(
            tokenizer,
            questions,
            [feature["pixel_values"].shape[0]],
            system_message,
        )
        feature["input_ids"] = model_inputs["input_ids"].squeeze(0)
        feature["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
        samples.append((feature, target))
    return drivevla_cached_collate(samples)


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _flatten_named_tensors(named_tensors: Iterable[Tuple[str, torch.Tensor]]):
    names = []
    tensors = []
    shapes = []
    for name, tensor in named_tensors:
        names.append(name)
        shapes.append(list(tensor.shape))
        tensors.append(tensor.detach().float().cpu().reshape(-1))
    flat = torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.float32)
    return names, shapes, flat


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _optimizer(agent):
    configured = agent.get_optimizers()
    if not isinstance(configured, list) or len(configured) != 1:
        raise RuntimeError(f"Unexpected optimizer payload: {type(configured)}")
    return configured[0]


def _run_path(
    agent,
    batch,
    initial_action_state: Dict[str, torch.Tensor],
    forward_seed: int,
):
    agent.action_head.load_state_dict(initial_action_state, strict=True)
    agent.zero_grad(set_to_none=True)
    optimizer = _optimizer(agent)
    captured: Dict[str, torch.Tensor] = {}

    def capture_action_input(_module, args):
        captured["last_hidden_state"] = (
            args[0]["last_hidden_state"].detach().float().cpu()
        )

    hook = agent.action_head.register_forward_pre_hook(capture_action_input)
    _seed_everything(forward_seed)
    features, targets = _clone_tree(batch)
    targets = {
        key: value.cuda(non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in targets.items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = agent.forward(features)
        loss_dict = agent.compute_loss(features, targets, prediction)
        loss = loss_dict["loss"]
    hook.remove()
    loss.backward()

    grad_names, grad_shapes, gradients = _flatten_named_tensors(
        (
            name,
            parameter.grad
            if parameter.grad is not None
            else torch.zeros_like(parameter),
        )
        for name, parameter in agent.action_head.named_parameters()
    )
    _, _, before = _flatten_named_tensors(agent.action_head.named_parameters())
    optimizer.step()
    _, _, after = _flatten_named_tensors(agent.action_head.named_parameters())

    result = {
        "last_hidden_state": captured["last_hidden_state"],
        "proposals": prediction["proposals"].detach().float().cpu(),
        "pdm_score": prediction["pdm_score"].detach().float().cpu(),
        "gradients": gradients,
        "parameter_delta": after - before,
        "loss": float(loss.detach().cpu()),
        "loss_terms": {
            key: float(value.detach().float().cpu())
            for key, value in loss_dict.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "gradient_names": grad_names,
        "gradient_shapes": grad_shapes,
    }
    return result


def _difference(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    delta = (left.float() - right.float()).abs()
    denominator = left.float().abs().clamp_min(1e-8)
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "max_relative": float((delta / denominator).max()) if delta.numel() else 0.0,
        "equal": bool(torch.equal(left, right)),
    }


def _compose_agent_config(args):
    config_dir = REPO_ROOT / "navsim/planning/script/config/training"
    overrides = [
        "train_test_split=navtrain",
        "agent.checkpoint_path=null",
        f"agent.stage1_checkpoint_path={args.stage1_checkpoint}",
        "agent.cache_data=false",
        "agent.vlm_config.freeze_backbone=true",
        f"agent.vlm_config.vlm_path={args.vlm_path}",
        "agent.vlm_config.cache_hidden_state=false",
        "agent.vlm_config.cache_mode=false",
        "agent.vlm_config.initialize_from_config=true",
        f"agent.vlm_config.use_flash_attn={str(args.flash_attention).lower()}",
        f"agent.vlm_config.frozen_backbone_mode={args.frozen_backbone_mode}",
        "agent.vlm_config.extra_token_count=8",
        "agent.vlm_config.target_vocab_size=151682",
        "agent.lora_config.use_lora=true",
        "agent.batch_size=2",
        "agent.num_gpus=8",
        "agent.lr_args.name=AdamW",
        "agent.lr_args.base_lr=1e-4",
        "agent.lr_args.base_batch_size=16",
        f"agent.lr_args.decay_norm_and_bias={str(args.decay_norm_and_bias).lower()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--metric-cache", type=Path, default=DEFAULT_METRIC_CACHE
    )
    parser.add_argument(
        "--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1
    )
    parser.add_argument("--vlm-path", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-seed", type=int, default=20260830)
    parser.add_argument(
        "--frozen-backbone-mode", choices=("eval", "train"), default="eval"
    )
    parser.add_argument(
        "--flash-attention", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--decay-norm-and-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    for path in (
        args.cache_root,
        args.metric_cache,
        args.stage1_checkpoint,
        args.vlm_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ["NAVSIM_TRAIN_METRIC_CACHE"] = str(args.metric_cache)
    os.environ["DRIVEVLA_SCORE_RAY"] = "0"
    os.environ["DRIVEVLA_SCORE_PROCESSES"] = "0"
    _seed_everything(args.seed)
    cfg = _compose_agent_config(args)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.to("cuda")
    agent.train(True)

    if any(parameter.requires_grad for parameter in agent.backbone.parameters()):
        raise RuntimeError("Frozen backbone unexpectedly contains trainable parameters")
    expected_backbone_training = args.frozen_backbone_mode == "train"
    if agent.backbone.training != expected_backbone_training:
        raise RuntimeError(
            "Frozen-backbone mode mismatch: "
            f"expected training={expected_backbone_training}, "
            f"got {agent.backbone.training}"
        )

    sample_dirs = _find_samples(args.cache_root, count=2)
    raw_batch = _raw_batch(sample_dirs)
    optimized_batch = _optimized_batch(sample_dirs, agent.backbone.tokenizer)
    initial_action_state = {
        name: tensor.detach().clone()
        for name, tensor in agent.action_head.state_dict().items()
    }
    _, _, initial_flat = _flatten_named_tensors(
        agent.action_head.named_parameters()
    )

    raw_result = _run_path(
        agent, raw_batch, initial_action_state, args.forward_seed
    )
    optimized_result = _run_path(
        agent, optimized_batch, initial_action_state, args.forward_seed
    )
    comparisons = {
        key: _difference(raw_result[key], optimized_result[key])
        for key in (
            "last_hidden_state",
            "proposals",
            "pdm_score",
            "gradients",
            "parameter_delta",
        )
    }
    comparisons["loss_abs"] = abs(raw_result["loss"] - optimized_result["loss"])

    artifact = {
        "name": args.name,
        "config": {
            "seed": args.seed,
            "forward_seed": args.forward_seed,
            "flash_attention": args.flash_attention,
            "frozen_backbone_mode": args.frozen_backbone_mode,
            "decay_norm_and_bias": args.decay_norm_and_bias,
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "metric_cache": str(args.metric_cache),
            "sample_dirs": [str(path) for path in sample_dirs],
        },
        "initial_action_sha256": _tensor_sha256(initial_flat),
        "raw": raw_result,
        "optimized": optimized_result,
        "raw_vs_optimized": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"{args.name}.pt"
    summary_path = args.output_dir / f"{args.name}.json"
    torch.save(artifact, artifact_path)
    summary = {
        "name": args.name,
        "config": artifact["config"],
        "initial_action_sha256": artifact["initial_action_sha256"],
        "backbone_training": agent.backbone.training,
        "raw_loss": raw_result["loss"],
        "optimized_loss": optimized_result["loss"],
        "raw_vs_optimized": comparisons,
        "tensor_sha256": {
            path_name: {
                key: _tensor_sha256(result[key])
                for key in (
                    "last_hidden_state",
                    "proposals",
                    "pdm_score",
                    "gradients",
                    "parameter_delta",
                )
            }
            for path_name, result in (
                ("raw", raw_result),
                ("optimized", optimized_result),
            )
        },
        "artifact": str(artifact_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
