#!/usr/bin/env python3
"""Record the software and public-config evidence relevant to Stage-2 replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = Path(
    "/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope/config.json"
)
DEFAULT_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)


def _run(arguments: list[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def _version(*distribution_names: str) -> Optional[str]:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config_summary(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text())
    vision = payload.get("vision_config", {})
    language = payload.get("llm_config", {})
    return {
        "path": str(path),
        "exists": True,
        "torch_dtype": payload.get("torch_dtype"),
        "vision_transformers_version": vision.get("transformers_version"),
        "language_transformers_version": language.get("transformers_version"),
        "vision_drop_path_rate": vision.get("drop_path_rate"),
        "vision_dropout": vision.get("dropout"),
        "vision_attention_dropout": vision.get("attention_dropout"),
        "language_attention_dropout": language.get("attention_dropout"),
    }


def _released_config_summary(repo_root: Path) -> dict:
    agent_path = (
        repo_root
        / "navsim/planning/script/config/common/agent/episode_drive.yaml"
    )
    training_path = (
        repo_root / "navsim/planning/script/config/training/default_training.yaml"
    )
    agent = yaml.safe_load(agent_path.read_text())
    training = yaml.safe_load(training_path.read_text())
    base_lr = float(agent["lr_args"]["base_lr"])
    base_batch_size = int(agent["lr_args"]["base_batch_size"])
    declared_agent_batch_size = int(agent["batch_size"])
    declared_agent_gpu_count = int(agent["num_gpus"])
    inferred_global_batch_size = 16
    return {
        "agent_config": str(agent_path),
        "training_config": str(training_path),
        "use_flash_attn": agent["vlm_config"].get("use_flash_attn"),
        "lora_dropout": agent["lora_config"].get("lora_dropout"),
        "optimizer": agent["lr_args"].get("name"),
        "configured_base_lr": base_lr,
        "configured_base_batch_size": base_batch_size,
        "declared_agent_batch_size": declared_agent_batch_size,
        "declared_agent_gpu_count": declared_agent_gpu_count,
        "configured_scheduler": agent.get("scheduler_args"),
        "yaml_effective_lr_at_declared_agent_batch": base_lr
        * (
            declared_agent_batch_size
            * declared_agent_gpu_count
            / base_batch_size
        )
        ** 0.5,
        "yaml_effective_lr_at_global_batch_16": base_lr
        * (inferred_global_batch_size / base_batch_size) ** 0.5,
        "precision": training["trainer"]["params"].get("precision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = {
        "path": str(args.checkpoint),
        "exists": args.checkpoint.is_file(),
        "size_bytes": args.checkpoint.stat().st_size
        if args.checkpoint.is_file()
        else None,
    }
    if args.hash_checkpoint and args.checkpoint.is_file():
        checkpoint["sha256"] = _sha256(args.checkpoint)

    report = {
        "repo": {
            "path": str(args.repo_root.resolve()),
            "branch": _run(["git", "branch", "--show-current"], args.repo_root),
            "commit": _run(["git", "rev-parse", "HEAD"], args.repo_root),
            "status_short": _run(["git", "status", "--short"], args.repo_root),
            "remotes": _run(["git", "remote", "-v"], args.repo_root),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "executable_resolved": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "packages": {
                "torch": torch.__version__,
                "torchvision": _version("torchvision"),
                "pytorch_lightning": _version("pytorch-lightning"),
                "transformers": _version("transformers"),
                "tokenizers": _version("tokenizers"),
                "peft": _version("peft"),
                "flash_attn": _version("flash-attn", "flash_attn"),
                "numpy": _version("numpy"),
                "scipy": _version("scipy"),
                "shapely": _version("shapely"),
                "torchmetrics": _version("torchmetrics"),
                "hydra_core": _version("hydra-core"),
                "omegaconf": _version("omegaconf"),
            },
            "cuda": {
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "nvidia_smi": _run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,driver_version,memory.total",
                        "--format=csv,noheader",
                    ]
                ),
            },
        },
        "model_config": _model_config_summary(args.model_config),
        "released_config": _released_config_summary(args.repo_root),
        "checkpoint": checkpoint,
        "reproduction_evidence": {
            "checkpoint_epoch_index": 26,
            "checkpoint_global_step": 174_312,
            "completed_epochs": 27,
            "derived_steps_per_epoch": 6_456,
            "navtrain_scene_count": 103_288,
            "derived_global_batch_size": 16,
            "paper_gpu_count": 16,
            "inferred_per_gpu_batch_size": 1,
            "paper_reported_base_model_lr": 1e-4,
            "lr_semantics": {
                "paper_value_as_actual_optimizer_lr": 1e-4,
                "paper_value_as_base_lr_through_released_scaling": 1e-4
                * (16 / 64) ** 0.5,
                "released_yaml_with_declared_agent_batch_fields": 5e-4
                * (2 / 64) ** 0.5,
                "released_yaml_with_inferred_global_batch_16": 5e-4
                * (16 / 64) ** 0.5,
            },
            "paper_reported_optimizer": "AdamW",
            "paper_source": "https://arxiv.org/html/2608.10413v1#A2",
            "inference_note": (
                "Global/per-GPU batch sizes are inferred from the checkpoint "
                "name, scene count, epoch count, and paper GPU count; the "
                "private launcher was not released. The paper does not say "
                "whether its 1e-4 value is the optimizer-group LR after the "
                "released square-root batch scaling or the pre-scaling "
                "base_lr configuration value."
            ),
        },
        "environment_paths": {
            key: os.environ.get(key)
            for key in (
                "NAVSIM_DEVKIT_ROOT",
                "NAVSIM_EXP_ROOT",
                "NUPLAN_MAPS_ROOT",
                "DRIVEVLA_VLM_CONFIG",
                "DRIVEVLA_STAGE1_CHECKPOINT",
            )
        },
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
