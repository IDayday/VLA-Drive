#!/usr/bin/env python3
"""Run one real NAVSIM sample through a deployment-only PlanReg checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data._utils.collate import default_collate

from navsim.common.dataclasses import SceneFilter
from navsim.planning.script.run_training_full import (
    build_datasets,
    resolve_data_protocol,
)


FUTURE_ONLY_KEYS = {
    "future_image_paths",
    "future_image_path_lengths",
    "future_valid_mask",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Use the exact student-only formal topology and strict loader",
    )
    return parser.parse_args()


def _compose_config(args: argparse.Namespace) -> DictConfig:
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "navsim/planning/script/config/training"
    )
    agent_config = (
        "episode_drive_planreg_wm_formal_student"
        if args.formal
        else "episode_drive_planreg_wm_v1"
    )
    overrides = [
        f"train_test_split={args.split}",
        f"agent={agent_config}",
        f"agent.checkpoint_path={args.checkpoint.resolve()}",
        "agent.stage1_checkpoint_path=null",
        f"agent.vlm_config.vlm_path={args.vlm_path.resolve()}",
        "agent.world_model.enabled=false",
        "agent.ema.enabled=false",
        "agent.batch_size=1",
        "agent.num_gpus=1",
        f"navsim_log_path={args.navsim_log_path.resolve()}",
        f"sensor_blobs_path={args.sensor_blobs_path.resolve()}",
        "load_image_path=true",
        "use_cache_without_dataset=false",
        "force_cache_computation=false",
        "train_test_split.scene_filter.max_scenes=1",
        "trainer.params.devices=1",
        "dataloader.params.batch_size=1",
        "dataloader.params.num_workers=0",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name="planreg_student_smoke",
    ):
        return compose(config_name="default_training", overrides=overrides)


def _batch_features(features: Dict[str, object]) -> Dict[str, object]:
    batched = default_collate([features])
    for key in FUTURE_ONLY_KEYS:
        batched.pop(key, None)
    return batched


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError(
            "The real InternVL3-2B deployment smoke requires one CUDA device"
        )
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    os.environ.setdefault("DRIVEVLA_SCORE_RAY", "0")
    os.environ.setdefault("DRIVEVLA_SCORE_PROCESSES", "0")
    os.environ.setdefault("LOCAL_RANK", str(torch.device(args.device).index or 0))
    if args.formal:
        os.environ["PLANREG_STUDENT_CHECKPOINT"] = str(checkpoint)
        os.environ["PLANREG_FORMAL_VLM_PATH"] = str(args.vlm_path.resolve())

    cfg = _compose_config(args)
    agent = instantiate(cfg.agent)
    agent.initialize()
    # Unlike Lightning predict, this standalone deployment path has no trainer
    # to place the FP32 action/scorer modules. The VLM is already initialized on
    # CUDA; moving the complete student is idempotent for it and places the
    # remaining modules on the same explicit device.
    agent.to(torch.device(args.device))
    if agent.world_model_enabled:
        raise AssertionError("Deployment config unexpectedly enabled world model")
    if agent.future_register_predictor is not None:
        raise AssertionError("Deployment initialized future_register_predictor")
    if agent.ema_register_target is not None:
        raise AssertionError("Deployment initialized EMA teacher")

    filter_logs = instantiate(cfg.train_test_split.scene_filter).log_names
    protocol = resolve_data_protocol(cfg, scene_filter_log_names=filter_logs)
    _, val_data = build_datasets(cfg, agent, protocol)
    if len(val_data) < 1:
        raise RuntimeError("No validation scene is available for deployment smoke")
    features, targets = val_data[0]
    future_keys_in_target = sorted(FUTURE_ONLY_KEYS.intersection(targets))
    runtime_features = _to_device(
        _batch_features(features), torch.device(args.device)
    )
    if FUTURE_ONLY_KEYS.intersection(runtime_features):
        raise AssertionError("Future-only keys leaked into deployment inference")

    agent.eval()
    with torch.inference_mode():
        prediction = agent.forward(runtime_features)
    trajectory = prediction["trajectory"]
    if tuple(trajectory.shape) != (1, 8, 3):
        raise AssertionError(
            f"Expected deployment trajectory [1,8,3], got {tuple(trajectory.shape)}"
        )
    if not bool(torch.isfinite(trajectory).all()):
        raise FloatingPointError("Deployment trajectory contains non-finite values")

    report = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_finite": True,
        "world_model_enabled": False,
        "future_predictor_constructed": False,
        "ema_teacher_constructed": False,
        "runtime_future_keys": [],
        "exact_formal_student_loader": bool(args.formal),
        "target_future_keys_present_but_unused": future_keys_in_target,
        "train_log_count": protocol["train_log_count"],
        "val_log_count": protocol["val_log_count"],
        "overlap_count": protocol["overlap_count"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
