#!/usr/bin/env python3
"""One optimizer step through production Qwen_PI + DDP-DRS training forward."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Mapping, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import get_scheduler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starVLA.model.framework.QwenPI import Qwen_PI  # noqa: E402
from starVLA.model.modules.action_model.multi_trajectory.checkpointing import (  # noqa: E402
    load_base_checkpoint_strict,
)
from starVLA.model.modules.action_model.multi_trajectory.losses import (  # noqa: E402
    DRIVOR_SCORE_NAMES,
    SUPRIM_SCORE_NAMES,
)
from starVLA.training.trainer_utils.trainer_tools import (  # noqa: E402
    build_param_lr_groups,
)


QWEN_CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0")
TRAINING_STAGES = (
    "train_drivor",
    "train_suprim_static",
    "train_suprim_joint",
    "joint_finetune",
)
SMOKE_STAGES = TRAINING_STAGES + ("inference",)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=SMOKE_STAGES, default="train_drivor")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--base-ddp-checkpoint", type=Path)
    parser.add_argument("--scene-checkpoint", type=Path)
    parser.add_argument("--drivor-checkpoint", type=Path)
    parser.add_argument("--suprim-checkpoint", type=Path)
    parser.add_argument("--suprim-vocab", type=Path)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _binary_targets(names: tuple[str, ...], count: int) -> dict[str, np.ndarray]:
    values = np.arange(count, dtype=np.float32) % 2
    return {name: values.copy() for name in names}


def _sample(metadata_path: Path, stage: str) -> tuple[dict, list[str], int]:
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    if not isinstance(metadata, Mapping) or "glo_images" not in metadata:
        raise TypeError("sample metadata must contain glo_images")
    camera_data = metadata["glo_images"]
    missing = [name for name in QWEN_CAMERA_ORDER if name not in camera_data]
    if missing:
        raise KeyError(f"sample is missing Qwen cameras: {missing}")
    paths = [Path(camera_data[name]["image_paths"][3]) for name in QWEN_CAMERA_ORDER]
    images = [_load_image(path) for path in paths]
    if stage == "inference":
        candidate_count = 8192 + 16
        targets = None
    elif stage == "train_drivor":
        candidate_count = 64
        targets = {"drivor_scores": _binary_targets(DRIVOR_SCORE_NAMES, 64)}
    else:
        candidate_count = 8192 + (
            16 if stage in {"train_suprim_joint", "joint_finetune"} else 0
        )
        targets = {
            "coarse_scores": _binary_targets(SUPRIM_SCORE_NAMES, candidate_count),
            "trajectory": np.zeros((8, 3), dtype=np.float32),
        }
        if stage == "joint_finetune":
            targets["drivor_scores"] = _binary_targets(DRIVOR_SCORE_NAMES, 64)
    return (
        {
            "image": images,
            "lang": "Drive safely and follow the planned route.",
            "state": np.asarray([[0.25, -0.10, 0.05]], dtype=np.float32),
            "action": np.zeros((8, 3), dtype=np.float32),
            **(
                {}
                if targets is None
                else {"multi_trajectory_targets": targets}
            ),
            **(
                {
                    "multi_trajectory_candidates": np.zeros(
                        (64, 8, 3), dtype=np.float32
                    )
                }
                if stage
                in {"train_drivor", "train_suprim_joint", "joint_finetune"}
                else {}
            ),
        },
        [str(path.resolve()) for path in paths],
        candidate_count,
    )


def _load_base(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError("base DDP checkpoint must contain a state dict")
    return state


def _optional_existing(path: Optional[Path], label: str) -> Optional[str]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def main() -> None:
    args = _arguments()
    required = (
        args.config,
        args.base_vlm / "config.json",
        args.base_vlm / "model.safetensors",
        args.sample_metadata,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"framework smoke inputs are missing: {missing}")
    if args.stage != "train_drivor" and args.suprim_vocab is None:
        raise FileNotFoundError("DriveSuprim stages require --suprim-vocab")
    if args.stage in {"train_suprim_static", "train_suprim_joint", "inference"} and args.scene_checkpoint is None:
        raise FileNotFoundError(f"{args.stage} requires --scene-checkpoint")
    if args.stage in {"train_suprim_joint", "inference"} and args.drivor_checkpoint is None:
        raise FileNotFoundError(f"{args.stage} requires --drivor-checkpoint")
    if args.stage == "inference" and args.suprim_checkpoint is None:
        raise FileNotFoundError("inference requires --suprim-checkpoint")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("framework smoke requires the requested CUDA device")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    config = OmegaConf.load(args.config)
    overrides = {
        "framework.qwenvl.base_vlm": str(args.base_vlm),
        "framework.qwenvl.attn_implementation": "sdpa",
        "datasets.vla_data.image_size": [224, 224],
        "multi_trajectory.enabled": True,
        "multi_trajectory.training_stage": args.stage,
        "multi_trajectory.strict_inference": False,
        "multi_trajectory.diagnostics_enabled": args.stage == "inference",
        "multi_trajectory.deterministic_seed": args.seed,
        "multi_trajectory.scene_compressor.checkpoint_path": _optional_existing(
            args.scene_checkpoint, "scene checkpoint"
        ),
        "multi_trajectory.drivor.checkpoint_path": _optional_existing(
            args.drivor_checkpoint, "DrivoR checkpoint"
        ),
        "multi_trajectory.suprim.checkpoint_path": _optional_existing(
            args.suprim_checkpoint, "DriveSuprim checkpoint"
        ),
        "multi_trajectory.suprim.vocab_path": _optional_existing(
            args.suprim_vocab, "DriveSuprim vocabulary"
        ),
    }
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, force_add=True)

    sample, image_paths, candidate_count = _sample(args.sample_metadata, args.stage)
    started = time.perf_counter()
    model = Qwen_PI(config)
    checkpoint_kind = "pretrained_qwen_random_ddp_computation_smoke"
    if args.base_ddp_checkpoint is not None:
        load_base_checkpoint_strict(model, _load_base(args.base_ddp_checkpoint))
        checkpoint_kind = "strict_base_ddp_checkpoint"
    if args.stage == "inference":
        model.to(device).eval()
        planner = model.multi_trajectory_planner
        counts = {"qwen": 0, "qformer": 0}
        qwen_hook = model.qwen_vl_interface.register_forward_hook(
            lambda *_: counts.__setitem__("qwen", counts["qwen"] + 1)
        )
        qformer_hook = planner.scene_compressor.register_forward_hook(
            lambda *_: counts.__setitem__("qformer", counts["qformer"] + 1)
        )
        try:
            output = model.predict_action(examples=[sample])
        finally:
            qwen_hook.remove()
            qformer_hook.remove()
        trajectory = output["normalized_actions"]
        if trajectory.shape != (1, 8, 3):
            raise RuntimeError(f"unexpected inference output {trajectory.shape}")
        if counts != {"qwen": 1, "qformer": 1}:
            raise RuntimeError(
                "expected one Qwen and one Q-Former inference call, got "
                f"{counts}"
            )
        diagnostics = output.get("multi_trajectory_diagnostics")
        if diagnostics is None:
            raise RuntimeError("inference diagnostics were not returned")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "stage": args.stage,
                    "checkpoint_kind": checkpoint_kind,
                    "component_checkpoint_kind": "one_step_training_smoke",
                    "output_shape": list(trajectory.shape),
                    "batch_size": 1,
                    "sequence_length": int(
                        diagnostics.dense_scene_memory_bytes
                        // (2048 * 2)
                    ),
                    "dynamic_candidate_count": 64,
                    "static_candidate_count": 8192,
                    "joint_candidate_count": candidate_count,
                    "qwen_forward_count": counts["qwen"],
                    "qformer_forward_count": counts["qformer"],
                    "qformer_parameter_count": sum(
                        parameter.numel()
                        for parameter in planner.scene_compressor.parameters()
                    ),
                    "latencies_seconds": {
                        "qwen": diagnostics.latency_qwen,
                        "ddp_sampling": diagnostics.latency_ddp_sampling,
                        "scene_compressor": diagnostics.latency_scene_compressor,
                        "drivor_scorer": diagnostics.latency_drivor_scorer,
                        "suprim_coarse": diagnostics.latency_suprim_coarse,
                        "suprim_fine": diagnostics.latency_suprim_refinement,
                        "full_inference": diagnostics.latency_total_inference,
                    },
                    "global_scene_tokens_bytes": diagnostics.global_scene_tokens_bytes,
                    "dense_scene_memory_bytes": diagnostics.dense_scene_memory_bytes,
                    "image_paths": image_paths,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "gpu": torch.cuda.get_device_name(device),
                    "dtype": "bfloat16 autocast",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    model.to(device).train()
    if model.qwen_vl_interface.training or model.action_model.training:
        raise RuntimeError("Qwen or DDP entered training mode")
    planner = model.multi_trajectory_planner
    if args.stage not in {"train_drivor", "joint_finetune"} and planner.scene_compressor.training:
        raise RuntimeError("frozen scene compressor entered training mode")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
        not name.startswith("multi_trajectory_planner.") for name in trainable
    ):
        raise RuntimeError("optimizer boundary includes Qwen or original DDP parameters")
    representative_name, representative_parameter = next(iter(trainable.items()))
    representative_before = representative_parameter.detach().cpu().clone()
    groups = build_param_lr_groups(model=model, cfg=config)
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(config.trainer.learning_rate.base),
        betas=tuple(config.trainer.optimizer.betas),
        weight_decay=float(config.trainer.optimizer.weight_decay),
        eps=float(config.trainer.optimizer.eps),
    )
    scheduler = get_scheduler(
        name=config.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(config.trainer.num_warmup_steps),
        num_training_steps=int(config.trainer.max_train_steps),
        scheduler_specific_kwargs=config.trainer.scheduler_specific_kwargs,
    )
    counts = {"qwen": 0, "qformer": 0}
    qwen_hook = model.qwen_vl_interface.register_forward_hook(
        lambda *_: counts.__setitem__("qwen", counts["qwen"] + 1)
    )
    qformer_hook = planner.scene_compressor.register_forward_hook(
        lambda *_: counts.__setitem__("qformer", counts["qformer"] + 1)
    )
    initial_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    optimizer_steps = 2 if all(value == 0.0 for value in initial_learning_rates) else 1
    learning_rate_history = [initial_learning_rates[0]]
    try:
        for step_index in range(optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model.forward([sample])
                loss = output["action_loss"]
            expected_calls = step_index + 1
            if counts != {"qwen": expected_calls, "qformer": expected_calls}:
                raise RuntimeError(
                    "expected one Qwen and Q-Former forward per step, got "
                    f"{counts} after {expected_calls} steps"
                )
            details = output["multi_trajectory_loss_details"]
            actual_count = (
                details["dynamic_output"].aggregate_score.shape[1]
                if args.stage == "train_drivor"
                else details["joint_output"].coarse_scores["aggregate_score"].shape[1]
            )
            if actual_count != candidate_count:
                raise RuntimeError(
                    f"candidate count {actual_count} does not match {candidate_count}"
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"framework smoke produced non-finite loss {loss}")
            loss.backward()
            qwen_grads = [
                name
                for name, parameter in model.qwen_vl_interface.named_parameters()
                if parameter.grad is not None
            ]
            ddp_grads = [
                name
                for name, parameter in model.action_model.named_parameters()
                if parameter.grad is not None
            ]
            trainable_grads = [
                name for name, parameter in trainable.items() if parameter.grad is not None
            ]
            if qwen_grads or ddp_grads or not trainable_grads:
                raise RuntimeError(
                    f"gradient boundary failed: qwen={qwen_grads[:3]} "
                    f"ddp={ddp_grads[:3]} trainable_grad_count={len(trainable_grads)}"
                )
            optimizer.step()
            scheduler.step()
            learning_rate_history.append(float(scheduler.get_last_lr()[0]))
    finally:
        qwen_hook.remove()
        qformer_hook.remove()
    if torch.equal(
        representative_before, representative_parameter.detach().cpu()
    ):
        raise RuntimeError(f"optimizer did not update {representative_name}")
    print(
        json.dumps(
            {
                "status": "passed",
                "stage": args.stage,
                "checkpoint_kind": checkpoint_kind,
                "loss": float(loss.detach().float().cpu()),
                "candidate_count": candidate_count,
                "qwen_forward_count": counts["qwen"],
                "qformer_forward_count": counts["qformer"],
                "optimizer_steps": optimizer_steps,
                "learning_rate_history": learning_rate_history,
                "trainable_parameter_count": sum(p.numel() for p in trainable.values()),
                "trainable_tensors_with_grad": len(trainable_grads),
                "qwen_tensors_with_grad": 0,
                "ddp_tensors_with_grad": 0,
                "image_paths": image_paths,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "gpu": torch.cuda.get_device_name(device),
                "dtype": "bfloat16 autocast",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
