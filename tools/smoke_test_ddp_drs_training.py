#!/usr/bin/env python3
"""One-step production-dimension DDP-DRS component training/inference smoke.

Inputs are synthetic Qwen hidden states and deterministic score labels.  The
actual 2048-wide Q-Former, 256/2048 asymmetric decoders, official vocabulary,
losses, optimizer discovery, freezing rules, and 8192+16 global selection run
unchanged.  Paths are explicit and no download is attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starVLA.model.modules.action_model.multi_trajectory.config import (  # noqa: E402
    DrivoRConfig,
    MultiTrajectoryConfig,
    SceneCompressorConfig,
    SuprimConfig,
)
from starVLA.model.modules.action_model.multi_trajectory.losses import (  # noqa: E402
    DRIVOR_SCORE_NAMES,
    SUPRIM_SCORE_NAMES,
)
from starVLA.model.modules.action_model.multi_trajectory.planner import (  # noqa: E402
    DDPDrivoRSuprimPlanner,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (  # noqa: E402
    STATIC_SAMPLE_INDICES,
)
from starVLA.training.trainer_utils.trainer_tools import (  # noqa: E402
    build_param_lr_groups,
)


TRAINING_STAGES = (
    "train_drivor",
    "train_suprim_static",
    "train_suprim_joint",
    "joint_finetune",
)


class FrozenSmokeActionHead(nn.Module):
    action_horizon = 8
    action_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(state_dim=3)
        self.sentinel = nn.Parameter(torch.ones(()))

    def predict_action(
        self, vl_embs_list: list[Tensor], state: Optional[Tensor]
    ) -> Tensor:
        reference = vl_embs_list[-1]
        return torch.randn(
            reference.shape[0],
            8,
            3,
            device=reference.device,
            dtype=reference.dtype,
        ).clamp_(-1.0, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=TRAINING_STAGES + ("inference",),
        required=True,
    )
    parser.add_argument("--scene-checkpoint", type=Path)
    parser.add_argument("--drivor-checkpoint", type=Path)
    parser.add_argument("--suprim-checkpoint", type=Path)
    parser.add_argument("--suprim-vocab", type=Path)
    parser.add_argument("--save-trained-components-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--qwen-hidden-dim", type=int, default=2048)
    return parser.parse_args()


def _require_file(path: Optional[Path], label: str) -> None:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} is required and must exist: {path}")


def _configuration(args: argparse.Namespace) -> MultiTrajectoryConfig:
    selector_stage = args.stage in {
        "train_suprim_static",
        "train_suprim_joint",
        "joint_finetune",
        "inference",
    }
    if selector_stage:
        _require_file(args.suprim_vocab, "--suprim-vocab")
    if args.stage in {"train_suprim_static", "train_suprim_joint"}:
        _require_file(args.scene_checkpoint, "--scene-checkpoint")
    if args.stage == "train_suprim_joint":
        _require_file(args.drivor_checkpoint, "--drivor-checkpoint")
    for path, label in (
        (args.scene_checkpoint, "scene checkpoint"),
        (args.drivor_checkpoint, "DrivoR checkpoint"),
        (args.suprim_checkpoint, "DriveSuprim checkpoint"),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return MultiTrajectoryConfig(
        enabled=True,
        num_dynamic_candidates=64,
        deterministic_seed=args.seed,
        scene_compressor=SceneCompressorConfig(
            checkpoint_path=(
                None if args.scene_checkpoint is None else str(args.scene_checkpoint)
            )
        ),
        drivor=DrivoRConfig(
            checkpoint_path=(
                None if args.drivor_checkpoint is None else str(args.drivor_checkpoint)
            )
        ),
        suprim=SuprimConfig(
            vocab_path=None if args.suprim_vocab is None else str(args.suprim_vocab),
            checkpoint_path=(
                None if args.suprim_checkpoint is None else str(args.suprim_checkpoint)
            ),
        ),
        training_stage=args.stage,
        strict_inference=False,
        diagnostics_enabled=args.stage == "inference",
    )


def _optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    cfg = OmegaConf.create(
        {"trainer": {"learning_rate": {"base": learning_rate}, "freeze_modules": ""}}
    )
    groups = build_param_lr_groups(model=model, cfg=cfg)
    if not groups:
        raise RuntimeError("existing optimizer discovery found no trainable parameters")
    return torch.optim.AdamW(groups, lr=learning_rate)


def _binary_targets(
    names: Iterable[str], shape: tuple[int, ...], device: torch.device
) -> Dict[str, Tensor]:
    values = torch.arange(shape[-1], device=device).remainder(2).float()
    values = values.view(*((1,) * (len(shape) - 1)), shape[-1]).expand(shape)
    return {name: values.clone() for name in names}


def _representative_snapshots(
    planner: DDPDrivoRSuprimPlanner,
) -> Dict[str, Tensor]:
    snapshots = {}
    for prefix in ("scene_compressor", "dynamic_scorer", "suprim_selector"):
        module = getattr(planner, prefix, None)
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                snapshots[f"{prefix}.{name}"] = parameter.detach().cpu().clone()
                break
    return snapshots


def _save_components(
    planner: DDPDrivoRSuprimPlanner, directory: Path
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for component in ("scene_compressor", "dynamic_scorer", "suprim_selector"):
        if not hasattr(planner, component):
            continue
        path = directory / f"{component}_one_step_smoke.pth"
        torch.save(
            planner.component_checkpoint_payload(component, inference_ready=False),
            path,
        )
        written.append(str(path.resolve()))
    return written


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.sequence_length <= 0 or args.qwen_hidden_dim <= 0:
        raise ValueError("batch-size, sequence-length, and qwen-hidden-dim must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    action_head = FrozenSmokeActionHead()
    planner = DDPDrivoRSuprimPlanner(
        action_head=action_head,
        config=_configuration(args),
        qwen_hidden_dim=args.qwen_hidden_dim,
    ).to(device)
    batch_size = args.batch_size
    vl_embs = [torch.zeros(batch_size, 8, 32, device=device)]
    full_hidden = torch.randn(
        batch_size,
        args.sequence_length,
        args.qwen_hidden_dim,
        device=device,
    )
    attention_mask = torch.ones(
        batch_size, args.sequence_length, device=device, dtype=torch.bool
    )
    if args.sequence_length > 1:
        attention_mask[:, -1] = False
    state = torch.zeros(batch_size, 1, 3, device=device)
    started = time.perf_counter()

    if args.stage == "inference":
        planner.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            trajectory = planner(vl_embs, state, full_hidden, attention_mask)
        if trajectory.shape != (batch_size, 8, 3):
            raise RuntimeError(f"unexpected inference output {tuple(trajectory.shape)}")
        diagnostics = planner.last_diagnostics
        result = {
            "stage": args.stage,
            "status": "passed",
            "output_shape": list(trajectory.shape),
            "batch_size": batch_size,
            "sequence_length": args.sequence_length,
            "dynamic_candidate_count": 64,
            "static_candidate_count": 8192,
            "dtype": "bfloat16" if device.type == "cuda" else "float32",
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "qformer_parameters": sum(
                p.numel() for p in planner.scene_compressor.parameters()
            ),
            "latencies": {
                "ddp_sampling": diagnostics.latency_ddp_sampling,
                "scene_compressor": diagnostics.latency_scene_compressor,
                "drivor_scorer": diagnostics.latency_drivor_scorer,
                "suprim_coarse": diagnostics.latency_suprim_coarse,
                "suprim_fine": diagnostics.latency_suprim_refinement,
                "planner_total": diagnostics.latency_total_inference,
            },
            "global_scene_tokens_bytes": diagnostics.global_scene_tokens_bytes,
            "dense_scene_memory_bytes": diagnostics.dense_scene_memory_bytes,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    planner.train()
    if action_head.training:
        raise RuntimeError("frozen DDP entered training mode")
    trainable = {
        name: parameter
        for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter
        for name, parameter in planner.named_parameters()
        if not parameter.requires_grad
    }
    snapshots = _representative_snapshots(planner)
    optimizer = _optimizer(planner, args.learning_rate)
    if args.stage == "train_drivor":
        targets = {
            "drivor_scores": _binary_targets(
                DRIVOR_SCORE_NAMES, (batch_size, 64), device
            )
        }
        candidate_count = 64
    else:
        candidate_count = 8192 + (
            16 if args.stage in {"train_suprim_joint", "joint_finetune"} else 0
        )
        target = planner.suprim_selector.static_vocab[
            0, list(STATIC_SAMPLE_INDICES)
        ][None].expand(batch_size, -1, -1).to(device)
        targets = {
            "coarse_scores": _binary_targets(
                SUPRIM_SCORE_NAMES, (batch_size, candidate_count), device
            ),
            "trajectory": target,
        }
        if args.stage == "joint_finetune":
            targets["drivor_scores"] = _binary_targets(
                DRIVOR_SCORE_NAMES, (batch_size, 64), device
            )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = planner.compute_training_loss(
            vl_embs_list=vl_embs,
            state=state,
            full_hidden_state=full_hidden,
            attention_mask=attention_mask,
            targets=targets,
        )
        loss = output["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite {args.stage} loss: {loss}")
    loss.backward()
    trainable_with_grad = [
        name for name, parameter in trainable.items() if parameter.grad is not None
    ]
    frozen_with_grad = [
        name for name, parameter in frozen.items() if parameter.grad is not None
    ]
    if not trainable_with_grad:
        raise RuntimeError("no trainable gradient was produced")
    if frozen_with_grad or action_head.sentinel.grad is not None:
        raise RuntimeError(f"gradient leaked to frozen parameters: {frozen_with_grad}")
    if not all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
        if parameter.grad is not None
    ):
        raise RuntimeError("training produced NaN/Inf gradients")
    optimizer.step()
    changed = [
        name
        for name, before in snapshots.items()
        if not torch.equal(
            before,
            dict(planner.named_parameters())[name].detach().cpu(),
        )
    ]
    if not changed:
        raise RuntimeError("optimizer changed no representative trainable parameter")
    written = (
        []
        if args.save_trained_components_dir is None
        else _save_components(planner, args.save_trained_components_dir)
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "passed",
                "loss": float(loss.detach().float().cpu()),
                "batch_size": batch_size,
                "sequence_length": args.sequence_length,
                "dynamic_candidate_count": 64,
                "static_candidate_count": 8192 if candidate_count > 64 else 0,
                "candidate_count": candidate_count,
                "dtype": "bfloat16" if device.type == "cuda" else "float32",
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "qformer_parameters": sum(
                    p.numel()
                    for p in getattr(planner, "scene_compressor", nn.Module()).parameters()
                ),
                "trainable_parameter_count": sum(p.numel() for p in trainable.values()),
                "trainable_tensors_with_grad": len(trainable_with_grad),
                "frozen_tensors_with_grad": 0,
                "changed_representative_parameters": changed,
                "saved_component_checkpoints": written,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
