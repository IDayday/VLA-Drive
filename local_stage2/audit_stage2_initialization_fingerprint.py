#!/usr/bin/env python3
"""Recover the released Stage-2 action-head initialization seed.

With the released ``prev_weight=0`` loss, only the final trajectory head is
connected to ``trajectory_loss``.  Heads 0--3 are also absent from the scorer
path, so their parameters receive no gradient and remain an exact fingerprint
of ActionDecoder initialization in a completed checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "navsim/planning/script/config/common/agent/episode_drive.yaml"
)
DEFAULT_PUBLIC_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-modelscope/"
    "best-epoch_26-step_174312.server_merged.ckpt"
)
DEFAULT_LOCAL_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-stage2/runs/training/"
    "stage2_full_seed0_pipeline_v8_restart/lightning_logs/version_0/"
    "checkpoints/best-epoch=25-step=167856.ckpt"
)
DEFAULT_LOCAL_LAST_CHECKPOINT = Path(
    "/mnt/project/DriveVLA-M0-stage2/runs/training/"
    "stage2_full_seed0_pipeline_v8_restart/lightning_logs/version_0/"
    "checkpoints/last.ckpt"
)
FINGERPRINT_HEADS = (0, 1, 2, 3)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must use nonempty NAME=PATH")
    return name, Path(raw_path)


def _seed_action_decoder(seed: int, config_path: Path) -> dict[str, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = OmegaConf.load(config_path).action_head_config
    model = ActionDecoder(config)
    return model.state_dict()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint state at {path}: {type(state)}")
    return state


def _fingerprint_keys(state: dict[str, torch.Tensor]) -> list[str]:
    prefixes = tuple(
        f"agent.action_head.traj_head.{head_index}."
        for head_index in FINGERPRINT_HEADS
    )
    keys = sorted(key for key in state if key.startswith(prefixes))
    if not keys:
        raise RuntimeError("Checkpoint contains no trajectory-head fingerprint tensors")
    return keys


def _tensor_digest(state: dict[str, torch.Tensor], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _compare(
    checkpoint_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    keys: list[str],
) -> dict:
    square_error = 0.0
    element_count = 0
    max_abs = 0.0
    exact_tensor_count = 0
    per_head = {}
    for head_index in FINGERPRINT_HEADS:
        head_prefix = f"agent.action_head.traj_head.{head_index}."
        head_keys = [key for key in keys if key.startswith(head_prefix)]
        head_square_error = 0.0
        head_element_count = 0
        head_max_abs = 0.0
        head_exact_count = 0
        for checkpoint_key in head_keys:
            initial_key = checkpoint_key.removeprefix("agent.action_head.")
            checkpoint_tensor = checkpoint_state[checkpoint_key]
            initial_tensor = initial_state[initial_key]
            exact = torch.equal(
                checkpoint_tensor,
                initial_tensor.to(dtype=checkpoint_tensor.dtype),
            )
            head_exact_count += int(exact)
            difference = checkpoint_tensor.float() - initial_tensor.float()
            head_square_error += difference.square().sum().item()
            head_element_count += difference.numel()
            head_max_abs = max(head_max_abs, difference.abs().max().item())
        square_error += head_square_error
        element_count += head_element_count
        max_abs = max(max_abs, head_max_abs)
        exact_tensor_count += head_exact_count
        per_head[str(head_index)] = {
            "tensor_count": len(head_keys),
            "element_count": head_element_count,
            "all_tensors_exact": head_exact_count == len(head_keys),
            "rms_difference": (head_square_error / head_element_count) ** 0.5,
            "max_abs_difference": head_max_abs,
        }
    return {
        "tensor_count": len(keys),
        "element_count": element_count,
        "exact_tensor_count": exact_tensor_count,
        "all_tensors_exact": exact_tensor_count == len(keys),
        "rms_difference": (square_error / element_count) ** 0.5,
        "max_abs_difference": max_abs,
        "per_head": per_head,
    }


def _displacement_from_initialization(
    checkpoint_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    fingerprint_keys: list[str],
) -> dict:
    """Summarize how far all action-head tensors moved from initialization."""

    fingerprint_key_set = set(fingerprint_keys)
    groups: dict[str, dict[str, float | int]] = {}
    missing_keys = []
    for initial_key, initial_tensor in initial_state.items():
        checkpoint_key = f"agent.action_head.{initial_key}"
        if checkpoint_key not in checkpoint_state:
            missing_keys.append(checkpoint_key)
            continue
        checkpoint_tensor = checkpoint_state[checkpoint_key]
        if not (
            torch.is_floating_point(initial_tensor)
            and torch.is_floating_point(checkpoint_tensor)
        ):
            continue
        group_name = initial_key.split(".", 1)[0]
        group = groups.setdefault(
            group_name,
            {
                "element_count": 0,
                "square_difference": 0.0,
                "square_initial": 0.0,
                "max_abs_difference": 0.0,
                "effective_element_count": 0,
                "effective_square_difference": 0.0,
                "effective_square_initial": 0.0,
            },
        )
        difference = checkpoint_tensor.float() - initial_tensor.float()
        element_count = difference.numel()
        square_difference = difference.square().sum().item()
        square_initial = initial_tensor.float().square().sum().item()
        group["element_count"] += element_count
        group["square_difference"] += square_difference
        group["square_initial"] += square_initial
        group["max_abs_difference"] = max(
            float(group["max_abs_difference"]),
            difference.abs().max().item(),
        )
        if checkpoint_key not in fingerprint_key_set:
            group["effective_element_count"] += element_count
            group["effective_square_difference"] += square_difference
            group["effective_square_initial"] += square_initial

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing_keys)} action-head tensors: "
            f"{missing_keys[:5]}"
        )

    def finalize(group: dict[str, float | int]) -> dict:
        count = int(group["element_count"])
        effective_count = int(group["effective_element_count"])
        square_difference = float(group["square_difference"])
        square_initial = float(group["square_initial"])
        effective_square_difference = float(group["effective_square_difference"])
        effective_square_initial = float(group["effective_square_initial"])
        return {
            "element_count": count,
            "rms_difference": (square_difference / count) ** 0.5,
            "initial_rms": (square_initial / count) ** 0.5,
            "relative_l2_difference": (
                (square_difference / square_initial) ** 0.5
                if square_initial
                else None
            ),
            "max_abs_difference": float(group["max_abs_difference"]),
            "effective_element_count": effective_count,
            "effective_rms_difference": (
                (effective_square_difference / effective_count) ** 0.5
                if effective_count
                else 0.0
            ),
            "effective_relative_l2_difference": (
                (effective_square_difference / effective_square_initial) ** 0.5
                if effective_square_initial
                else None
            ),
        }

    total = {
        key: sum(float(group[key]) for group in groups.values())
        for key in (
            "element_count",
            "square_difference",
            "square_initial",
            "effective_element_count",
            "effective_square_difference",
            "effective_square_initial",
        )
    }
    total["max_abs_difference"] = max(
        float(group["max_abs_difference"]) for group in groups.values()
    )
    return {
        "all_action_head": finalize(total),
        "by_module": {
            name: finalize(group) for name, group in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        help="Checkpoint as NAME=PATH; may be repeated.",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoints = args.checkpoint or [
        ("released", DEFAULT_PUBLIC_CHECKPOINT),
        ("local_seed0_best", DEFAULT_LOCAL_CHECKPOINT),
        ("local_seed0_last", DEFAULT_LOCAL_LAST_CHECKPOINT),
    ]
    seeds = args.seed or [0, 1, 2, 3]
    initial_states = {
        seed: _seed_action_decoder(seed, args.config) for seed in seeds
    }

    report = {
        "method": {
            "fingerprint_heads": list(FINGERPRINT_HEADS),
            "reason": (
                "With prev_weight=0, heads 0-3 do not contribute to the final "
                "trajectory loss or scorer and retain their initialization."
            ),
            "config": str(args.config),
            "candidate_seeds": seeds,
        },
        "checkpoints": {},
    }
    for name, path in checkpoints:
        state = _checkpoint_state(path)
        keys = _fingerprint_keys(state)
        comparisons = {
            str(seed): _compare(state, initial_states[seed], keys)
            for seed in seeds
        }
        matching_seeds = [
            seed
            for seed in seeds
            if comparisons[str(seed)]["all_tensors_exact"]
        ]
        report["checkpoints"][name] = {
            "path": str(path),
            "fingerprint_sha256": _tensor_digest(state, keys),
            "fingerprint_tensor_count": len(keys),
            "fingerprint_element_count": sum(state[key].numel() for key in keys),
            "matching_seeds": matching_seeds,
            "comparisons": comparisons,
        }
        if len(matching_seeds) == 1:
            report["checkpoints"][name]["displacement_from_initialization"] = (
                _displacement_from_initialization(
                    state,
                    initial_states[matching_seeds[0]],
                    keys,
                )
            )

    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
