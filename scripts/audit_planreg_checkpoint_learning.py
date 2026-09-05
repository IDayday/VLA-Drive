#!/usr/bin/env python3
"""Measure which PlanReg trainable tensors actually moved from shared init."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from navsim.agents.EpisodeDrive.shared_planreg_initialization import (
    classify_shared_trainable_parameter,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _accumulator() -> dict:
    return {
        "parameter_count": 0,
        "tensor_count": 0,
        "unchanged_parameter_count": 0,
        "unchanged_tensor_count": 0,
        "initial_l2_sq": 0.0,
        "checkpoint_l2_sq": 0.0,
        "delta_l2_sq": 0.0,
        "dot": 0.0,
        "max_abs_delta": 0.0,
    }


def _add(acc: dict, initial: torch.Tensor, current: torch.Tensor) -> dict:
    left = initial.detach().float().reshape(-1)
    right = current.detach().float().reshape(-1)
    delta = right - left
    unchanged = torch.equal(initial, current)
    count = left.numel()
    acc["parameter_count"] += count
    acc["tensor_count"] += 1
    if unchanged:
        acc["unchanged_parameter_count"] += count
        acc["unchanged_tensor_count"] += 1
    acc["initial_l2_sq"] += float(torch.dot(left, left))
    acc["checkpoint_l2_sq"] += float(torch.dot(right, right))
    acc["delta_l2_sq"] += float(torch.dot(delta, delta))
    acc["dot"] += float(torch.dot(left, right))
    if count:
        acc["max_abs_delta"] = max(acc["max_abs_delta"], float(delta.abs().max()))
    return {
        "parameter_count": count,
        "unchanged": unchanged,
        "initial_rms": float(torch.sqrt(torch.mean(left.square()))),
        "checkpoint_rms": float(torch.sqrt(torch.mean(right.square()))),
        "delta_rms": float(torch.sqrt(torch.mean(delta.square()))),
        "max_abs_delta": float(delta.abs().max()) if count else 0.0,
    }


def _finalize(acc: dict) -> dict:
    count = max(1, acc["parameter_count"])
    initial_norm = math.sqrt(acc["initial_l2_sq"])
    checkpoint_norm = math.sqrt(acc["checkpoint_l2_sq"])
    delta_norm = math.sqrt(acc["delta_l2_sq"])
    cosine = acc["dot"] / max(initial_norm * checkpoint_norm, 1e-30)
    return {
        "parameter_count": acc["parameter_count"],
        "tensor_count": acc["tensor_count"],
        "unchanged_parameter_fraction": acc["unchanged_parameter_count"] / count,
        "unchanged_tensor_count": acc["unchanged_tensor_count"],
        "initial_rms": math.sqrt(acc["initial_l2_sq"] / count),
        "checkpoint_rms": math.sqrt(acc["checkpoint_l2_sq"] / count),
        "delta_rms": math.sqrt(acc["delta_l2_sq"] / count),
        "relative_l2_delta": delta_norm / max(initial_norm, 1e-30),
        "initial_checkpoint_cosine": cosine,
        "max_abs_delta": acc["max_abs_delta"],
    }


def _effective_lora_norm(state: Mapping[str, torch.Tensor], stem: str) -> float:
    a = state[f"agent.{stem}_lora_a.weight"].detach().float()
    b = state[f"agent.{stem}_lora_b.weight"].detach().float()
    # ||BA||_F^2 = trace((B^T B)(A A^T)); avoids forming a 1024x1024 matrix.
    gram_b = b.T @ b
    gram_a = a @ a.T
    return float(torch.sqrt(torch.sum(gram_b * gram_a.T).clamp_min(0.0)))


def _special_diagnostics(state: Mapping[str, torch.Tensor]) -> dict:
    prefix = "agent."
    result: Dict[str, object] = {}
    gate = state.get(prefix + "action_head.semantic_gate")
    if gate is not None:
        result["semantic_gate_probability"] = float(torch.sigmoid(gate.float()).mean())
    tile_gate = state.get(prefix + "backbone.planning_register_adapter.tile_gate")
    if tile_gate is not None:
        result["tile_gate_tanh"] = float(torch.tanh(tile_gate.float()).mean())
    registers = state.get(prefix + "backbone.planning_register_adapter.planning_registers")
    if registers is not None:
        value = registers.float().squeeze(0)
        normalized = torch.nn.functional.normalize(value, dim=-1)
        cosine = normalized @ normalized.T
        mask = ~torch.eye(len(value), dtype=torch.bool)
        singular = torch.linalg.svdvals(value - value.mean(dim=0, keepdim=True))
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-30)
        result["planning_registers"] = {
            "rms": float(value.square().mean().sqrt()),
            "pairwise_cosine_mean": float(cosine[mask].mean()),
            "effective_rank": float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())),
        }
    residual_stem = prefix + "future_register_predictor.residual_output"
    result["future_predictor_residual_output"] = {
        suffix: float(state[residual_stem + suffix].float().norm())
        for suffix in (".weight", ".bias")
        if residual_stem + suffix in state
    }

    layers = []
    for layer in range(24):
        root = f"backbone.model.vision_model.encoder.layers.{layer}.attn.qkv."
        if prefix + root + "q_lora_a.weight" not in state:
            continue
        layers.append(
            {
                "layer": layer,
                "q_effective_delta_frobenius": _effective_lora_norm(state, root + "q"),
                "v_effective_delta_frobenius": _effective_lora_norm(state, root + "v"),
            }
        )
    result["vision_qv_lora_layers"] = layers
    if layers:
        result["vision_qv_lora_summary"] = {
            "adapted_layer_count": len(layers),
            "q_effective_delta_mean": float(np.mean([row["q_effective_delta_frobenius"] for row in layers])),
            "q_effective_delta_min": float(np.min([row["q_effective_delta_frobenius"] for row in layers])),
            "v_effective_delta_mean": float(np.mean([row["v_effective_delta_frobenius"] for row in layers])),
            "v_effective_delta_min": float(np.min([row["v_effective_delta_frobenius"] for row in layers])),
        }
    return result


def audit_checkpoint(initial_state: Mapping[str, torch.Tensor], path: Path) -> dict:
    payload = _load(path)
    state = payload.get("state_dict", payload)
    group_accumulators: Dict[str, dict] = {}
    head_accumulators: Dict[str, dict] = {}
    tensor_rows = []
    missing = []
    for name, initial in initial_state.items():
        checkpoint_name = "agent." + name
        current = state.get(checkpoint_name)
        if current is None:
            missing.append(checkpoint_name)
            continue
        if tuple(initial.shape) != tuple(current.shape):
            raise RuntimeError(
                f"Shape mismatch for {checkpoint_name}: {tuple(initial.shape)} vs {tuple(current.shape)}"
            )
        group = classify_shared_trainable_parameter(name)
        tensor_result = _add(group_accumulators.setdefault(group, _accumulator()), initial, current)
        tensor_result.update({"name": name, "logical_module": group})
        tensor_rows.append(tensor_result)
        if name.startswith("action_head.traj_head."):
            head = name.split(".")[2]
            _add(head_accumulators.setdefault(f"traj_head_{head}", _accumulator()), initial, current)
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} trainable tensors: {missing[:20]}")
    unexpected_trainable = []
    report = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "logical_modules": {name: _finalize(acc) for name, acc in sorted(group_accumulators.items())},
        "trajectory_heads": {name: _finalize(acc) for name, acc in sorted(head_accumulators.items())},
        "special_diagnostics": _special_diagnostics(state),
        "unchanged_tensors": sorted(row["name"] for row in tensor_rows if row["unchanged"]),
        "largest_tensor_delta_rms": sorted(tensor_rows, key=lambda row: row["delta_rms"], reverse=True)[:30],
        "unexpected_trainable": unexpected_trainable,
    }
    del payload
    return report


def _rename_pairwise_summary(summary: dict) -> dict:
    """Give the generic accumulator fields checkpoint-to-checkpoint semantics."""
    return {
        "parameter_count": summary["parameter_count"],
        "tensor_count": summary["tensor_count"],
        "unchanged_parameter_fraction": summary["unchanged_parameter_fraction"],
        "unchanged_tensor_count": summary["unchanged_tensor_count"],
        "source_rms": summary["initial_rms"],
        "target_rms": summary["checkpoint_rms"],
        "delta_rms": summary["delta_rms"],
        "relative_l2_delta_to_source": summary["relative_l2_delta"],
        "source_target_cosine": summary["initial_checkpoint_cosine"],
        "max_abs_delta": summary["max_abs_delta"],
    }


def compare_checkpoints(
    initial_state: Mapping[str, torch.Tensor],
    source_path: Path,
    target_path: Path,
) -> dict:
    """Measure additional learning between two ordered training checkpoints."""
    source_payload = _load(source_path)
    target_payload = _load(target_path)
    source_state = source_payload.get("state_dict", source_payload)
    target_state = target_payload.get("state_dict", target_payload)
    groups: Dict[str, dict] = {}
    heads: Dict[str, dict] = {}
    for name in initial_state:
        checkpoint_name = "agent." + name
        if checkpoint_name not in source_state or checkpoint_name not in target_state:
            raise RuntimeError(
                f"Pairwise checkpoint audit is missing {checkpoint_name}"
            )
        source = source_state[checkpoint_name]
        target = target_state[checkpoint_name]
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Pairwise shape mismatch for {checkpoint_name}: "
                f"{tuple(source.shape)} vs {tuple(target.shape)}"
            )
        group = classify_shared_trainable_parameter(name)
        _add(groups.setdefault(group, _accumulator()), source, target)
        if name.startswith("action_head.traj_head."):
            head = name.split(".")[2]
            _add(heads.setdefault(f"traj_head_{head}", _accumulator()), source, target)
    report = {
        "source": {
            "path": str(source_path.resolve()),
            "epoch": int(source_payload.get("epoch", -1)),
            "global_step": int(source_payload.get("global_step", -1)),
        },
        "target": {
            "path": str(target_path.resolve()),
            "epoch": int(target_payload.get("epoch", -1)),
            "global_step": int(target_payload.get("global_step", -1)),
        },
        "logical_modules": {
            name: _rename_pairwise_summary(_finalize(acc))
            for name, acc in sorted(groups.items())
        },
        "trajectory_heads": {
            name: _rename_pairwise_summary(_finalize(acc))
            for name, acc in sorted(heads.items())
        },
    }
    del source_payload, target_payload
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-init", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=/absolute/path.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial_payload = _load(args.shared_init)
    initial_state = initial_payload["trainable_state_dict"]
    expected_count = int(initial_payload["metadata"]["trainable_parameter_count"])
    actual_count = sum(tensor.numel() for tensor in initial_state.values())
    if expected_count != actual_count:
        raise RuntimeError(f"Shared-init parameter-count mismatch: {expected_count} != {actual_count}")
    checkpoints = {}
    ordered_specs: Sequence[Tuple[str, Path]] = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError("--checkpoint must be LABEL=PATH")
        label, path_text = spec.split("=", 1)
        path = Path(path_text)
        checkpoints[label] = audit_checkpoint(initial_state, path)
        ordered_specs.append((label, path))
    successive_checkpoint_deltas = {}
    for (source_label, source_path), (target_label, target_path) in zip(
        ordered_specs, ordered_specs[1:]
    ):
        successive_checkpoint_deltas[f"{source_label}_to_{target_label}"] = (
            compare_checkpoints(initial_state, source_path, target_path)
        )
    report = {
        "schema_version": 1,
        "shared_init": str(args.shared_init.resolve()),
        "shared_init_sha256": _sha256(args.shared_init),
        "shared_trainable_parameter_count": actual_count,
        "checkpoints": checkpoints,
        "successive_checkpoint_deltas": successive_checkpoint_deltas,
        "interpretation": {
            "unchanged_tensor": "bitwise equal to the shared random initialization",
            "effective_lora_delta": "Frobenius norm of the learned low-rank BA update, excluding the frozen base QKV",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        label: {
            "step": checkpoint["global_step"],
            "modules": {name: value["relative_l2_delta"] for name, value in checkpoint["logical_modules"].items()},
            "trajectory_heads_unchanged": {name: value["unchanged_parameter_fraction"] for name, value in checkpoint["trajectory_heads"].items()},
        }
        for label, checkpoint in checkpoints.items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
