#!/usr/bin/env python3
"""Audit whether the training EMA can numerically move in its stored dtype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


STUDENT_VISION = "agent.backbone.model.vision_model."
TEACHER_VISION = "agent.ema_register_target.vision_model."
STUDENT_ADAPTER = "agent.backbone.planning_register_adapter."
TEACHER_ADAPTER = "agent.ema_register_target.planning_register_adapter."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    return payload, payload["state_dict"]


def _pairs(state: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, str, str]]:
    for student_name in state:
        if student_name.startswith(STUDENT_VISION) and any(
            marker in student_name
            for marker in (".q_lora_a.", ".q_lora_b.", ".v_lora_a.", ".v_lora_b.")
        ):
            suffix = student_name[len(STUDENT_VISION) :]
            yield "vision_qv_lora", student_name, TEACHER_VISION + suffix
        elif student_name.startswith(STUDENT_ADAPTER):
            suffix = student_name[len(STUDENT_ADAPTER) :]
            if suffix == "planning_registers":
                group = "planning_register_tokens"
            elif suffix.startswith("tile_"):
                group = "tile_aggregator"
            else:
                group = "register_neck"
            yield group, student_name, TEACHER_ADAPTER + suffix


def _aggregate(state: Dict[str, torch.Tensor], momentum: float) -> dict:
    totals = {}
    for group, student_name, teacher_name in _pairs(state):
        if teacher_name not in state:
            raise KeyError(f"Missing EMA tensor matching {student_name}: {teacher_name}")
        student = state[student_name].detach().cpu()
        teacher = state[teacher_name].detach().cpu()
        if student.shape != teacher.shape or student.dtype != teacher.dtype:
            raise RuntimeError(f"EMA topology mismatch: {student_name}, {teacher_name}")
        row = totals.setdefault(
            group,
            {
                "elements": 0,
                "tensors": 0,
                "exact_equal_elements": 0,
                "squared_delta": 0.0,
                "squared_student": 0.0,
                "max_abs_delta": 0.0,
                "simulated_one_step_unchanged_elements": 0,
                "dtype": str(student.dtype),
            },
        )
        delta = student.float() - teacher.float()
        simulated = teacher.clone()
        simulated.mul_(momentum)
        simulated.add_(student, alpha=1.0 - momentum)
        row["elements"] += student.numel()
        row["tensors"] += 1
        row["exact_equal_elements"] += int((student == teacher).sum())
        row["squared_delta"] += float(delta.square().sum())
        row["squared_student"] += float(student.float().square().sum())
        row["max_abs_delta"] = max(
            row["max_abs_delta"], float(delta.abs().max())
        )
        row["simulated_one_step_unchanged_elements"] += int(
            (simulated == teacher).sum()
        )
    result = {}
    for group, row in totals.items():
        elements = row.pop("elements")
        squared_delta = row.pop("squared_delta")
        squared_student = row.pop("squared_student")
        equal = row.pop("exact_equal_elements")
        unchanged = row.pop("simulated_one_step_unchanged_elements")
        result[group] = {
            **row,
            "parameter_count": elements,
            "student_teacher_exact_equal_fraction": equal / elements,
            "student_teacher_delta_rms": (squared_delta / elements) ** 0.5,
            "student_teacher_relative_l2": (
                squared_delta / max(squared_student, 1e-30)
            )
            ** 0.5,
            "simulated_next_update_unchanged_fraction": unchanged / elements,
        }
    return result


def _teacher_change(first_state, final_state) -> dict:
    totals = {}
    first_pairs = list(_pairs(first_state))
    for group, _, teacher_name in first_pairs:
        first = first_state[teacher_name].detach().float().cpu()
        final = final_state[teacher_name].detach().float().cpu()
        row = totals.setdefault(
            group,
            {"elements": 0, "squared_change": 0.0, "exact_equal": 0},
        )
        row["elements"] += first.numel()
        row["squared_change"] += float((final - first).square().sum())
        row["exact_equal"] += int((final == first).sum())
    return {
        group: {
            "teacher_checkpoint_change_rms": (
                row["squared_change"] / row["elements"]
            )
            ** 0.5,
            "teacher_exactly_unchanged_element_fraction": (
                row["exact_equal"] / row["elements"]
            ),
        }
        for group, row in totals.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    loaded = []
    for path in args.checkpoints:
        payload, state = _load(path)
        momentum = float(state["agent._ema_current_momentum"])
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "epoch": int(payload.get("epoch", -1)),
                "global_step": int(payload.get("global_step", -1)),
                "ema_optimizer_step": int(state["agent._ema_optimizer_step"]),
                "ema_start_momentum": float(
                    state["agent._ema_actual_start_momentum"]
                ),
                "ema_end_momentum": float(
                    state["agent._ema_actual_end_momentum"]
                ),
                "ema_current_momentum": momentum,
                "groups": _aggregate(state, momentum),
            }
        )
        loaded.append((payload, state))
    first_state = loaded[0][1]
    final_state = loaded[-1][1]
    result = {
        "schema_version": 1,
        "checkpoints": rows,
        "teacher_change_first_to_last": _teacher_change(
            first_state, final_state
        ),
        "risk_definition": (
            "simulated_next_update_unchanged_fraction reproduces the in-place "
            "mul-then-add EMA update in the checkpoint tensor dtype. A high "
            "fraction means per-step EMA changes are quantized away."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
