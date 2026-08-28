#!/usr/bin/env python3
"""Audit a No-VQA dense checkpoint against its raw InternVL initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


PROBES = {
    "vision": "vision_model.embeddings.patch_embedding.weight",
    "projector": "mlp1.3.weight",
    "language": "language_model.model.layers.0.self_attn.q_proj.weight",
    "lm_head": "language_model.lm_head.weight",
}

EXPECTED_GROUPS = {
    "action_head_decay": (1e-4, 1e-4),
    "action_head_no_decay": (1e-4, 0.0),
    "vlm_vision_decay": (1e-5, 0.05),
    "vlm_vision_no_decay": (1e-5, 0.0),
    "vlm_projector_decay": (2e-5, 0.05),
    "vlm_projector_no_decay": (2e-5, 0.0),
    "vlm_language_decay": (1e-5, 0.05),
    "vlm_language_no_decay": (1e-5, 0.0),
}


def checkpoint_key(state: dict[str, torch.Tensor], suffix: str) -> str:
    matches = [
        key
        for key in state
        if key.startswith("agent.backbone.") and key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one checkpoint tensor ending in {suffix!r}, found {matches}"
        )
    return matches[0]


def compare_probe(
    state: dict[str, torch.Tensor],
    raw_file,
    name: str,
    raw_key: str,
) -> dict[str, object]:
    saved_key = checkpoint_key(state, raw_key)
    saved = state[saved_key]
    raw = raw_file.get_tensor(raw_key)
    if saved.shape != raw.shape:
        raise RuntimeError(
            f"{name} shape mismatch: checkpoint={tuple(saved.shape)}, "
            f"raw={tuple(raw.shape)}"
        )
    difference = saved != raw
    changed = int(difference.count_nonzero().item())
    maximum_error = float((saved.float() - raw.float()).abs().max().item())
    return {
        "checkpoint_key": saved_key,
        "shape": tuple(saved.shape),
        "changed_values": changed,
        "total_values": saved.numel(),
        "max_abs_delta": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-epoch", type=int)
    args = parser.parse_args()

    raw_path = args.raw_model
    if raw_path.is_dir():
        raw_path /= "model.safetensors"
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no state_dict")

    global_step = int(payload.get("global_step", -1))
    epoch = int(payload.get("epoch", -1))
    if args.expected_step is not None and global_step != args.expected_step:
        raise RuntimeError(
            f"global_step={global_step}; expected {args.expected_step}"
        )
    if args.expected_epoch is not None and epoch != args.expected_epoch:
        raise RuntimeError(f"epoch={epoch}; expected {args.expected_epoch}")

    backbone_keys = [key for key in state if key.startswith("agent.backbone.")]
    action_keys = [key for key in state if key.startswith("agent.action_head.")]
    if len(backbone_keys) < 685:
        raise RuntimeError(
            f"Dense VLM state is incomplete: only {len(backbone_keys)} tensors"
        )
    if not action_keys:
        raise RuntimeError("Checkpoint has no action-head tensors")

    with safe_open(raw_path, framework="pt", device="cpu") as raw_file:
        probe_results = {
            name: compare_probe(state, raw_file, name, raw_key)
            for name, raw_key in PROBES.items()
            if name != "lm_head"
        }

        lm_head_key = checkpoint_key(state, PROBES["lm_head"])
        saved_lm_head = state[lm_head_key]
        raw_lm_head = raw_file.get_tensor(PROBES["lm_head"])
        if saved_lm_head.shape[1:] != raw_lm_head.shape[1:]:
            raise RuntimeError(
                "lm_head hidden dimension differs: "
                f"checkpoint={tuple(saved_lm_head.shape)}, "
                f"raw={tuple(raw_lm_head.shape)}"
            )
        if saved_lm_head.shape[0] < raw_lm_head.shape[0]:
            raise RuntimeError(
                "Checkpoint lm_head has fewer vocabulary rows than the raw model"
            )
        frozen_lm_head_equal = torch.equal(
            saved_lm_head[: raw_lm_head.shape[0]], raw_lm_head
        )

    unchanged_trainable = [
        name
        for name, result in probe_results.items()
        if result["changed_values"] == 0
    ]
    if unchanged_trainable:
        raise RuntimeError(
            "Trainable VLM probes did not update: " + ", ".join(unchanged_trainable)
        )
    if not frozen_lm_head_equal:
        raise RuntimeError("Raw lm_head rows changed even though lm_head is frozen")

    optimizer_states = payload.get("optimizer_states", [])
    if len(optimizer_states) != 1:
        raise RuntimeError(
            f"Expected one optimizer state, found {len(optimizer_states)}"
        )
    param_groups = optimizer_states[0].get("param_groups", [])
    groups_by_name = {group.get("name"): group for group in param_groups}
    if set(groups_by_name) != set(EXPECTED_GROUPS):
        raise RuntimeError(
            "Optimizer groups differ: "
            f"actual={sorted(groups_by_name)}, expected={sorted(EXPECTED_GROUPS)}"
        )
    for name, (expected_initial_lr, expected_weight_decay) in EXPECTED_GROUPS.items():
        group = groups_by_name[name]
        initial_lr = float(group.get("initial_lr", -1.0))
        weight_decay = float(group.get("weight_decay", -1.0))
        if initial_lr != expected_initial_lr:
            raise RuntimeError(
                f"{name} initial_lr={initial_lr}; expected {expected_initial_lr}"
            )
        if weight_decay != expected_weight_decay:
            raise RuntimeError(
                f"{name} weight_decay={weight_decay}; expected {expected_weight_decay}"
            )

    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"global_step={global_step}")
    print(f"epoch={epoch}")
    print(f"backbone_tensors={len(backbone_keys)}")
    print(f"action_head_tensors={len(action_keys)}")
    for name, result in probe_results.items():
        print(
            f"{name}: changed={result['changed_values']}/{result['total_values']} "
            f"max_abs_delta={result['max_abs_delta']}"
        )
    print(
        f"lm_head_raw_rows_equal={frozen_lm_head_equal} "
        f"raw_rows={raw_lm_head.shape[0]} saved_rows={saved_lm_head.shape[0]}"
    )
    print(f"optimizer_groups={','.join(sorted(groups_by_name))}")
    print("No-VQA full checkpoint audit: PASS")


if __name__ == "__main__":
    main()
