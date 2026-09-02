"""Shared random trainable-state artifact for paired formal experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import torch
from torch import nn


SCHEMA_VERSION = 1


def classify_shared_trainable_parameter(name: str) -> str:
    """Map every formal trainable tensor to one declared logical module."""
    if name.startswith("backbone.planning_register_adapter."):
        return "planning_adapter"
    if name.startswith("backbone.") and any(
        marker in name
        for marker in (".q_lora_a.", ".q_lora_b.", ".v_lora_a.", ".v_lora_b.")
    ):
        return "vision_qv_lora"
    if name.startswith("future_register_predictor."):
        return "future_predictor"
    if name == "action_head.scene_embeds" or name.startswith("action_head.q_former."):
        return "semantic_qformer"
    if name.startswith(
        (
            "action_head.planning_norm.",
            "action_head.semantic_norm.",
            "action_head.semantic_cross_attention.",
            "action_head.output_norm.",
        )
    ) or name == "action_head.semantic_gate":
        return "semantic_fusion"
    if name.startswith(
        (
            "action_head.scorer_attention.",
            "action_head.pos_embed.",
            "action_head.scorer.",
        )
    ):
        return "scorer"
    if name.startswith(
        (
            "action_head.hist_encoding.",
            "action_head.init_feature.",
            "action_head.trajectory_decoder.",
            "action_head.traj_head.",
        )
    ):
        return "action_generator"
    raise RuntimeError(
        "Unclassified trainable parameter in formal shared initialization: "
        f"{name}"
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(tensor_sha256(state[name])))
    return digest.hexdigest()


def capture_shared_trainable_state(
    module: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    state: Dict[str, torch.Tensor] = {}
    grouped: MutableMapping[str, Dict[str, torch.Tensor]] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("ema_register_target."):
            raise RuntimeError("EMA teacher must not enter shared trainable initialization")
        group = classify_shared_trainable_parameter(name)
        value = parameter.detach().cpu().clone()
        state[name] = value
        grouped.setdefault(group, {})[name] = value
    if not state:
        raise RuntimeError("No trainable parameters found for shared initialization")
    group_metadata = {
        group: {
            "key_count": len(values),
            "parameter_count": sum(value.numel() for value in values.values()),
            "sha256": state_sha256(values),
        }
        for group, values in sorted(grouped.items())
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "trainable_state_key_count": len(state),
        "trainable_parameter_count": sum(value.numel() for value in state.values()),
        "trainable_state_sha256": state_sha256(state),
        "logical_modules": group_metadata,
    }
    return state, metadata


def save_shared_trainable_initialization(
    module: nn.Module,
    output: str,
    *,
    seed: int,
    architecture_config_sha256: str,
    source_git_commit: str,
) -> Dict[str, Any]:
    state, metadata = capture_shared_trainable_state(module)
    metadata.update(
        {
            "seed": int(seed),
            "architecture_config_sha256": architecture_config_sha256,
            "source_git_commit": source_git_commit,
        }
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {"metadata": metadata, "trainable_state_dict": state}, temporary
    )
    temporary.replace(destination)
    return metadata


def load_shared_trainable_initialization(
    module: nn.Module,
    artifact_path: str,
) -> Dict[str, Any]:
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Shared PlanReg initialization not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Shared initialization payload must be a mapping")
    metadata = dict(payload.get("metadata", {}))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported shared PlanReg initialization schema: "
            f"{metadata.get('schema_version')}"
        )
    state = payload.get("trainable_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Shared initialization lacks trainable_state_dict")

    expected = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and not name.startswith("ema_register_target.")
    }
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = {
        name: {"expected": list(expected[name].shape), "artifact": list(state[name].shape)}
        for name in sorted(set(expected) & set(state))
        if tuple(expected[name].shape) != tuple(state[name].shape)
    }
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Shared PlanReg initialization topology mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={shape_mismatch}"
        )
    calculated_sha = state_sha256(state)
    if calculated_sha != metadata.get("trainable_state_sha256"):
        raise RuntimeError(
            "Shared PlanReg initialization content hash mismatch: "
            f"expected={metadata.get('trainable_state_sha256')} actual={calculated_sha}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    restored, restored_metadata = capture_shared_trainable_state(module)
    if restored_metadata["trainable_state_sha256"] != calculated_sha:
        raise RuntimeError(
            "Shared initialization was not restored bitwise; this commonly indicates "
            "a dtype mismatch between the artifact and the constructed topology"
        )
    return metadata


__all__ = [
    "capture_shared_trainable_state",
    "classify_shared_trainable_parameter",
    "load_shared_trainable_initialization",
    "save_shared_trainable_initialization",
    "state_sha256",
    "tensor_sha256",
]
