"""Input-only cache contract for dynamic PlanReg-WM representation training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

from navsim.agents.EpisodeDrive.drivevla_backbone import system_message
from navsim.agents.EpisodeDrive.utils.prompt_contract import resolve_system_message, prompt_sha256
from navsim.agents.EpisodeDrive.layers.world_model.future_image_io import (
    decode_path_tensor,
    encode_path_tensor,
)
from navsim.agents.EpisodeDrive.utils.internvl_preprocess import (
    tile_metadata_from_image_size,
)
from navsim.agents.EpisodeDrive.utils.internvl_tokenize import (
    build_internvl_model_inputs,
)
from navsim.agents.EpisodeDrive.utils.utils import build_drivevla_questions


INPUT_ONLY_CACHE_SCHEMA_VERSION = 1
INPUT_ONLY_CACHE_NAME = "planreg_input_only"
DYNAMIC_FEATURE_CACHE_KEYS = frozenset(
    {
        "last_hidden_state",
        "patch_features",
        "semantic_tokens",
        "planning_registers",
        "future_registers",
        "ema_registers",
    }
)


def reject_dynamic_feature_cache(
    values: Mapping[str, Any],
    *,
    enabled: bool,
    source: str,
) -> None:
    if not enabled:
        return
    forbidden = sorted(DYNAMIC_FEATURE_CACHE_KEYS.intersection(values))
    if forbidden:
        raise RuntimeError(
            f"Dynamic PlanReg feature cache rejected from {source}: {forbidden}. "
            "These representations change as vision Q/V LoRA, planning registers, "
            "Q-Former, or the EMA teacher update; reading a static cache would "
            "change the scientific method. Use cache_policy.mode=input_only."
        )


def validate_input_only_cache_policy(policy: Any) -> None:
    if policy is None:
        return
    mode = str(getattr(policy, "mode", ""))
    if mode != "input_only":
        raise ValueError(f"Formal PlanReg cache_policy.mode must be input_only, got {mode!r}")
    prohibited_flags = {
        "cache_vlm_hidden_state": bool(
            getattr(policy, "cache_vlm_hidden_state", False)
        ),
        "cache_patch_features": bool(getattr(policy, "cache_patch_features", False)),
        "cache_semantic_tokens": bool(
            getattr(policy, "cache_semantic_tokens", False)
        ),
        "cache_planning_registers": bool(
            getattr(policy, "cache_planning_registers", False)
        ),
        "cache_future_ema_registers": bool(
            getattr(policy, "cache_future_ema_registers", False)
        ),
    }
    enabled = sorted(name for name, value in prohibited_flags.items() if value)
    if enabled:
        raise ValueError(
            "Formal input-only cache enables forbidden dynamic representations: "
            f"{enabled}"
        )


def _decode_legacy_current_path(path_tensor: torch.Tensor) -> str:
    if path_tensor.ndim != 1:
        path_tensor = path_tensor.squeeze(0)
    return "".join(chr(int(value)) for value in path_tensor.tolist() if int(value))


def _image_geometry(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image:
        width, height = image.size
    return (
        torch.tensor([width, height], dtype=torch.long),
        tile_metadata_from_image_size(width, height),
    )


def build_input_only_cache_record(
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    tokenizer,
    max_path_bytes: int = 1024,
) -> Dict[str, Any]:
    """Consolidate raw feature/target caches without storing model outputs."""
    reject_dynamic_feature_cache(
        features, enabled=True, source="input-only record construction"
    )
    reject_dynamic_feature_cache(
        targets, enabled=True, source="input-only target construction"
    )
    required_features = (
        "history_trajectory",
        "high_command_one_hot",
        "status_feature",
        "image_path_tensor",
    )
    required_targets = (
        "trajectory",
        "trajectory_long",
        "future_image_paths",
        "future_image_path_lengths",
        "future_valid_mask",
    )
    missing = [name for name in required_features if name not in features]
    missing += [name for name in required_targets if name not in targets]
    if missing:
        raise KeyError(f"Cannot build formal input-only cache; missing fields: {missing}")

    current_path = _decode_legacy_current_path(features["image_path_tensor"])
    current_path_tensor, current_path_length = encode_path_tensor(
        current_path, max_bytes=max_path_bytes
    )
    current_size, current_metadata = _image_geometry(current_path)
    num_current_tiles = int(current_metadata.shape[0])
    question = build_drivevla_questions(
        features["history_trajectory"], features["high_command_one_hot"]
    )[0]
    model_inputs = build_internvl_model_inputs(
        tokenizer,
        [question],
        [num_current_tiles],
        resolve_system_message(system_message, os.getenv("PLANREG_PROMPT_VERSION", "legacy")),
    )

    future_paths = torch.as_tensor(targets["future_image_paths"]).clone()
    future_lengths = torch.as_tensor(targets["future_image_path_lengths"]).clone()
    future_valid = torch.as_tensor(targets["future_valid_mask"]).bool().clone()
    if future_paths.shape != (3, max_path_bytes):
        raise ValueError(
            f"future_image_paths must be [3,{max_path_bytes}], got {tuple(future_paths.shape)}"
        )
    future_sizes = torch.zeros(3, 2, dtype=torch.long)
    future_tile_metadata = []
    future_num_patches = torch.zeros(3, dtype=torch.long)
    for index in range(3):
        if bool(future_valid[index]):
            path = decode_path_tensor(future_paths[index], future_lengths[index])
            size, metadata = _image_geometry(path)
        else:
            size, metadata = current_size.clone(), current_metadata.clone()
        future_sizes[index] = size
        future_tile_metadata.append(metadata)
        future_num_patches[index] = metadata.shape[0]

    cached_features = {
        "history_trajectory": torch.as_tensor(features["history_trajectory"]).clone(),
        "high_command_one_hot": torch.as_tensor(
            features["high_command_one_hot"]
        ).clone(),
        "status_feature": torch.as_tensor(features["status_feature"]).clone(),
        "image_path_tensor": current_path_tensor,
        "image_path_length": current_path_length,
        "image_original_size": current_size,
        "tile_metadata_cached": current_metadata,
        "num_patches_cached": torch.tensor(num_current_tiles, dtype=torch.long),
        "input_ids": model_inputs["input_ids"].squeeze(0).cpu(),
        "attention_mask": model_inputs["attention_mask"].squeeze(0).cpu(),
        "prompt_contract_hash": torch.tensor(list(bytes.fromhex(prompt_sha256(
            resolve_system_message(system_message, os.getenv("PLANREG_PROMPT_VERSION", "legacy"))
        ))), dtype=torch.uint8),
        "future_image_original_sizes": future_sizes,
        "future_tile_metadata_cached": future_tile_metadata,
        "future_num_patches_cached": future_num_patches,
    }
    cached_targets = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in targets.items()
    }
    return {
        "schema_version": INPUT_ONLY_CACHE_SCHEMA_VERSION,
        "features": cached_features,
        "targets": cached_targets,
    }


def validate_input_only_cache_record(record: Mapping[str, Any]) -> None:
    if int(record.get("schema_version", -1)) != INPUT_ONLY_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            "Stale PlanReg input-only cache schema; rebuild the cache rather than "
            "reusing dynamic feature artifacts"
        )
    features = record.get("features")
    targets = record.get("targets")
    if not isinstance(features, Mapping) or not isinstance(targets, Mapping):
        raise RuntimeError("Input-only cache record must contain feature/target mappings")
    reject_dynamic_feature_cache(features, enabled=True, source="input-only cache")
    reject_dynamic_feature_cache(targets, enabled=True, source="input-only cache")


def validate_input_only_manifest(cache_root: Path) -> Dict[str, Any]:
    path = Path(cache_root) / "planreg_input_only_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Formal training requires planreg_input_only_manifest.json; static VLM "
            "feature caches are not accepted"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != INPUT_ONLY_CACHE_SCHEMA_VERSION:
        raise RuntimeError("Input-only cache manifest schema is stale")
    if manifest.get("cache_mode") != "input_only":
        raise RuntimeError("Input cache manifest does not declare cache_mode=input_only")
    forbidden = set(manifest.get("cached_fields", ())).intersection(
        DYNAMIC_FEATURE_CACHE_KEYS
    )
    if forbidden:
        raise RuntimeError(
            f"Input-only cache manifest advertises forbidden fields: {sorted(forbidden)}"
        )
    return manifest


__all__ = [
    "DYNAMIC_FEATURE_CACHE_KEYS",
    "INPUT_ONLY_CACHE_NAME",
    "INPUT_ONLY_CACHE_SCHEMA_VERSION",
    "build_input_only_cache_record",
    "reject_dynamic_feature_cache",
    "validate_input_only_cache_policy",
    "validate_input_only_cache_record",
    "validate_input_only_manifest",
]
