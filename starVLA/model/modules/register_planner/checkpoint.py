"""Strict component checkpoints for staged Register64 training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn


REGISTER_CHECKPOINT_SCHEMA_VERSION = 1
GENERATOR_METADATA_FIELDS = (
    "schema_version",
    "stage",
    "qwen_base_model",
    "qwen_trainable_manifest_hash",
    "proposal_num",
    "num_poses",
    "state_dim",
    "scene_queries",
    "scene_dim",
    "decoder_layers",
    "decoder_heads",
    "proposal_head_style",
    "stage_loss_mode",
    "proposal_head_count",
    "commit",
    "config_hash",
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_container"):
        value = value.to_container(resolve=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_config_hash(config: Any) -> str:
    payload = json.dumps(
        _jsonable(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen_trainable_state(
    module: nn.Module, names: Optional[set[str]] = None
) -> dict[str, Tensor]:
    selected = (
        {name for name, parameter in module.named_parameters() if parameter.requires_grad}
        if names is None
        else set(names)
    )
    available = dict(module.named_parameters())
    missing = selected.difference(available)
    if missing:
        raise KeyError(f"unknown Qwen checkpoint parameters: {sorted(missing)}")
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if name in selected
    }


def trainable_manifest_hash(module: nn.Module) -> str:
    return parameter_manifest_hash(
        module,
        {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        },
    )


def parameter_manifest_hash(module: nn.Module, names: set[str]) -> str:
    names = set(names)
    available = dict(module.named_parameters())
    missing = names.difference(available)
    if missing:
        raise KeyError(f"parameter manifest contains unknown names: {sorted(missing)}")
    manifest = [
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in module.named_parameters()
        if name in names
    ]
    return stable_config_hash(manifest)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def save_register_generator_checkpoint(
    path: os.PathLike[str] | str,
    *,
    qwen_vl_interface: nn.Module,
    action_input_model: nn.Module,
    scene_encoder: nn.Module,
    register_generator: nn.Module,
    metadata: Mapping[str, Any],
    full_model_state_dict: Optional[Mapping[str, Tensor]] = None,
    qwen_state_names: Optional[set[str]] = None,
) -> Path:
    metadata = dict(metadata)
    metadata.setdefault("schema_version", REGISTER_CHECKPOINT_SCHEMA_VERSION)
    metadata.setdefault("stage", "register_generator")
    actual_qwen_manifest = (
        trainable_manifest_hash(qwen_vl_interface)
        if qwen_state_names is None
        else parameter_manifest_hash(qwen_vl_interface, qwen_state_names)
    )
    metadata.setdefault("qwen_trainable_manifest_hash", actual_qwen_manifest)
    missing = set(GENERATOR_METADATA_FIELDS).difference(metadata)
    if missing:
        raise KeyError(f"generator checkpoint metadata is missing {sorted(missing)}")
    if metadata["schema_version"] != REGISTER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported generator checkpoint schema version")
    if metadata["stage"] != "register_generator":
        raise ValueError("generator checkpoint stage must be 'register_generator'")
    if metadata["qwen_trainable_manifest_hash"] != actual_qwen_manifest:
        raise ValueError("generator checkpoint Qwen trainable manifest is incorrect")
    actual_architecture = {
        "proposal_num": int(register_generator.proposal_num),
        "num_poses": int(register_generator.num_poses),
        "state_dim": int(register_generator.state_dim),
        "scene_queries": int(scene_encoder.num_queries),
        "scene_dim": int(scene_encoder.output_dim),
        "decoder_layers": int(register_generator.num_layers),
        "decoder_heads": int(register_generator.num_heads),
        "proposal_head_style": str(register_generator.proposal_head_style),
        "stage_loss_mode": str(register_generator.stage_loss_mode),
        "proposal_head_count": int(register_generator.proposal_head_count),
    }
    for name, value in actual_architecture.items():
        if metadata[name] != value:
            raise ValueError(
                f"generator checkpoint metadata {name}={metadata[name]!r} "
                f"does not match module value {value!r}"
            )
    if full_model_state_dict is None:
        qwen_state = qwen_trainable_state(qwen_vl_interface, qwen_state_names)
        action_state = {
            key: value.detach().cpu()
            for key, value in action_input_model.state_dict().items()
        }
        scene_state = {
            key: value.detach().cpu()
            for key, value in scene_encoder.state_dict().items()
        }
        generator_state = {
            key: value.detach().cpu()
            for key, value in register_generator.state_dict().items()
        }
    else:
        full_model_state_dict = dict(full_model_state_dict)

        def component(prefix: str, expected_keys) -> dict[str, Tensor]:
            expected_keys = set(expected_keys)
            result = {
                key[len(prefix) :]: value.detach().cpu()
                for key, value in full_model_state_dict.items()
                if key.startswith(prefix) and key[len(prefix) :] in expected_keys
            }
            if set(result) != expected_keys:
                raise RuntimeError(
                    f"gathered component state mismatch for {prefix.rstrip('.')}"
                )
            return result

        trainable_names = (
            {
                name
                for name, parameter in qwen_vl_interface.named_parameters()
                if parameter.requires_grad
            }
            if qwen_state_names is None
            else set(qwen_state_names)
        )
        qwen_state = component("qwen_vl_interface.", trainable_names)
        action_state = component(
            "action_input_model.", action_input_model.state_dict().keys()
        )
        scene_state = component("scene_encoder.", scene_encoder.state_dict().keys())
        generator_state = component(
            "register_generator.", register_generator.state_dict().keys()
        )
    payload = {
        "metadata": metadata,
        "state_dict": {
            "qwen_trainable": qwen_state,
            "action_input_model": action_state,
            "scene_encoder": scene_state,
            "register_generator": generator_state,
        },
    }
    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return destination


def _validate_metadata(
    metadata: Mapping[str, Any], expected: Optional[Mapping[str, Any]]
) -> None:
    missing = set(GENERATOR_METADATA_FIELDS).difference(metadata)
    if missing:
        raise RuntimeError(f"checkpoint metadata is missing {sorted(missing)}")
    if metadata["schema_version"] != REGISTER_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version mismatch")
    if metadata["stage"] != "register_generator":
        raise RuntimeError("checkpoint is not a register-generator component")
    for name, expected_value in (expected or {}).items():
        if name not in metadata:
            raise RuntimeError(f"checkpoint metadata has no field {name!r}")
        if metadata[name] != expected_value:
            raise RuntimeError(
                f"checkpoint metadata mismatch for {name}: "
                f"expected {expected_value!r}, found {metadata[name]!r}"
            )


def _load_trainable_qwen(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise RuntimeError(
            "Qwen trainable parameter manifest mismatch: "
            f"missing={sorted(expected.difference(state))} "
            f"unexpected={sorted(set(state).difference(expected))}"
        )
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            target = parameters[name]
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(f"Qwen checkpoint shape mismatch for {name}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def load_register_generator_checkpoint(
    path: os.PathLike[str] | str,
    *,
    qwen_vl_interface: nn.Module,
    action_input_model: nn.Module,
    scene_encoder: nn.Module,
    register_generator: nn.Module,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
        raise RuntimeError("invalid register-generator checkpoint envelope")
    metadata = dict(payload["metadata"])
    _validate_metadata(metadata, expected_metadata)
    expected_manifest = trainable_manifest_hash(qwen_vl_interface)
    if metadata["qwen_trainable_manifest_hash"] != expected_manifest:
        raise RuntimeError("Qwen trainable manifest hash mismatch")
    actual_architecture = {
        "proposal_num": int(register_generator.proposal_num),
        "num_poses": int(register_generator.num_poses),
        "state_dim": int(register_generator.state_dim),
        "scene_queries": int(scene_encoder.num_queries),
        "scene_dim": int(scene_encoder.output_dim),
        "decoder_layers": int(register_generator.num_layers),
        "decoder_heads": int(register_generator.num_heads),
        "proposal_head_style": str(register_generator.proposal_head_style),
        "stage_loss_mode": str(register_generator.stage_loss_mode),
        "proposal_head_count": int(register_generator.proposal_head_count),
    }
    for name, value in actual_architecture.items():
        if metadata[name] != value:
            raise RuntimeError(
                f"generator checkpoint metadata {name} does not match loaded modules"
            )
    state = payload["state_dict"]
    expected_components = {
        "qwen_trainable",
        "action_input_model",
        "scene_encoder",
        "register_generator",
    }
    if set(state) != expected_components:
        raise RuntimeError("generator checkpoint component set mismatch")
    _load_trainable_qwen(qwen_vl_interface, state["qwen_trainable"])
    action_input_model.load_state_dict(state["action_input_model"], strict=True)
    scene_encoder.load_state_dict(state["scene_encoder"], strict=True)
    register_generator.load_state_dict(state["register_generator"], strict=True)
    return metadata


def save_stage_component_checkpoint(
    path: os.PathLike[str] | str,
    *,
    stage: str,
    module: nn.Module,
    metadata: Mapping[str, Any],
    state_dict: Optional[Mapping[str, Tensor]] = None,
) -> Path:
    envelope = dict(metadata)
    envelope.setdefault("schema_version", REGISTER_CHECKPOINT_SCHEMA_VERSION)
    envelope["stage"] = stage
    destination = Path(path)
    _atomic_torch_save(
        {
            "metadata": envelope,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in (state_dict or module.state_dict()).items()
            },
        },
        destination,
    )
    return destination


def load_stage_component_checkpoint(
    path: os.PathLike[str] | str,
    *,
    stage: str,
    module: nn.Module,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("schema_version") != REGISTER_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("component checkpoint schema version mismatch")
    if metadata.get("stage") != stage:
        raise RuntimeError(
            f"component checkpoint stage mismatch: expected {stage!r}, "
            f"found {metadata.get('stage')!r}"
        )
    for key, expected in (expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"component checkpoint metadata mismatch for {key}")
    module.load_state_dict(payload["state_dict"], strict=True)
    return metadata
