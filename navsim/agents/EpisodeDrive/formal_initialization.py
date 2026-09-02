"""Auditable VLM-only initialization for the formal PlanReg-WM runs.

The formal experiment deliberately starts from a standalone InternVL VLM and
then restores a separately generated, shared random planning stack.  It must
never accept a DriveVLA/M0 agent checkpoint through this path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - exercised by the CLI preflight.
    safe_open = None


FORMAL_INITIALIZATION_MODE = "vlm_pretrained_random_planning_stack"
FORMAL_INITIALIZATION_VARIANTS = frozenset({"base", "driving_vqa"})
FORBIDDEN_AGENT_STATE_MARKERS = (
    "action_head.",
    "scorer.",
    "future_register_predictor.",
    "planning_register_adapter.",
)
FORBIDDEN_TRAINING_STATE_KEYS = frozenset(
    {"optimizer_states", "lr_schedulers"}
)
TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combined_file_sha256(paths: Iterable[Path]) -> str:
    """Hash names, sizes, and bytes so a sharded checkpoint has one identity."""
    digest = hashlib.sha256()
    resolved = sorted((Path(path) for path in paths), key=lambda value: value.name)
    if not resolved:
        raise FileNotFoundError("No files were supplied for SHA-256 fingerprinting")
    for path in resolved:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def discover_weight_files(checkpoint_path: Path) -> Tuple[Path, ...]:
    candidates = []
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        candidates.extend(checkpoint_path.glob(pattern))
    return tuple(
        sorted(
            path
            for path in candidates
            if not path.name.startswith("adapter_model")
        )
    )


def scan_forbidden_state_keys(keys: Iterable[str]) -> Tuple[str, ...]:
    findings = []
    for key in keys:
        normalized = str(key)
        if normalized in FORBIDDEN_TRAINING_STATE_KEYS or any(
            marker in normalized for marker in FORBIDDEN_AGENT_STATE_MARKERS
        ):
            findings.append(normalized)
    return tuple(sorted(set(findings)))


def _state_keys_from_weights(weight_files: Sequence[Path]) -> Tuple[str, ...]:
    keys = []
    for path in weight_files:
        if path.suffix == ".safetensors":
            if safe_open is None:
                raise RuntimeError(
                    "safetensors is required for the formal VLM checkpoint audit"
                )
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
        else:
            # Loading legacy pickle weights is intentionally avoided in the
            # formal preflight.  Convert them to safetensors first.
            raise RuntimeError(
                f"Formal VLM checkpoint audit requires safetensors, found {path}"
            )
    return tuple(keys)


def _architecture_signature(config_dict: Mapping[str, Any]) -> Dict[str, Any]:
    vision = config_dict.get("vision_config", {})
    language = config_dict.get("llm_config", {})
    return {
        "model_architectures": list(config_dict.get("architectures", [])),
        "vision_architectures": list(vision.get("architectures", [])),
        "language_architectures": list(language.get("architectures", [])),
        "vision_block_count": int(vision.get("num_hidden_layers", -1)),
        "vision_hidden_size": int(vision.get("hidden_size", -1)),
        "patch_size": int(vision.get("patch_size", -1)),
        "image_size": int(vision.get("image_size", -1)),
        "llm_hidden_size": int(language.get("hidden_size", -1)),
        "vocab_size": int(language.get("vocab_size", -1)),
        "prompt_template": config_dict.get("template"),
    }


def validate_formal_initialization_config(
    initialization: Any,
    *,
    checkpoint_path: Optional[str],
    stage1_checkpoint_path: Optional[str],
    vlm_config: Any,
) -> Dict[str, Any]:
    """Validate the in-memory Hydra contract before any large model is built."""
    mode = str(_cfg_get(initialization, "mode", ""))
    if mode != FORMAL_INITIALIZATION_MODE:
        raise ValueError(
            "Formal PlanReg training requires initialization.mode="
            f"{FORMAL_INITIALIZATION_MODE!r}, got {mode!r}"
        )
    variant = str(_cfg_get(initialization, "variant", ""))
    if variant not in FORMAL_INITIALIZATION_VARIANTS:
        raise ValueError(
            "initialization.variant must be one of "
            f"{sorted(FORMAL_INITIALIZATION_VARIANTS)}, got {variant!r}"
        )
    if not bool(_cfg_get(initialization, "prohibit_agent_checkpoint", True)):
        raise ValueError("Formal runs must set prohibit_agent_checkpoint=true")
    if checkpoint_path:
        raise ValueError(
            "Formal VLM-only initialization prohibits checkpoint_path; an M0/full-agent "
            "checkpoint would invalidate the experiment"
        )
    if stage1_checkpoint_path:
        raise ValueError(
            "Formal VLM-only initialization prohibits stage1_checkpoint_path; load the "
            "standalone VLM with AutoModel.from_pretrained instead"
        )
    if bool(_cfg_get(vlm_config, "initialize_from_config", True)):
        raise ValueError(
            "Formal VLM initialization requires vlm_config.initialize_from_config=false"
        )
    vlm_path = str(_cfg_get(vlm_config, "vlm_path", "") or "")
    if not vlm_path:
        raise ValueError("Formal VLM initialization requires a non-empty vlm_path")
    return {
        "mode": mode,
        "variant": variant,
        "vlm_path": str(Path(vlm_path).expanduser().resolve()),
        "agent_checkpoint_loaded": False,
    }


def validate_formal_scientific_contract(
    *,
    vlm_config: Any,
    vision_adaptation: Any,
    planning_registers: Any,
    scene_fusion: Any,
    semantic_path: Any,
    world_model: Any,
    ema: Any,
    action_head_config: Any,
) -> None:
    """Reject accidental ablations in either official formal launcher."""
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(str(_cfg_get(vlm_config, "vlm_type", "")) == "internvl", "VLM must be InternVL")
    require(not bool(_cfg_get(vlm_config, "cache_hidden_state", True)), "cache_hidden_state must be false")
    require(not bool(_cfg_get(vlm_config, "cache_mode", True)), "cache_mode must be false")
    require(bool(_cfg_get(vlm_config, "freeze_language_model", False)), "LLM must remain frozen")
    require(bool(_cfg_get(vlm_config, "planning_registers_enabled", False)), "planning registers must be enabled")
    require(int(_cfg_get(vlm_config, "num_planning_registers", -1)) == 16, "exactly 16 planning registers are required")
    require(str(_cfg_get(vlm_config, "tile_register_aggregation", "")) == "thumbnail_query_attention", "tile aggregation must be thumbnail_query_attention")
    require(str(_cfg_get(vision_adaptation, "mode", "")) == "qv_lora", "vision adaptation must be Q/V LoRA")
    require(str(_cfg_get(vision_adaptation, "layers", "")) == "all", "all vision blocks must be adapted")
    require(int(_cfg_get(vision_adaptation, "rank", -1)) == 32, "vision Q/V LoRA rank must be 32")
    require(bool(_cfg_get(vision_adaptation, "train_q", False)), "vision Q must be trainable through LoRA")
    require(not bool(_cfg_get(vision_adaptation, "train_k", True)), "vision K must not be trained")
    require(bool(_cfg_get(vision_adaptation, "train_v", False)), "vision V must be trainable through LoRA")
    require(str(_cfg_get(planning_registers, "attention_mode", "")) == "read_only", "planning registers must use read_only attention")
    require(str(_cfg_get(scene_fusion, "mode", "")) == "planning_primary_semantic_xattn", "formal fusion must be planning-primary semantic cross-attention")
    require(bool(_cfg_get(semantic_path, "frozen_llm_no_grad", False)), "semantic LLM must run under no_grad")
    require(not bool(_cfg_get(semantic_path, "backprop_to_vision", True)), "semantic path must not backpropagate to vision")
    require(bool(_cfg_get(semantic_path, "train_qformer", False)), "semantic Q-Former must be trainable")
    require(bool(_cfg_get(world_model, "enabled", False)), "world model must be enabled")
    require(str(_cfg_get(world_model, "future_mode", "")) == "correct", "future_mode must be correct")
    require(not bool(_cfg_get(world_model, "predictor_only", True)), "predictor_only must be false")
    require(float(_cfg_get(world_model, "min_weight", 0.0)) > 0.0, "WM weight must be positive from optimizer step zero")
    require(float(_cfg_get(world_model, "start_fraction", -1.0)) == 0.0, "WM start_fraction must be zero")
    require(int(_cfg_get(world_model, "candidate_count", -1)) == 1, "WM candidate_count must be one")
    require(str(_cfg_get(world_model, "trajectory_source", "")) == "gt", "WM trajectory source must be GT")
    require(bool(_cfg_get(ema, "enabled", False)), "online EMA teacher must be enabled")
    for key, expected in {
        "proposal_num": 64,
        "num_poses": 8,
        "ref_num": 4,
        "scorer_ref_num": 4,
        "long_trajectory_additional_poses": 2,
    }.items():
        require(int(_cfg_get(action_head_config, key, -1)) == expected, f"action_head_config.{key} must be {expected}")
    require(bool(_cfg_get(action_head_config, "one_token_per_traj", False)), "one_token_per_traj must be true")
    require(list(_cfg_get(action_head_config, "cam_f0", [])) == [3], "front camera history must be cam_f0=[3]")
    for camera in ("cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"):
        require(not list(_cfg_get(action_head_config, camera, [])), f"{camera} must be disabled")
    if errors:
        raise ValueError("Invalid formal PlanReg-WM scientific contract: " + "; ".join(errors))


def _tokenizer_artifact_fingerprint(checkpoint_path: Path) -> Dict[str, Any]:
    artifacts = [
        checkpoint_path / name
        for name in TOKENIZER_ARTIFACT_NAMES
        if (checkpoint_path / name).is_file()
    ]
    if not artifacts:
        raise FileNotFoundError(
            f"No tokenizer artifacts were found under {checkpoint_path}"
        )
    return {
        "sha256": combined_file_sha256(artifacts),
        "files": {path.name: sha256_file(path) for path in artifacts},
    }


def audit_vlm_checkpoint(
    checkpoint_path: str,
    *,
    variant: str,
    load_runtime_classes: bool = False,
) -> Dict[str, Any]:
    """Inspect a standalone InternVL checkpoint without accepting agent state."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"VLM checkpoint directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"VLM checkpoint lacks config.json: {path}")
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    weight_files = discover_weight_files(path)
    if not weight_files:
        raise FileNotFoundError(f"No standalone VLM weights found under {path}")
    state_keys = _state_keys_from_weights(weight_files)
    forbidden_keys = scan_forbidden_state_keys(state_keys)
    forbidden_files = sorted(
        item.name
        for item in path.iterdir()
        if item.name in {"optimizer.pt", "scheduler.pt"}
    )
    if forbidden_keys or forbidden_files:
        raise RuntimeError(
            "Formal initialization rejected agent/training checkpoint content: "
            f"state_keys={list(forbidden_keys[:20])}, files={forbidden_files}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True, use_fast=False, local_files_only=True
    )
    vocabulary = tokenizer.get_vocab()
    token_id_map = {
        token: int(index)
        for token, index in sorted(vocabulary.items(), key=lambda item: item[1])
    }
    tokenizer_artifacts = _tokenizer_artifact_fingerprint(path)
    signature = _architecture_signature(config_dict)
    weight_file_hashes = {file.name: sha256_file(file) for file in weight_files}
    result: Dict[str, Any] = {
        "variant": variant,
        "checkpoint_path": str(path),
        "checkpoint_sha256": (
            weight_file_hashes[weight_files[0].name]
            if len(weight_files) == 1
            else combined_file_sha256(weight_files)
        ),
        "weight_files": [file.name for file in weight_files],
        "weight_file_sha256": weight_file_hashes,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": tokenizer_artifacts["sha256"],
        "tokenizer_file_sha256": tokenizer_artifacts["files"],
        "tokenizer_vocab_sha256": canonical_sha256(token_id_map),
        "tokenizer_length": len(tokenizer),
        "token_id_map": token_id_map,
        "state_dict_key_count": len(state_keys),
        "state_dict_key_prefixes": sorted(
            {key.split(".", 1)[0] for key in state_keys}
        ),
        "forbidden_agent_state_detected": False,
        "agent_checkpoint_loaded": False,
        **signature,
    }
    driving_markers = (
        "<loc>",
        "</loc>",
        "<FRONT VIEW>",
        "<BACK VIEW>",
    )
    marker_ids = {
        token: token_id_map[token]
        for token in driving_markers
        if token in token_id_map
    }
    provenance = config_dict.get("planreg_formal_provenance", {})
    source_variant = str(provenance.get("source_variant", ""))
    path_marker = any(
        marker in str(path).lower()
        for marker in ("driving-vqa", "driving_vqa", "recogdrive", "vqa")
    )
    result["driving_vqa_training_marker_detected"] = (
        source_variant == "driving_vqa" or path_marker
    )
    result["driving_vqa_marker_evidence"] = {
        "source_variant": source_variant or None,
        "path_marker": path_marker,
        "driving_tokens_present": bool(marker_ids),
    }
    result["driving_token_ids"] = marker_ids

    config = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    result["model_class"] = list(getattr(config, "architectures", []) or [""])[0]
    result["vision_model_class"] = "InternVisionModel"
    result["language_model_class"] = list(
        getattr(config.llm_config, "architectures", []) or [""]
    )[0]
    if load_runtime_classes:
        # This mode is intentionally explicit because it maps the full VLM.
        model = AutoModel.from_pretrained(
            str(path),
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        result["model_class"] = (
            f"{type(model).__module__}.{type(model).__qualname__}"
        )
        result["vision_model_class"] = (
            f"{type(model.vision_model).__module__}."
            f"{type(model.vision_model).__qualname__}"
        )
        result["language_model_class"] = (
            f"{type(model.language_model).__module__}."
            f"{type(model.language_model).__qualname__}"
        )
        del model
    return result


def compare_formal_vlm_audits(
    base: Mapping[str, Any], vqa: Mapping[str, Any]
) -> Dict[str, Any]:
    """Require the two formal VLMs to differ only in pretrained tensors."""
    architecture_fields = (
        "model_architectures",
        "vision_architectures",
        "language_architectures",
        "vision_block_count",
        "vision_hidden_size",
        "patch_size",
        "image_size",
        "llm_hidden_size",
        "vocab_size",
        "prompt_template",
    )
    architecture_differences = {
        field: {"base": base.get(field), "driving_vqa": vqa.get(field)}
        for field in architecture_fields
        if base.get(field) != vqa.get(field)
    }
    base_vocab = dict(base.get("token_id_map", {}))
    vqa_vocab = dict(vqa.get("token_id_map", {}))
    token_differences = {
        token: {"base": base_vocab.get(token), "driving_vqa": vqa_vocab.get(token)}
        for token in sorted(set(base_vocab) | set(vqa_vocab))
        if base_vocab.get(token) != vqa_vocab.get(token)
    }
    tokenizer_equal = not token_differences
    result = {
        "architecture_equal": not architecture_differences,
        "tokenizer_token_ids_equal": tokenizer_equal,
        "architecture_differences": architecture_differences,
        "tokenizer_difference_count": len(token_differences),
        "tokenizer_differences": dict(list(token_differences.items())[:100]),
        "formal_pair_compatible": not architecture_differences and tokenizer_equal,
    }
    if not result["formal_pair_compatible"]:
        raise RuntimeError(
            "Base and Driving-VQA VLM initializations are not formally compatible. "
            "Silent vocabulary resizing is prohibited. Details: "
            + json.dumps(result, ensure_ascii=False, sort_keys=True)
        )
    return result


__all__ = [
    "FORMAL_INITIALIZATION_MODE",
    "FORMAL_INITIALIZATION_VARIANTS",
    "audit_vlm_checkpoint",
    "canonical_sha256",
    "combined_file_sha256",
    "compare_formal_vlm_audits",
    "discover_weight_files",
    "scan_forbidden_state_keys",
    "sha256_file",
    "validate_formal_initialization_config",
    "validate_formal_scientific_contract",
]
