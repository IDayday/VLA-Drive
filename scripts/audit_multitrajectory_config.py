#!/usr/bin/env python3
"""Audit matched-control configs against the formal multi-trajectory method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.training.config_loader import load_training_config


_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}$")


def _raw_container(config) -> dict:
    return OmegaConf.to_container(config, resolve=False)


def _get(mapping: dict, path: str, default=None):
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return default
        current = current[component]
    return current


def _resolve_env_string(value: Any) -> str:
    text = str(value)
    match = _ENV_PATTERN.fullmatch(text)
    if match is None:
        return text
    variable, default = match.groups()
    if variable in os.environ:
        return os.environ[variable]
    if default is not None and default.lower() not in {"null", "none"}:
        return default
    raise RuntimeError(
        f"cannot resolve {text}; export {variable} before config audit"
    )


def _checkpoint_keys(model_path: str) -> set[str]:
    root = Path(model_path).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as stream:
            return set(json.load(stream)["weight_map"])
    tensor_path = root / "model.safetensors"
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Qwen safetensors checkpoint not found under {root}")
    with safe_open(tensor_path, framework="pt", device="cpu") as stream:
        return set(stream.keys())


def qwen_parameter_manifests(config) -> tuple[set[str], set[str]]:
    """Derive full wrapper parameter-name sets without materializing 2B tensors."""

    raw = _raw_container(config)
    model_path = _resolve_env_string(_get(raw, "framework.qwenvl.base_vlm"))
    checkpoint_names = _checkpoint_keys(model_path)
    # _QWen3_VL_Interface owns the HF model as ``model``.
    names = {f"model.{name}" for name in checkpoint_names}
    config_path = Path(model_path).expanduser().resolve() / "config.json"
    with config_path.open("r", encoding="utf-8") as stream:
        model_config = json.load(stream)
    tied_embeddings = bool(
        model_config.get(
            "tie_word_embeddings",
            model_config.get("text_config", {}).get("tie_word_embeddings", False),
        )
    )

    raw_freeze = _get(raw, "trainer.freeze_modules", "")
    freeze_paths = (
        [part.strip() for part in raw_freeze.split(",") if part.strip()]
        if isinstance(raw_freeze, str)
        else []
    )
    frozen: set[str] = set()
    for path in freeze_paths:
        if path == "qwen_vl_interface":
            frozen.update(names)
        elif path == "qwen_vl_interface.model.visual":
            frozen.update(
                name for name in names if name.startswith("model.model.visual.")
            )
        elif path == "qwen_vl_interface.model.lm_head":
            frozen.update(
                name for name in names if name.startswith("model.lm_head.")
            )
            if tied_embeddings:
                frozen.add("model.model.language_model.embed_tokens.weight")
        elif path.startswith("qwen_vl_interface."):
            relative = path[len("qwen_vl_interface.") :] + "."
            frozen.update(name for name in names if name.startswith(relative))
    unknown = frozen.difference(names)
    if unknown:
        raise RuntimeError(f"derived frozen Qwen names are absent: {sorted(unknown)}")
    return names.difference(frozen), frozen


def _manifest_digest(names: set[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flow_repeats(raw: dict) -> int:
    action = _get(raw, "framework.action_model", {})
    return int(
        action.get(
            "flow_train_repeats",
            action.get("repeated_diffusion_steps", 8),
        )
    )


def _auxiliary_manifest(raw: dict) -> dict:
    return {
        "video": _get(raw, "datasets.video_data.load_2d_data"),
        "depth": _get(raw, "w_depth"),
        "gs": _get(raw, "datasets.gs_data.load_3d_data"),
        "rgb_query_loss": _get(raw, "rgb_query_loss"),
        "gs_query_loss": _get(raw, "gs_query_loss"),
        "reward": _get(raw, "datasets.reward_data.load_reward_data"),
    }


def audit_configs(baseline_path: str, method_path: str) -> list[str]:
    baseline = load_training_config(baseline_path)
    method = load_training_config(method_path)
    base = _raw_container(baseline)
    full = _raw_container(method)
    failures: list[str] = []

    def display(value) -> str:
        if isinstance(value, (dict, list)):
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return f"matched sha256={hashlib.sha256(payload.encode()).hexdigest()}"
        return repr(value)

    def require_equal(label: str, base_value, method_value) -> None:
        if base_value != method_value:
            failures.append(
                f"{label} differs: baseline={base_value!r} method={method_value!r}"
            )
        else:
            print(f"PASS {label}: {display(base_value)}")

    require_equal(
        "Qwen model path",
        _get(base, "framework.qwenvl.base_vlm"),
        _get(full, "framework.qwenvl.base_vlm"),
    )
    require_equal(
        "Qwen attention implementation",
        _get(base, "framework.qwenvl.attn_implementation"),
        _get(full, "framework.qwenvl.attn_implementation"),
    )
    base_trainable, base_frozen = qwen_parameter_manifests(baseline)
    method_trainable, method_frozen = qwen_parameter_manifests(method)
    if base_trainable != method_trainable:
        failures.append(
            "Qwen trainable parameter names differ: "
            f"missing={sorted(base_trainable - method_trainable)[:8]} "
            f"extra={sorted(method_trainable - base_trainable)[:8]}"
        )
    else:
        print(f"PASS Qwen trainable parameter names: exact set ({len(base_trainable)})")
    if base_frozen != method_frozen:
        failures.append(
            "Qwen frozen parameter names differ: "
            f"missing={sorted(base_frozen - method_frozen)[:8]} "
            f"extra={sorted(method_frozen - base_frozen)[:8]}"
        )
    else:
        print(f"PASS Qwen frozen parameter names: exact set ({len(base_frozen)})")
    print(
        "Qwen manifests: "
        f"trainable={len(method_trainable)} sha256={_manifest_digest(method_trainable)} "
        f"frozen={len(method_frozen)} sha256={_manifest_digest(method_frozen)}"
    )

    comparisons = (
        ("DiT hidden size", "framework.action_model.hidden_size"),
        ("DiT layer count", "framework.action_model.diffusion_model_cfg.num_layers"),
        ("action horizon", "framework.action_model.action_horizon"),
        ("action dim", "framework.action_model.action_dim"),
        ("dataset", "datasets"),
        ("optimizer", "trainer.optimizer"),
        ("learning rates", "trainer.learning_rate"),
        ("scheduler", "trainer.lr_scheduler_type"),
        ("scheduler kwargs", "trainer.scheduler_specific_kwargs"),
        ("warmup steps", "trainer.num_warmup_steps"),
        ("training steps", "trainer.max_train_steps"),
        ("epochs", "trainer.epochs"),
    )
    for label, path in comparisons:
        require_equal(label, _get(base, path), _get(full, path))
    require_equal("Flow train repeats", _flow_repeats(base), _flow_repeats(full))
    require_equal(
        "auxiliary branch flags",
        _auxiliary_manifest(base),
        _auxiliary_manifest(full),
    )

    if int(_get(full, "framework.action_model.hidden_size", -1)) != 1536:
        failures.append("method DiT hidden size is not the required 1536")
    if int(
        _get(full, "framework.action_model.diffusion_model_cfg.num_layers", -1)
    ) != 24:
        failures.append("method DiT layer count is not the required 24")
    if _flow_repeats(full) != 8:
        failures.append("method flow_train_repeats is not the required 8")
    if int(_get(full, "framework.action_model.action_horizon", -1)) != 8:
        failures.append("method action_horizon is not the required 8")
    if int(_get(full, "framework.action_model.action_dim", -1)) != 4:
        failures.append("method action_dim is not the required 4")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--method-config", required=True)
    args = parser.parse_args()
    failures = audit_configs(args.baseline_config, args.method_config)
    if failures:
        print("CONFIG AUDIT FAILED")
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("CONFIG AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
