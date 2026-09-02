#!/usr/bin/env python3
"""Prove the formal Base/VQA resolved configs differ only in VLM identity."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from omegaconf import OmegaConf


ALLOWED_DIFFERENCES = frozenset(
    {
        "agent.initialization.variant",
        "agent.initialization.vlm_checkpoint_sha256",
        "agent.initialization.vlm_config_sha256",
        "agent.vlm_config.vlm_path",
        "experiment_name",
        "output_dir",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _differences(left: Any, right: Any, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: Dict[str, Dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                result[path] = {"base": "<missing>", "driving_vqa": right[key]}
            elif key not in right:
                result[path] = {"base": left[key], "driving_vqa": "<missing>"}
            else:
                result.update(_differences(left[key], right[key], path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = {}
        if len(left) != len(right):
            return {prefix: {"base": left, "driving_vqa": right}}
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            result.update(
                _differences(left_value, right_value, f"{prefix}[{index}]")
            )
        return result
    if left != right:
        return {prefix: {"base": left, "driving_vqa": right}}
    return {}


def load_resolved_config(path: Path) -> Dict[str, Any]:
    config = OmegaConf.load(path)
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise RuntimeError(f"Resolved config must be a mapping: {path}")
    return value


def audit_formal_config_pair(
    base: Mapping[str, Any], driving_vqa: Mapping[str, Any]
) -> Dict[str, Any]:
    normalized_base = deepcopy(base)
    normalized_vqa = deepcopy(driving_vqa)
    for config in (normalized_base, normalized_vqa):
        trainer_root = config.get("trainer", {}).get("params", {}).get(
            "default_root_dir"
        )
        if trainer_root == config.get("output_dir"):
            config["trainer"]["params"]["default_root_dir"] = "${output_dir}"
    differences = _differences(normalized_base, normalized_vqa)
    forbidden = sorted(set(differences) - ALLOWED_DIFFERENCES)
    if forbidden:
        detail = {name: differences[name] for name in forbidden}
        raise RuntimeError(
            "Formal Base/VQA configs have forbidden differences: "
            + json.dumps(detail, sort_keys=True)
        )
    base_agent = base["agent"]
    vqa_agent = driving_vqa["agent"]
    if base_agent["initialization"]["variant"] != "base":
        raise RuntimeError("Base formal config must use initialization.variant=base")
    if vqa_agent["initialization"]["variant"] != "driving_vqa":
        raise RuntimeError(
            "VQA formal config must use initialization.variant=driving_vqa"
        )
    if (
        base_agent["initialization"]["shared_trainable_init_path"]
        != vqa_agent["initialization"]["shared_trainable_init_path"]
    ):
        raise RuntimeError("Formal pair must load the same shared trainable init")
    for label, config in (("base", base), ("driving_vqa", driving_vqa)):
        agent = config["agent"]
        failures = []
        if agent.get("checkpoint_path") is not None:
            failures.append("agent.checkpoint_path must be null")
        if agent.get("stage1_checkpoint_path") is not None:
            failures.append("agent.stage1_checkpoint_path must be null")
        if not agent["world_model"]["enabled"]:
            failures.append("world model must be enabled")
        if agent["world_model"]["future_mode"] != "correct":
            failures.append("future_mode must be correct")
        if agent["world_model"]["candidate_count"] != 1:
            failures.append("world-model candidate_count must be one")
        if config["formal_training"]["dataset_epochs"] != 27:
            failures.append("dataset_epochs must be 27")
        if not config["data_protocol"]["include_val_in_train"]:
            failures.append("full trainval must be enabled")
        if float(config["trainer"]["params"]["limit_val_batches"]) != 0.0:
            failures.append("validation must be disabled")
        if failures:
            raise RuntimeError(f"Invalid {label} formal config: {failures}")
    return {
        "schema_version": 1,
        "pair_equal_outside_allowlist": True,
        "allowed_difference_paths": sorted(ALLOWED_DIFFERENCES),
        "observed_differences": differences,
        "shared_trainable_init_path": base_agent["initialization"][
            "shared_trainable_init_path"
        ],
        "world_model_enabled_both": True,
        "multi_trajectory_consequence_modeling_implemented": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--driving-vqa", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = load_resolved_config(args.base)
    driving_vqa = load_resolved_config(args.driving_vqa)
    report = audit_formal_config_pair(base, driving_vqa)
    report["base_resolved_config"] = str(args.base.resolve())
    report["base_resolved_config_sha256"] = _sha256(args.base)
    report["driving_vqa_resolved_config"] = str(args.driving_vqa.resolve())
    report["driving_vqa_resolved_config_sha256"] = _sha256(args.driving_vqa)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
