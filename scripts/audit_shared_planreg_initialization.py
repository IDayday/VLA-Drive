#!/usr/bin/env python3
"""Prove BaseInit and VQAInit restore identical trainable tensors."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import sha256_file  # noqa: E402
from navsim.agents.EpisodeDrive.layers.planning_registers import (  # noqa: E402
    iter_qv_lora_modules,
)
from navsim.agents.EpisodeDrive.shared_planreg_initialization import (  # noqa: E402
    capture_shared_trainable_state,
)


def _compose_agent(config_path: Path, vlm_path: str, shared_path: str):
    with initialize_config_dir(
        version_base=None, config_dir=str(config_path.parent.resolve())
    ):
        config = compose(config_name=config_path.stem)
    OmegaConf.update(config, "vlm_config.vlm_path", vlm_path)
    OmegaConf.update(
        config, "initialization.shared_trainable_init_path", shared_path
    )
    return config


def _load_state(
    config_path: Path, vlm_path: str, shared_path: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    agent = instantiate(_compose_agent(config_path, vlm_path, shared_path))
    agent.initialize()
    state, metadata = capture_shared_trainable_state(agent)
    qv_modules = list(iter_qv_lora_modules(agent.backbone.model.vision_model))
    frozen_vision_base = all(
        not parameter.requires_grad
        for name, parameter in agent.backbone.model.vision_model.named_parameters()
        if not any(
            marker in name
            for marker in (
                ".q_lora_a.",
                ".q_lora_b.",
                ".v_lora_a.",
                ".v_lora_b.",
            )
        )
    )
    frozen_llm = all(
        not parameter.requires_grad
        for parameter in agent.backbone.model.language_model.parameters()
    )
    metadata.update(
        {
            "agent_checkpoint_loaded": bool(agent._agent_checkpoint_loaded),
            "shared_initialization_loaded": (
                agent._shared_trainable_initialization_metadata is not None
            ),
            "vision_adapted_block_count": len(qv_modules),
            "vision_qv_adapter_count": len(qv_modules) * 2,
            "vision_qv_ab_linear_count": len(qv_modules) * 4,
            "all_base_internvit_parameters_frozen": frozen_vision_base,
            "all_language_model_parameters_frozen": frozen_llm,
            "ema_initialized_after_shared_restore": agent.ema_register_target is not None,
        }
    )
    del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--vqa-config", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--vqa-vlm", required=True)
    parser.add_argument("--shared-init", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("DRIVEVLA_SCORE_RAY", "0")

    base_state, base_metadata = _load_state(
        Path(args.base_config).resolve(), args.base_vlm, args.shared_init
    )
    vqa_state, vqa_metadata = _load_state(
        Path(args.vqa_config).resolve(), args.vqa_vlm, args.shared_init
    )
    names_equal = set(base_state) == set(vqa_state)
    shape_differences = {
        name: {
            "base": list(base_state[name].shape),
            "driving_vqa": list(vqa_state[name].shape),
        }
        for name in sorted(set(base_state) & set(vqa_state))
        if base_state[name].shape != vqa_state[name].shape
    }
    value_differences = [
        name
        for name in sorted(set(base_state) & set(vqa_state))
        if base_state[name].shape == vqa_state[name].shape
        and not torch.equal(base_state[name], vqa_state[name])
    ]
    bitwise_equal = names_equal and not shape_differences and not value_differences
    report = {
        "schema_version": 1,
        "shared_init_path": str(Path(args.shared_init).expanduser().resolve()),
        "shared_init_sha256": sha256_file(
            Path(args.shared_init).expanduser().resolve()
        ),
        "base": base_metadata,
        "driving_vqa": vqa_metadata,
        "trainable_key_sets_equal": names_equal,
        "trainable_parameter_counts_equal": (
            base_metadata["trainable_parameter_count"]
            == vqa_metadata["trainable_parameter_count"]
        ),
        "shape_differences": shape_differences,
        "value_difference_count": len(value_differences),
        "value_differences": value_differences[:100],
        "all_trainable_tensors_bitwise_equal": bitwise_equal,
        "agent_checkpoint_loaded": False,
    }
    if not bitwise_equal:
        raise RuntimeError(
            "BaseInit and VQAInit trainable tensors are not bitwise identical: "
            + json.dumps(report, sort_keys=True)
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
