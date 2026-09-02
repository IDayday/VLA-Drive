#!/usr/bin/env python3
"""Create one shared random trainable-state artifact for both formal VLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from navsim.agents.EpisodeDrive.layers.planning_registers import (  # noqa: E402
    iter_qv_lora_modules,
)
from navsim.agents.EpisodeDrive.shared_planreg_initialization import (  # noqa: E402
    save_shared_trainable_initialization,
)


def _load_architecture_config(path: Path):
    if path.suffix.lower() == ".json":
        config = OmegaConf.create(json.loads(path.read_text(encoding="utf-8")))
    else:
        raw = OmegaConf.load(path)
        if "defaults" in raw:
            with initialize_config_dir(
                version_base=None, config_dir=str(path.parent.resolve())
            ):
                config = compose(config_name=path.stem)
        else:
            config = raw
    if "agent" in config and "_target_" not in config:
        config = config.agent
    if "_target_" not in config:
        raise ValueError(
            "Architecture config must resolve to an agent node containing _target_"
        )
    OmegaConf.update(
        config,
        "initialization.shared_trainable_init_path",
        None,
        force_add=True,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--architecture-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    args = parser.parse_args()

    config_path = Path(args.architecture_config).expanduser().resolve()
    agent_config = _load_architecture_config(config_path)
    resolved = OmegaConf.to_container(agent_config, resolve=True)
    architecture_sha = canonical_sha256(resolved)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("DRIVEVLA_SCORE_RAY", "0")

    agent = instantiate(agent_config)
    if not getattr(agent, "_formal_initialization", False):
        raise RuntimeError("Architecture config is not a formal PlanReg agent")
    if getattr(agent, "_agent_checkpoint_loaded", True):
        raise RuntimeError("Agent checkpoint was unexpectedly loaded")
    if agent.ema_register_target is not None:
        raise RuntimeError("EMA must be initialized only after shared state restoration")
    agent._freeze_backbone_for_planreg()

    qv_modules = list(iter_qv_lora_modules(agent.backbone.model.vision_model))
    qv_linear_count = sum(
        4
        for _, module in qv_modules
        for _unused in [module]
    )
    qv_parameter_count = sum(
        parameter.numel()
        for _, module in qv_modules
        for branch in (
            module.q_lora_a,
            module.q_lora_b,
            module.v_lora_a,
            module.v_lora_b,
        )
        for parameter in branch.parameters()
    )
    if len(qv_modules) != 24 or qv_linear_count != 96:
        raise RuntimeError(
            "Formal InternVL3-2B must expose 24 adapted blocks / 96 A/B Linear "
            f"modules, got {len(qv_modules)} / {qv_linear_count}"
        )
    if qv_parameter_count != 3_145_728:
        raise RuntimeError(
            "Unexpected rank-32 dim-1024 Q/V LoRA parameter count: "
            f"{qv_parameter_count:,}"
        )

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    metadata = save_shared_trainable_initialization(
        agent,
        args.output,
        seed=args.seed,
        architecture_config_sha256=architecture_sha,
        source_git_commit=git_commit,
    )
    artifact_path = Path(args.output).expanduser().resolve()
    non_ema_parameters = [
        parameter
        for name, parameter in agent.named_parameters()
        if not name.startswith("ema_register_target.")
    ]
    non_ema_total = sum(parameter.numel() for parameter in non_ema_parameters)
    non_ema_trainable = sum(
        parameter.numel()
        for parameter in non_ema_parameters
        if parameter.requires_grad
    )
    metadata.update(
        {
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "vision_adapted_block_count": len(qv_modules),
            "vision_qv_adapter_count": len(qv_modules) * 2,
            "vision_qv_ab_linear_count": qv_linear_count,
            "vision_qv_lora_parameter_count": qv_parameter_count,
            "non_ema_model_parameter_count": non_ema_total,
            "non_ema_trainable_parameter_count": non_ema_trainable,
            "non_ema_frozen_parameter_count": non_ema_total - non_ema_trainable,
            "agent_checkpoint_loaded": False,
            "ema_initialized_during_creation": False,
            "paired_runtime_bitwise_verification": "required_before_launch",
        }
    )
    metadata_path = Path(
        args.metadata_output
        or artifact_path.parent / "run_metadata" / "shared_trainable_init.json"
    ).expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
