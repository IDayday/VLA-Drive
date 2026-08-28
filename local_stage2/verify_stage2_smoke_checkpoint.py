#!/usr/bin/env python3

"""Audit a one-step Stage-2 checkpoint against its warm start and seed-0 head."""

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder


def tensor_hash(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_checkpoint", type=Path)
    parser.add_argument("stage1_checkpoint", type=Path)
    args = parser.parse_args()

    smoke = torch.load(args.smoke_checkpoint, map_location="cpu")
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu")
    smoke_state = smoke["state_dict"]
    stage1_state = stage1["state_dict"]

    backbone_keys = sorted(
        key for key in smoke_state if key.startswith("agent.backbone.")
    )
    action_keys = sorted(
        key for key in smoke_state if key.startswith("agent.action_head.")
    )
    backbone_mismatches = [
        key
        for key in backbone_keys
        if key not in stage1_state
        or not torch.equal(smoke_state[key], stage1_state[key])
    ]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = OmegaConf.load(
        Path(__file__).parents[1]
        / "navsim/planning/script/config/common/agent/episode_drive.yaml"
    )
    initial_head = ActionDecoder(config.action_head_config).state_dict()
    initial_items = []
    smoke_items = []
    changed = []
    missing = []
    for checkpoint_key in action_keys:
        local_key = checkpoint_key.removeprefix("agent.action_head.")
        if local_key not in initial_head:
            missing.append(checkpoint_key)
            continue
        initial_tensor = initial_head[local_key]
        smoke_tensor = smoke_state[checkpoint_key]
        initial_items.append((local_key, initial_tensor))
        smoke_items.append((local_key, smoke_tensor))
        if not torch.equal(initial_tensor, smoke_tensor):
            changed.append(checkpoint_key)

    print(f"global_step: {smoke.get('global_step')}")
    print(f"epoch: {smoke.get('epoch')}")
    print(f"optimizer states: {len(smoke.get('optimizer_states', []))}")
    print(f"backbone tensors: {len(backbone_keys):,}")
    print(f"backbone mismatches vs Stage 1: {len(backbone_mismatches):,}")
    print(f"action-head tensors: {len(action_keys):,}")
    print(f"action tensors changed after one step: {len(changed):,}")
    print(f"action tensors missing from seed-0 head: {len(missing):,}")
    print(f"seed-0 action hash: {tensor_hash(initial_items)}")
    print(f"one-step action hash: {tensor_hash(smoke_items)}")

    if smoke.get("global_step") != 1:
        raise SystemExit("smoke checkpoint global_step is not 1")
    if len(backbone_keys) != 1_005 or backbone_mismatches:
        raise SystemExit("frozen backbone changed during the smoke step")
    if len(action_keys) != 318 or missing or not changed:
        raise SystemExit("Stage-2 action-head update audit failed")
    if not smoke.get("optimizer_states"):
        raise SystemExit("smoke checkpoint has no optimizer state")
    print("Stage-2 smoke checkpoint audit: PASS")


if __name__ == "__main__":
    main()
