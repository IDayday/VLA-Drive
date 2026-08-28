#!/usr/bin/env python3

"""Verify the released DriveVLA Stage-1/VQA backbone boundary."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


DRIVEVLA_PREFIX = "agent.backbone.base_model.model.model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("drivevla_checkpoint", type=Path)
    parser.add_argument("recogdrive_safetensors", type=Path)
    return parser.parse_args()


def checkpoint_key(recogdrive_key: str, state_dict: dict) -> str:
    direct = DRIVEVLA_PREFIX + recogdrive_key
    if direct in state_dict:
        return direct

    module_name, tensor_name = recogdrive_key.rsplit(".", 1)
    peft_wrapped = DRIVEVLA_PREFIX + module_name + ".base_layer." + tensor_name
    if peft_wrapped in state_dict:
        return peft_wrapped

    raise KeyError(f"No DriveVLA tensor corresponds to {recogdrive_key}")


def main() -> None:
    args = parse_args()
    payload = torch.load(
        args.drivevla_checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state_dict = payload.get("state_dict", payload)

    backbone_keys = [key for key in state_dict if key.startswith("agent.backbone.")]
    action_keys = [key for key in state_dict if key.startswith("agent.action_head.")]
    lora_keys = [key for key in backbone_keys if ".lora_" in key]
    lora_b_keys = [key for key in lora_keys if ".lora_B." in key]

    mismatched = []
    checked = 0
    with safe_open(args.recogdrive_safetensors, framework="pt", device="cpu") as source:
        for source_key in source.keys():
            target_key = checkpoint_key(source_key, state_dict)
            source_tensor = source.get_tensor(source_key)
            target_tensor = state_dict[target_key]
            if source_tensor.shape != target_tensor.shape or not torch.equal(
                source_tensor, target_tensor
            ):
                mismatched.append((source_key, target_key))
            checked += 1

    nonzero_lora_b = sum(
        int(torch.count_nonzero(state_dict[key])) > 0 for key in lora_b_keys
    )

    print(f"ReCogDrive dense tensors checked: {checked:,}")
    print(f"Dense tensor mismatches:         {len(mismatched):,}")
    print(f"DriveVLA backbone tensors:       {len(backbone_keys):,}")
    print(f"DriveVLA backbone LoRA tensors:  {len(lora_keys):,}")
    print(f"Nonzero LoRA-B tensors:          {nonzero_lora_b:,}/{len(lora_b_keys):,}")
    print(f"DriveVLA action-head tensors:    {len(action_keys):,}")

    if mismatched:
        for source_key, target_key in mismatched[:10]:
            print(f"MISMATCH {source_key} -> {target_key}")
        raise SystemExit("Stage-1 dense VLM verification failed")
    if nonzero_lora_b != len(lora_b_keys):
        raise SystemExit("One or more released Stage-1 LoRA-B tensors are zero")


if __name__ == "__main__":
    main()
