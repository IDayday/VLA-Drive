#!/usr/bin/env python3
"""Probe one CUDA-compatible PPU per distributed launcher rank."""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.expected_world_size <= 0 or world_size != args.expected_world_size:
        raise RuntimeError(
            f"distributed runtime world size {world_size} != expected "
            f"{args.expected_world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible PPU runtime is unavailable")
    visible = torch.cuda.device_count()
    if not 0 <= local_rank < visible:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside {visible} visible devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    left = torch.arange(256, device=device, dtype=torch.bfloat16).reshape(16, 16)
    result = left @ left.transpose(0, 1)
    torch.cuda.synchronize(device)
    if not torch.isfinite(result).all():
        raise RuntimeError("PPU BF16 matmul probe produced NaN or Inf")
    properties = torch.cuda.get_device_properties(device)
    print(
        f"PPU runtime rank={rank}/{world_size} local_rank={local_rank} "
        f"device={properties.name} bf16_matmul=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
