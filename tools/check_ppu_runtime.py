#!/usr/bin/env python3
"""Fail-fast PPU BF16/SDPA/collective smoke test for a DLC node.

Launch this file with one process per visible PPU.  PAI-PPU exposes its
CUDA-compatible PyTorch runtime through ``torch.cuda`` and its ACCL-P
collectives through the NCCL-compatible backend.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The PAI-PPU CUDA-compatible PyTorch runtime is unavailable. "
            "Use a PAI-PPU training image whose SDK matches the host driver."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    visible_devices = torch.cuda.device_count()
    if not 0 <= local_rank < visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside {visible_devices} visible PPU devices"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    # The VGGT cache and planner both depend on BF16 and native SDPA.  Keep
    # this allocation deliberately tiny so the check does not perturb memory.
    q = torch.randn(1, 4, 16, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attended = F.scaled_dot_product_attention(q, k, v)
    assert attended.shape == (1, 4, 16, 32)
    if not torch.isfinite(attended).all():
        raise RuntimeError("PPU BF16 SDPA produced non-finite values")

    collective = torch.ones(1, device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(collective)
    if collective.item() != float(world_size):
        raise RuntimeError(
            f"ACCL-P/NCCL-compatible all-reduce mismatch: {collective.item()} != {world_size}"
        )

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "runtime": "PAI-PPU CUDA-compatible PyTorch",
                    "torch": torch.__version__,
                    "visible_ppus": visible_devices,
                    "world_size": world_size,
                    "collective_backend": dist.get_backend() if dist.is_initialized() else "none",
                    "bf16_sdpa_shape": list(attended.shape),
                    "device_name_rank0": torch.cuda.get_device_name(0),
                },
                sort_keys=True,
            )
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
