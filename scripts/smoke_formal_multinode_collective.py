#!/usr/bin/env python3
"""Small NCCL gate used before retrying a formal two-node layout."""

from __future__ import annotations

from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = int(os.environ.get("FORMAL_EXPECTED_WORLD_SIZE", "16"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} formal ranks, found {world_size}"
        )
    # Large enough to exercise the socket transport, small enough to be a
    # sub-second gate on healthy nodes.
    value = torch.full((4 * 1024 * 1024,), float(rank + 1), device="cuda")
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(5):
        dist.all_reduce(value)
        value.div_(world_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not torch.isfinite(value).all():
        raise RuntimeError("Non-finite value in formal multi-node NCCL smoke")
    dist.barrier()
    if rank == 0:
        print(
            "FORMAL_MULTINODE_COLLECTIVE "
            + json.dumps(
                {
                    "world_size": world_size,
                    "iterations": 5,
                    "tensor_bytes": value.numel() * value.element_size(),
                    "elapsed_seconds": elapsed,
                    "finite": True,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
