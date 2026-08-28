#!/usr/bin/env python3
"""Fail-fast, non-mutating PPU runtime check for DLC evaluation jobs."""

from __future__ import annotations

import json
import os
from importlib.metadata import version

import torch
import torch.nn.functional as functional


EXPECTED_FLASH_ATTN = "2.8.2+v0.1.0.ppu2.1.0.oe"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The PAI-PPU CUDA-compatible runtime is unavailable. Select the "
            "audited PPU image; do not reinstall torch, triton, or flash-attn."
        )
    expected_devices = int(os.environ.get("EXPECTED_VISIBLE_DEVICES", "1"))
    visible_devices = torch.cuda.device_count()
    if visible_devices != expected_devices:
        raise RuntimeError(
            f"Expected {expected_devices} visible PPU, found {visible_devices}"
        )
    flash_attn = version("flash-attn")
    if flash_attn != EXPECTED_FLASH_ATTN:
        raise RuntimeError(
            f"flash-attn changed: {flash_attn} != {EXPECTED_FLASH_ATTN}. "
            "Do not modify the installed vendor package."
        )

    devices: list[dict[str, object]] = []
    for index in range(visible_devices):
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
        query = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16, device=device)
        attended = functional.scaled_dot_product_attention(query, query, query)
        if attended.shape != (1, 2, 8, 16) or not torch.isfinite(attended).all():
            raise RuntimeError(f"PPU {index} BF16 SDPA smoke failed")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "bf16_sdpa": "PASS",
            }
        )
        del query, attended
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": "PASS",
                "runtime": "PAI-PPU CUDA-compatible PyTorch",
                "torch": torch.__version__,
                "flash_attn": flash_attn,
                "flash_attn_mutated": False,
                "visible_ppus": visible_devices,
                "devices": devices,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
