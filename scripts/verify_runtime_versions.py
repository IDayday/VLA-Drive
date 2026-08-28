#!/usr/bin/env python3
"""Fail if the vendor PPU/PyTorch runtime differs from the audited image."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import sys


EXPECTED_RUNTIME = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "torchaudio": "2.4.0",
    "triton": "3.0.0+ppu1.7.0.oe",
    "flash-attn": "2.8.2+v0.1.0.ppu2.1.0.oe",
    "deepspeed": "0.16.9",
    "pytorch-lightning": "2.2.1",
    "transformers": "4.57.0",
    "accelerate": "1.5.2",
    "peft": "0.17.0",
    "timm": "1.0.20",
}


def main() -> int:
    failures: list[str] = []
    for package, expected in EXPECTED_RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            failures.append(f"{package}: missing (expected {expected})")
            continue
        print(f"{package}=={actual}")
        if actual != expected:
            failures.append(f"{package}: found {actual}, expected {expected}")

    if failures:
        print("Vendor runtime contract failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "Do not repair this by reinstalling PyTorch packages; select the audited "
            "PPU image or update the contract deliberately.",
            file=sys.stderr,
        )
        return 1

    print("Vendor PPU/PyTorch runtime contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
