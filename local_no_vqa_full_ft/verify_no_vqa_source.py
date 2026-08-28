#!/usr/bin/env python3
"""Prove that the selected InternVL source is not the ReCogDrive VQA state."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_internvl", type=Path)
    parser.add_argument("vqa_internvl", type=Path)
    return parser.parse_args()


def tensor_file(path: Path) -> Path:
    candidate = path / "model.safetensors" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> None:
    args = parse_args()
    raw_path = tensor_file(args.raw_internvl)
    vqa_path = tensor_file(args.vqa_internvl)

    with safe_open(raw_path, framework="pt", device="cpu") as raw, safe_open(
        vqa_path, framework="pt", device="cpu"
    ) as vqa:
        common_keys = sorted(set(raw.keys()) & set(vqa.keys()))
        if not common_keys:
            raise RuntimeError("The raw and VQA checkpoints have no common tensor keys")

        equal = 0
        different = 0
        shape_mismatch = 0
        examples = []
        for key in common_keys:
            raw_tensor = raw.get_tensor(key)
            vqa_tensor = vqa.get_tensor(key)
            if raw_tensor.shape != vqa_tensor.shape:
                shape_mismatch += 1
                continue
            if torch.equal(raw_tensor, vqa_tensor):
                equal += 1
            else:
                different += 1
                if len(examples) < 8:
                    examples.append(key)

    print(f"raw checkpoint: {raw_path}")
    print(f"VQA checkpoint: {vqa_path}")
    print(f"common tensors: {len(common_keys):,}")
    print(f"exactly equal: {equal:,}")
    print(f"different: {different:,}")
    print(f"shape mismatch: {shape_mismatch:,}")
    print("different examples:")
    for key in examples:
        print(f"  {key}")
    if different == 0 and shape_mismatch == 0:
        raise SystemExit("Selected source is identical to the VQA checkpoint")
    print("No-VQA source audit: PASS")


if __name__ == "__main__":
    main()
