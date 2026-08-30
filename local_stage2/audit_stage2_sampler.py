#!/usr/bin/env python3
"""Quantify the ordering change caused by padding before distributed shuffle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _reference_order(dataset_size: int, global_batch: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    padding = (-dataset_size) % global_batch
    return indices + indices[:padding]


def _prepad_order(dataset_size: int, global_batch: int, seed: int) -> list[int]:
    padding = (-dataset_size) % global_batch
    padded_size = dataset_size + padding
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(padded_size, generator=generator).tolist()
    # The legacy ConcatDataset appended Subset(dataset, range(padding)).
    return [index if index < dataset_size else index - dataset_size for index in indices]


def _batch_sets(indices: list[int], global_batch: int) -> list[set[int]]:
    return [
        set(indices[offset : offset + global_batch])
        for offset in range(0, len(indices), global_batch)
    ]


def audit(dataset_size: int, global_batch: int, seed: int) -> dict:
    reference = _reference_order(dataset_size, global_batch, seed)
    prepad = _prepad_order(dataset_size, global_batch, seed)
    if len(reference) != len(prepad):
        raise RuntimeError("Reference and pre-padded epochs have different lengths")

    reference_batches = _batch_sets(reference, global_batch)
    prepad_batches = _batch_sets(prepad, global_batch)
    position_matches = sum(left == right for left, right in zip(reference, prepad))
    batch_overlaps = [
        len(left & right)
        for left, right in zip(reference_batches, prepad_batches)
    ]
    exact_batches = sum(
        left == right for left, right in zip(reference_batches, prepad_batches)
    )
    return {
        "dataset_size": dataset_size,
        "global_batch_size": global_batch,
        "seed": seed,
        "padding_samples": len(reference) - dataset_size,
        "optimizer_steps": len(reference_batches),
        "same_position_count": position_matches,
        "same_position_fraction": position_matches / len(reference),
        "exact_same_global_batch_count": exact_batches,
        "exact_same_global_batch_fraction": exact_batches / len(reference_batches),
        "mean_batch_member_overlap": sum(batch_overlaps) / len(batch_overlaps),
        "mean_batch_member_overlap_fraction": (
            sum(batch_overlaps) / (len(batch_overlaps) * global_batch)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=103_288)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.dataset_size, args.global_batch_size, args.seed)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
