#!/usr/bin/env python3
"""Separate direct Stage-2 batch facts from conditional layout inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def standard_optimizer_steps(
    dataset_size: int,
    world_size: int,
    per_device_batch: int,
    accumulation: int,
) -> int:
    """Model released DDP sampler + drop_last DataLoader + Lightning accumulation."""

    samples_per_rank = math.ceil(dataset_size / world_size)
    micro_batches = samples_per_rank // per_device_batch
    return math.ceil(micro_batches / accumulation)


def audit_layout(
    dataset_size: int,
    total_optimizer_steps: int,
    completed_epochs: int,
    paper_gpu_count: int,
) -> dict[str, Any]:
    if total_optimizer_steps % completed_epochs:
        raise ValueError("Total optimizer steps are not divisible by completed epochs")
    steps_per_epoch = total_optimizer_steps // completed_epochs

    effective_batch_candidates = [
        batch
        for batch in range(1, dataset_size + 1)
        if math.ceil(dataset_size / batch) == steps_per_epoch
    ]
    if len(effective_batch_candidates) != 1:
        raise ValueError(
            "The one-pass arithmetic does not identify a unique effective batch"
        )
    inferred_effective_batch = effective_batch_candidates[0]

    layouts = []
    for world_size in range(1, paper_gpu_count + 1):
        if paper_gpu_count % world_size:
            continue
        for per_device_batch in range(1, inferred_effective_batch + 1):
            for accumulation in range(1, inferred_effective_batch + 1):
                effective_batch = world_size * per_device_batch * accumulation
                if effective_batch != inferred_effective_batch:
                    continue
                steps = standard_optimizer_steps(
                    dataset_size, world_size, per_device_batch, accumulation
                )
                layouts.append(
                    {
                        "world_size": world_size,
                        "per_device_batch": per_device_batch,
                        "gradient_accumulation": accumulation,
                        "effective_batch": effective_batch,
                        "standard_release_steps_per_epoch": steps,
                        "matches_checkpoint_steps": steps == steps_per_epoch,
                        "uses_all_paper_reported_gpus": world_size == paper_gpu_count,
                    }
                )

    preferred = next(
        layout
        for layout in layouts
        if layout["world_size"] == paper_gpu_count
        and layout["per_device_batch"] == 1
        and layout["gradient_accumulation"] == 1
    )
    matching = [layout for layout in layouts if layout["matches_checkpoint_steps"]]

    return {
        "audit": "stage2_batch_layout_evidence_levels",
        "direct_artifact_facts": {
            "total_optimizer_steps": total_optimizer_steps,
            "completed_epochs": completed_epochs,
            "steps_per_epoch": steps_per_epoch,
            "checkpoint_stores_global_batch": False,
            "checkpoint_stores_per_device_batch": False,
            "checkpoint_stores_gradient_accumulation": False,
        },
        "external_inputs": {
            "local_reconstructed_training_scene_count": dataset_size,
            "paper_reported_gpu_count": paper_gpu_count,
        },
        "conditional_inference": {
            "assumptions": [
                "The private run used the same 103,288-scene concatenated set.",
                "Each epoch traversed that set once without private resampling.",
                "Checkpoint global_step counts ordinary optimizer updates.",
            ],
            "effective_batch_candidates_under_one_pass_ceiling": (
                effective_batch_candidates
            ),
            "inferred_effective_global_batch": inferred_effective_batch,
            "is_direct_fact": False,
        },
        "counterfactual_batch_32": {
            "one_pass_steps_per_epoch": math.ceil(dataset_size / 32),
            "one_pass_total_optimizer_steps": (
                math.ceil(dataset_size / 32) * completed_epochs
            ),
            "matches_checkpoint_under_one_pass": (
                math.ceil(dataset_size / 32) == steps_per_epoch
            ),
            "double_dataset_size": dataset_size * 2,
            "double_pass_steps_per_epoch": math.ceil(dataset_size * 2 / 32),
            "matches_checkpoint_if_dataset_is_repeated_twice": (
                math.ceil(dataset_size * 2 / 32) == steps_per_epoch
            ),
            "required_unpublished_mechanism": (
                "A 32-sample effective batch needs approximately two traversals "
                "of the reconstructed dataset per reported epoch, a doubled "
                "private dataset, or another nonstandard step-count mechanism."
            ),
        },
        "standard_release_layouts_with_effective_batch": layouts,
        "standard_release_layouts_matching_step_count": matching,
        "preferred_layout_hypothesis": preferred,
        "preferred_layout_is_uniquely_proven": False,
        "interpretation": (
            "Effective global batch 16 is a strong conditional reconstruction. "
            "The 16x1x1 layout is preferred because it uses all 16 GPUs reported "
            "by the paper and matches released accumulation defaults, but neither "
            "batch value is stored directly in the public checkpoint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=103_288)
    parser.add_argument("--total-optimizer-steps", type=int, default=174_312)
    parser.add_argument("--completed-epochs", type=int, default=27)
    parser.add_argument("--paper-gpu-count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = audit_layout(
        args.dataset_size,
        args.total_optimizer_steps,
        args.completed_epochs,
        args.paper_gpu_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
