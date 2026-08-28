#!/usr/bin/env python3
"""Verify worker-side image preprocessing after the parent initializes CUDA."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from navsim.planning.training.dataset import CacheOnlyDataset, drivevla_cached_collate


class Builder:
    def __init__(self, name: str):
        self.name = name

    def get_unique_name(self) -> str:
        return self.name


def main() -> None:
    mp.set_forkserver_preload(
        [
            "navsim.planning.training.dataset",
            "navsim.agents.EpisodeDrive.utils.internvl_preprocess",
            "navsim.agents.EpisodeDrive.score_module.compute_navsim_score",
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope",
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                f"<DRIVEVLA_EXTRA_{index}>" for index in range(8)
            ]
        }
    )
    cuda_probe = torch.ones(1, device="cuda:0")
    assert cuda_probe.item() == 1

    cache_root = Path(
        "/mnt/project/DriveVLA-M0-stage2/cache/feature_cache_navtrain_full"
    )
    log_name = sorted(path.name for path in cache_root.iterdir() if path.is_dir())[0]
    dataset = CacheOnlyDataset(
        str(cache_root),
        [Builder("internvl_feature")],
        [Builder("trajectory_target")],
        [log_name],
        preprocess_images=True,
        preprocess_image_dtype="bfloat16",
        pretokenize_inputs=True,
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        collate_fn=drivevla_cached_collate,
    )
    batches = 0
    for features, targets in loader:
        assert features["pixel_values"].shape == (2, 9, 3, 448, 448)
        assert features["pixel_values"].dtype == torch.bfloat16
        assert features["input_ids"].shape == (2, 2800)
        assert features["attention_mask"].shape == (2, 2800)
        assert "questions" not in features
        assert len(targets["token"]) == 2
        batches += 1
    print(f"forkserver_dataloader_after_cuda=True batches={batches}")


if __name__ == "__main__":
    main()
