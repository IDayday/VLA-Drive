"""Bank-only dataset for DrivoR and optional DriveSuprim training."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .reader import CandidateBankReader


class CandidateBankDataset(Dataset):
    """Lazy LMDB dataset; it never imports raw-image, Qwen, or NAVSIM code."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        expected_generator_checkpoint_sha256: Optional[str] = None,
        expected_generator_config_hash: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        self.reader = CandidateBankReader(
            root,
            expected_generator_checkpoint_sha256=expected_generator_checkpoint_sha256,
            expected_generator_config_hash=expected_generator_config_hash,
            strict=strict,
        )
        self._records = tuple(self.reader.manifest.records)

    @property
    def manifest(self):
        return self.reader.manifest

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.reader.get(self._records[index].token)

    def close(self) -> None:
        self.reader.close()


def candidate_bank_collate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("candidate-bank batch cannot be empty")
    result: dict[str, Any] = {
        "token": [str(record["token"]) for record in records],
        "ego_state": torch.stack([record["ego_state"] for record in records]),
        "scene_global_tokens": torch.stack(
            [record["scene_global_tokens"] for record in records]
        ),
        "proposals": torch.stack([record["proposals"] for record in records]),
        "gt_trajectory": torch.stack(
            [record["gt_trajectory"] for record in records]
        ),
        "metrics": {
            name: torch.stack([record["metrics"][name] for record in records])
            for name in records[0]["metrics"]
        },
    }
    if "scene_dense_memory" in records[0]:
        max_length = max(
            int(record["scene_dense_memory"].shape[0]) for record in records
        )
        scene_dim = int(records[0]["scene_dense_memory"].shape[-1])
        memory = records[0]["scene_dense_memory"].new_zeros(
            (len(records), max_length, scene_dim)
        )
        mask = torch.zeros(
            (len(records), max_length), dtype=torch.bool
        )
        for index, record in enumerate(records):
            length = int(record["scene_dense_memory"].shape[0])
            memory[index, :length] = record["scene_dense_memory"]
            mask[index, :length] = record["attention_mask"]
        result["scene_dense_memory"] = memory
        result["attention_mask"] = mask
    return result


def build_candidate_bank_dataloader(
    dataset: CandidateBankDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    distributed: bool = False,
) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    )
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=candidate_bank_collate,
        **kwargs,
    )
