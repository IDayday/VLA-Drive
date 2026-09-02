"""Streaming access to the immutable DrivoR current-observation replay cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import torch


def stable_shard(token: str, count: int) -> int:
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


@dataclass(frozen=True)
class ReplayBatch:
    tokens: List[str]
    log_names: List[str]
    visual_tokens: torch.Tensor
    status_feature: torch.Tensor


def replay_manifests(root: Path) -> List[Path]:
    return sorted(root.glob("*/manifest.json"))


def validate_replay_lineage(root: Path, expected_checkpoint_sha256: str) -> Dict[str, object]:
    manifests = replay_manifests(root)
    if not manifests:
        raise FileNotFoundError(f"No replay manifests under {root}")
    payloads = [json.loads(path.read_text()) for path in manifests]
    hashes = {str(value["checkpoint_sha256"]) for value in payloads}
    if hashes != {expected_checkpoint_sha256}:
        raise RuntimeError(f"Replay checkpoint mismatch: {sorted(hashes)}")
    total = sum(int(value["scene_count"]) for value in payloads)
    return {
        "root": str(root.resolve()),
        "manifest_count": len(manifests),
        "scene_count": total,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "producer_sha256": sorted({value["producer_sha256"] for value in payloads}),
    }


def iter_replay_batches(
    root: Path,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
    max_scenes: int = 0,
) -> Iterator[ReplayBatch]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    emitted = 0
    for path in sorted(root.glob("*/chunk_*.pt")):
        payload = torch.load(path, map_location="cpu")
        tokens = [str(value) for value in payload["tokens"]]
        log_names = [str(value) for value in payload["log_names"]]
        keep = [
            index
            for index, token in enumerate(tokens)
            if stable_shard(token, shard_count) == shard_index
        ]
        if max_scenes:
            keep = keep[: max(0, max_scenes - emitted)]
        if keep:
            indices = torch.as_tensor(keep, dtype=torch.long)
            yield ReplayBatch(
                tokens=[tokens[index] for index in keep],
                log_names=[log_names[index] for index in keep],
                visual_tokens=payload["visual_tokens"].index_select(0, indices).float(),
                status_feature=payload["status_feature"].index_select(0, indices).float(),
            )
            emitted += len(keep)
        if max_scenes and emitted >= max_scenes:
            return
