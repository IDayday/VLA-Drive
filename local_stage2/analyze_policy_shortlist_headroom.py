#!/usr/bin/env python3
"""Measure Base plus independent-scorer shortlist oracle headroom.

This is an offline held-out diagnostic.  Target scores are used only to report
the upper bound of a shortlist fixed by deployable scorer outputs; they are not
an inference input or a candidate-selection rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import torch


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _parse_top_k(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",")}))
    if not values or values[0] <= 0 or values[-1] > 64:
        raise ValueError("top-k values must lie in [1, 64]")
    return values


def _load_base_indices(path: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = str(row["token"])
            if token in result:
                raise RuntimeError(f"duplicate selection token: {token}")
            result[token] = int(row["base_index"])
    return result


def _load_targets(root: Path, wanted: set[str]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("**/chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        valid = payload["valid_mask"].bool()
        factors = payload["target_factors"].float()
        if factors.shape[1:] != (64, 7):
            raise RuntimeError(f"unexpected target shape in {path}: {factors.shape}")
        for index, raw_token in enumerate(payload["tokens"]):
            token = str(raw_token)
            if token not in wanted or not bool(valid[index]):
                continue
            if token in result:
                raise RuntimeError(f"duplicate target token: {token}")
            result[token] = factors[index, :, -1]
    return result


def _load_scores(root: Path) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("**/chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scores = payload["scores"].float()
        if scores.shape[1:] != (64,):
            raise RuntimeError(f"unexpected score shape in {path}: {scores.shape}")
        for index, raw_token in enumerate(payload["tokens"]):
            token = str(raw_token)
            if token in result:
                raise RuntimeError(f"duplicate scorer token: {token}")
            result[token] = scores[index]
    return result


def _manifest_hashes(root: Path) -> Dict[str, str]:
    paths: Iterable[Path] = sorted(
        set(root.glob("**/lineage.json")) | set(root.glob("**/manifest.json"))
    )
    return {str(path.relative_to(root)): _sha256(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", default="1,2,4,8,16,32,64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.score_root, args.label_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.selection_csv.is_file():
        raise FileNotFoundError(args.selection_csv)
    top_k = _parse_top_k(args.top_k)
    base_indices = _load_base_indices(args.selection_csv)
    targets = _load_targets(args.label_root, set(base_indices))
    scores = _load_scores(args.score_root)
    expected = set(base_indices)
    if set(targets) != expected or set(scores) != expected:
        raise RuntimeError(
            "token-set mismatch: "
            f"selection={len(expected)}, targets={len(targets)}, scores={len(scores)}"
        )

    base_sum = 0.0
    oracle_sum = 0.0
    shortlist_sum = {value: 0.0 for value in top_k}
    oracle_hits = {value: 0 for value in top_k}
    for token, base_index in base_indices.items():
        target = targets[token]
        order = scores[token].argsort(descending=True)
        oracle_index = int(target.argmax())
        base_sum += float(target[base_index])
        oracle_sum += float(target[oracle_index])
        for value in top_k:
            shortlist = torch.cat(
                (torch.tensor([base_index], dtype=torch.long), order[:value])
            ).unique()
            shortlist_sum[value] += float(target[shortlist].max())
            oracle_hits[value] += int(bool((shortlist == oracle_index).any()))

    count = len(base_indices)
    base_mean = base_sum / count
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_scope": "offline_heldout_shortlist_oracle",
        "inference_uses_target_score": False,
        "scene_count": count,
        "base_selected_pdms": base_mean,
        "full_candidate_oracle_pdms": oracle_sum / count,
        "shortlists": {
            str(value): {
                "base_plus_top_k_union_oracle_pdms": shortlist_sum[value] / count,
                "gain_over_base": shortlist_sum[value] / count - base_mean,
                "full_oracle_candidate_recall": oracle_hits[value] / count,
            }
            for value in top_k
        },
        "score_root": str(args.score_root.resolve()),
        "score_root_manifest_sha256": _manifest_hashes(args.score_root),
        "selection_csv": str(args.selection_csv.resolve()),
        "selection_csv_sha256": _sha256(args.selection_csv),
        "label_root": str(args.label_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
