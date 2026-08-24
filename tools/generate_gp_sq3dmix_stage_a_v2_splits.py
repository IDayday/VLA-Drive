#!/usr/bin/env python3
"""Create immutable disjoint 8k/1k/1k Stage-A-v2 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path


SPLITS = (("train", 8000), ("model_selection", 1000), ("final_gate", 1000))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    tokens = json.loads(source.read_text(encoding="utf-8"))
    total = sum(size for _, size in SPLITS)
    if (
        not isinstance(tokens, list)
        or len(tokens) < total
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError("source datalist is too small or contains duplicates")
    indices = list(range(len(tokens)))
    random.Random(args.seed).shuffle(indices)
    selected = {}
    offset = 0
    for name, size in SPLITS:
        # Restore source order inside each split.  Selection is random and
        # immutable; evaluation/training iteration remains deterministic.
        chosen = sorted(indices[offset : offset + size])
        selected[name] = [tokens[index] for index in chosen]
        offset += size
    if any(
        set(selected[left]).intersection(selected[right])
        for left, _ in SPLITS
        for right, _ in SPLITS
        if left < right
    ):
        raise RuntimeError("Stage-A-v2 splits are not pairwise disjoint")
    manifest = {
        "schema_version": 2,
        "source_datalist": str(source),
        "source_datalist_sha256": sha256_file(source),
        "source_sample_count": len(tokens),
        "selection_seed": args.seed,
        "splits": {name: len(values) for name, values in selected.items()},
        "pairwise_disjoint": True,
        "code_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    paths = {
        name: output_dir / f"gp_sq3dmix_stage_a_v2_{name}.json"
        for name in selected
    }
    manifest_path = output_dir / "gp_sq3dmix_stage_a_v2_splits.manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if any(path.exists() for path in (*paths.values(), manifest_path)):
        raise FileExistsError("Refusing to overwrite Stage-A-v2 split artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in selected.items():
        atomic_json(paths[name], values)
        manifest.setdefault("token_list_sha256", {})[name] = sha256_file(paths[name])
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
