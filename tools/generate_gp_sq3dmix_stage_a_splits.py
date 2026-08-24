#!/usr/bin/env python3
"""Create disjoint immutable Stage-A train/model-selection/final-gate splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SPLITS = (("train", 2000), ("model_selection", 500), ("final_gate", 500))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir)
    tokens = json.loads(source.read_text(encoding="utf-8"))
    total = sum(size for _, size in SPLITS)
    if len(tokens) != len(set(tokens)) or len(tokens) < total:
        raise ValueError("source datalist is too small or contains duplicate tokens")
    indices = list(range(len(tokens)))
    random.Random(args.seed).shuffle(indices)
    offset = 0
    selected = {}
    for name, size in SPLITS:
        chosen = sorted(indices[offset : offset + size])
        selected[name] = [tokens[index] for index in chosen]
        offset += size
    manifest = {
        "schema_version": 1,
        "source_datalist": source.name,
        "source_datalist_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sample_count": len(tokens),
        "selection_seed": args.seed,
        "splits": {name: len(values) for name, values in selected.items()},
        "pairwise_disjoint": True,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    paths = {
        name: output_dir / f"gp_sq3dmix_stage_a_{name}.json" for name in selected
    }
    manifest_path = output_dir / "gp_sq3dmix_stage_a_splits.manifest.json"
    if any(path.exists() for path in (*paths.values(), manifest_path)):
        raise FileExistsError("Refusing to overwrite Stage-A split artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in selected.items():
        path = paths[name]
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        manifest.setdefault("token_list_sha256", {})[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
