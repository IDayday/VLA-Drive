#!/usr/bin/env python3
"""Generate the immutable GP-SQ3D-Mix 2k navtest token subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    tokens = json.loads(source.read_text(encoding="utf-8"))
    if len(tokens) != len(set(tokens)) or args.count > len(tokens):
        raise ValueError("source tokens must be unique and contain the requested count")
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(tokens)), args.count))
    selected = [tokens[index] for index in selected_indices]
    manifest = {
        "schema_version": 1,
        "source_datalist": source.name,
        "source_datalist_sha256": sha256(source),
        "source_sample_count": len(tokens),
        "subset_sample_count": len(selected),
        "selection_seed": args.seed,
        "selection_algorithm": "python_random_sample_then_source_order",
        "token_list_file": output.name,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite fixed split artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
    output_tmp.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    os.replace(output_tmp, output)
    manifest["token_list_sha256"] = sha256(output)
    manifest_tmp = manifest_path.with_name(
        manifest_path.name + f".tmp-{os.getpid()}"
    )
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, manifest_path)


if __name__ == "__main__":
    main()
