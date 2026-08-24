#!/usr/bin/env python3
"""Derive fixed smoke prefixes from immutable Stage-A-v2 splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    args = parser.parse_args()
    root = Path(args.split_dir).resolve()
    specifications = (
        ("train", 256, "gp_sq3dmix_stage_a_v2_smoke_train_256.json"),
        ("model_selection", 128, "gp_sq3dmix_stage_a_v2_smoke_selection_128.json"),
    )
    for source_name, count, output_name in specifications:
        source = root / f"gp_sq3dmix_stage_a_v2_{source_name}.json"
        output = root / output_name
        if output.exists():
            raise FileExistsError(output)
        tokens = json.loads(source.read_text(encoding="utf-8"))
        if len(tokens) < count:
            raise RuntimeError(f"{source} contains fewer than {count} samples")
        atomic_json(output, tokens[:count])


if __name__ == "__main__":
    main()
