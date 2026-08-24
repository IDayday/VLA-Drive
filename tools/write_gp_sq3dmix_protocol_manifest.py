#!/usr/bin/env python3
"""Atomically bind a GP-SQ3D-Mix experiment phase to immutable inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_value(raw: str) -> tuple[str, str]:
    key, separator, value = raw.partition("=")
    if not separator or not key or not value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return key, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--input", action="append", default=[], type=key_value)
    parser.add_argument("--value", action="append", default=[], type=key_value)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inputs: dict[str, dict[str, str]] = {}
    for name, raw_path in args.input:
        if name in inputs:
            raise ValueError(f"duplicate input name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    values: dict[str, str] = {}
    for name, value in args.value:
        if name in values:
            raise ValueError(f"duplicate value name: {name}")
        values[name] = value

    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "code_commit": args.code_commit,
        "inputs": inputs,
        "resolved_values": values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
