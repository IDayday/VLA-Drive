#!/usr/bin/env python3
"""Create and strictly validate a deterministic navtrain holdout split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokens(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"source datalist must be a non-empty JSON list: {path}")
    if any(not isinstance(token, str) or not token for token in value):
        raise TypeError("datalist entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError("source datalist contains duplicate tokens")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "train.json", root / "val.json", root / "manifest.json"


def _identity(source: Path, validation_size: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "validation_size": validation_size,
        "seed": seed,
        "selection": "lowest_sha256(seed:token)",
    }


def _validate(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    train_path, val_path, manifest_path = _paths(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"split identity mismatch for {key}: {manifest.get(key)!r} != {value!r}"
            )
    train_tokens = _load_tokens(train_path)
    val_tokens = _load_tokens(val_path)
    if set(train_tokens) & set(val_tokens):
        raise RuntimeError("train and validation token sets overlap")
    source_tokens = _load_tokens(Path(identity["source"]))
    if set(train_tokens) | set(val_tokens) != set(source_tokens):
        raise RuntimeError("train/validation union differs from source datalist")
    checks = {
        "train_count": len(train_tokens),
        "val_count": len(val_tokens),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"split manifest mismatch for {key}")
    return manifest


def main() -> None:
    args = _parse_args()
    source = Path(args.source).expanduser().resolve()
    root = Path(args.output_dir).expanduser().resolve()
    tokens = _load_tokens(source)
    if not 1 <= args.validation_size < len(tokens):
        raise ValueError("validation size must be between 1 and source_size - 1")
    identity = _identity(source, args.validation_size, args.seed)
    train_path, val_path, manifest_path = _paths(root)

    if args.validate_only:
        print(json.dumps(_validate(root, identity), indent=2, sort_keys=True))
        return
    if (
        any(path.exists() for path in (train_path, val_path, manifest_path))
        and not args.overwrite
    ):
        print(json.dumps(_validate(root, identity), indent=2, sort_keys=True))
        return

    ranked = sorted(
        tokens,
        key=lambda token: _sha256_bytes(f"{args.seed}:{token}".encode("utf-8")),
    )
    validation = set(ranked[: args.validation_size])
    train_tokens = [token for token in tokens if token not in validation]
    val_tokens = [token for token in tokens if token in validation]
    _atomic_json(train_path, train_tokens)
    _atomic_json(val_path, val_tokens)
    manifest = {
        **identity,
        "source_count": len(tokens),
        "train_count": len(train_tokens),
        "val_count": len(val_tokens),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(_validate(root, identity), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
