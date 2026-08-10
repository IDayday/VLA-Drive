"""Build a deterministic token datalist from processed NAVSIM metadata."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def collect_tokens(meta_dir: Path) -> list[str]:
    """Return sorted unique tokens for ``meta_dir/{token}.pkl`` files."""
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"processed metadata directory does not exist: {meta_dir}")
    tokens = sorted(
        path.stem
        for path in meta_dir.glob("*.pkl")
        if not path.name.endswith("-depth.pkl")
    )
    if not tokens:
        raise ValueError(f"no processed NAVSIM metadata found in {meta_dir}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"duplicate tokens found in {meta_dir}")
    return tokens


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically serialize JSON in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = collect_tokens(args.meta_dir)
    atomic_write_json(args.output, tokens)
    print(f"wrote {len(tokens)} tokens to {args.output}")


if __name__ == "__main__":
    main()
