"""Portable access helpers for NAVSIM metric caches used only as targets."""

from __future__ import annotations

import json
import lzma
import pickle
from pathlib import Path
from typing import Any, Iterator, Mapping


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield object rows from a UTF-8 JSONL file."""

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            yield value


def load_relative_metric_cache_index(cache_root: Path) -> dict[str, Path]:
    """Resolve a portable token-to-file index below ``cache_root``.

    The official NAVSIM metadata CSV stores absolute paths and becomes stale
    when a cache is moved.  Research scripts therefore use this relative index
    while still publishing the official CSV for evaluator compatibility.
    """

    index_path = cache_root / "metric_cache_index.json"
    with index_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise TypeError(f"metric cache index must be a mapping: {index_path}")
    resolved = {str(token): cache_root / str(relative) for token, relative in raw.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        preview = missing[:3]
        raise FileNotFoundError(f"metric cache index has missing files: {preview}")
    return resolved


def load_metric_cache(path: Path) -> Any:
    """Load one official XZ-compressed NAVSIM ``MetricCache`` object."""

    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)
