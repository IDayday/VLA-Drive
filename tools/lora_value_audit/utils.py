"""I/O, hashing, deterministic bootstrap, and lineage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")


def physical_log_name(log_name: str) -> str:
    return _SEGMENT_SUFFIX.sub("", str(log_name))


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_proposal_pickle(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected non-empty token dictionary: {path}")
    return payload


def token_index(tokens: Sequence[str]) -> Dict[str, int]:
    result = {str(token): index for index, token in enumerate(tokens)}
    if len(result) != len(tokens):
        raise ValueError("Duplicate scene token")
    return result


def assert_score_range(values: np.ndarray, name: str = "true_pdms") -> None:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN/Inf")
    low = float(values.min(initial=0.0))
    high = float(values.max(initial=1.0))
    if low < -1e-7 or high > 1.0 + 1e-7:
        raise ValueError(f"{name} outside [0,1]: min={low}, max={high}")


def bootstrap_mean(
    values: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260902,
    chunk_size: int = 256,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "standard_error": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(n_bootstrap, start + chunk_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(len(values)),
        "bootstrap_replicates": int(n_bootstrap),
        "seed": int(seed),
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260902,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    valid = np.isfinite(values)
    values, groups = values[valid], groups[valid]
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(unique), size=len(unique))
        means[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "mean": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "scene_count": int(len(values)),
        "cluster_count": int(len(unique)),
        "bootstrap_replicates": int(n_bootstrap),
        "seed": int(seed),
    }


def safe_divide(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / max(denominator, eps))
