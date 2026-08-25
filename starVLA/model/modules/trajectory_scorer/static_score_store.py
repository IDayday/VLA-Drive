# SPDX-License-Identifier: Apache-2.0
# Cache layout compatibility is adapted from William-Yao-2000/DriveSuprim
# commit 80fe792d7654a596d92e20d030d1650f6f605c02,
# navsim/agents/drivesuprim/drivesuprim_agent.py and
# navsim/agents/tools/gen_vocab_full_score.py.  This adapter adds lazy
# per-token files, strict schemas, and a worker-local LRU cache.

"""Lazy CPU store for per-scene static-vocabulary metric targets."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .losses import SUPRIM_METRICS


class StaticVocabScoreStore:
    """Load only current-batch ``[vocab_size]`` score rows onto the device.

    Supported production layouts are a donor aggregate ``.pkl`` mapping token
    to metric dictionaries, or a directory containing ``<token>.npz`` /
    ``<token>.pkl`` (optionally below ``split/`` or a two-character shard).
    """

    def __init__(
        self,
        cache_root: str,
        *,
        split: str = "train",
        vocab_size: int = 8192,
        cache_size: int = 64,
        mmap: bool = True,
        include_aggregate_score: bool = False,
    ) -> None:
        if not cache_root:
            raise FileNotFoundError("static_score_store.cache_root is required")
        self.root = Path(cache_root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"static score cache does not exist: {self.root}")
        if vocab_size <= 0 or cache_size < 0:
            raise ValueError("vocab_size must be positive and cache_size non-negative")
        self.split = str(split)
        self.vocab_size = int(vocab_size)
        self.cache_size = int(cache_size)
        self.mmap = bool(mmap)
        self.include_aggregate_score = bool(include_aggregate_score)
        self._cache: OrderedDict[str, Dict[str, Tensor]] = OrderedDict()
        self._aggregate: Mapping[str, object] | None = None

    def _candidate_paths(self, token: str) -> list[Path]:
        suffixes = (".npz", ".pkl", ".pickle", ".pt", ".pth")
        bases = (
            self.root / self.split / token,
            self.root / token[:2] / token,
            self.root / token,
        )
        return [base.with_suffix(suffix) for base in bases for suffix in suffixes]

    @staticmethod
    def _as_mapping(value: object, token: str) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            if "scores" in value and isinstance(value["scores"], Mapping):
                return value["scores"]
            return value
        raise TypeError(f"static score entry for token {token!r} is not a mapping")

    def _load_file(self, path: Path, token: str) -> Mapping[str, object]:
        if path.suffix == ".npz":
            with np.load(
                path, allow_pickle=False, mmap_mode="r" if self.mmap else None
            ) as data:
                return {key: np.asarray(data[key]) for key in data.files}
        if path.suffix in {".pt", ".pth"}:
            return self._as_mapping(
                torch.load(path, map_location="cpu", weights_only=True), token
            )
        with path.open("rb") as stream:
            return self._as_mapping(pickle.load(stream), token)

    def _load_aggregate(self) -> Mapping[str, object]:
        if self._aggregate is None:
            if self.root.suffix not in {".pkl", ".pickle"} or not self.root.is_file():
                raise RuntimeError("aggregate cache loader called for a directory")
            with self.root.open("rb") as stream:
                loaded = pickle.load(stream)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"aggregate static cache is not a mapping: {self.root}")
            self._aggregate = loaded
        return self._aggregate

    def _validate_entry(
        self, token: str, raw: Mapping[str, object], source: Path
    ) -> Dict[str, Tensor]:
        missing = set(SUPRIM_METRICS).difference(raw)
        if missing:
            raise KeyError(
                f"static score token {token!r} in {source} is missing keys {sorted(missing)}"
            )
        result: Dict[str, Tensor] = {}
        for name in SUPRIM_METRICS:
            value = torch.as_tensor(np.asarray(raw[name])).detach().to(dtype=torch.float32)
            value = value.squeeze()
            if tuple(value.shape) != (self.vocab_size,):
                raise ValueError(
                    f"static score {token}/{name} has shape {tuple(value.shape)}, "
                    f"expected ({self.vocab_size},)"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"static score {token}/{name} contains NaN or Inf")
            result[name] = value.contiguous()
        aggregate_key = (
            next(
                (name for name in ("aggregate_score", "pdm_score") if name in raw),
                None,
            )
            if self.include_aggregate_score
            else None
        )
        if aggregate_key is not None:
            value = torch.as_tensor(np.asarray(raw[aggregate_key])).detach().float().squeeze()
            if tuple(value.shape) != (self.vocab_size,):
                raise ValueError(
                    f"static score {token}/{aggregate_key} has shape {tuple(value.shape)}, "
                    f"expected ({self.vocab_size},)"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"static score {token}/{aggregate_key} contains NaN or Inf")
            result["aggregate_score"] = value.contiguous()
        return result

    def _get_cpu(self, token: str) -> Dict[str, Tensor]:
        token = str(token)
        if token in self._cache:
            value = self._cache.pop(token)
            self._cache[token] = value
            return value
        if self.root.is_file():
            aggregate = self._load_aggregate()
            if token not in aggregate:
                raise KeyError(
                    f"static score cache {self.root} has no entry for token {token!r}"
                )
            raw = self._as_mapping(aggregate[token], token)
            source = self.root
        else:
            candidates = self._candidate_paths(token)
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                expected = ", ".join(str(path) for path in candidates[:5])
                raise FileNotFoundError(
                    f"static score cache is missing token {token!r}; expected one of: {expected}"
                )
            raw = self._load_file(source, token)
        value = self._validate_entry(token, raw, source)
        if self.cache_size:
            self._cache[token] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def get(
        self,
        tokens: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """Return current-batch named targets as ``[B,vocab_size]`` tensors."""

        if not tokens:
            raise ValueError("static score lookup requires at least one token")
        rows = [self._get_cpu(str(token)) for token in tokens]
        names = list(SUPRIM_METRICS)
        aggregate_presence = ["aggregate_score" in row for row in rows]
        if any(aggregate_presence) and not all(aggregate_presence):
            raise ValueError("static score batch mixes rows with and without aggregate_score")
        if all(aggregate_presence):
            names.append("aggregate_score")
        return {
            name: torch.stack([row[name] for row in rows], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
            for name in names
        }
