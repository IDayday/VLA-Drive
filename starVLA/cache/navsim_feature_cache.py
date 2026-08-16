"""Shared-storage feature cache helpers for NAVSIM v1.

The cache is split by component and by the rank that generated the record.
Each rank owns one LMDB, so a 16-device pre-cache job has no cross-rank write
lock.  Training derives the owning LMDB from the stable dataset index.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import lmdb
import torch


CACHE_SCHEMA_VERSION = 1
CACHE_COMPONENTS = ("qwen", "wan", "ppd", "agent_dino")
CACHE_SCHEMA_VERSIONS = {
    "qwen": (1,),
    "wan": (1,),
    "ppd": (1,),
    "agent_dino": (1, 2, 3),
}

ROBOT_HISTORY_TOKEN = "<robot_history_action_0>"
RGB_QUERY_TOKENS = tuple(f"<2d_world_{index}>" for index in range(64))
GS_QUERY_TOKENS = tuple(f"<3d_world_{index}>" for index in range(64))
MINE_AGENT_QUERY_TOKENS = tuple(f"<mine_agent_{index}>" for index in range(32))
REWARD_QUERY_TOKENS = ("<reward_0>",)


def action_query_tokens(count: int) -> tuple[str, ...]:
    return tuple(f"<robot_action_{index}>" for index in range(count))


def append_world_action_tokens(
    instruction: str,
    act_token_count: int,
    w_depth: bool,
    with_mine_agent: bool = False,
) -> str:
    """Append the exact released special-token suffix to one instruction."""
    history = ROBOT_HISTORY_TOKEN
    rgb = "".join(RGB_QUERY_TOKENS)
    gs = "".join(GS_QUERY_TOKENS)
    mine_agent = "".join(MINE_AGENT_QUERY_TOKENS) if with_mine_agent else ""
    actions = "".join(action_query_tokens(act_token_count))
    reward = "".join(REWARD_QUERY_TOKENS)
    if w_depth:
        suffix = f" {history}{gs}{rgb}{mine_agent}{actions}{reward}"
    else:
        suffix = f" {history}{rgb}{gs}{mine_agent}{actions}{reward}"
    return instruction + suffix


def parse_components(value: Optional[str]) -> tuple[str, ...]:
    if value is None:
        return CACHE_COMPONENTS
    components = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(components) - set(CACHE_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown NAVSIM cache component(s): {unknown}")
    return components


def _serialize(payload: Mapping[str, torch.Tensor]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _deserialize(value: bytes) -> Dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def component_dir(cache_root: os.PathLike[str] | str, component: str) -> Path:
    if component not in CACHE_COMPONENTS:
        raise ValueError(f"Unsupported cache component: {component}")
    return Path(cache_root) / component


def rank_db_path(cache_root: os.PathLike[str] | str, component: str, rank: int) -> Path:
    return component_dir(cache_root, component) / f"rank_{rank:05d}.lmdb"


class RankCacheWriter:
    """Single-writer LMDB used by one pre-cache rank."""

    def __init__(
        self,
        cache_root: os.PathLike[str] | str,
        component: str,
        rank: int,
        map_size_bytes: int,
        commit_interval: int = 16,
    ) -> None:
        self.path = rank_db_path(cache_root, component, rank)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            subdir=True,
            map_size=map_size_bytes,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=64,
        )
        self.commit_interval = max(1, int(commit_interval))
        self.transaction = self.env.begin(write=True)
        self.pending = 0
        self.written = 0
        self.skipped = 0

    def contains(self, token: str) -> bool:
        return self.transaction.get(token.encode("utf-8")) is not None

    def put(self, token: str, payload: Mapping[str, torch.Tensor], overwrite: bool = False) -> bool:
        key = token.encode("utf-8")
        if not overwrite and self.transaction.get(key) is not None:
            self.skipped += 1
            return False
        self.transaction.put(key, _serialize(payload), overwrite=True)
        self.pending += 1
        self.written += 1
        if self.pending >= self.commit_interval:
            self.commit()
        return True

    def commit(self) -> None:
        if self.transaction is None:
            return
        self.transaction.commit()
        self.transaction = self.env.begin(write=True)
        self.pending = 0

    def close(self) -> None:
        if self.transaction is not None:
            self.transaction.commit()
            self.transaction = None
        self.env.sync()
        self.env.close()

    def __enter__(self) -> "RankCacheWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            if self.transaction is not None:
                self.transaction.abort()
                self.transaction = None
            self.env.close()


class NavsimFeatureCacheReader:
    """Fork-safe, lazy LMDB reader used inside DataLoader workers."""

    def __init__(
        self,
        cache_root: os.PathLike[str] | str,
        components: Iterable[str] = CACHE_COMPONENTS,
        strict: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.components = tuple(components)
        self.strict = bool(strict)
        self.manifests: Dict[str, dict] = {}
        for component in self.components:
            manifest_path = component_dir(self.cache_root, component) / "manifest.json"
            if not manifest_path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing completed cache manifest: {manifest_path}")
                continue
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            allowed_versions = CACHE_SCHEMA_VERSIONS.get(component, (CACHE_SCHEMA_VERSION,))
            if manifest.get("schema_version") not in allowed_versions:
                raise RuntimeError(
                    f"Cache schema mismatch for {component}: "
                    f"expected one of {allowed_versions}, found {manifest.get('schema_version')}"
                )
            if not manifest.get("complete", False):
                raise RuntimeError(f"Cache manifest is not marked complete: {manifest_path}")
            self.manifests[component] = manifest
        self._pid: Optional[int] = None
        self._environments: Dict[tuple[str, int], lmdb.Environment] = {}
        self._token_rank_cache: Dict[tuple[str, str], int] = {}

    def has_component(self, component: str) -> bool:
        return component in self.manifests

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        for environment in self._environments.values():
            environment.close()
        self._environments = {}
        self._pid = pid

    def _environment(self, component: str, rank: int) -> lmdb.Environment:
        self._reset_after_fork()
        key = (component, rank)
        if key not in self._environments:
            path = rank_db_path(self.cache_root, component, rank)
            if not path.is_dir():
                raise FileNotFoundError(f"Missing cache rank database: {path}")
            self._environments[key] = lmdb.open(
                str(path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        return self._environments[key]

    def get(self, component: str, sample_index: int, token: str) -> Optional[Dict[str, torch.Tensor]]:
        manifest = self.manifests.get(component)
        if manifest is None:
            if self.strict:
                raise KeyError(f"Cache component {component!r} is unavailable")
            return None

        token_key = (component, token)
        if token_key in self._token_rank_cache:
            owner_rank = self._token_rank_cache[token_key]
            environment = self._environment(component, owner_rank)
            with environment.begin(write=False, buffers=True) as transaction:
                value = transaction.get(token.encode("utf-8"))
            if value is not None:
                return _deserialize(bytes(value))

        world_size = int(manifest["world_size"])
        owner_rank = int(sample_index) % world_size
        search_ranks = [owner_rank] + [rank for rank in range(world_size) if rank != owner_rank]
        for rank in search_ranks:
            environment = self._environment(component, rank)
            with environment.begin(write=False, buffers=True) as transaction:
                value = transaction.get(token.encode("utf-8"))
            if value is not None:
                self._token_rank_cache[token_key] = rank
                return _deserialize(bytes(value))

        if self.strict:
            raise KeyError(
                f"Cache miss: component={component} token={token} "
                f"sample_index={sample_index} owner_rank={owner_rank}"
            )
        return None


def write_rank_completion(
    cache_root: os.PathLike[str] | str,
    component: str,
    rank: int,
    payload: Mapping[str, object],
) -> Path:
    path = component_dir(cache_root, component) / f"rank_{rank:05d}.complete.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path


def write_manifest(
    cache_root: os.PathLike[str] | str,
    component: str,
    payload: Mapping[str, object],
) -> Path:
    path = component_dir(cache_root, component) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest = dict(payload)
    manifest["schema_version"] = CACHE_SCHEMA_VERSION
    manifest["complete"] = True
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path
