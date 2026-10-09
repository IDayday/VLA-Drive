"""Content-addressed, crash-safe artifacts. Only completion receipts permit reuse."""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import tempfile
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An artifact, dependency or execution contract is incompatible."""


class BlockedError(RuntimeError):
    """A required real resource or binding is unavailable."""


def json_bytes(value: Any) -> bytes:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    p = Path(path).resolve()
    stat = p.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _verified_file_hash(str(p), identity)


@lru_cache(maxsize=4096)
def _verified_file_hash(path, identity):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    s = Path(path).stat()
    if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns) != identity:
        raise ContractError(f"file changed during SHA256 verification: {path}")
    return h.hexdigest()


@contextmanager
def lock(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def atomic_bytes(path: str | Path, data: bytes, *, immutable: bool = False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with lock(p.with_suffix(p.suffix + ".lock")):
        if immutable and p.exists():
            if p.read_bytes() != data:
                raise ContractError(f"immutable artifact conflict: {p}")
            return
        fd, temporary = tempfile.mkstemp(prefix="." + p.name, dir=p.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, p)
            directory = os.open(p.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def atomic_json(path: str | Path, value: Any, *, immutable=False):
    atomic_bytes(path, json_bytes(value) + b"\n", immutable=immutable)


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def atomic_torch(path: str | Path, value: Any, *, immutable=False):
    import torch
    # Stream large model/Adam states once; BytesIO duplicated entire multi-GB
    # checkpoints in RAM. Serialization and exact-resume contents are unchanged.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with lock(p.with_suffix(p.suffix + ".lock")):
        fd, temporary = tempfile.mkstemp(prefix="." + p.name, dir=p.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(value, f)
                f.flush()
                os.fsync(f.fileno())
            if immutable and p.exists():
                if file_hash(p) != file_hash(temporary):
                    raise ContractError(f"immutable artifact conflict: {p}")
                return
            os.replace(temporary, p)
            directory = os.open(p.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def complete(path: str | Path, inputs: dict, outputs: list[str | Path], **details):
    receipt = {"status": "COMPLETE", "inputs_hash": digest(inputs), "inputs": inputs,
               "outputs": {str(p): file_hash(p) for p in outputs}, **details}
    atomic_json(path, receipt, immutable=True)
    return receipt


def reusable(path: str | Path, inputs: dict) -> bool:
    if not Path(path).exists():
        return False
    receipt = read_json(path)
    if receipt["status"] != "COMPLETE" or receipt["inputs_hash"] != digest(inputs):
        raise ContractError(f"stage dependency conflict: {path}; use a new output version")
    for p, expected in receipt["outputs"].items():
        if not Path(p).is_file() or file_hash(p) != expected:
            raise ContractError(f"incomplete/corrupt output: {p}")
    return True
