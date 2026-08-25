"""Per-rank resumable LMDB writer for Register64 candidate banks."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import lmdb
import torch

from .schema import (
    CandidateBankBuildIdentity,
    CandidateBankManifest,
    CandidateBankRecordRef,
    build_identity_hash,
    validate_candidate_record,
)


BANK_IDENTITY_FILENAME = "build_identity.json"
_RANK_ARTIFACT = re.compile(
    r"rank_\d{5}\.(?:lmdb|complete\.json|report\.json)$"
)
_ROOT_ARTIFACTS = {
    BANK_IDENTITY_FILENAME,
    "manifest.json",
    "candidate_bank_report.json",
}


def rank_bank_path(root: os.PathLike[str] | str, rank: int) -> Path:
    return Path(root) / f"rank_{int(rank):05d}.lmdb"


def rank_completion_path(root: os.PathLike[str] | str, rank: int) -> Path:
    return Path(root) / f"rank_{int(rank):05d}.complete.json"


def _serialize(record: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(record), stream)
    return stream.getvalue()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _validate_split_root(root: Path, split: str) -> Path:
    resolved = root.expanduser().resolve()
    if resolved.name != split:
        raise ValueError(
            f"candidate-bank root must end in the split name {split!r}: {resolved}"
        )
    forbidden = {Path("/").resolve(), Path.home().resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"refusing unsafe candidate-bank root: {resolved}")
    return resolved


def _known_bank_artifact(path: Path) -> bool:
    return path.name in _ROOT_ARTIFACTS or bool(_RANK_ARTIFACT.fullmatch(path.name))


def read_candidate_bank_build_identity(
    root: os.PathLike[str] | str,
) -> CandidateBankBuildIdentity:
    path = Path(root) / BANK_IDENTITY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"candidate-bank build identity is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return CandidateBankBuildIdentity.from_dict(json.load(stream))


def prepare_candidate_bank_root(
    root: os.PathLike[str] | str,
    *,
    identity: CandidateBankBuildIdentity,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    """Prepare one exact split root and enforce immutable resume topology.

    ``--overwrite`` only removes recognized candidate-bank artifacts below the
    exact ``.../<split>`` directory. Unknown files fail closed.
    """

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    identity.validate()
    root = _validate_split_root(Path(root), identity.split)
    root.mkdir(parents=True, exist_ok=True)
    children = list(root.iterdir())
    unknown = [path for path in children if not _known_bank_artifact(path)]
    if unknown:
        raise RuntimeError(
            "candidate-bank root contains unrelated artifacts: "
            + ", ".join(str(path) for path in sorted(unknown))
        )

    identity_path = root / BANK_IDENTITY_FILENAME
    if overwrite:
        for path in children:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    elif children:
        if not resume:
            raise FileExistsError(
                f"candidate-bank root already contains artifacts; pass --resume "
                f"or --overwrite: {root}"
            )
        existing = read_candidate_bank_build_identity(root)
        if existing != identity:
            raise RuntimeError(
                "candidate-bank resume identity mismatch; generator, split, "
                "world size, data list, metric cache, storage, or code changed"
            )
    _atomic_json(identity_path, identity.to_dict())
    return build_identity_hash(identity)


class CandidateBankWriter:
    """One writer owns one LMDB and never mixes incompatible build identities."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        rank: int,
        proposal_num: int = 64,
        scene_queries: int = 16,
        scene_dim: int = 256,
        include_dense_memory: bool = False,
        map_size_bytes: int = 1 << 40,
        commit_interval: int = 16,
        resume: bool = False,
        overwrite: bool = False,
        expected_build_identity_hash: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.rank = int(rank)
        self.path = rank_bank_path(root, rank)
        self.proposal_num = int(proposal_num)
        self.scene_queries = int(scene_queries)
        self.scene_dim = int(scene_dim)
        self.include_dense_memory = bool(include_dense_memory)
        self.overwrite = bool(overwrite)
        self.expected_build_identity_hash = expected_build_identity_hash
        completion_path = rank_completion_path(self.root, self.rank)
        if self.path.exists() and overwrite:
            if self.path.parent.resolve() != self.root.resolve() or not re.fullmatch(
                r"rank_\d{5}\.lmdb", self.path.name
            ):
                raise RuntimeError(f"refusing unsafe rank-bank overwrite: {self.path}")
            shutil.rmtree(self.path)
            completion_path.unlink(missing_ok=True)
        if self.path.exists() and not (resume or overwrite):
            raise FileExistsError(
                f"candidate-bank rank already exists; pass --resume or --overwrite: {self.path}"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            subdir=True,
            map_size=int(map_size_bytes),
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        self.transaction = self.env.begin(write=True)
        self.commit_interval = max(1, int(commit_interval))
        self.pending = 0
        self.written = 0
        self.skipped = 0
        # Recover committed keys from LMDB itself instead of trusting a prior
        # completion marker.  This also makes --resume safe after a process was
        # interrupted between the last transaction commit and close().
        self.tokens: list[str] = [
            bytes(key).decode("utf-8")
            for key in self.transaction.cursor().iternext(
                keys=True, values=False
            )
        ]

    def contains(self, token: str) -> bool:
        return self.transaction.get(token.encode("utf-8")) is not None

    def put(self, record: Mapping[str, Any]) -> bool:
        validate_candidate_record(
            record,
            proposal_num=self.proposal_num,
            scene_queries=self.scene_queries,
            scene_dim=self.scene_dim,
            include_dense_memory=self.include_dense_memory,
        )
        token = str(record["token"])
        key = token.encode("utf-8")
        if not self.overwrite and self.transaction.get(key) is not None:
            self.skipped += 1
            return False
        cpu_record = {
            name: (
                value.detach().cpu().contiguous()
                if torch.is_tensor(value)
                else {
                    metric: tensor.detach().cpu().to(dtype=torch.float32).contiguous()
                    for metric, tensor in value.items()
                }
                if name == "metrics"
                else value
            )
            for name, value in record.items()
        }
        self.transaction.put(key, _serialize(cpu_record), overwrite=True)
        self.tokens.append(token)
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

    def close(self, *, complete: bool = True) -> None:
        if self.transaction is not None:
            self.transaction.commit()
            self.transaction = None
        self.env.sync()
        self.env.close()
        if complete:
            _atomic_json(
                rank_completion_path(self.root, self.rank),
                {
                    "rank": self.rank,
                    "tokens": sorted(set(self.tokens)),
                    "written": self.written,
                    "skipped": self.skipped,
                    "build_identity_hash": self.expected_build_identity_hash,
                },
            )

    def __enter__(self) -> "CandidateBankWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close(complete=True)
        else:
            if self.transaction is not None:
                self.transaction.abort()
                self.transaction = None
            self.env.close()


def finalize_candidate_bank(
    root: os.PathLike[str] | str,
    *,
    manifest_fields: Mapping[str, Any],
    world_size: int,
    expected_build_identity_hash: str | None = None,
) -> CandidateBankManifest:
    root = Path(root)
    records: list[CandidateBankRecordRef] = []
    for rank in range(int(world_size)):
        completion = rank_completion_path(root, rank)
        if not completion.is_file():
            raise FileNotFoundError(f"candidate-bank rank is incomplete: {completion}")
        with completion.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if int(payload.get("rank", -1)) != rank:
            raise RuntimeError(f"candidate-bank rank completion mismatch: {completion}")
        if (
            expected_build_identity_hash is not None
            and payload.get("build_identity_hash") != expected_build_identity_hash
        ):
            raise RuntimeError(
                f"candidate-bank rank identity mismatch: {completion}"
            )
        records.extend(
            CandidateBankRecordRef(token=str(token), rank=rank)
            for token in payload.get("tokens", [])
        )
    records.sort(key=lambda value: value.token)
    fields = dict(manifest_fields)
    fields.update(
        num_scenes=len(records),
        records=records,
        world_size=int(world_size),
        build_identity_hash=(
            expected_build_identity_hash or fields.get("build_identity_hash", "")
        ),
    )
    manifest = CandidateBankManifest(**fields)
    manifest.validate()
    _atomic_json(root / "manifest.json", manifest.to_dict())
    return manifest
