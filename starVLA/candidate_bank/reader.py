"""Fork-safe lazy reader for per-rank candidate-bank LMDBs."""

from __future__ import annotations

import io
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import lmdb
import torch

from .schema import (
    CandidateBankManifest,
    build_identity_hash,
    validate_candidate_record,
)
from .writer import rank_bank_path, read_candidate_bank_build_identity


def read_candidate_bank_manifest(
    root: os.PathLike[str] | str,
) -> CandidateBankManifest:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"candidate-bank manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return CandidateBankManifest.from_dict(json.load(stream))


class CandidateBankReader:
    """Read one record at a time without loading proposal tensors into RAM."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        expected_generator_checkpoint_sha256: Optional[str] = None,
        expected_generator_config_hash: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest = read_candidate_bank_manifest(root)
        identity_path = self.root / "build_identity.json"
        raw_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        identity = read_candidate_bank_build_identity(root)
        canonical_hash = build_identity_hash(identity)
        # label_protocol was appended to schema-v1 with a v2 default. Preserve
        # read compatibility with banks produced before that field existed;
        # newly built banks always hash the complete canonical identity.
        legacy_hash = hashlib.sha256(
            json.dumps(
                raw_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.manifest.build_identity_hash not in {canonical_hash, legacy_hash}:
            raise RuntimeError(
                "candidate-bank manifest does not match its immutable build identity"
            )
        if identity.world_size != self.manifest.world_size:
            raise RuntimeError("candidate-bank identity/manifest world_size mismatch")
        self.strict = bool(strict)
        if (
            expected_generator_checkpoint_sha256 is not None
            and self.manifest.generator_checkpoint_sha256
            != expected_generator_checkpoint_sha256
        ):
            raise RuntimeError("candidate bank was built by a different generator checkpoint")
        if (
            expected_generator_config_hash is not None
            and self.manifest.generator_config_hash != expected_generator_config_hash
        ):
            raise RuntimeError("candidate bank generator configuration hash mismatch")
        self._token_to_rank = {
            record.token: int(record.rank) for record in self.manifest.records
        }
        self._pid: Optional[int] = None
        self._environments: dict[int, lmdb.Environment] = {}

    def __len__(self) -> int:
        return self.manifest.num_scenes

    def tokens(self) -> tuple[str, ...]:
        return tuple(record.token for record in self.manifest.records)

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        for environment in self._environments.values():
            environment.close()
        self._environments = {}
        self._pid = pid

    def _environment(self, rank: int) -> lmdb.Environment:
        self._reset_after_fork()
        if rank not in self._environments:
            path = rank_bank_path(self.root, rank)
            if not path.is_dir():
                raise FileNotFoundError(f"candidate-bank rank database is missing: {path}")
            self._environments[rank] = lmdb.open(
                str(path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )
        return self._environments[rank]

    def get(self, token: str) -> dict[str, Any]:
        rank = self._token_to_rank.get(token)
        if rank is None:
            raise KeyError(f"candidate-bank cache miss for token {token!r}")
        with self._environment(rank).begin(write=False, buffers=True) as transaction:
            value = transaction.get(token.encode("utf-8"))
            if value is None:
                raise KeyError(
                    f"candidate-bank manifest references missing token {token!r}"
                )
            record = torch.load(
                io.BytesIO(bytes(value)), map_location="cpu", weights_only=False
            )
        if self.strict:
            validate_candidate_record(
                record,
                proposal_num=self.manifest.proposal_num,
                scene_queries=self.manifest.scene_queries,
                scene_dim=self.manifest.scene_dim,
                include_dense_memory=self.manifest.include_dense_memory,
            )
        return record

    def close(self) -> None:
        for environment in self._environments.values():
            environment.close()
        self._environments = {}
