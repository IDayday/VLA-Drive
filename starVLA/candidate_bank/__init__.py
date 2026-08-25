"""Offline Register64 proposal/metric bank."""

from .dataset import (
    CandidateBankDataset,
    build_candidate_bank_dataloader,
    candidate_bank_collate,
)
from .reader import CandidateBankReader, read_candidate_bank_manifest
from .schema import (
    CANDIDATE_BANK_SCHEMA_VERSION,
    CANDIDATE_METRICS,
    CandidateBankBuildIdentity,
    CandidateBankManifest,
    CandidateBankRecordRef,
    build_identity_hash,
    estimate_record_bytes,
    manifest_hash,
    validate_candidate_record,
)
from .writer import (
    CandidateBankWriter,
    finalize_candidate_bank,
    prepare_candidate_bank_root,
    rank_bank_path,
    read_candidate_bank_build_identity,
)

__all__ = [
    "CANDIDATE_BANK_SCHEMA_VERSION",
    "CANDIDATE_METRICS",
    "CandidateBankBuildIdentity",
    "CandidateBankDataset",
    "CandidateBankManifest",
    "CandidateBankReader",
    "CandidateBankRecordRef",
    "CandidateBankWriter",
    "build_candidate_bank_dataloader",
    "build_identity_hash",
    "candidate_bank_collate",
    "estimate_record_bytes",
    "finalize_candidate_bank",
    "manifest_hash",
    "prepare_candidate_bank_root",
    "rank_bank_path",
    "read_candidate_bank_manifest",
    "read_candidate_bank_build_identity",
    "validate_candidate_record",
]
