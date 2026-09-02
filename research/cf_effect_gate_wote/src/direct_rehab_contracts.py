"""Data-boundary contracts for the current-only Direct Scorer rehabilitation.

This module deliberately has no NAVSIM or model imports.  Split construction and
access-policy validation therefore happen before any scene data can be opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLIT_SCHEMA = "direct_scorer_rehab_split.v1"
ACCESS_SCHEMA = "direct_scorer_rehab_access.v1"

SPLIT_FILENAMES = {
    "train": "direct_rehab_train_1024.txt",
    "val": "direct_rehab_val_256.txt",
    "dev": "direct_rehab_dev_512.txt",
    "holdout": "direct_rehab_holdout_512.txt",
    "reserved": "future_effect_reserved_512.txt",
}

SPLIT_SLICES = {
    "train": ("train_tokens.txt", 0, 1024),
    "val": ("val_tokens.txt", 0, 256),
    "dev": ("test_tokens.txt", 200, 712),
    "holdout": ("test_tokens.txt", 712, 1224),
    "reserved": ("test_tokens.txt", 1224, 1736),
}

ALLOWED_HOLDOUT_PHASES = frozenset({"asset_generation", "final_evaluation"})
FORBIDDEN_STORE_FRAGMENT = "effect"


class DirectRehabContractError(RuntimeError):
    """Raised before a split or access boundary can be violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise DirectRehabContractError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _token_bytes(tokens: Sequence[str]) -> bytes:
    return "".join(f"{token}\n" for token in tokens).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_source_tokens(path: Path) -> tuple[str, ...]:
    """Read an original, non-rehabilitation source split exactly once."""

    tokens = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not tokens or any(not token for token in tokens):
        raise DirectRehabContractError(f"empty token/source split: {path}")
    if len(tokens) != len(set(tokens)):
        raise DirectRehabContractError(f"duplicate tokens in source split: {path}")
    return tokens


def _validate_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    names = tuple(splits)
    for index, left_name in enumerate(names):
        left = set(splits[left_name])
        for right_name in names[index + 1 :]:
            overlap = left & set(splits[right_name])
            if overlap:
                sample = sorted(overlap)[:5]
                raise DirectRehabContractError(
                    f"split overlap {left_name}/{right_name}: {sample}"
                )


def build_direct_rehab_splits(
    source_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    access_policy_path: Path,
) -> Mapping[str, Any]:
    """Create all five immutable token files and their access policy.

    The reserved file is written from the in-memory original-test slice.  It is
    never reopened to compute its digest: the exact bytes are hashed before the
    write and the token deny-list is sealed into the separate policy document.
    """

    originals = {
        "train_tokens.txt": read_source_tokens(source_dir / "train_tokens.txt"),
        "val_tokens.txt": read_source_tokens(source_dir / "val_tokens.txt"),
        "test_tokens.txt": read_source_tokens(source_dir / "test_tokens.txt"),
    }
    splits: dict[str, tuple[str, ...]] = {}
    for name, (source_name, start, stop) in SPLIT_SLICES.items():
        source = originals[source_name]
        if len(source) < stop:
            raise DirectRehabContractError(
                f"{source_name} has {len(source)} tokens; need at least {stop}"
            )
        splits[name] = tuple(source[start:stop])

    expected_counts = {"train": 1024, "val": 256, "dev": 512, "holdout": 512, "reserved": 512}
    actual_counts = {name: len(tokens) for name, tokens in splits.items()}
    if actual_counts != expected_counts:
        raise DirectRehabContractError(
            f"unexpected rehabilitation split counts: {actual_counts}"
        )
    _validate_disjoint(splits)
    if set(splits["holdout"]) & set(originals["test_tokens.txt"][200:712]):
        raise DirectRehabContractError("fresh holdout overlaps old Oracle Effect test")

    if output_dir.exists() and any(output_dir / name for name in SPLIT_FILENAMES.values()):
        existing = [
            str(output_dir / filename)
            for filename in SPLIT_FILENAMES.values()
            if (output_dir / filename).exists()
        ]
        if existing:
            raise DirectRehabContractError(
                f"refusing existing rehabilitation split files: {existing}"
            )

    file_entries: dict[str, dict[str, Any]] = {}
    for name, filename in SPLIT_FILENAMES.items():
        payload = _token_bytes(splits[name])
        target = output_dir / filename
        _atomic_write_bytes(target, payload)
        file_entries[name] = {
            "path": str(target),
            "count": len(splits[name]),
            "sha256": sha256_bytes(payload),
            "source": {
                "file": SPLIT_SLICES[name][0],
                "slice": [SPLIT_SLICES[name][1], SPLIT_SLICES[name][2]],
            },
        }

    manifest: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA,
        "splits": file_entries,
        "pairwise_disjoint": True,
        "holdout_absent_from_old_oracle_effect_test": True,
        "old_oracle_effect_test": "original_test[200:712]",
        "holdout_contract": "original_test[712:1224]",
        "reserved_contract": "original_test[1224:1736]",
        "reserved_file_reopened_after_write": False,
    }
    _atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))

    policy = {
        "schema_version": ACCESS_SCHEMA,
        "reserved_split_path": str((output_dir / SPLIT_FILENAMES["reserved"]).resolve()),
        "reserved_split_sha256": file_entries["reserved"]["sha256"],
        "reserved_tokens": list(splits["reserved"]),
        "holdout_tokens": list(splits["holdout"]),
        "allowed_holdout_phases": sorted(ALLOWED_HOLDOUT_PHASES),
        "forbidden_input_store_fragment": FORBIDDEN_STORE_FRAGMENT,
    }
    _atomic_write_bytes(access_policy_path, _canonical_json_bytes(policy))
    return manifest


def assert_no_effect_input_stores(configured_input_stores: Iterable[object]) -> None:
    """Reject any configured model input store with an effect-bearing identity."""

    normalized = tuple(str(value).lower() for value in configured_input_stores)
    if any(FORBIDDEN_STORE_FRAGMENT in value for value in normalized):
        raise DirectRehabContractError(
            f"effect input stores are forbidden for Direct rehabilitation: {normalized}"
        )
    # Keep the scientific contract recognizable in runtime/source audits.
    assert "effect" not in normalized


@dataclass(frozen=True)
class AccessPolicy:
    reserved_split_path: Path
    reserved_split_sha256: str
    reserved_tokens: frozenset[str]
    holdout_tokens: frozenset[str]
    allowed_holdout_phases: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> "AccessPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ACCESS_SCHEMA:
            raise DirectRehabContractError("invalid Direct rehabilitation access policy")
        return cls(
            reserved_split_path=Path(payload["reserved_split_path"]).resolve(),
            reserved_split_sha256=str(payload["reserved_split_sha256"]),
            reserved_tokens=frozenset(str(token) for token in payload["reserved_tokens"]),
            holdout_tokens=frozenset(str(token) for token in payload["holdout_tokens"]),
            allowed_holdout_phases=frozenset(
                str(phase) for phase in payload["allowed_holdout_phases"]
            ),
        )

    def assert_token_access(self, token: str, phase: str) -> None:
        if token in self.reserved_tokens:
            raise DirectRehabContractError(
                f"reserved future-effect token access denied: {token}"
            )
        if token in self.holdout_tokens and phase not in self.allowed_holdout_phases:
            raise DirectRehabContractError(
                f"holdout token {token} cannot be opened during phase {phase!r}"
            )

    def read_token_file(self, path: Path, phase: str) -> tuple[str, ...]:
        resolved = path.resolve()
        if resolved == self.reserved_split_path:
            raise DirectRehabContractError(
                "future_effect_reserved_512.txt is sealed and must never be opened"
            )
        tokens = tuple(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        )
        if not tokens or any(not token for token in tokens):
            raise DirectRehabContractError(f"empty or malformed token file: {path}")
        for token in tokens:
            self.assert_token_access(token, phase)
        return tokens


class AccessAuditLog:
    """Append-only JSONL audit for every raw scene-data access."""

    def __init__(self, path: Path, policy: AccessPolicy, phase: str):
        self.path = path
        self.policy = policy
        self.phase = str(phase)

    def record(self, token: str, purpose: str) -> None:
        self.policy.assert_token_access(token, self.phase)
        event = {
            "schema_version": ACCESS_SCHEMA,
            "phase": self.phase,
            "purpose": str(purpose),
            "token": str(token),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def audit_access_log(path: Path, policy: AccessPolicy) -> Mapping[str, Any]:
    phases: dict[str, int] = {}
    reserved_accesses: list[str] = []
    holdout_accesses: list[dict[str, str]] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            event = json.loads(line)
            token = str(event["token"])
            phase = str(event["phase"])
            phases[phase] = phases.get(phase, 0) + 1
            if token in policy.reserved_tokens:
                reserved_accesses.append(f"line {line_number}: {token}")
            if token in policy.holdout_tokens:
                holdout_accesses.append({"phase": phase, "token": token})
                if phase not in policy.allowed_holdout_phases:
                    raise DirectRehabContractError(
                        f"holdout access outside allowed phases at line {line_number}"
                    )
    if reserved_accesses:
        raise DirectRehabContractError(
            f"reserved scene data was accessed: {reserved_accesses[:5]}"
        )
    return {
        "schema_version": ACCESS_SCHEMA,
        "status": "PASS",
        "event_count": sum(phases.values()),
        "phase_counts": phases,
        "reserved_token_access_count": 0,
        "holdout_access_count": len(holdout_accesses),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-splits")
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--access-policy", type=Path, required=True)
    audit = commands.add_parser("audit-access")
    audit.add_argument("--access-policy", type=Path, required=True)
    audit.add_argument("--log", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-splits":
        result = build_direct_rehab_splits(
            args.source_dir, args.output_dir, args.manifest, args.access_policy
        )
    else:
        result = audit_access_log(args.log, AccessPolicy.load(args.access_policy))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
