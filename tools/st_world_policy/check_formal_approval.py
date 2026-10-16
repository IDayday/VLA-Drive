"""Verify actual campaign dependencies and independent reviews before launch.

This is an audit gate, not a cryptographic authorization system. Reviewers create
their own reports; this module never manufactures PASS reviews. Runtime trainers
must pass their effective execution settings, not merely the registration path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


ARTIFACTS = frozenset({
    "config", "data_manifest", "target_contract", "teacher_manifest",
    "base_binding", "environment", "resource_lease",
})
REVIEW_ROLES = ("evidence", "correctness", "efficiency")
EXECUTION_FIELDS = frozenset({
    "scope", "global_batch", "micro_batch", "world_size", "max_updates",
    "stop_after", "sample_limit", "precision_policy", "family", "seed",
})


class ApprovalError(ValueError):
    """A real dependency or authorization does not match the reviewed run."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ApprovalError(f"Expected JSON object: {path}")
    return value


def _artifact(record: Mapping[str, str]) -> str:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ApprovalError("Artifact requires exactly path and sha256")
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file():
        raise ApprovalError(f"Missing actual absolute artifact: {path}")
    actual = file_hash(path)
    if actual != record["sha256"]:
        raise ApprovalError(f"Artifact content changed: {path}")
    return actual


def source_fingerprint(repo: Path, ref: str = "HEAD") -> str:
    """Hash tracked execution inputs; prose-only commits need no new profile."""
    rows = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", ref], cwd=repo
    ).split(b"\0")
    included = []
    for row in rows:
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        path = name.decode()
        if not path.startswith(("reports/", "docs/")) and (path.endswith((".py", ".sh", ".toml", ".yaml", ".yml", ".lock",
                           ".json", ".cpp", ".cu", ".h", ".c", ".ini", ".cfg"))
                or Path(path).name.startswith("requirements")):
            included.append([path, meta.decode()])
    if not included:
        raise ApprovalError("No tracked execution source found")
    return digest(included)


def validate_execution(execution: dict) -> None:
    if not isinstance(execution, dict) or set(execution) != EXECUTION_FIELDS:
        raise ApprovalError("Execution fields differ from registered schema")
    for k in ("global_batch", "micro_batch", "world_size", "max_updates"):
        if type(execution[k]) is not int or execution[k] <= 0:
            raise ApprovalError(f"Positive integer required: {k}")
    for k in ("stop_after", "sample_limit", "seed"):
        if type(execution[k]) is not int or execution[k] < 0:
            raise ApprovalError(f"Nonnegative integer required: {k}")
    if execution["global_batch"] % (execution["micro_batch"] * execution["world_size"]):
        raise ApprovalError("Global batch must match whole accumulation microbatches")
    if not 0 < execution["stop_after"] <= execution["max_updates"]:
        raise ApprovalError("Explicit finite stop_after is required")
    scope = execution["scope"]
    if scope == "SMOKE":
        if not 1 <= execution["sample_limit"] <= 64 or execution["stop_after"] > 64:
            raise ApprovalError("SMOKE is limited to 64 scenes and 64 updates")
    elif scope == "PROFILE":
        if execution["sample_limit"] <= 0 or execution["stop_after"] > 256:
            raise ApprovalError("PROFILE needs an explicit population and <=256 updates")
    elif scope == "FORMAL":
        ceilings = {"spatial": 20000, "temporal": 20000, "consequence": 4096}
        if execution["family"] not in ceilings:
            raise ApprovalError("Conditional stages require their own reviewed budget schema")
        if execution["max_updates"] > ceilings[execution["family"]]:
            raise ApprovalError("Initial wave budget exceeded")
        if execution["sample_limit"] != 0:
            raise ApprovalError("Formal run consumes the complete registered manifest")
    else:
        raise ApprovalError(f"Unknown scope: {scope}")
    if not isinstance(execution["precision_policy"], str) or not execution["precision_policy"]:
        raise ApprovalError("An explicit numeric execution identity is required")


def verify_registration(path: str | Path, *, repo: str | Path,
                        effective_execution: dict | None = None) -> dict:
    reg = read_json(path)
    required = {"schema", "run_id", "source_sha", "source_fingerprint", "artifacts",
                "execution", "implementer_agents", "weights_usable_for_formal_init"}
    if set(reg) != required or reg["schema"] != "st_world_run_v1":
        raise ApprovalError("Unknown/incomplete run registration schema")
    if not isinstance(reg["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", reg["run_id"]):
        raise ApprovalError("A safe explicit run identity is required")
    if not isinstance(reg["source_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", reg["source_sha"]):
        raise ApprovalError("An exact source commit SHA is required")
    implementers = reg["implementer_agents"]
    if (not isinstance(implementers, list) or not implementers
            or any(not isinstance(x, str) or not x for x in implementers)
            or len(set(implementers)) != len(implementers)):
        raise ApprovalError("Unique implementer identities are required")
    if type(reg["weights_usable_for_formal_init"]) is not bool:
        raise ApprovalError("Explicit boolean weight provenance is required")
    validate_execution(reg["execution"])
    if effective_execution is not None and effective_execution != reg["execution"]:
        raise ApprovalError("Runtime arguments differ from reviewed execution")
    if reg["execution"]["scope"] != "FORMAL" and reg["weights_usable_for_formal_init"]:
        raise ApprovalError("Diagnostic weights cannot initialize formal runs")
    if set(reg["artifacts"]) != ARTIFACTS:
        raise ApprovalError("All model/data/target/environment/resource artifacts are required")
    hashes = {k: _artifact(v) for k, v in reg["artifacts"].items()}
    config = read_json(reg["artifacts"]["config"]["path"])
    if config.get("execution") != reg["execution"]:
        raise ApprovalError("Config execution differs from registration")
    repo = Path(repo).resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise ApprovalError("Runtime source must be committed and clean")
    if subprocess.run(["git", "merge-base", "--is-ancestor", reg["source_sha"], "HEAD"],
                      cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise ApprovalError("Reviewed source is not an ancestor of runtime source")
    expected = source_fingerprint(repo, reg["source_sha"])
    if reg["source_fingerprint"] != expected or source_fingerprint(repo) != expected:
        raise ApprovalError("Execution source changed; affected reviews must be renewed")
    return {"registration_sha256": file_hash(path), "source_fingerprint": expected,
            "artifacts": hashes, "execution": reg["execution"], "run_id": reg["run_id"]}


def check_formal_approval(registration: str | Path, approval: str | Path | None, *,
                          repo: str | Path, effective_execution: dict | None = None) -> dict:
    actual = verify_registration(registration, repo=repo,
                                 effective_execution=effective_execution)
    reg = read_json(registration)
    if actual["execution"]["scope"] != "FORMAL":
        return {"status": "DIAGNOSTIC_ONLY", "identity": digest(actual),
                "formal_launch_allowed": False, "weights_usable_for_formal_init": False}
    if approval is None:
        raise ApprovalError("Formal pilot requires independent evidence/correctness/efficiency reviews")
    approved = read_json(approval)
    if (approved.get("schema") != "st_world_approval_v1"
            or approved.get("fingerprint") != actual
            or approved.get("formal_launch_allowed") is not True):
        raise ApprovalError("Approval is absent, stale or bound to another run")
    if set(approved.get("reviews", {})) != set(REVIEW_ROLES):
        raise ApprovalError("Three independent review roles are required")
    reviewers = set()
    for role in REVIEW_ROLES:
        ref = approved["reviews"][role]
        _artifact(ref)
        review = read_json(ref["path"])
        reviewer = review.get("reviewer_agent")
        if (review.get("status") != "PASS" or review.get("role") != role
                or review.get("fingerprint") != actual or not reviewer
                or reviewer in reg["implementer_agents"] or reviewer in reviewers):
            raise ApprovalError(f"Invalid or nonindependent {role} review")
        evidence = review.get("evidence_artifacts")
        if not isinstance(evidence, list) or not evidence:
            raise ApprovalError(f"No actual evidence behind {role} PASS")
        for item in evidence:
            _artifact(item)
        reviewers.add(reviewer)
    return {"status": "VERIFIED", "identity": digest(actual), "formal_launch_allowed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    try:
        result = check_formal_approval(args.registration, args.approval, repo=args.repo)
    except (ApprovalError, OSError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
