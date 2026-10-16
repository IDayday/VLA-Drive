"""Launch admission must use the reviewed implementation and actual inputs."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools.st_world_policy.check_formal_approval import (
    ARTIFACTS, ApprovalError, check_formal_approval, file_hash,
    source_fingerprint, validate_execution, verify_registration,
)


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": file_hash(path)}


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def run(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@invalid.example")
    git(repo, "config", "user.name", "Launch gate fixture")
    (repo / "train.py").write_text("LOSS_WEIGHT = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "fixture")
    execution = dict(scope="FORMAL", global_batch=32, micro_batch=4, world_size=8,
                     max_updates=20000, stop_after=5000, sample_limit=0,
                     precision_policy="bf16_exec_fp32_master_v1", family="spatial", seed=42)
    artifacts = {}
    for name in ARTIFACTS:
        value = {"fixture": name}
        if name == "config":
            value["execution"] = execution
        artifacts[name] = put(tmp_path / (name + ".json"), value)
    registration = dict(schema="st_world_run_v1", run_id="SP0_seed42",
                        source_sha=git(repo, "rev-parse", "HEAD"),
                        source_fingerprint=source_fingerprint(repo), artifacts=artifacts,
                        execution=execution, implementer_agents=["/root/author"],
                        weights_usable_for_formal_init=True)
    reg = tmp_path / "registration.json"
    put(reg, registration)
    actual = verify_registration(reg, repo=repo, effective_execution=execution)
    proof = put(tmp_path / "real_test_receipt.json", {"fixture_test": "not a production approval"})
    reviews = {}
    for role in ("evidence", "correctness", "efficiency"):
        reviews[role] = put(tmp_path / (role + "_review.json"), dict(
            status="PASS", role=role, fingerprint=actual,
            reviewer_agent="/root/fixture_" + role, evidence_artifacts=[proof]))
    approval = tmp_path / "approval.json"
    put(approval, dict(schema="st_world_approval_v1", fingerprint=actual,
                       formal_launch_allowed=True, reviews=reviews))
    return repo, reg, approval, registration


def test_only_matching_independent_reviews_admit_formal(run):
    repo, reg, approval, record = run
    with pytest.raises(ApprovalError, match="independent"):
        check_formal_approval(reg, None, repo=repo)
    result = check_formal_approval(reg, approval, repo=repo,
                                   effective_execution=record["execution"])
    assert result["formal_launch_allowed"] is True


@pytest.mark.parametrize("artifact", sorted(ARTIFACTS))
def test_in_place_dependency_changes_invalidate_approval(run, artifact):
    repo, reg, approval, record = run
    path = Path(record["artifacts"][artifact]["path"])
    path.write_text('{"changed": true}')
    with pytest.raises(ApprovalError, match="content changed"):
        check_formal_approval(reg, approval, repo=repo)


def test_effective_runtime_must_match_not_just_config_file(run):
    repo, reg, approval, record = run
    runtime = copy.deepcopy(record["execution"])
    runtime["world_size"] = 4
    with pytest.raises(ApprovalError, match="Runtime arguments"):
        check_formal_approval(reg, approval, repo=repo, effective_execution=runtime)


def test_diagnostic_never_approves_or_exports_formal_initialization(run):
    repo, reg, approval, record = run
    record["execution"].update(scope="SMOKE", sample_limit=32, stop_after=32)
    record["artifacts"]["config"] = put(Path(record["artifacts"]["config"]["path"]),
                                        {"execution": record["execution"]})
    put(reg, record)
    with pytest.raises(ApprovalError, match="Diagnostic weights"):
        check_formal_approval(reg, approval, repo=repo)
    record["weights_usable_for_formal_init"] = False
    put(reg, record)
    result = check_formal_approval(reg, None, repo=repo)
    assert result["status"] == "DIAGNOSTIC_ONLY"
    assert not result["formal_launch_allowed"] and not result["weights_usable_for_formal_init"]


@pytest.mark.parametrize("scope,samples,steps", [
    ("SMOKE", 65, 32), ("SMOKE", 64, 65), ("PROFILE", 1, 257),
    ("PROFILE", 0, 200), ("FORMAL", 32, 5000),
])
def test_budgets_are_enforced(run, scope, samples, steps):
    execution = copy.deepcopy(run[3]["execution"])
    execution.update(scope=scope, sample_limit=samples, stop_after=steps)
    with pytest.raises(ApprovalError):
        validate_execution(execution)


@pytest.mark.parametrize("mutation", ["author", "same_reviewer", "missing_evidence", "wrong_run"])
def test_review_status_alone_does_not_grant_approval(run, mutation):
    repo, reg, approval, record = run
    container = json.loads(approval.read_text())
    ref = container["reviews"]["correctness"]
    path = Path(ref["path"])
    review = json.loads(path.read_text())
    if mutation == "author":
        review["reviewer_agent"] = "/root/author"
    elif mutation == "same_reviewer":
        review["reviewer_agent"] = "/root/fixture_evidence"
    elif mutation == "missing_evidence":
        review["evidence_artifacts"] = []
    else:
        review["fingerprint"]["run_id"] = "different_run"
    container["reviews"]["correctness"] = put(path, review)
    put(approval, container)
    with pytest.raises(ApprovalError):
        check_formal_approval(reg, approval, repo=repo)


def test_changed_review_evidence_rejected(run):
    repo, reg, approval, record = run
    (reg.parent / "real_test_receipt.json").write_text('{}')
    with pytest.raises(ApprovalError, match="content changed"):
        check_formal_approval(reg, approval, repo=repo)


def test_code_change_requires_review_but_prose_commit_does_not(run):
    repo, reg, approval, record = run
    (repo / "README.md").write_text("Documentation correction.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "prose only")
    assert check_formal_approval(reg, approval, repo=repo)["formal_launch_allowed"]
    (repo / "train.py").write_text("LOSS_WEIGHT = 2\n")
    with pytest.raises(ApprovalError, match="committed and clean"):
        check_formal_approval(reg, approval, repo=repo)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "changed objective")
    with pytest.raises(ApprovalError, match="source changed"):
        check_formal_approval(reg, approval, repo=repo)


def test_cli_returns_json_failure_for_broken_registration(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"broken":')
    import sys
    result = subprocess.run([sys.executable, "-m", "tools.st_world_policy.check_formal_approval",
                             "--registration", str(broken)], text=True, capture_output=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "BLOCKED"
