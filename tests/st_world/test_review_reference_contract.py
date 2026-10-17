"""Independent A6 counterexamples: no simulation, GPU, or scientific labels.

The positive/negative integration cases use explicit synthetic native arrays.
The paired-analysis test exposes the remaining input/output conflation consumer.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from starVLA.model.modules.st_world import consequence as c
from tools.st_world_policy import evaluate_campaign as evaluation
from tools.st_world_policy.train_flow_policy import OfficialFeedbackLedger, feedback_from_records
from test_flow_training import FixtureScorer, scored, setup_rollout
from test_evaluate_campaign import DEFINITIONS


def replace_reference_execution(row, request):
    """Different exact numerical output; every immutable input stays identical."""
    before_input = deepcopy(row["reference_contract"]["input"])
    reference = row["reference_interpolated_states"].copy()
    executed = np.stack((row["reference_executed_states"], row["executed_states"]))
    executed[0, 7, 9] = np.nextafter(executed[0, 7, 9], np.inf)
    row.update(c._reference_evidence(request, row["context_hash"], reference, executed,
        {"dependencies": row["reference_identity"]["software_identity"]}))
    assert c.validated_reference_input(row, request) == before_input


@pytest.mark.parametrize("cell", ["C1", "C2", "C3", "C4"])
def test_review_feedback_preserves_reward_with_distinct_valid_reference_execution(cell):
    *_, rollout, requests = setup_rollout(1)
    rows = [scored(r, score=s, nc=n, dac=d, ttc=t) for r, s, n, d, t in zip(
        requests, [0., .1, .8, 1.], [0., .5, 1., 1.], [0., 1., 1., 1.], [1., 0., 1., 1.])]
    before = feedback_from_records(rollout, requests, rows, cell, "c" * 64)
    changed = deepcopy(rows)
    replace_reference_execution(changed[3], requests[3])
    assert changed[3]["reference_identity"] != rows[3]["reference_identity"]
    after = feedback_from_records(rollout, requests, changed, cell, "c" * 64)
    torch.testing.assert_close(before.rewards, after.rewards, rtol=0, atol=0)
    torch.testing.assert_close(before.valid, after.valid, rtol=0, atol=0)
    assert before.context_hashes == after.context_hashes
    assert before.reward_kind == after.reward_kind
    for original, final in zip(rows, changed):
        assert original["metrics"] == final["metrics"]
        assert original["consequences"] == final["consequences"]


@pytest.mark.parametrize("field", [
    "reference_interpolated_states", "reference_executed_states", "executed_states"])
def test_review_feedback_rejects_one_ulp_array_corruption(field):
    *_, rollout, requests = setup_rollout(1)
    rows = [scored(r) for r in requests]
    rows[3][field][7, 9] = np.nextafter(rows[3][field][7, 9], np.inf)
    with pytest.raises(c.ContractMismatch, match="checksum|provenance"):
        feedback_from_records(rollout, requests, rows, "C1", "c" * 64)


@pytest.mark.parametrize("change", ["input_reference", "software", "scene"])
def test_review_group_rejects_resealed_different_inputs(change):
    *_, rollout, requests = setup_rollout(1)
    rows = [scored(r) for r in requests]
    row, request = rows[3], requests[3]
    reference = row["reference_interpolated_states"].copy()
    execution = np.stack((row["reference_executed_states"], row["executed_states"]))
    dependencies = dict(row["reference_identity"]["software_identity"])
    if change == "input_reference": reference[7, 0] = 1.
    elif change == "software": dependencies["numpy"] = "different"
    else:
        from dataclasses import replace
        requests[3] = request = replace(request, scene_id="foreign")
        row["identity"] = request.identity()
    row.update(c._reference_evidence(request, row["context_hash"], reference, execution,
                                     {"dependencies": dependencies}))
    c.validated_reference_input(row, request)
    with pytest.raises(ValueError, match="identical official reference"):
        feedback_from_records(rollout, requests, rows, "C1", "c" * 64)


@pytest.mark.parametrize("damage", ["legacy", "reference_interpolated_states",
    "reference_executed_states", "executed_states"])
def test_review_ledger_rejects_stale_or_corrupt_complete_without_rewrite(tmp_path, damage):
    *_, requests = setup_rollout(1)
    service = FixtureScorer()
    ledger = OfficialFeedbackLedger(tmp_path, identity="fixture-v2", max_calls=4, retry_attempts=1)
    ledger.score(requests, policy_version="0", service=service)
    payload = next(tmp_path.glob("*/attempt_00/RESULTS.pt"))
    complete = payload.with_name("COMPLETE.json")
    rows = torch.load(payload, map_location="cpu", weights_only=False)
    if damage == "legacy":
        for row in rows:
            for field in ("reference_contract", "reference_interpolated_states", "reference_executed_states"):
                row.pop(field)
    else:
        rows[3][damage][7, 9] = np.nextafter(rows[3][damage][7, 9], np.inf)
    # Model a complete saved artifact with an internally stale/invalid schema;
    # even a valid outer checksum must not replace native per-array validation.
    torch.save(rows, payload)
    receipt = json.loads(complete.read_text())
    receipt["result"]["sha256"] = c.file_sha256(payload)
    complete.write_text(json.dumps(receipt))
    before = {str(p): c.file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    restarted = OfficialFeedbackLedger(tmp_path, identity="fixture-v2", max_calls=4, retry_attempts=1)
    with pytest.raises(c.ContractMismatch):
        restarted.score(requests, policy_version="0", service=service)
    assert len(service.calls) == 4  # No implicit re-score or fake upgrade.
    after = {str(p): c.file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_review_paired_evaluation_accepts_same_input_distinct_native_execution(tmp_path, monkeypatch):
    """Regression must PASS after the paired-analysis consumer is connected.

    Loading/provenance was tested separately; isolate analyze's actual pair
    guard with two individually valid v2 records and unchanged official labels.
    """
    *_, requests = setup_rollout(1)
    request = requests[0]
    first = scored(request)
    second = deepcopy(first)
    replace_reference_execution(second, request)
    assert c.validate_reference_group([first, second], [request, request])
    token = request.scene_id
    metadata = {"identity": "same_immutable_export", "observations": {}, "environment": {},
                "checkpoint": {"run_identity": "fixture"}}

    def run(record, path):
        return {"export": metadata, "rows": {token: {"source_log": request.source_log, **record["metrics"]}},
                "requests": {token: request.identity()}, "records": {token: record},
                "score_root": str(path), "export_root": str(tmp_path / "export"),
                "scenes_csv_sha256": "0" * 64, "failed_attempts": [], "score": {"identity": "test-only"}}

    a, b = run(first, tmp_path / "a"), run(second, tmp_path / "b")
    selected = {"scenes": [{"token": token, "log": request.source_log}]}
    monkeypatch.setattr(evaluation, "load_population", lambda *args: selected)
    monkeypatch.setattr(evaluation, "load_run", lambda score, *args: a if str(score).endswith("/a") else b)
    monkeypatch.setattr(evaluation, "trajectory_diagnostics", lambda *args: {token: {"ade_m": 0.}})
    population = tmp_path / "population.json"
    population.write_text(json.dumps(selected))
    plan = {"schema": "st_world_evaluation_plan_v1", "statistics": DEFINITIONS}
    result, _ = evaluation.analyze(plan, population, tmp_path / "a", tmp_path / "export",
                                  tmp_path / "b", tmp_path / "export", scope="SMOKE",
                                  comparison="SMOKE_AA", stage="SMOKE")
    assert result["status"] == "COMPLETE"
    assert result["statistics"]["metrics"]["score"]["delta"]["mean"] == 0
