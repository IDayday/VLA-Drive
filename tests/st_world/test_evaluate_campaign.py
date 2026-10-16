"""Behavioral fixtures only; no simulated score here is scientific evidence."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from starVLA.dataloader.foresight_dataset import decode_ego
from starVLA.model.modules.st_world.consequence import (
    METRICS, STATE_COLUMNS, file_sha256, fingerprint, trajectory_sha256,
)
from tools.st_world_policy import evaluate_campaign as evaluation


DEFINITIONS = {"bootstrap_replicates": 2000, "bootstrap_seed": 20261017,
               "zero_epsilon": 1e-8, "high_score_threshold": .9, "high_drop_threshold": .05,
               "ep_drop_threshold": .05, "moving_reference_m": 5., "low_progress_m": 1.,
               "near_stop_speed_mps": .5}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def seal(value, key="identity"):
    value[key] = fingerprint({k: v for k, v in value.items() if k != key})
    return value


def ref(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "source_log", *METRICS])
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def campaign(tmp_path):
    """A complete tiny receipt chain, with no model, map, images or scorer."""
    torch.set_num_threads(2)
    data = tmp_path / "data"; (data / "ego").mkdir(parents=True)
    ego = {"schema": "foresight_ego_labels_v1", "split": "dev", "index_sha256": "index", "identity": "ego"}
    write_json(data / "ego_identity.json", ego)
    scenes = []
    raw = torch.zeros(8, 4); raw[:, 3] = 1
    physical = decode_ego(raw).numpy()
    for token, log in (("a", "one"), ("b", "one"), ("c", "two")):
        label = data / "ego" / (token + ".pt")
        torch.save({"identity": "ego", "token": token, "ego": raw}, label)
        scenes.append({"token": token, "log": log, "current_record_sha256": "current",
                       "ego_record_sha256": file_sha256(label)})
    selected = {"schema": "st_fixed_scene_list_v1", "split": "dev", "source_index_hash": "index",
                "source_student_identity": "student", "ego_identity_sha256": file_sha256(data / "ego_identity.json"),
                "scenes": scenes}
    population = tmp_path / "SELECTED_SCENES.json"; write_json(population, selected)
    export_dir = tmp_path / "export"; (export_dir / "predictions").mkdir(parents=True)
    checkpoint = seal({"scope": "SMOKE", "weights_usable_for_formal_init": False,
                       "base_binding": "base", "native_checkpoint_sha256": "c" * 64,
                       "deployment_source_contract": {"local": "d" * 64}, "run_identity": "run",
                       "completed": 2, "training_source_sha": "f" * 40}, "sha256")
    export = seal({"schema": "st_single_trajectory_export_v1", "checkpoint": checkpoint,
                   "observations": {"split": "dev", "identity": "obs", "history": None,
                                    "manifest_sha256": file_sha256(population), "current_source": "student"},
                   "selected_scenes_sha256": fingerprint(scenes), "requested_scenes": 3, "world_size": 1,
                   "seed": 42, "environment": {"test_only": True}, "solver": "original_Euler_10",
                   "precision": "FP32_master_eval_TF32_off", "scene_noise": "foresight-action-v1",
                   "candidates_per_scene": 1, "scorer": None, "targets_read": False, "teacher_cache_read": False})
    write_json(export_dir / "IDENTITY.json", export)
    write_json(export_dir / "OBSERVATIONS.json", {"scenes": scenes, "identity": "obs"})
    receipts, requests, records, contexts, csv_rows = [], [], [], [], []
    protocol = seal({"fixture": "unit_test_only"}, "sha256")
    runtime = fingerprint({k: export[k] for k in ("environment", "seed", "solver", "precision", "scene_noise")})
    for i, scene in enumerate(scenes):
        token, log = scene["token"], scene["log"]
        path = export_dir / "predictions" / (token + ".npz")
        trajectory = physical.copy(); trajectory[:, 0] += 1
        np.savez(path, raw=raw.numpy(), trajectory=trajectory)
        write_json(path.with_suffix(".json"), {"identity": export["identity"], "token": token, "log": log,
                                              "status": "VALID_PREDICTION", "sha256": file_sha256(path)})
        receipts.append({"token": token, "receipt_sha256": file_sha256(path.with_suffix(".json"))})
        context = tmp_path / "unread_cache" / (token + ".pkl")
        contexts.append({"token": token, "log": log, "cache_path": str(context)})
        request = {"scene_id": token, "source_log": log, "metric_cache_path": str(context),
                   "metric_cache_sha256": fingerprint(token), "candidate_id": "single_policy", "generator_kind": "policy",
                   "policy_version": checkpoint["sha256"], "generator_checkpoint_sha256": "c" * 64,
                   "generator_source_hash": fingerprint(checkpoint["deployment_source_contract"]),
                   "inference_runtime_hash": runtime, "score_protocol_hash": protocol["sha256"],
                   "trajectory_sha256": trajectory_sha256(trajectory)}
        requests.append(request)
        values = {k: [0., .9, 1.][i] if k == "score" else 1. for k in METRICS}
        csv_rows.append({"scene_id": token, "source_log": log, **values})
        states = np.zeros((41, 11)); states[:, 3] = 2
        records.append({"valid": True, "status": "VALID_SCORED", "identity": request,
                        "metrics": values, "metrics_valid": {k: True for k in METRICS},
                        "context_unchanged": True, "context_hash": fingerprint([token, log]),
                        "reference_identity": {"full_cache_sha256": fingerprint(token), "score_protocol_hash": protocol["sha256"]},
                        "progress": {"raw_meters": 6., "reference_raw_meters": 7.},
                        "executed_states": states, "state_columns": STATE_COLUMNS,
                        "executed_state_times_seconds": np.arange(41) * .1})
    write_json(export_dir / "COMPLETE_rank0.json", {"identity": export["identity"], "rank": 0,
                                                   "requested": 3, "receipts": receipts})
    index = tmp_path / "cache_index.json"; write_json(index, contexts)
    score_dir = tmp_path / "score"; score_dir.mkdir()
    score = seal({"schema": "st_world_official_evaluation_v1", "checkpoint": checkpoint,
                  "export_identity": export["identity"], "export_file_sha256": file_sha256(export_dir / "IDENTITY.json"),
                  "protocol": protocol, "spec": {"score_protocol_hash": protocol["sha256"]},
                  "request_identities": requests, "metric_index": ref(index), "retry_attempts": 1})
    write_json(score_dir / "IDENTITY.json", score)
    write_csv(score_dir / "SCENES.csv", csv_rows)
    complete = {"status": "COMPLETE", "identity": score["identity"], "requested_scenes": 3, "valid_scored": 3,
                "invalid": 0, "unresolved": 0, "official_single_candidate_reference": True,
                "deployment_candidates_per_scene": 1, "checkpoint_scope": "SMOKE", "diagnostic_weights": True,
                "score_units": "0..1", "scenes_csv_sha256": file_sha256(score_dir / "SCENES.csv")}
    write_json(score_dir / "COMPLETE.json", complete)
    ledger = score_dir / "ledger"; key = fingerprint(requests); folder = ledger / key
    write_json(ledger / "IDENTITY.json", {"identity": score["identity"], "retry_attempts": 1})
    write_json(folder / "REQUESTS.json", {"policy_version": checkpoint["sha256"], "requests": requests})
    attempt = folder / "attempt_00"
    write_json(attempt / "ADMITTED.json", {"indices": [0, 1, 2], "calls": 3,
                                           "policy_version": checkpoint["sha256"], "request_hash": key})
    torch.save(records, attempt / "RESULTS.pt")
    write_json(attempt / "COMPLETE.json", {"result": ref(attempt / "RESULTS.pt"),
                                           "admission_sha256": file_sha256(attempt / "ADMITTED.json")})
    plan = {"schema": "st_world_evaluation_plan_v1", "formal_runs": {}, "base_binding": {"binding_sha256": "base"},
            "inference_seed": 42, "score_protocol_sha256": protocol["sha256"], "statistics": DEFINITIONS,
            "populations": {"dev": {"manifest": ref(population), "scenes": 3, "logs": 2,
                                    "source_index_hash": "index", "source_student_identity": "student", "current_root": str(data)}}}
    return {"plan": plan, "population": population, "score": score_dir, "export": export_dir,
            "csv_rows": csv_rows, "records": records, "attempt": attempt, "tmp": tmp_path}


def analyze(fixture):
    return evaluation.analyze(fixture["plan"], fixture["population"], fixture["score"], fixture["export"],
                              fixture["score"], fixture["export"], scope="SMOKE", comparison="SMOKE_AA", stage="SMOKE")


def update_csv(fixture, rows):
    path = fixture["score"] / "SCENES.csv"; write_csv(path, rows)
    complete = evaluation.read_json(fixture["score"] / "COMPLETE.json")
    complete["scenes_csv_sha256"] = file_sha256(path)
    write_json(fixture["score"] / "COMPLETE.json", complete)


def update_records(fixture):
    attempt = fixture["attempt"]
    torch.save(fixture["records"], attempt / "RESULTS.pt")
    receipt = evaluation.read_json(attempt / "COMPLETE.json")
    receipt["result"] = ref(attempt / "RESULTS.pt")
    write_json(attempt / "COMPLETE.json", receipt)


def test_real_shape_receipts_aa_are_zero_and_csv_order_is_irrelevant(campaign):
    result, rows = analyze(campaign)
    assert result["scientific_result"] is False
    assert result["validity"] == {"requested": 3, "paired_valid": 3, "invalid": 0, "unresolved": 0}
    assert result["statistics"]["metrics"]["score"]["control"]["mean"] == pytest.approx(1.9 / 3)
    assert result["statistics"]["metrics"]["ade_m"]["control"]["mean"] == pytest.approx(1.)
    assert all(m["delta"]["mean"] == 0 and m["delta"]["ci95"] == [0, 0] for m in result["statistics"]["metrics"].values())
    update_csv(campaign, list(reversed(campaign["csv_rows"])))
    shuffled, shuffled_rows = analyze(campaign)
    assert shuffled["statistics"] == result["statistics"] and shuffled_rows == rows
    assert not (campaign["tmp"] / "unread_cache").exists()  # Cache payloads were never needed.


@pytest.mark.parametrize("change,match", [
    ("missing", "missing scenes"), ("duplicate", "duplicate scene"), ("extra", "extra scenes"),
    ("log", "foreign log"), ("nan", "Nonfinite"), ("range", "out-of-range"),
    ("fake_metric", "saved per-scene score"),
])
def test_forged_complete_csv_cannot_shrink_relabel_or_reconstruct_population(campaign, change, match):
    rows = copy.deepcopy(campaign["csv_rows"])
    if change == "missing": rows.pop()
    elif change == "duplicate": rows.append(rows[0])
    elif change == "extra": rows.append(dict(rows[0], scene_id="extra"))
    elif change == "log": rows[0]["source_log"] = "foreign"
    elif change == "nan": rows[0]["score"] = float("nan")
    elif change == "range": rows[0]["score"] = 1.1
    elif change == "fake_metric": rows[0]["score"] = .123
    update_csv(campaign, rows)  # Even a refreshed outer hash cannot hide bad scene records.
    with pytest.raises(ValueError, match=match): analyze(campaign)


@pytest.mark.parametrize("change,match", [
    ("invalid", "Invalid/unresolved"), ("metric_valid", "Invalid/unresolved"),
    ("mutated_context", "Invalid/unresolved"), ("reference", "reference differs"),
    ("missing_record", "Missing saved"), ("state_nan", "state/time/units"),
])
def test_ledger_failures_are_never_zeros_or_ignored_scenes(campaign, change, match):
    record = campaign["records"][0]
    if change == "invalid": record["valid"] = False
    elif change == "metric_valid": record["metrics_valid"]["comfort"] = False
    elif change == "mutated_context": record["context_unchanged"] = False
    elif change == "reference": record["reference_identity"]["full_cache_sha256"] = "x" * 64
    elif change == "missing_record": campaign["records"].pop()
    elif change == "state_nan": record["executed_states"][0, 0] = np.nan
    update_records(campaign)
    with pytest.raises(ValueError, match=match): analyze(campaign)


def test_changes_to_prediction_and_metadata_are_detected(campaign):
    path = campaign["export"] / "predictions/a.npz"
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum/status"): analyze(campaign)


def test_non_development_population_and_missing_formal_bindings_rejected(campaign):
    selection = evaluation.read_json(campaign["population"])
    selection["split"] = "navtest"
    alternate = campaign["tmp"] / "wrong_role.json"; write_json(alternate, selection)
    with pytest.raises(ValueError, match="registered dev"):
        evaluation.load_population(campaign["plan"], alternate, "SMOKE")
    with pytest.raises(ValueError, match="binding is pending"):
        evaluation._formal_binding(campaign["plan"], "C0", {}, "COMMON_ENDPOINT")
    with pytest.raises(ValueError, match="engineering A/A"):
        evaluation.analyze(campaign["plan"], campaign["population"], None, None, None, None,
                           scope="SMOKE", comparison="C1_minus_C0", stage="SMOKE")


def test_formal_population_cannot_be_a_pinned_subset(campaign):
    selection = evaluation.read_json(campaign["population"])
    selection["scenes"] = selection["scenes"][:2]
    path = campaign["tmp"] / "subset.json"; write_json(path, selection)
    with pytest.raises(ValueError, match="complete registered dev"):
        evaluation.load_population(campaign["plan"], path, "FORMAL")


def test_formal_receipts_require_independent_roles_and_the_exact_endpoint(tmp_path):
    # Synthetic review records in a temporary fixture exercise rejection paths.
    # They are never campaign approvals or committed PASS artifacts.
    cfg = {"cell": "C0", "execution": {"scope": "FORMAL", "sample_limit": 0, "seed": 42,
                                         "max_updates": 4096}, "optimizer": {"lr": 1e-5}}
    cp = tmp_path / "CONFIG.json"; write_json(cp, cfg)
    reg = {"source_sha": "source", "source_fingerprint": "source_fp", "run_id": "run",
           "execution": cfg["execution"], "artifacts": {"config": ref(cp)}, "implementer_agents": ["writer"]}
    rp = tmp_path / "REGISTRATION.json"; write_json(rp, reg)
    frozen = {"registration_sha256": file_sha256(rp), "source_fingerprint": "source_fp",
              "artifacts": {"config": file_sha256(cp)}, "execution": cfg["execution"], "run_id": "run"}
    review_refs = {}
    for role in ("evidence", "correctness", "efficiency"):
        path = tmp_path / (role + ".json")
        write_json(path, {"status": "PASS", "role": role, "reviewer_agent": role,
                          "fingerprint": frozen, "evidence_artifacts": ["fixture_only"]})
        review_refs[role] = ref(path)
    ap = tmp_path / "APPROVAL.json"
    approval = {"schema": "st_world_approval_v1", "formal_launch_allowed": True,
                "fingerprint": frozen, "reviews": review_refs}; write_json(ap, approval)
    identity = seal({"config": cfg, "registration_sha256": file_sha256(rp)})
    ip = tmp_path / "IDENTITY.json"; write_json(ip, identity)
    plan = {"training_seed": 42, "formal_runs": {"C0": {"registration": ref(rp), "approval": ref(ap),
                                                         "run_identity_file": ref(ip)}},
            "arms": {"C0": {"family": "consequence", "config_cell": "C0"}},
            "endpoints": {"consequence": {"COMMON_ENDPOINT": 4096}}, "optimizer": {"consequence": cfg["optimizer"]}}
    export = {"checkpoint": {"scope": "FORMAL", "weights_usable_for_formal_init": True,
                             "run_identity": identity["identity"], "training_source_sha": "source", "completed": 4096}}
    evaluation._formal_binding(plan, "C0", export, "COMMON_ENDPOINT")
    for key, value, match in (("completed", 1024, "common endpoint"), ("scope", "SMOKE", "lineage differs")):
        changed = copy.deepcopy(export); changed["checkpoint"][key] = value
        with pytest.raises(ValueError, match=match):
            evaluation._formal_binding(plan, "C0", changed, "COMMON_ENDPOINT")
    path = tmp_path / "correctness.json"; review = evaluation.read_json(path)
    review["reviewer_agent"] = "writer"; write_json(path, review)
    approval["reviews"]["correctness"] = ref(path); write_json(ap, approval)
    plan["formal_runs"]["C0"]["approval"] = ref(ap)
    with pytest.raises(ValueError, match="nonindependent"):
        evaluation._formal_binding(plan, "C0", export, "COMMON_ENDPOINT")


def test_log_bootstrap_preserves_scene_weighting_pairing_and_empty_denominators():
    logs = ["a", "a", "a", "b"]
    unique, inverse, draws = evaluation.cluster_draws(logs, replicates=2000, seed=20261017)
    value = evaluation.estimate([0, 0, 0, 1], inverse, draws)
    assert value["mean"] == .25  # An incorrect mean of log means would be .5.
    assert value["ci95"] == [0., 1.] and value["bootstrap_defined"] == 2000
    assert np.array_equal(draws, evaluation.cluster_draws(logs, replicates=2000, seed=20261017)[2])
    empty = evaluation.estimate([0, 0, 0, 1], inverse, draws, [False] * 4)
    assert empty["mean"] is None and empty["ci95"] is None and empty["denominator_scenes"] == 0
    assert empty["bootstrap_undefined"] == 2000
    rare = evaluation.estimate([0, 0, 0, 1], inverse, draws, [False, False, False, True])
    assert 0 < rare["bootstrap_undefined"] < 2000 and rare["ci95"] == [1, 1]


def test_zero_transitions_high_degradation_and_stopping_have_explicit_denominators():
    a = {k: np.ones(4) for k in METRICS}; b = copy.deepcopy(a)
    a["score"] = np.array([0., .95, .4, 1.]); b["score"] = np.array([1., 0., 0., .8])
    a["ego_progress"] = np.array([.8] * 4); b["ego_progress"] = np.array([.9, .1, .1, .8])
    a["no_at_fault_collisions"][1] = .5
    for x in (a, b): x["reference_progress_m"] = np.array([7.] * 4)
    a["raw_progress_m"] = np.array([6.] * 4); b["raw_progress_m"] = np.array([6., 0., 1., 6.])
    a["mean_executed_speed_mps"] = np.array([2.] * 4); b["mean_executed_speed_mps"] = np.array([2., 0., .5, 2.])
    result, flags = evaluation.paired_statistics(a, b, ["a", "a", "b", "b"], DEFINITIONS)
    events = result["events"]
    assert events["zero_repaired"]["count"] == 1 and events["new_zero"]["count"] == 2
    assert events["zero_repaired"]["conditional_rate"]["denominator_scenes"] == 1
    assert events["new_zero"]["conditional_rate"]["denominator_scenes"] == 3
    assert events["high_large_drop"]["count"] == 2 and events["high_to_zero"]["count"] == 1
    assert events["high_large_drop"]["conditional_rate"]["denominator_scenes"] == 2
    assert events["safety_gain_with_ep_drop"]["count"] == 1
    assert result["stopping_proxy"]["new_stop_count"] == 2
    assert result["stopping_proxy"]["delta_full_population_rate"]["mean"] == .5


def test_cli_writes_immutable_reusable_outputs(campaign, monkeypatch):
    plan = campaign["tmp"] / "plan.json"; write_json(plan, campaign["plan"])
    output = campaign["tmp"] / "analysis"
    argv = ["evaluate_campaign", "--plan", str(plan), "--population", str(campaign["population"]),
            "--control-score", str(campaign["score"]), "--control-export", str(campaign["export"]),
            "--treatment-score", str(campaign["score"]), "--treatment-export", str(campaign["export"]),
            "--scope", "SMOKE", "--comparison", "SMOKE_AA", "--stage", "SMOKE", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluation.main() == 0 and evaluation.main() == 0
    (output / "PAIRED_SCENES.csv").write_text("changed")
    assert evaluation.main() == 2
