"""Synthetic transaction/gradient tests; no claim of real S0 or simulation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from tools.action_effect.common import artifact, digest, write_json
from tools.action_effect.train_refiner import RepairDataset, make_step, ACTIVE_GRID_COEFFICIENTS, build_model
from tools.action_effect.evaluate_refiner import _summary, load_refiner
from tools.action_effect.training import run_training
from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner
from starVLA.model.modules.action_effect.effect_model import ActionEffectModel
from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner, RefinerConfig
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec, reconstruct
from test_model_effect import cfg, spec, batch, state_hash


def recipe():
    return {"training": {"hidden": 16, "layers": 2, "heads": 4, "max_updates": 4,
            "batch_scenes": 4, "microbatch_scenes": 1, "warmup_updates": 1, "validation_every": 2,
            "learning_rate": 1e-3, "weight_decay": .01, "gradient_clip": 1.}}


def fixture_dataset(tmp_path):
    context, a, _ = batch(4)
    labels = spec(); entries, contexts, context_rows = [], {}, {}
    for i in range(4):
        sid, log, role = f"scene{i}", f"log{i}", "validation" if i == 3 else "fit"
        entry = {"scene_id": sid, "source_log": log, "role": role, "observation_hash": "f"*64}
        context_rows[sid] = entry
        contexts[sid] = {"scene_id": sid, "memory": context["memory"][i], "ego": context["ego"][i],
                         "physical": a[i], "W_count": 3}
        coeff = [0.]*8
        if i % 2: coeff[4] = .15
        target = reconstruct(a[i].double().numpy(), coeff, BasisSpec())
        before = {"names": list(labels.names), "units": list(labels.units), "values": [.5]*len(labels.names), "valid": [True]*len(labels.names)}
        after = deepcopy(before); after["values"][0] += .1 if i % 2 else 0
        after["valid"][-2:] = [False, False]
        entries.append({**entry, "context_entry": entry, "base_physical": a[i].tolist(),
            "target_physical": target.tolist(), "coefficients": coeff, "base_labels": before, "selected_labels": after,
            "base_feedback": None, "selected_feedback": None, "base_canonical_receipt": None, "selected_canonical_receipt": None,
            "trainable": True, "target_kind": "verified_local_repair" if i % 2 else "identity_preservation"})
    context_manifest = tmp_path/"CONTEXT.json"; write_json(context_manifest, {"test": "synthetic"})
    reg = {"base_checkpoint": {"path": "/synthetic/not-a-checkpoint", "sha256": "e"*64},
           "context_manifest": artifact(context_manifest), "numeric_runtime_hash": "d"*64,
           "spec": {"score_protocol_hash": "c"*64}}
    reg["identity"] = digest(reg); write_json(tmp_path/"REGISTRATION.json", reg)
    complete = {"registration_identity": reg["identity"], "scenes": [None]*4}
    write_json(tmp_path/"COMPLETE.json", complete)
    m = {"schema": "action_effect_repair_dataset_v1", "status": "COMPLETE", "split": "train", "deployment_inputs": False,
         "registration": artifact(tmp_path/"REGISTRATION.json"), "population_completion": artifact(tmp_path/"COMPLETE.json"),
         "registration_identity": reg["identity"], "requested_scenes": 4, "entries": entries,
         "base_checkpoint": reg["base_checkpoint"], "numeric_runtime_hash": reg["numeric_runtime_hash"],
         "score_protocol_hash": reg["spec"]["score_protocol_hash"]}
    m["identity"] = digest(m); write_json(tmp_path/"REPAIR.json", m)
    class Reader:
        binding = {"base_checkpoint": reg["base_checkpoint"], "base_checkpoint_binding_hash": "b"*64}
        rows = context_rows
        manifest_path = context_manifest
        def __getitem__(self, scene_id): return contexts[scene_id]
    # Test-only bypass is explicit. Production prepare_dataset never disables receipts.
    return RepairDataset(tmp_path/"REPAIR.json", Reader(), verify_feedback=False), contexts


def refiner(teacher=None):
    return EffectGuidedRefiner(CoefficientRefiner(RefinerConfig(model=cfg(), active_coefficients=ACTIVE_GRID_COEFFICIENTS)),
                               effect_teacher=teacher)


def test_repair_role_and_observation_contracts_reject_invalid_joins(tmp_path):
    dataset, _ = fixture_dataset(tmp_path)
    assert len(dataset.fit) == 3 and len(dataset.validation) == 1
    for change in ("log", "base", "window", "identity", "unknown_target"):
        m = deepcopy(dataset.manifest)
        if change == "log":
            m["entries"][-1]["source_log"] = "log0"
        elif change == "base": m["entries"][0]["base_physical"][0][0] += 1
        elif change == "window": m["entries"][1]["coefficients"][1] = .1
        elif change == "identity": m["entries"][0]["target_physical"][0][0] += .1
        else: m["entries"][0]["target_kind"] = "GT_unverified"
        m.pop("identity"); m["identity"] = digest(m)
        p = tmp_path/(change+".json"); write_json(p,m)
        with pytest.raises(ValueError): RepairDataset(p,dataset.reader,verify_feedback=False)


def test_compact_self_hash_cannot_replace_native_row_provenance(tmp_path, monkeypatch):
    dataset, _ = fixture_dataset(tmp_path)
    from tools.action_effect import build_effect_pairs
    calls = []
    def native_rebuild(root):
        calls.append(root)
        return {"repair_dataset": {"sha256": "not-the-compact-file-checksum"}}
    monkeypatch.setattr(build_effect_pairs, "build", native_rebuild)
    with pytest.raises(ValueError, match="deterministic native receipt"):
        RepairDataset(dataset.path, dataset.reader)
    assert calls == [tmp_path]


@pytest.mark.parametrize("arm", ["A0", "A1"])
def test_entry_microbatch_preserves_complete_update_objective_and_frozen_teacher(tmp_path, arm):
    dataset, _ = fixture_dataset(tmp_path)
    first = refiner(ActionEffectModel(cfg(), spec()) if arm == "A1" else None)
    second = refiner(ActionEffectModel(cfg(), spec()) if arm == "A1" else None)
    one = make_step(dataset,arm=arm,microbatch_scenes=4,device="cpu")(first,[0,1,2,1],0)
    micro = make_step(dataset,arm=arm,microbatch_scenes=1,device="cpu")(second,[0,1,2,1],0)
    assert one["preserve_denominator"] == micro["preserve_denominator"] == 2
    for name in one: assert one[name] == pytest.approx(micro[name], abs=2e-7)
    for p,q in zip(first.parameters(), second.parameters()):
        if p.requires_grad: torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-7)
        else: assert p.grad is q.grad is None


def test_thin_a_training_resume_export_and_terminal_no_extra_step(tmp_path):
    dataset, _ = fixture_dataset(tmp_path)
    cfg_t = recipe()
    def run(out, model, **kw):
        identity = {"source_commit": "synthetic-test-only", "arm":"A0", "base_binding_hash":"b"*64,
                    "data_artifact": artifact(dataset.path), "model_config": asdict(model.refiner.config)}
        return run_training(model,output=out,identity=identity,dataset_size=3,
            step_fn=make_step(dataset,arm="A0",microbatch_scenes=1,device="cpu"),cfg=cfg_t,device="cpu",seed=5,max_updates=4,**kw)
    continuous = refiner(); run(tmp_path/"continuous", continuous)
    run(tmp_path/"resumed", refiner(), stop_after=2)
    resumed = refiner(); run(tmp_path/"resumed", resumed, resume=True)
    assert state_hash(continuous) == state_hash(resumed)
    out = run(tmp_path/"resumed", resumed, resume=True)
    assert out["segment_updates"] == 0 and out["termination_reason"] == "MAX_UPDATES"
    loaded = load_refiner(tmp_path/"resumed"/"step_000004.pt",expected_base_binding_hash="b"*64,
                           expected_data_artifact=artifact(dataset.path))
    resumed.eval()  # Match deployment attention backend; training mode is a different kernel path.
    context, a, _ = batch()
    with torch.no_grad():
        torch.testing.assert_close(loaded(context,a)["trajectory"],resumed(context,a)["trajectory"],rtol=0,atol=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_refiner(tmp_path/"resumed"/"step_000004.pt",expected_base_binding_hash="x"*64)


def test_evaluation_denominator_does_not_promote_valid_subset_to_full_pdms():
    metrics = {"score":.9,"no_at_fault_collisions":1.,"drivable_area_compliance":1.,"time_to_collision_within_bound":1.}
    rows = [{"status":"VALID_SCORED","metrics":metrics,"base_metrics":{**metrics,"score":0.},"distance_m":.1},
            {"status":"PARTIAL"}]
    result = _summary(rows,3)
    assert result["PDMS"] is None and result["PDMS_valid_subset"] == .9
    assert result["requested_scenes"] == 3 and result["valid_paired_scenes"] == 1
    assert result["transient_failures"] == 1 and result["unprocessed_scenes"] == 1 and result["zero_repaired"] == 1


def test_arm_source_whitelist_rejects_external_or_missing_teacher(tmp_path):
    dataset, _ = fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match="A0 must not"):
        build_model(recipe(),dataset,arm="A0",seed=1,device="cpu",effect_checkpoint="anything")
    with pytest.raises(ValueError, match="require an explicit"):
        build_model(recipe(),dataset,arm="A1",seed=1,device="cpu")


def _synthetic_scoring_input(tmp_path):
    from tools.action_effect.common import read_json
    for name in ("checkpoint.pt", "repair.json", "context.json"):
        (tmp_path/name).write_text("marked synthetic transaction fixture")
    source = {"spec": {"navsim_root": "/fixture", "maps_root": "/fixture", "cache_roots": ["/fixture"],
                       "campaign_id": "test_campaign", "score_protocol_hash": "7"*64, "allow_engineering_controls": True},
              "base_checkpoint": {"sha256": "4"*64}, "numeric_runtime_hash": "5"*64,
              "population": [{"context": {"scene_id": "scene_1"}, "observation": {"observation_hash": "9"*64,
                    "metric_cache": {"path": "/fixture/log_1/unknown/scene_1/metric_cache.pkl", "sha256": "6"*64}}}]}
    source["identity"] = digest(source); write_json(tmp_path/"G0.json",source)
    reg = {"scene_ids": ["scene_1"], "role": "validation", "arm": "A0", "base_checkpoint": source["base_checkpoint"],
           "numeric_runtime_hash": source["numeric_runtime_hash"], "score_protocol_hash": "7"*64,
           "checkpoint": artifact(tmp_path/"checkpoint.pt"), "repair_dataset": artifact(tmp_path/"repair.json"),
           "context_manifest": artifact(tmp_path/"context.json"), "g0_registration": artifact(tmp_path/"G0.json")}
    reg["identity"] = digest(reg); write_json(tmp_path/"PREDREG.json", reg)
    physical = np.column_stack((np.arange(1,9)*.5,np.zeros(8),np.zeros(8))).tolist()
    row = {"scene_id": "scene_1", "source_log": "log_1", "role": "validation", "physical": physical,
           "base_physical": physical, "base_metrics": None, "base_feedback": None, "base_canonical_receipt": None}
    row["identity"] = digest(row)
    predictions = {"schema":"action_effect_refiner_predictions_v1", "status":"COMPLETE", "registration":artifact(tmp_path/"PREDREG.json"),
                   "registration_identity":reg["identity"],"requested_scenes":1,"entries":[row]}
    predictions["identity"] = digest(predictions); write_json(tmp_path/"PRED.json",predictions)
    return tmp_path/"PRED.json"


def test_offline_scoring_retry_and_completed_reuse_preserve_trajectory(tmp_path, monkeypatch):
    from test_g0_pipeline import factory as fixture_factory, prepared_record
    from tools.action_effect import simulate_interventions as g0
    from tools.action_effect.evaluate_refiner import score
    factory = fixture_factory.__wrapped__(tmp_path)
    path = _synthetic_scoring_input(tmp_path)
    observed_trajectories = []
    class Engine:
        calls = 0
        def load_context(self, request):
            return SimpleNamespace(ego_state=SimpleNamespace(dynamic_car_state=SimpleNamespace(
                rear_axle_velocity_2d=SimpleNamespace(array=np.array([1.,0.])))))
        def score(self, request, policy):
            self.calls += 1; observed_trajectories.append(request.trajectory)
            if self.calls == 1:
                return {"schema":"st_world_canonical_consequence_v1","identity":request.identity(),
                    "valid":False,"status":"TRANSIENT_SCORING_FAILURE","metrics":None,"metrics_valid":{},
                    "error":{"retryable":True,"type":"TimeoutError","message":"synthetic timeout"}}
            return prepared_record(factory,request)
    engine = Engine(); monkeypatch.setattr(g0,"worker_engine",lambda spec:engine)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): monkeypatch.setenv(key,"1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = {"g0":{"max_simulation_budget":2,"retry_budget":1}}
    result = score(config,predictions_path=path,output=tmp_path/"score",max_simulation_budget=2,max_seconds=30)
    assert result["status"] == "COMPLETE" and engine.calls == 2
    assert observed_trajectories[0] == observed_trajectories[1]
    assert result["summary"]["valid_refined_scenes"] == 1 and result["summary"]["valid_paired_scenes"] == 0
    again = score(config,predictions_path=path,output=tmp_path/"score",max_simulation_budget=2,max_seconds=30,resume=True)
    assert again["status"] == "COMPLETE" and engine.calls == 2
    candidate = result["entries"][0]
    receipt = json.loads(Path(candidate["canonical_receipt"]["path"]).read_text())
    Path(receipt["canonical"]["path"]).write_bytes(b"corrupt synthetic native record")
    with pytest.raises(ValueError, match="stale"):
        score(config,predictions_path=path,output=tmp_path/"score",max_simulation_budget=2,max_seconds=30,resume=True)


def test_scoring_budget_exhaustion_remains_partial_without_valid_zero(tmp_path, monkeypatch):
    from tools.action_effect import simulate_interventions as g0
    from tools.action_effect.evaluate_refiner import score
    path = _synthetic_scoring_input(tmp_path)
    class Engine:
        def load_context(self, request):
            return SimpleNamespace(ego_state=SimpleNamespace(dynamic_car_state=SimpleNamespace(
                rear_axle_velocity_2d=SimpleNamespace(array=np.array([1.,0.])))))
        def score(self, request, policy):
            return {"schema":"st_world_canonical_consequence_v1","identity":request.identity(),
                "valid":False,"status":"TRANSIENT_SCORING_FAILURE","metrics":None,"metrics_valid":{},
                "error":{"retryable":True,"type":"TimeoutError","message":"synthetic timeout"}}
    monkeypatch.setattr(g0,"worker_engine",lambda spec:Engine())
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): monkeypatch.setenv(key,"1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = score({"g0":{"max_simulation_budget":1,"retry_budget":0}},predictions_path=path,
                   output=tmp_path/"score",max_simulation_budget=1,max_seconds=30)
    assert result["status"] == "PARTIAL" and result["summary"]["PDMS"] is None
    assert result["summary"]["valid_refined_scenes"] == 0 and not (tmp_path/"score"/"COMPLETE.json").exists()
