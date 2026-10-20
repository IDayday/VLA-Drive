"""Marked synthetic accounting/comparison tests, not real simulator results."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tools.action_effect import audit_navsim_feedback as audit
from tools.action_effect import simulate_interventions as g0
from tools.action_effect.common import artifact, digest, read_json, write_json
from starVLA.model.modules.st_world.consequence import ConsequenceEngine, ConsequenceSpec


@pytest.fixture
def factory(tmp_path):
    spec = importlib.util.spec_from_file_location("_feedback_replay_marked_fixture", Path(__file__).with_name("test_feedback_schema.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.record_factory.__wrapped__(tmp_path)


def synthetic_run(tmp_path, monkeypatch, factory, *, fail_once=False, invalid_prediction=False, budget=2, retry_budget=0):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    record, request, wrapped = factory()
    cfg = {"artifact_root": str(tmp_path)}
    spec = ConsequenceSpec("/fixture_navsim", "/fixture_maps", ("/fixture_cache",), request.campaign_id, request.score_protocol_hash, True)
    root = tmp_path/"g0"/"population_000064"
    reg = {"identity":"fixture_registration", "config_hash":digest(cfg), "spec":asdict(spec),
           "numeric_runtime_hash":request.inference_runtime_hash}
    write_json(root/"REGISTRATION.json",reg); write_json(root/"COMPLETE.json",{"fixture":True})
    pin = artifact(tmp_path/"FIXTURE_ONLY_NOT_REAL_SIMULATION.json")
    item = {"context":{"scene_id":request.scene_id,"source_log":request.source_log,"role":"fit"},
        "observation":{"observation_hash":"9"*64},"source_feedback":pin,"source_canonical_receipt":pin,
        "source_scene_receipt":pin,"request":request}
    monkeypatch.setattr(audit,"select_inputs",lambda *_args,**_kwargs:(reg,[item]))
    original = g0.verified_feedback
    monkeypatch.setattr(audit,"verified_feedback",lambda row: wrapped if row["feedback"]==pin else original(row))
    class Engine:
        calls = 0
        def __init__(self,_spec): pass
        def score(self,req,version):
            Engine.calls += 1
            if invalid_prediction:
                return ConsequenceEngine._failure(req.identity(),"INVALID_PREDICTION",ValueError("fixture invalid prediction"),False)
            if fail_once and Engine.calls==1:
                return ConsequenceEngine._failure(req.identity(),"TRANSIENT_SCORING_FAILURE",TimeoutError("fixture timeout"),True)
            return deepcopy(record)
    monkeypatch.setattr(audit,"ConsequenceEngine",Engine)
    kwargs = dict(g0_root=root,output=tmp_path/"feedback_audit",max_samples=1,repeats=2,
                  max_simulation_budget=budget,retry_budget=retry_budget,max_seconds=30)
    return audit.run(cfg,**kwargs), Engine, cfg, kwargs


def test_exact_comparison_excludes_cost_and_preserves_zero(factory):
    left, _, _ = factory(collision=True)
    right = deepcopy(left);right["cost"]={"worker_pid":999,"worker_wall_seconds":100.}
    result = audit.compare_records(left,right)
    assert result["status"]=="PASS" and left["metrics"]["score"]==0


@pytest.mark.parametrize("field",["metrics","consequences","reference_contract"])
def test_changed_metrics_events_or_reference_are_not_repeat_pass(factory,field):
    left, _, _ = factory();right=deepcopy(left)
    if field=="metrics":right[field]["score"]+=.01
    elif field=="consequences":right[field]["road"]["event"]=True
    else:right[field]["input_sha256"]="a"*64
    assert audit.compare_records(left,right)["status"]=="FAIL"


def test_one_ulp_state_difference_is_reported_not_rounded_to_pass(factory):
    left, _, _ = factory();right=deepcopy(left)
    right["executed_states"][1,0]=np.nextafter(right["executed_states"][1,0],np.inf)
    result=audit.compare_records(left,right)
    assert result["status"]=="INCONCLUSIVE"
    assert result["array_checks"]["executed_states"]["max_absolute_difference"]>0


def test_two_slots_are_two_real_backend_calls_and_resume_is_zero_new(tmp_path,monkeypatch,factory):
    report,Engine,cfg,kwargs=synthetic_run(tmp_path,monkeypatch,factory)
    assert report["status"]=="PASS" and Engine.calls==2
    assert report["reserved_calls"]==2 and report["completed_replays"]==2
    result=audit.run(cfg,**kwargs,resume=True)
    assert result["new_calls"]==0 and result["reused"] and Engine.calls==2
    assert read_json(kwargs["output"]/"calls"/"COUNTER.json")=={"reserved_calls":2,"reserved_retries":0}


def test_transient_retry_is_separate_and_charged(tmp_path,monkeypatch,factory):
    report,Engine,_,_=synthetic_run(tmp_path,monkeypatch,factory,fail_once=True,budget=3,retry_budget=1)
    assert report["status"]=="PASS" and Engine.calls==3
    assert report["reserved_calls"]==3 and report["reserved_retries"]==1


def test_exhausted_retry_does_not_mark_complete_or_fill_zero(tmp_path,monkeypatch,factory):
    report,Engine,_,kwargs=synthetic_run(tmp_path,monkeypatch,factory,fail_once=True)
    assert report["status"]=="PARTIAL" and Engine.calls==1
    assert not (kwargs["output"]/"COMPLETE.json").exists()
    assert report["completed_replays"]==0 and report["errors"]
    assert report["scenes"][0]["pair_comparison"] is None


def test_invalid_prediction_stays_invalid_not_successful_zero_score(tmp_path,monkeypatch,factory):
    report,Engine,_,kwargs=synthetic_run(tmp_path,monkeypatch,factory,invalid_prediction=True)
    assert report["status"]=="PARTIAL" and Engine.calls==1
    assert report["completed_replays"]==0 and report["recorded_replay_slots"]==1
    assert report["scenes"][0]["comparisons_to_source"]==[]
    assert not (kwargs["output"]/"COMPLETE.json").exists()


def test_completed_audit_missing_feedback_is_rejected_without_rescore(tmp_path,monkeypatch,factory):
    report,Engine,cfg,kwargs=synthetic_run(tmp_path,monkeypatch,factory)
    Path(report["scenes"][0]["replays"][0]["feedback"]["path"]).unlink()
    with pytest.raises(ValueError,match="Missing/stale"):
        audit.run(cfg,**kwargs,resume=True)
    assert Engine.calls==2


def test_over_budget_is_not_silently_executed(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):monkeypatch.setenv(key,"1")
    with pytest.raises(ValueError,match="sixteen"):
        audit.run({"artifact_root":str(tmp_path)},g0_root=tmp_path/"g0",output=tmp_path/"audit",max_simulation_budget=17)
