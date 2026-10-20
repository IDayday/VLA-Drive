"""CPU admission/fault tests with explicit synthetic feature records, not DEV results."""
from collections import OrderedDict
from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.action_effect.common import artifact, digest, write_json
from tools.action_effect.evaluation_context_data import (
    EvaluationContextReader, FEATURE_SCHEMA, REGISTRATION_SCHEMA, MANIFEST_SCHEMA, SHARD_SCHEMA,
    validate_registration, validate_feature_record, validate_shard, validate_manifest, shard_population,
)
from starVLA.model.modules.flow_posttraining.policy_snapshot import tensor_hash


def registration():
    rows=[{"scene_id":f"scene{i:04d}","source_log":f"log{i%16}","role":"dev_report","split":"dev",
           "observation_hash":"a"*64} for i in range(1696)]
    reg={"schema":REGISTRATION_SCHEMA,"split":"dev","role":"dev_report","for_training":False,
         "requested_scenes":1696,"requested_logs":16,"population":rows,"train_dev_log_overlap":0,
         "dtype":"float32","shard_scenes":16,"base_checkpoint_binding_hash":"b"*64,"numeric_runtime_hash":"c"*64,"seed":42}
    reg["identity"]=digest(reg);return reg


def reidentify(d):
    d.pop("identity",None);d["identity"]=digest(d);return d


def feature(tmp_path,reg,index=0):
    expected=deepcopy(reg["population"][index]);memory=torch.arange(152*2048,dtype=torch.float32).reshape(152,2048)/10000
    physical=torch.zeros(8,3);physical[:,0]=torch.arange(1,9)*2.5
    path=tmp_path/f"historical{index}.npz";np.savez_compressed(path,trajectory=physical.numpy())
    expected["historical_prediction"]=artifact(path)
    from tools.foresight.checkpoints import scene_noise
    record={"schema":FEATURE_SCHEMA,**{k:expected[k] for k in ("scene_id","source_log","role","split","observation_hash")},
            "base_binding_hash":reg["base_checkpoint_binding_hash"],"evaluation_binding":reg["identity"],
            "memory":memory,"ego":torch.zeros(1,4),"condition":memory[144:].clone(),"W_count":144,
            "raw":torch.zeros(8,4),"physical":physical,"noise":scene_noise(expected["scene_id"],42,"cpu")[0],"historical_physical_exact":True}
    record["raw_hash"]=tensor_hash(record["raw"][None]);record["physical_hash"]=tensor_hash(physical[None])
    return expected,record


def test_registered_population_is_full_unselected_development_and_shard_ranks_do_not_change_order():
    reg=registration();validate_registration(reg)
    chunks=shard_population(reg)
    assert len(chunks)==106 and all(len(c)==16 for c in chunks)
    for workers in (1,2,4,8):
        assigned={i:[(j,c) for j,c in enumerate(chunks) if j%workers==i] for i in range(workers)}
        joined=sorted([item for parts in assigned.values() for item in parts])
        assert [r["scene_id"] for _,c in joined for r in c]==[r["scene_id"] for r in reg["population"]]


@pytest.mark.parametrize("change",["train","navtest","fit","partial","overlap","dtype","duplicate"])
def test_wrong_roles_population_and_precision_reject_before_gpu(change):
    reg=registration()
    if change in ("train","navtest"):reg["split"]=change
    elif change=="fit":reg["population"][0]["role"]="fit"
    elif change=="partial":reg["population"].pop()
    elif change=="overlap":reg["train_dev_log_overlap"]=1
    elif change=="dtype":reg["dtype"]="bfloat16"
    else:reg["population"][1]["scene_id"]=reg["population"][0]["scene_id"]
    with pytest.raises(ValueError):validate_registration(reidentify(reg))


def test_feature_whitelist_exact_planner_and_historical_values(tmp_path):
    reg=registration();expected,record=feature(tmp_path,reg)
    validate_feature_record(record,expected,reg,compare_historical=True)
    for key,value in [("target_raw",torch.ones(8,4)),("score",1.),("metric_cache","forbidden"),("future",torch.ones(8,3))]:
        with pytest.raises(ValueError,match="whitelist"):validate_feature_record({**record,key:value},expected,reg)
    bad={**record,"condition":record["condition"].clone()};bad["condition"][0,0]+=1
    with pytest.raises(ValueError,match="planner"):validate_feature_record(bad,expected,reg)
    bad={**record,"memory":record["memory"].bfloat16()}
    with pytest.raises(ValueError,match="FP32"):validate_feature_record(bad,expected,reg)
    bad={**record,"noise":record["noise"]+.01}
    with pytest.raises(ValueError,match="registered original scene seed"):validate_feature_record(bad,expected,reg)
    bad={**record,"physical":record["physical"]+.01};bad["physical_hash"]=tensor_hash(bad["physical"][None])
    with pytest.raises(ValueError,match="original FP32"):validate_feature_record(bad,expected,reg,compare_historical=True)


def test_feature_different_scene_and_nonfinite_reject(tmp_path):
    reg=registration();expected,record=feature(tmp_path,reg)
    bad={**record,"scene_id":"wrong"}
    with pytest.raises(ValueError,match="identity"):validate_feature_record(bad,expected,reg)
    bad={**record,"memory":record["memory"].clone()};bad["memory"][0,0]=float("nan")
    with pytest.raises(ValueError,match="FP32"):validate_feature_record(bad,expected,reg)
    bad={**record,"ego":record["ego"].requires_grad_()}
    with pytest.raises(ValueError,match="FP32"):validate_feature_record(bad,expected,reg)


def written_shard(tmp_path,reg,index=0):
    expected,record=feature(tmp_path,reg,index)
    tag=digest([reg["identity"],[expected["scene_id"]]])
    path=tmp_path/(tag+".pt");torch.save([record],path)
    entry={k:expected[k] for k in ("scene_id","source_log","role","split","observation_hash")}
    entry.update(shard=artifact(path),shard_offset=0)
    done={"schema":SHARD_SCHEMA,"evaluation_binding":reg["identity"],"split":"dev","for_training":False,
          "single_encoder_per_scene":True,"historical_physical_exact":True,"shard":artifact(path),"entries":[entry]}
    reidentify(done);return expected,record,done


def test_complete_shard_receipt_requires_hash_coverage_and_exact_offset(tmp_path):
    reg=registration();expected,_,done=written_shard(tmp_path,reg)
    validate_shard(done,reg,[expected])
    bad=deepcopy(done);bad["entries"][0]["shard_offset"]=1
    with pytest.raises(ValueError,match="offset"):validate_shard(reidentify(bad),reg,[expected])
    bad=deepcopy(done);bad["historical_physical_exact"]=False
    with pytest.raises(ValueError,match="foreign"):validate_shard(reidentify(bad),reg,[expected])
    Path(done["shard"]["path"]).write_bytes(b"partial write")
    with pytest.raises(ValueError,match="stale"):validate_shard(done,reg,[expected])


def test_reader_record_access_has_bounded_lru_and_detects_changed_payload(tmp_path):
    reg=registration();e0,_,d0=written_shard(tmp_path,reg,0);e1,_,d1=written_shard(tmp_path,reg,1)
    # Unit-test the read primitive; this bypasses full-manifest construction and
    # therefore cannot be interpreted as a complete DEV cache qualification.
    reader=EvaluationContextReader.__new__(EvaluationContextReader)
    reader.rows={d["entries"][0]["scene_id"]:d["entries"][0] for d in (d0,d1)}
    reader.population={e["scene_id"]:e for e in (e0,e1)};reader.binding=reg
    reader.cache=OrderedDict();reader.checked={};reader.max_shards=1;reader.loads=0
    for e in (e0,e1,e0):assert reader[e["scene_id"]]["role"]=="dev_report"
    assert reader.loads==3 and len(reader.cache)==1
    Path(d0["shard"]["path"]).write_bytes(b"corrupt after validated access")
    with pytest.raises(ValueError,match="stale"):reader[e0["scene_id"]]


def test_partial_dev_never_becomes_manifest_and_training_reader_rejects_dev(tmp_path):
    reg=registration();m={"schema":MANIFEST_SCHEMA,"status":"COMPLETE","split":"dev","role":"dev_report",
        "for_training":False,"dtype":"float32","evaluation_binding":reg["identity"],
        "base_checkpoint_binding_hash":reg["base_checkpoint_binding_hash"],"numeric_runtime_hash":reg["numeric_runtime_hash"],
        "requested_scenes":1696,"requested_logs":16,"entries":[],"shard_receipts":[]}
    reidentify(m)
    with pytest.raises(ValueError,match="Partial"):validate_manifest(m,reg)
    cohort={"entries":[]};reidentify(cohort);write_json(tmp_path/"cohort.json",cohort)
    binding={"cohort":artifact(tmp_path/"cohort.json")};reidentify(binding);write_json(tmp_path/"binding.json",binding)
    m["max_samples"]=1696;reidentify(m);write_json(tmp_path/"dev.json",m)
    from tools.action_effect.context_data import ContextReader
    with pytest.raises(ValueError):ContextReader(tmp_path/"dev.json",tmp_path/"binding.json")
    train={"schema":"action_effect_context_manifest_v1","for_training":True};reidentify(train);write_json(tmp_path/"train.json",train)
    with pytest.raises(ValueError,match="DEV-only"):EvaluationContextReader(tmp_path/"train.json")


def test_finalization_cannot_promote_missing_shards_by_marker_presence(tmp_path):
    from tools.action_effect.collect_evaluation_context import finalize
    reg=registration();write_json(tmp_path/"EVAL_REGISTRATION.json",reg)
    write_json(tmp_path/"COMPLETE.json",{"previous_marker":"does not prove complete valid current shards"})
    with pytest.raises(FileNotFoundError):finalize(tmp_path)
    assert not (tmp_path/"EVAL_CONTEXT_MANIFEST.json").exists()
