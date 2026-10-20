"""Resource-fork negative tests use small synthetic files, not actual model data."""
from copy import deepcopy
from pathlib import Path
import os

import pytest

from tools.action_effect import migrate_execution_namespace as migration
from tools.action_effect.common import artifact, digest, read_json, write_json


def signed(row):
    row = {k:v for k,v in row.items() if k!="identity"};return {**row,"identity":digest(row)}


@pytest.fixture
def setup(tmp_path,monkeypatch):
    oldroot,newroot=tmp_path/"old",tmp_path/"new"
    oldcfg={"artifact_root":str(oldroot),"resources":{"minimum_free_bytes":100,"artifact_cap_bytes":1000},
            "feature_shard_scenes":2,"seed":42,"training":{"learning_rate":1e-4}}
    newcfg={**oldcfg,"artifact_root":str(newroot),"resources":{"minimum_free_bytes":50,"artifact_cap_bytes":500}}
    scenes=[{"scene_id":"scene_1","source_log":"log_1","effect_role":"fit","observation_hash":"a"*64},
            {"scene_id":"scene_2","source_log":"log_2","effect_role":"validation","observation_hash":"b"*64}]
    write_json(oldroot/"COHORT.json",signed({"entries":scenes,"fixture":True}))
    binding=signed({"config_hash":digest(oldcfg),"cohort":artifact(oldroot/"COHORT.json"),
        "base_checkpoint":{"sha256":"c"*64},"base_checkpoint_binding_hash":"d"*64,"extraction_contract":{"source":"same"}})
    write_json(oldroot/"CAMPAIGN_BINDING.json",binding)
    contexts=oldroot/"contexts";contexts.mkdir()
    head=contexts/"S0_ACTION_HEAD.pt";head.write_bytes(b"synthetic head, not actual model")
    headreceipt={"schema":"action_effect_original_action_head_v1","artifact":artifact(head),
        "base_binding_hash":binding["base_checkpoint_binding_hash"],"strict_original_master":True}
    write_json(contexts/"S0_ACTION_HEAD.COMPLETE.json",headreceipt)
    tag=digest([binding["identity"],[r["scene_id"] for r in scenes]])
    shard=contexts/(tag+".pt");shard.write_bytes(b"synthetic exact payload, not actual features")
    entries=[{"scene_id":r["scene_id"],"source_log":r["source_log"],"role":r["effect_role"],
              "observation_hash":r["observation_hash"],"shard_offset":i,"shard":artifact(shard)} for i,r in enumerate(scenes)]
    done=signed({"schema":"action_effect_context_shard_v1","binding_identity":binding["identity"],
        "base_binding_hash":binding["base_checkpoint_binding_hash"],"single_encoder_per_scene":True,
        "shard":artifact(shard),"entries":entries})
    receipt=contexts/(tag+".COMPLETE.json");write_json(receipt,done)
    cache=oldroot/"g0"/"candidate_cache";cache.mkdir(parents=True)
    alters={};calls=[]
    def prepare(cfg):
        calls.append(cfg);newroot.mkdir(exist_ok=True)
        write_json(newroot/"COHORT.json",read_json(oldroot/"COHORT.json"))
        new=signed({**binding,"config_hash":digest(cfg),"cohort":artifact(newroot/"COHORT.json")})
        new.update(alters)
        write_json(newroot/"CAMPAIGN_BINDING.json",new)
        return new
    monkeypatch.setattr(migration,"prepare",prepare)
    return {"old":oldcfg,"new":newcfg,"binding":binding,"shard":shard,"receipt":receipt,
            "head_receipt":contexts/"S0_ACTION_HEAD.COMPLETE.json","cache":cache,"alters":alters,"prepare_calls":calls}


def test_exact_hardlink_rebind_preserves_source_and_repeats_idempotently(setup):
    s=setup;oldbytes=s["shard"].read_bytes();receiptbytes=s["receipt"].read_bytes()
    first=migration.migrate(s["old"],s["new"])
    second=migration.migrate(s["old"],s["new"])
    assert first==second and first["scenes_reused"]==2 and first["completed_shards_reused"]==1
    newreceipt=read_json(first["payloads"][0]["new_receipt"]["path"])
    newpath=Path(newreceipt["shard"]["path"])
    assert newpath.stat().st_ino==s["shard"].stat().st_ino and newpath.stat().st_dev==s["shard"].stat().st_dev
    assert newpath.read_bytes()==oldbytes and s["receipt"].read_bytes()==receiptbytes
    assert newreceipt["binding_identity"]!=s["binding"]["identity"]
    assert first["new_inference_calls"]==first["new_simulation_calls"]==first["new_model_updates"]==0
    assert (Path(s["new"]["artifact_root"])/"g0/candidate_cache").resolve()==s["cache"].resolve()


def test_scientific_recipe_change_rejects_before_new_prepare(setup):
    s=setup;s["new"]["seed"]=43
    with pytest.raises(ValueError,match="Only explicit resource"):migration.migrate(s["old"],s["new"])
    assert not s["prepare_calls"]


def test_source_config_must_match_its_original_binding(setup):
    s=setup;s["old"]["resources"]["minimum_free_bytes"]=101
    with pytest.raises(ValueError,match="Source config"):migration.migrate(s["old"],s["new"])


@pytest.mark.parametrize("key,value",[("base_checkpoint",{"sha256":"e"*64}),
    ("base_checkpoint_binding_hash","f"*64),("extraction_contract",{"source":"changed"}),
    ("cohort",{"sha256":"1"*64})])
def test_base_cohort_or_source_changes_cannot_be_resource_migration(setup,key,value):
    s=setup;s["alters"][key]=value
    with pytest.raises(ValueError,match="Scientific/numeric/cohort"):migration.migrate(s["old"],s["new"])


@pytest.mark.parametrize("action",["delete_payload","corrupt_payload","delete_head","delete_cache","wrong_destination","dangling_destination"])
def test_missing_or_foreign_source_and_links_reject(setup,action):
    s=setup
    if action=="delete_payload":s["shard"].unlink()
    elif action=="corrupt_payload":s["shard"].write_bytes(b"corrupt")
    elif action=="delete_head":s["head_receipt"].unlink()
    elif action=="delete_cache":s["cache"].rmdir()
    else:
        p=Path(s["new"]["artifact_root"])/"g0/candidate_cache";p.parent.mkdir(parents=True)
        if action=="wrong_destination":p.mkdir()
        else:p.symlink_to(p.parent/"missing",target_is_directory=True)
    with pytest.raises((OSError,ValueError)):migration.migrate(s["old"],s["new"])
