"""Full artifact/state-machine regression with explicit synthetic tensors/labels.

This is NOT real S0 or a real NAVSIM smoke; those have separate recorded evidence.
"""
from dataclasses import asdict, replace
from pathlib import Path
import torch
import pytest
from iqe.config import load_config
from iqe.contracts import FeatureBundle, ScoreRecord
from iqe.io import atomic_torch, atomic_json, file_hash, digest, read_json
from iqe.data.manifests import save_scenes
from iqe.pipeline import Pipeline
from iqe.rounds import run_round
from iqe.scoring.reference import COMPONENTS
from conftest import scene, TensorAdapter
from test_cli_registry_export import UnitFramework


class SyntheticLabelBackend:
    protocol_hash = "unit_protocol"
    source_hash = "unit_navsim"
    def score(self,candidate,scene,repetition=0):
        value = .95 if candidate.expert_id == "ground_truth" else .1 + .2 * int(candidate.expert_id[7:])
        return ScoreRecord(1,candidate.key,self.protocol_hash,scene.metric_context_hash,"unit_reference",value,
            {c:1. for c in COMPONENTS},{c:True for c in COMPONENTS},"SYNTHETIC_UNIT_ONLY","unit",42,repetition,True,None)


def pipeline_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr("iqe.s0_adapter.build_query_framework",lambda *args,**kwargs:UnitFramework())
    monkeypatch.setattr("iqe.query_base.build_query_framework",lambda *args,**kwargs:UnitFramework())
    config=load_config("configs/iqe/main.yaml")
    config["output_root"]=str(tmp_path/"out")
    config["s0"]["contract"]=str(tmp_path/"contract.json")
    config["expert_train"].update(global_batch_size=4,micro_batch_size=2,max_optimizer_steps=2,warmup_steps=0,validation_every_steps=1)
    config["scorer"].update(global_scene_batch_size=4,micro_batch_size=2,max_optimizer_steps=2,warmup_steps=0,validation_every_steps=1,hidden_size=16,num_layers=1,attention_heads=2)
    config["expert_data"]["support_limit"]=1
    config["selection"]["delta_grid"]=[0.,.1];config["selection"]["eta_grid"]=[0.,.9]
    config["execution"]["scientific_expansion"]=False
    checkpoint=tmp_path/"base.pt"
    torch.manual_seed(42)
    framework=UnitFramework()
    atomic_torch(checkpoint,{"kind":"iqe_query_base","framework_contract_hash":"unit_framework","model":framework.state_dict()})
    contract={"binding_status":"LOCKED_QUERY_S0","query_checkpoint":str(checkpoint),"query_checkpoint_hash":file_hash(checkpoint),
        "framework_contract_hash":"unit_framework","source_config":{},"source_root":"unit","source_commit":"unit",
        "query_architecture":{"width":8},"trajectory":asdict(TensorAdapter().contract),"tokenizer_hash":"tok","prompt_hash":"prompt",
        "transform_hash":"transform","normalizer_hash":digest(asdict(TensorAdapter().contract)),"camera_time_hash":"camera","iqe_code_hash":"unit_code",
        "metric":{"python_source_hash":"unit_navsim"},"base_mode":"smoke"}
    atomic_json(tmp_path/"contract.json",contract)
    root=Path(config["output_root"]);atomic_json(root/"LOCKED_S0.json",contract)
    records=[]
    for role,offset,n in (("incremental_fit",0,12),("stage_val",20,4),("selector_cal",30,4),("dev_report",40,4)):
        for i in range(offset,offset+n):
            target=torch.zeros(4,4);target[:,3]=1.
            target_ref=tmp_path/f"target_{i}.pt";atomic_torch(target_ref,{"token":f"s{i}","ego":target})
            context=tmp_path/f"context_{i}.json";atomic_json(context,{"unit":i})
            records.append(scene(i,role,target_id=file_hash(target_ref),target_ref=str(target_ref),metric_context_ref=str(context),metric_context_hash=file_hash(context)))
    save_scenes(root/"scenes.json",records)
    pipeline=Pipeline(config,mode="smoke",max_samples=32)
    pipeline.use_locked_base();pipeline._backend=SyntheticLabelBackend()
    atomic_json(root/"protocol_audit/SCORE_CONTEXT_EVIDENCE.json", {"passed":True,"original_entry_equivalence":True,
        "protocol_hash":"unit_protocol","evidence_type":"SYNTHETIC_UNIT_ONLY"})
    pipeline.registry.append({"expert_id":"expert_0","created_round":0,"parent":"base","architecture":{"variant":"independent"},
        "checkpoint_hash":file_hash(checkpoint),"train_manifest_hash":"unit","shared_contract_hash":digest(contract),"status":"served","gates":{}})
    for s in records:
        f=FeatureBundle(torch.randn(1,3,8),torch.randn(1,1,8),torch.ones(1,3,dtype=torch.bool),digest(contract),(s.scene_id,))
        pipeline.features.put(pipeline.feature_record(s),f)
    return pipeline


def test_synthetic_two_round_actual_artifact_pipeline(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    first=run_round(pipeline,1,steps=2)
    assert first["status"]=="COMPLETE"
    first_bank=read_json(pipeline.round_root(1)/"scored.json")
    second=run_round(pipeline,2,steps=2)
    assert second["status"]=="COMPLETE"
    assert tuple(pipeline.model(2).experts)==("expert_0","expert_1","expert_2")
    second_bank=read_json(pipeline.round_root(2)/"scored.json")
    for sid,rows in first_bank["scenes"].items():
        assert all(second_bank["scenes"][sid][eid]==keys for eid,keys in rows.items())
    identity=read_json(pipeline.round_root(2)/"scorer/identity.json")
    assert identity["old_new_replay"] and identity["warm_started"] and identity["expert_ids"]==["expert_0","expert_1","expert_2"]
    assert read_json(pipeline.round_root(2)/"frozen_audit.json")["passed"]
    assert not (pipeline.root/"ACTIVE.json").exists()
    resumed=run_round(pipeline,2,resume=True,steps=2)
    assert all(s["status"]=="REUSED" for s in resumed["stages"])


def test_selector_only_round_preserves_registry_and_replays_pool(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    run_round(pipeline,1,steps=2)
    registry_before=read_json(pipeline.root/'registry.json')
    outcome=run_round(pipeline,2,steps=2,selector_only=True)
    assert outcome['status']=='COMPLETE'
    assert read_json(pipeline.root/'registry.json')==registry_before
    assert read_json(pipeline.round_root(2)/'oracle_stage_val.json')['selector_only']
    assert read_json(pipeline.round_root(2)/'scorer/identity.json')['expert_ids']==['expert_0','expert_1']


def test_new_round_uses_updated_coverage_not_original_failure_list(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    run_round(pipeline,1,steps=2)
    # The manifest is frozen before training, includes H/A/N/G and fixed target IDs.
    manifest=read_json(pipeline.round_root(1)/'expert_manifest.json')
    assert manifest['sampling_plan']['metadata']['experiment_kind']=='resampling_only'
    assert manifest['sampling_plan']['metadata']['actual_proportions']['global_anchor']>0
    pipeline.train_selector(1,router=True,steps=2)
    pipeline.calibrate(1,'selector_cal',router=True)
    assert pipeline.evaluate(1,'dev_report',router=True)['role']=='dev_report'
    assert read_json(pipeline.round_root(1)/'router/identity.json')['expert_ids']==['expert_0','expert_1']


def test_scoring_receipt_interruption_reconstructs_cost_without_inventing_time(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    pipeline.export_candidates(0,['incremental_fit'])
    pipeline.score_candidates(0)
    (pipeline.round_root(0)/'scoring_cost.json').unlink()
    monkeypatch.setattr(pipeline._backend,'score',lambda *a,**k:pytest.fail('completed labels rescored'))
    resumed=pipeline.score_candidates(0)
    assert resumed['reused'] and resumed['wall_seconds'] is None


def test_mining_uses_actual_served_winners_not_last_rejected_selector(tmp_path,monkeypatch):
    import numpy as np
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    run_round(pipeline,1,steps=2)
    fit=pipeline.scenes(['incremental_fit'])
    atomic_json(pipeline.round_root(1)/'selection_incremental_fit.json',{s.scene_id:'rejected_selector_winner' for s in fit})
    monkeypatch.setattr(pipeline,'previous_indices',lambda *a:np.zeros(len(fit),int))
    pipeline.build_round(2,steps=2)
    diagnoses=read_json(pipeline.round_root(2)/'gap_diagnoses.json')
    assert all(d['deployed_selected_expert_id']=='expert_0' for d in diagnoses)


def test_feature_shards_partition_named_scenes_without_extra_encoding(tmp_path,monkeypatch):
    pipeline=pipeline_fixture(tmp_path,monkeypatch)
    visited=[]
    monkeypatch.setattr(pipeline,'cache_scene_list',lambda model,scenes:visited.extend(s.scene_id for s in scenes))
    roles=['incremental_fit','stage_val']
    for i in range(4):pipeline.cache_features(roles,shard_index=i,num_shards=4)
    assert sorted(visited)==sorted(s.scene_id for s in pipeline.scenes(roles))
    assert len(set(visited))==len(visited)
