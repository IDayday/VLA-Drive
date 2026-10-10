"""Fault injection uses SYNTHETIC_UNIT_ONLY labels, never scientific evidence."""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pytest
import torch
from iqe.io import atomic_json,atomic_torch,read_json,file_hash,digest,BlockedError
from iqe.data.manifests import save_scenes,load_scenes
from iqe.contracts import FeatureBundle
from iqe.rounds import run_round
from iqe.workflow import check_expansion
from test_two_round_pipeline import pipeline_fixture
from conftest import scene


def incoming_fixture(pipeline,tmp_path,monkeypatch):
    target=tmp_path/'new_target.pt';raw=torch.zeros(4,4);raw[:,3]=1
    atomic_torch(target,{'token':'s100','ego':raw})
    context=tmp_path/'context.json';atomic_json(context,{'fixture':'SYNTHETIC_UNIT_ONLY'})
    observation=tmp_path/'observation.json';atomic_json(observation,{'fixture':'SYNTHETIC_UNIT_ONLY'})
    record=scene(100,source_kind='new_real',added_round=1,target_ref=str(target),target_id=file_hash(target),
        observation_ref=str(observation),metric_context_ref=str(context),metric_context_hash=file_hash(context))
    path=tmp_path/'incoming.json';save_scenes(path,[record]);pipeline.config['data']['new_scene_manifest']=str(path)
    monkeypatch.setattr('iqe.pipeline.validate_incoming',lambda *a,**k:{'unit_fixture':True})
    def cache(model,scenes):
        for s in scenes:
            f=FeatureBundle(torch.ones(1,3,8),torch.ones(1,1,8),torch.ones(1,3,dtype=torch.bool),digest(pipeline.contract),(s.scene_id,))
            pipeline.features.put(pipeline.feature_record(s),f)
    monkeypatch.setattr(pipeline,'cache_scene_list',cache)
    pipeline.export_candidates(0,['incremental_fit','stage_val','selector_cal','dev_report']);pipeline.score_candidates(0)
    return record,path


@pytest.mark.parametrize('fault',['new_data_audited','new_features_cached','old_pool_new_scene_scored',
    'admission_staging_written','before_expert_manifest_commit','committed_before_pointer'])
def test_new_data_transaction_faults_resume_identically(tmp_path,monkeypatch,fault):
    pipeline=pipeline_fixture(tmp_path/'interrupted',monkeypatch)
    new,path=incoming_fixture(pipeline,tmp_path,monkeypatch)
    old_hash=file_hash(pipeline.round_root(0)/'scored.json')
    fired=[]
    def inject(point):
        if point==fault and not fired:
            fired.append(point);raise RuntimeError('injected '+point)
    pipeline._data_fault_hook=inject
    with pytest.raises(RuntimeError,match='injected'):pipeline.build_round(1,2)
    if fault!='committed_before_pointer':
        assert new.scene_id not in {s.scene_id for s in pipeline.scenes()}
    pipeline._data_fault_hook=lambda point:None
    pipeline.build_round(1,2)
    manifest=read_json(pipeline.round_root(1)/'expert_manifest.json')
    checksum=file_hash(pipeline.round_root(1)/'DATA_COMMIT.json')
    pipeline.build_round(1,2)
    assert file_hash(pipeline.round_root(1)/'DATA_COMMIT.json')==checksum
    assert file_hash(pipeline.round_root(0)/'scored.json')==old_hash
    assert [s.scene_id for s in pipeline.scenes()].count(new.scene_id)==1
    assert manifest['buckets']['new_labeled']==[new.scene_id]
    clean=pipeline_fixture(tmp_path/'clean',monkeypatch)
    incoming_fixture(clean,tmp_path,monkeypatch);clean.build_round(1,2)
    expected=read_json(clean.round_root(1)/'expert_manifest.json')
    assert manifest['buckets']==expected['buckets']
    assert manifest['sampling_plan']==expected['sampling_plan']
    assert [(r['scene_id'],r['target_id']) for r in manifest['records']]==[(r['scene_id'],r['target_id']) for r in expected['records']]
    def new_labels(p):
        values=[]
        for pointer in (p.round_root(1)/'data_staging/old_pool_on_new_scenes').glob('*.json'):
            row=read_json(pointer);c=p.bank.candidate(row['candidate_key'])
            score=p.bank.score(row['score_key'],c,p.backend().protocol_hash,new.metric_context_hash)
            values.append((row['scene_id'],row['expert_id'],score.total_score_01))
        return values
    assert new_labels(pipeline)==new_labels(clean)


def test_same_path_changed_manifest_stale_and_legacy_admission_preserved(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);new,path=incoming_fixture(p,tmp_path,monkeypatch)
    p.build_round(1,2)
    save_scenes(path,[replace(new,target_provenance='optimized')],immutable=False)
    with pytest.raises(ValueError,match='STALE_ROUND_DATA'):p.build_round(1,2)
    legacy=p.root/'admitted_sources.json';save_scenes(legacy,[replace(new,scene_id='orphan',added_round=4)])
    checksum=file_hash(legacy)
    with pytest.raises(BlockedError,match='LEGACY_ADMISSION'):p.scenes()
    assert file_hash(legacy)==checksum and (p.root/'LEGACY_ADMISSION_REPAIR.json').exists()


def test_workflow_round_not_expansion_failure_and_rejected_not_accepted(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch)
    run_round(p,1,steps=2)
    run_round(p,2,steps=2,selector_only=True)
    assert read_json(p.round_root(2)/'WORKFLOW.json')['expansion_attempt_id'] is None
    # Use full scientific expansion gate for the third workflow round; no Base
    # qualification claim is made by this isolated tensor state-machine fixture.
    p.mode='full';p.config['execution']['scientific_expansion']=True
    state=check_expansion(p,3)
    assert state['expansion_attempt_id']==2 and len(state['prior_expansion_attempts'])==1
    run_round(p,3,steps=2)
    assert list(p.registry.latest(p.registry.read()))==['expert_0','expert_1','expert_3']
    assert p.registry.latest(p.registry.read())['expert_3']['status']=='rejected'
    assert 'expert_1' in p.registry.candidate_pool() and 'expert_3' not in p.registry.candidate_pool()
    with pytest.raises(BlockedError,match='NO_COVERAGE_GAIN'):check_expansion(p,4)
    check_expansion(p,4,selector_only=True)
    assert read_json(p.round_root(3)/'previous_serving.json')=={'policy':'locked_query_S0'}


def test_completed_workflow_detects_changed_manifest_content(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);run_round(p,1,steps=2)
    path=p.root/'scenes.json';rows=load_scenes(path)
    save_scenes(path,[replace(s,quality_audit_ref='changed') for s in rows],immutable=False)
    with pytest.raises(ValueError,match='version mismatch'):run_round(p,1,resume=True,steps=2)


def test_standalone_expansion_cannot_bypass_workflow_gate(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch)
    p.mode='full'
    for action in (lambda:p.build_round(1,2),lambda:p.train_expert(1,steps=2)):
        with pytest.raises(BlockedError,match='SCIENTIFIC_EXPANSION_DISABLED'):
            action()
    assert not (p.round_root(1)/'expert_manifest.json').exists()
    assert list(p.registry.candidate_pool())==['expert_0']
