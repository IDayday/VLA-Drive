from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import time
import pytest
import torch
from iqe.io import atomic_json,atomic_torch,read_json,file_hash,digest,BlockedError
from iqe.base_qualification import binding,require_qualification,qualify_base,evaluate_base
from iqe.data.manifests import save_scenes
from iqe.training.checkpoint import capture_rng
from iqe.evaluation.retention import module_hash
from test_two_round_pipeline import pipeline_fixture
from conftest import TensorAdapter


def prepared(p,tmp_path):
    manifest=tmp_path/'stage_val.json';save_scenes(manifest,p.scenes(['stage_val']))
    p.config['base_evaluation']['manifest']=str(manifest)
    ck=tmp_path/'quality_base.pt'
    state={'kind':'iqe_query_base','framework_contract_hash':p.contract['framework_contract_hash'],
           'model':{},'optimizer':{},'scheduler':{},'optimizer_step':1,'epoch':0,'offset':1,'rng_by_rank':[],
           'resume_boundary':'optimizer_step','mode':'full','dependencies':{'config':{}},
           'learned_driving_weights_loaded':False,'fit_scene_ids':['fit_only']}
    atomic_torch(ck,state);atomic_json(str(ck)+'.COMPLETE.json',{'checksum':file_hash(ck),'optimizer_step':1,'dependencies':state['dependencies']})
    p.config['data']['train_root']=str(tmp_path/'train')
    atomic_json(tmp_path/'train/index.json',[{'token':'fit_only','log':'training_only_log'}])
    p.contract['metric']['components']=['no_at_fault_collisions','drivable_area_compliance']
    return ck,manifest


def test_full_few_updates_without_quality_never_locks(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);ck,manifest=prepared(p,tmp_path);p.mode='full'
    with pytest.raises(BlockedError,match='BASE_QUALITY_UNTESTED'):p.freeze_base(ck)
    assert 'base_qualification' not in read_json(p.root/'LOCKED_S0.json')


def test_evaluate_base_read_only_without_lock_or_selector(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);ck,manifest=prepared(p,tmp_path)
    adapter=TensorAdapter().eval().requires_grad_(False)
    adapter.predict_s0=lambda f:adapter.action(f)
    monkeypatch.setattr('iqe.base_qualification.S0Adapter.load_and_validate_s0',lambda *a:adapter)
    monkeypatch.setattr('iqe.base_qualification.load_observation',lambda *a:torch.ones(1,3,8))
    # encode receives a one-observation list in real adapter; fixture accepts tensor.
    encode=adapter.encode_scene;adapter.encode_scene=lambda batch:encode(batch[0])
    before=module_hash(adapter);ckhash=file_hash(ck);rng=capture_rng();locked=file_hash(p.root/'LOCKED_S0.json')
    result=evaluate_base(p,ck)
    assert result['summary']['scenes']==4 and result['summary']['valid_official_scores']==4
    assert module_hash(adapter)==before and file_hash(ck)==ckhash and file_hash(p.root/'LOCKED_S0.json')==locked
    assert torch.equal(rng['cpu'],capture_rng()['cpu']) and rng['python']==capture_rng()['python']


def test_qualification_binds_rules_manifest_checkpoint_and_protocol(tmp_path,monkeypatch):
    p=pipeline_fixture(tmp_path,monkeypatch);ck,manifest=prepared(p,tmp_path)
    bind=binding(p.contract,file_hash(ck),file_hash(manifest),p.backend().protocol_hash)
    rules={'status':'PREREGISTERED','registered_unix':1,'validation_manifest_hash':file_hash(manifest),'score_protocol_hash':p.backend().protocol_hash,
           'minimum_mean_PDMS_01':.8,'maximum_zero_rate':0.,'maximum_NC_violation_rate':0.,'maximum_DAC_violation_rate':0.,'minimum_scenes':4,'minimum_logs':4}
    rulefile=tmp_path/'rules.json';atomic_json(rulefile,rules);p.config['base_evaluation']['rules']=str(rulefile)
    metrics=['no_at_fault_collisions','drivable_area_compliance']
    evaluation={'status':'COMPLETE','binding':bind,'optimizer_step':1,'identity':{'rules_hash':file_hash(rulefile)},'started_unix':2,'smoke_or_profile':False,
        'summary':{'scenes':4,'valid_official_scores':4,'nonfinite':0,'PDMS_01':.9,'zero_rate':0.},'per_log':dict.fromkeys(['a','b','c','d']),
        'rows':[{'score':{'score_valid':True,'component_valid_masks':{c:True for c in metrics},'named_components':{c:1. for c in metrics}}}]*4}
    ep=tmp_path/'evaluation.json';atomic_json(ep,evaluation)
    assert qualify_base(p,ck,ep,select=True)['status']=='QUALIFIED'
    assert require_qualification(p,ck)['binding']==bind
    changed=tmp_path/'changed.pt';atomic_torch(changed,{'different':True})
    with pytest.raises(ValueError,match='mismatch'):require_qualification(p,changed)
    original=p.contract['normalizer_hash'];p.contract['normalizer_hash']='wrong'
    with pytest.raises(ValueError,match='mismatch'):require_qualification(p,ck)
    p.contract['normalizer_hash']=original
    original_manifest=manifest.read_bytes()
    from iqe.data.manifests import load_scenes
    from dataclasses import replace
    save_scenes(manifest,[replace(x,quality_audit_ref='different') for x in load_scenes(manifest)],immutable=False)
    with pytest.raises(ValueError,match='mismatch'):require_qualification(p,ck)
    manifest.write_bytes(original_manifest)
    atomic_json(rulefile,rules|{'minimum_mean_PDMS_01':.7})
    with pytest.raises(ValueError,match='rules changed'):require_qualification(p,ck)
