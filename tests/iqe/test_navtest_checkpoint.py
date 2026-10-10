from dataclasses import asdict
import pytest
from iqe.contracts import strict_record,SceneRecord
from iqe.scoring.navtest_schema import NavtestInput
from iqe.navtest import load_locked,lock_dataset,payload_for
from iqe.io import atomic_json,digest


def test_navtest_inference_schema_has_no_training_label_or_training_role():
    scene=NavtestInput(1,'id','log','obs-hash','obs.json','metric.pkl','metric-hash')
    assert 'target_ref' not in asdict(scene)
    with pytest.raises(ValueError):strict_record(SceneRecord,asdict(scene))
    with pytest.raises(ValueError):NavtestInput(1,'id','log','hash','obs','metric','h','incremental_fit')


def test_navtest_lock_rejects_changed_scene_or_code(tmp_path,monkeypatch):
    from iqe import navtest
    data={'records':[{'scene_id':'a'}]};data['dataset_hash']=digest(data['records'])
    atomic_json(tmp_path/'data.json',data)
    record={'dataset_hash':data['dataset_hash'],'code_hash':'code'};record['identity']=digest(record)
    atomic_json(tmp_path/'EVALUATION_LOCK.json',record)
    monkeypatch.setattr(navtest,'code_hash',lambda:'code')
    assert load_locked(tmp_path/'data.json',tmp_path)[1]==record
    monkeypatch.setattr(navtest,'code_hash',lambda:'changed')
    with pytest.raises(ValueError,match='code changed'):load_locked(tmp_path/'data.json',tmp_path)
    data['records'].append({'scene_id':'b'});atomic_json(tmp_path/'data.json',data)
    with pytest.raises(ValueError,match='dataset changed'):load_locked(tmp_path/'data.json',tmp_path)


def test_navtest_dataset_cannot_silently_use_partial_population(tmp_path):
    identity={'split':'navtest','future_in_cache':False,'annotations_in_cache':False,
              'cameras':['CAM_F0','CAM_L0','CAM_R0'],'image_size':[1024,576],'history_images':0,'index_sha256':digest([])}
    atomic_json(tmp_path/'identity.json',identity);atomic_json(tmp_path/'index.json',[]);atomic_json(tmp_path/'metric.json',[])
    with pytest.raises(ValueError,match='complete official'):lock_dataset({},tmp_path,tmp_path/'metric.json',tmp_path/'out',1)
    assert not (tmp_path/'out/DATASET.json').exists()


def test_navtest_payload_checks_current_metric_and_trajectory_hash(tmp_path):
    from types import SimpleNamespace
    from iqe.io import file_hash
    from iqe.contracts import CandidateRecord
    a=tmp_path/'metric';a.write_text('cache');b=tmp_path/'traj';b.write_text('trajectory')
    scene=NavtestInput(1,'id','log','o','obs',str(a),file_hash(a))
    c=CandidateRecord(1,'id','expert_0','ck','contract',file_hash(b),str(b),'normalized_xy_sincos','ego_relative_rear_axle',8,.5,True,'code','cfg','time')
    backend=SimpleNamespace(protocol_hash='p',root='root',source_hash='s',config={},eval_seed=42)
    payload=payload_for(scene,c,backend)
    assert payload['scene_schema']=='navtest_input_v1' and 'target_ref' not in payload['scene']
    b.write_text('changed')
    with pytest.raises(ValueError,match='trajectory changed'):payload_for(scene,c,backend)
