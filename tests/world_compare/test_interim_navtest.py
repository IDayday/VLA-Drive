import json
from pathlib import Path

import pytest

from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_compare.interim_navtest import SCHEMA, validate_registration


def build(tmp_path):
    export=tmp_path/'export';export.mkdir();(export/'export.json').write_text('sealed export')
    parent=dict(path='/immutable/A5000',model_sha256='native',seal=dict(updates=5000,files_sha256='files'),
                identity=dict(architecture_id='A',qualification=False))
    meta=dict(architecture_id='A',model_sha256='export_model',IL_parent=parent,qualification=False,RL_parent=None)
    row=dict(schema=SCHEMA,user_instruction='评测最近的ckpts',seed=2101,count=12146,
        inference='single_candidate_cleanx_K10_shift5',Test_observed=True,
        used_for_parent_selection=False,used_for_actor_update=False,architecture_id='A',
        model_sha256='export_model',native_checkpoint=parent['path'],native_model_sha256='native',
        native_files_sha256='files',updates=5000,export=str(export),export_sha256=file_sha(export/'export.json'),
        data_sha256='data',protocol_sha256='protocol')
    return meta,row


def save(tmp_path,row):
    p=tmp_path/'registration.json';p.write_text(json.dumps(dict(row,sha256=fingerprint(row))));return p


def test_full_user_observation_is_separate_from_parent_selection(tmp_path):
    meta,row=build(tmp_path);p=save(tmp_path,row)
    result=validate_registration(p,meta,data_sha256='data',protocol_sha256='protocol')
    assert result['updates']==5000 and result['used_for_parent_selection']is False


@pytest.mark.parametrize('key,value',[('count',100),('seed',42),('architecture_id','B'),
    ('used_for_parent_selection',True),('used_for_actor_update',True),('native_files_sha256','other'),('updates',4000)])
def test_resealed_wrong_scope_cannot_bypass_formal_gate(tmp_path,key,value):
    meta,row=build(tmp_path);row[key]=value;p=save(tmp_path,row)
    with pytest.raises(ValueError):validate_registration(p,meta,data_sha256='data',protocol_sha256='protocol')


def test_qualification_weights_and_altered_protocol_rejected(tmp_path):
    meta,row=build(tmp_path);p=save(tmp_path,row);meta['qualification']=True
    with pytest.raises(ValueError):validate_registration(p,meta,data_sha256='data',protocol_sha256='protocol')
    meta['qualification']=False
    with pytest.raises(ValueError):validate_registration(p,meta,data_sha256='data',protocol_sha256='changed')
