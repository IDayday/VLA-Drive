import copy
import json

import pytest

from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.loading import validate_rl_parent_selection


def fixture(tmp_path):
    info=dict(path=str(tmp_path/'parent'),model_sha256='model',
              identity=dict(qualification=False),seal=dict(updates=40000,files_sha256='files'))
    data=dict(schema='world_postcot_explicit_IL40000_parent_v1',
        user_instruction='基于 40K的权重，启动Qwen-world 的RL训练',
        selection_method='explicit_user_IL40000',completed_IL_updates=40000,
        Navtest_used=True,training_seed=42,evaluation_seed=2101,
        selected=dict(checkpoint=info['path'],updates=40000,model_sha256='model',
            files_sha256='files',identity_sha256=fingerprint(info['identity'])),
        training_recipe_sha256='a'*64,hard_manifest_sha256='b'*64)
    path=tmp_path/'parent.json'
    def write(value):path.write_text(json.dumps(dict(value,sha256=fingerprint(value))))
    write(data)
    return path,data,info,write


def test_explicit40k_is_separate_from_completed100k_dev_selection(tmp_path):
    path,data,info,write=fixture(tmp_path)
    result=validate_rl_parent_selection(path,info)
    assert result['completed_IL_updates']==40000 and result['Navtest_used'] is True
    assert 'evaluated_candidates' not in result


@pytest.mark.parametrize('field,value',[
    ('user_instruction','other'),('completed_IL_updates',100000),
    ('Navtest_used',False),('training_seed',43),('evaluation_seed',2102),
    ('training_recipe_sha256',''),('hard_manifest_sha256','g'*64)])
def test_explicit40k_cannot_forge_parent_or_training_authorization(tmp_path,field,value):
    path,data,info,write=fixture(tmp_path)
    data[field]=value;write(data)
    with pytest.raises(ValueError):validate_rl_parent_selection(path,info)


@pytest.mark.parametrize('field,value',[
    ('updates',37000),('model_sha256','other'),('checkpoint','/other'),
    ('files_sha256','other'),('identity_sha256','other')])
def test_explicit40k_exact_parent_binding(tmp_path,field,value):
    path,data,info,write=fixture(tmp_path)
    data['selected'][field]=value;write(data)
    with pytest.raises(ValueError,match='identity'):validate_rl_parent_selection(path,info)


def test_explicit40k_rejects_qualification_and_unsealed_mutation(tmp_path):
    path,data,info,write=fixture(tmp_path)
    info['identity']['qualification']=True
    with pytest.raises(ValueError,match='Qualification'):validate_rl_parent_selection(path,info)
    info['identity']['qualification']=False
    data['completed_IL_updates']=37000;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Changed'):validate_rl_parent_selection(path,info)
