import pytest
import torch
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, ModelInputs


def contract(**kw):
    d=dict(image_hw=[288,384],patch_hw=[16,16],native_grid_hw=[18,24],output_grid_hw=[18,24],
           channels=1024,storage_dtype='float16',compute_dtype='float32',view_names=['CAM_F0','CAM_L0','CAM_R0'],
           teacher_identity='a'*64,transform_identity='b'*64,pooling={'kind':'none','kernel_hw':[1,1],'stride_hw':[1,1]})
    d.update(kw)
    return TeacherFeatureContract.from_dict(d)


def test_contract_roundtrip_and_real_pool():
    c=contract(); assert c.identity == TeacherFeatureContract.from_dict(c.to_dict()).identity
    assert c.native_grid_hw == (18,24)
    p=contract(output_grid_hw=[6,8],pooling={'kind':'avg','kernel_hw':[3,3],'stride_hw':[3,3]})
    assert c.identity != p.identity
    with pytest.raises(TypeError): c.pooling['kind']='avg'


@pytest.mark.parametrize('update',[
    {'bogus':True},{'channels':'1024'},{'storage_dtype':'bfloat16'}, {'compute_dtype':'bfloat16'},
    {'image_hw':[287,384]}, {'output_grid_hw':[9,12]}, {'view_names':['CAM_F0','CAM_F0']},
    {'pooling':{'kind':'avg','kernel_hw':[4,4],'stride_hw':[4,4]}}])
def test_contract_rejects_unresolved_or_false_native(update):
    with pytest.raises(ValueError): contract(**update)


def test_no_labels_allowed_in_current():
    obs=dict(image=[None]*3,state=torch.zeros(1,4),lang='straight',token='scene')
    assert ModelInputs(obs).legacy_current() is not obs
    for key in ['ego','future','score','metric_cache','teacher_features']:
        with pytest.raises(ValueError): ModelInputs({**obs,key:torch.zeros(1)})
