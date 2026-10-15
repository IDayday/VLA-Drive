import numpy as np
import pytest
import torch
from starVLA.dataloader.natural_local_dataset import NaturalLocalData
from tools.decision_local.common import atomic_json,digest,identity


def test_canonical_body_velocity_is_not_rotated_as_global_velocity():
    from tools.decision_local.review_repair_execution import relative_execution_states
    states=np.zeros((2,11));states[:,:3]=[[10,20,2.7],[9,21,2.8]]
    states[:,3:7]=[5.,.02,.3,.01]
    result=relative_execution_states(states,[10,20,2.7])
    np.testing.assert_array_equal(result[:,3:],states[:,3:])
    np.testing.assert_allclose(result[:,2],[0,.1],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(result[0,:2],[0.,0.])
    np.testing.assert_array_equal(states[0,:3],[10,20,2.7])


def fixture_bank(tmp_path,admitted=False,omit=False):
    labels=tmp_path/'labels';(labels/'scenes').mkdir(parents=True);(labels/'records').mkdir()
    values=dict(case_status=np.array([[0,1,2]]),target_eligible=np.array([[True,True,False]]),
        repair_target=np.concatenate((np.zeros((1,2,8,4)),np.full((1,1,8,4),np.nan)),axis=1),
        valid=np.ones((1,3,3,7,8,3),bool),relations=np.ones((1,3,3,7,8,3),np.float32))
    path=labels/'scenes'/'scene.npz';np.savez_compressed(path,**values)
    sha=digest(path);atomic_json(labels/'records'/'scene.json',dict(identity='labels1',sha256=sha))
    data=NaturalLocalData.__new__(NaturalLocalData);data.labels=labels;data._labels={};data.repair_review=None
    data.label_identity={'identity':'labels1'};data.index=[dict(token='scene',log='log')]
    row=dict(token='scene',anchor_index=0,plan_index=1,label_sha256=sha,admitted=admitted,checks={'tracked_geometry':admitted})
    review=dict(schema='natural_repair_execution_review_v1',labels_identity='labels1',
        all_label_repairs_checked=True,all_admitted_repairs_independently_checked=True,
        cases={} if omit else {'scene:0:1':row})
    review['identity']=identity(review);review_path=tmp_path/'review.json';atomic_json(review_path,review)
    return data,review_path,values,path,sha


@pytest.mark.parametrize('admitted',[False,True])
def test_review_only_changes_full_repair_eligibility(tmp_path,admitted):
    data,review,original,path,sha=fixture_bank(tmp_path,admitted)
    data.bind_repair_execution_review(review);values=data._scene_labels('scene')
    assert values['target_eligible'].tolist()==[[True,admitted,False]]
    for key in ('case_status','valid','relations'):
        assert torch.equal(values[key],torch.from_numpy(original[key]))
    assert torch.isnan(values['repair_target'][0,2]).all()
    assert digest(path)==sha


def test_review_cannot_silently_omit_a_strict_repair(tmp_path):
    data,review,_,_,_=fixture_bank(tmp_path,omit=True)
    data.bind_repair_execution_review(review)
    with pytest.raises(ValueError,match='omitted'):data._scene_labels('scene')


def test_preserved_failed_execution_review_is_not_usable(tmp_path):
    data,review,_,_,_=fixture_bank(tmp_path)
    atomic_json(review.parent/'CORRECTNESS_HOLD.json',dict(reason='known coordinate error'))
    with pytest.raises(ValueError,match='correctness hold'):data.bind_repair_execution_review(review)
