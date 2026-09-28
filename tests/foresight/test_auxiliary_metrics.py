import pytest
import torch
from tools.foresight.evaluate_auxiliary_tasks import visual_statistics,interaction_statistics,summarize


def test_copy_current_reference_and_invalid_views_preserve_denominators():
    target=torch.ones(3,2,4,6);current=torch.zeros_like(target);prediction=target*.5
    prediction[2]=current[2]=target[2]=float('nan')
    rows=visual_statistics(prediction,target,current,torch.tensor([True,True,False]))
    assert rows[0]=={'elements':48,'prediction_squared_error':12.,'copy_current_squared_error':48.}
    assert rows[2]['elements']==0 and rows[2]['prediction_squared_error']==0
    report=summarize([{'failure':None,'visual_1s_v0_elements':48,'visual_1s_v0_squared_error':12.,'visual_1s_v0_copy_squared_error':48.},
                      {'failure':None},{'failure':'missing prediction'}])
    assert report['scenes']==3 and report['failed']==1 and not report['valid_complete_result']
    assert report['visual']['visual_1s_v0']['relative_improvement_over_copy']==.75
    assert report['visual']['visual_1s_v0']['valid_scenes']==1
    with pytest.raises(ValueError):visual_statistics(prediction,target,current,torch.ones(3,dtype=torch.bool))


def test_interaction_same_fixed_normalization_and_empty_invalid_target():
    x=torch.arange(4096,dtype=torch.float32).reshape(8,512)
    assert interaction_statistics(x,x,True,1e-5)=={'elements':4096,'squared_error':0.}
    assert interaction_statistics(x,torch.full_like(x,float('nan')),False,1e-5)['elements']==0
    with pytest.raises(ValueError):interaction_statistics(x,torch.full_like(x,float('nan')),True,1e-5)


def test_gradient_calibration_uses_shared_paths_and_keeps_missing_scenes():
    from tools.foresight.calibrate_auxiliaries import weight_from_gradients
    rows=[{'ego_fm':{'norm':4.},'visual':{'norm':2.,'W':1.,'language':1.}},
          {'ego_fm':{'norm':6.},'visual':{'norm':3.,'W':1.,'language':2.}},
          {'ego_fm':{'norm':1.},'visual':{'norm':0.,'W':0.,'language':0.}}]
    result=weight_from_gradients(rows,'visual')
    assert result['weight']==.5 and result['weighted_shared_gradient_ratio_median']==.25
    assert result['valid_scenes']==2 and result['missing_or_zero_scenes']==1
    rows[0]['visual']['language']=0
    with pytest.raises(ValueError):weight_from_gradients(rows,'visual')


def test_teacher_choice_uses_both_registered_metrics_and_fixed_tie_break():
    from tools.foresight.freeze_teacher import choose_milestone
    rows=[{'epoch':2,'failures':0,'ego_with_peer_ADE':1.,'vehicle_with_peer_ADE':3.},
          {'epoch':4,'failures':0,'ego_with_peer_ADE':2.,'vehicle_with_peer_ADE':2.},
          {'epoch':8,'failures':0,'ego_with_peer_ADE':.5,'vehicle_with_peer_ADE':4.}]
    assert choose_milestone(rows)['epoch']==4
    rows[0]['failures']=1
    with pytest.raises(ValueError):choose_milestone(rows)
