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
