import torch
import pytest
from tools.full_foresight.diagnose_auxiliary_learning import (selected_indices, feature_error,
    trajectory_error, AcrossSceneVariance)


def test_selection_is_log_stratified_and_independent_of_input_order():
    rows = [{'token': str(i), 'log': str(i%3)} for i in range(30)]
    chosen = [rows[i] for i in selected_indices(rows, 2)]
    reverse = list(reversed(rows))
    other = [reverse[i] for i in selected_indices(reverse, 2)]
    assert len(chosen) == 6
    assert {r['token'] for r in chosen} == {r['token'] for r in other}
    assert {r['log'] for r in chosen} == {'0','1','2'}


def test_invalid_padding_does_not_pollute_feature_error():
    target = torch.ones(3, 2, 2, 2)
    pred = torch.zeros_like(target)
    valid = torch.ones(3, 2, 2, dtype=torch.bool)
    valid[:,0,0] = False
    target[:,:,0,0] = float('nan')
    pred[:,:,0,0] = float('inf')
    assert feature_error(pred, target, valid) == {'squared_error':18., 'elements':18}
    valid[:,0,0] = True
    with pytest.raises(ValueError, match='Nonfinite'):
        feature_error(pred, target, valid)


def test_coordinate_variance_detects_template_despite_channel_variance():
    # A fixed template can have large variance ACROSS channels while no scene variation.
    template = torch.arange(24.).reshape(3,2,2,2)
    valid = torch.ones(3,2,2,dtype=torch.bool)
    stat = AcrossSceneVariance()
    stat.add(template,valid)
    stat.add(template,valid)
    assert stat.result() == 0.
    changing = AcrossSceneVariance()
    changing.add(template,valid)
    changing.add(template+2,valid)
    assert changing.result() == 1.


def test_trajectory_endpoint_requires_real_last_point():
    p = torch.ones(8,2); t = torch.zeros_like(p)
    v = torch.ones(8,dtype=torch.bool);v[-1]=False
    t[-1]=float('nan')
    out = trajectory_error(p,t,v)
    assert out['valid_points']==7 and out['endpoint_FDE'] is None
    assert out['ADE']==pytest.approx(2**.5)


def test_template_reference_uses_same_effective_elements():
    t=torch.ones(3,2,2,2);v=torch.ones(3,2,2,dtype=torch.bool)
    assert feature_error(t,t,v)['squared_error']==0
    assert feature_error(torch.zeros_like(t),t,v)['elements']==24
