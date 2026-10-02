import pytest
import torch
from tools.action_video_foresight.evaluate_future_probe import normalized_values,errors_in_normalized_space
from tools.action_video_foresight.audit_changed_regions import change_values
from tools.action_video_foresight.summarize_frozen_probes import paired_interval


def test_reference_padding_and_empty_population_preserve_loss_space():
    target=torch.randn(2,3,2,2,3,4)
    valid=torch.ones(target.shape[:-1],dtype=torch.bool)
    valid[1]=False
    polluted=target.clone();polluted[~valid]=float('nan')
    clean=normalized_values(target,valid)
    actual=normalized_values(polluted,valid)
    torch.testing.assert_close(actual,clean)
    assert errors_in_normalized_space(clean,actual,valid)==[0.,None]
    polluted[0,0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Illegal valid'):
        normalized_values(polluted,valid)


def test_mean_template_must_not_be_layer_normalized_a_second_time():
    raw=torch.tensor([1.,2.,3.,4.]).reshape(1,1,1,1,1,4)
    valid=torch.ones(raw.shape[:-1],dtype=torch.bool)
    target=normalized_values(raw,valid)
    mean=target*.5
    arithmetic=errors_in_normalized_space(mean,target,valid)[0]
    unit_template=errors_in_normalized_space(normalized_values(mean,valid),target,valid)[0]
    assert arithmetic>.2
    assert unit_template<1e-7


def test_feature_change_uses_common_valid_positions_and_channel_mean():
    current=torch.zeros(4,2,3);future=current.clone();future[:,0,0]=2.
    valid=torch.ones(2,3,dtype=torch.bool);valid[1,2]=False
    current[:,1,2]=float('nan');future[:,1,2]=float('inf')
    delta=change_values(current,future,valid)
    assert delta[0,0]==4. and delta[1,2]==0.
    assert torch.isfinite(delta).all()
    future[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Illegal valid'):change_values(current,future,valid)


def test_probe_pairing_preserves_missing_targets_and_clusters_by_log():
    base=[{'token':'a','log':'one','normal':.4,'failure':None},
          {'token':'b','log':'one','normal':.5,'failure':None},
          {'token':'c','log':'two','normal':None,'failure':None}]
    improved=[{**row,'normal':row['normal']-.1 if row['normal'] is not None else None} for row in base]
    result=paired_interval(improved,base,'normal',draws=100)
    assert result['left_minus_right_normalized_mse']==pytest.approx(-.1)
    assert result['requested_scenes']==3 and result['missing_target_scenes_retained']==1
    assert result['logs_with_valid_targets']==1
    improved[-1]['normal']=.2
    with pytest.raises(ValueError,match='eligibility'):paired_interval(improved,base,'normal')
