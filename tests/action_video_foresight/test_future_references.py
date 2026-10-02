import pytest
import torch
from tools.action_video_foresight.evaluate_future_probe import normalized_values,errors_in_normalized_space


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
