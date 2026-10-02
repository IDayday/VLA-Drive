import pytest
import torch
from tools.action_video_foresight.verify_resume import compare


class StatefulScaler:
    def __init__(self, scale):
        self.scale = scale
        self.history = [torch.tensor([1., 2.])]


def test_serialized_scaler_state_and_tensor_mismatch():
    assert compare(StatefulScaler(1.), StatefulScaler(1.)) == 2
    with pytest.raises(AssertionError, match='Value differs'):
        compare(StatefulScaler(1.), StatefulScaler(2.))
    with pytest.raises(AssertionError, match='Tensor differs'):
        compare({'weight': torch.ones(2)}, {'weight': torch.zeros(2)})
    with pytest.raises(AssertionError, match='Tensor metadata'):
        compare(torch.ones(2), torch.ones(3))
