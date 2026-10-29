"""Native ZeRO objects must compare serialized state, never object identity."""
import io

import pytest
import torch

from tools.qwen_world_compare.qualify import tree_compare


def roundtrip(value):
    payload = io.BytesIO()
    torch.save(value, payload)
    payload.seek(0)
    return torch.load(payload, map_location='cpu', weights_only=False)


def test_independently_loaded_actual_deepspeed_scaler_compares_all_fields():
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    value = dict(loss_scaler=LossScaler(scale=1), exp_avg=torch.arange(4.))
    left, right = roundtrip(value), roundtrip(value)
    assert left['loss_scaler'] is not right['loss_scaler']
    tree_compare(left, right, 'optimizer')
    right['loss_scaler'].cur_scale = 2.
    with pytest.raises(ValueError, match='cur_scale'):
        tree_compare(left, right, 'optimizer')


def test_dynamic_scaler_history_and_class_cannot_be_skipped():
    from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler, LossScaler
    left = DynamicLossScaler()
    right = roundtrip(left)
    tree_compare(left, right, 'scaler')
    right.cur_iter += 1
    with pytest.raises(ValueError, match='cur_iter'):
        tree_compare(left, right, 'scaler')
    with pytest.raises(ValueError, match='type differs'):
        tree_compare(left, LossScaler(), 'scaler')


def test_other_optimizer_values_still_require_exact_equality():
    left = dict(exp_avg=torch.ones(4), step=4)
    right = roundtrip(left)
    right['exp_avg'][0] += 1e-6
    with pytest.raises(ValueError, match='exp_avg'):
        tree_compare(left, right, 'optimizer')
