from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.agents.EpisodeDrive.layers.world_model import (
    scale_ema_momentum_for_global_batch,
)


@pytest.mark.parametrize("global_batch", [16, 32, 64, 96, 128])
def test_ema_reference_momentum_is_scaled_per_global_batch(global_batch):
    assert scale_ema_momentum_for_global_batch(
        0.996, global_batch, 16
    ) == pytest.approx(0.996 ** (global_batch / 16))
    assert scale_ema_momentum_for_global_batch(
        0.9999, global_batch, 16
    ) == pytest.approx(0.9999 ** (global_batch / 16))


def test_agent_resolves_actual_ema_endpoints():
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.batch_size = 4
    agent.num_gpus = 8
    agent.ema_config = SimpleNamespace(
        reference_global_batch=16,
        scale_momentum_by_global_batch=True,
        start_momentum_reference=0.996,
        end_momentum_reference=0.9999,
    )
    start, end = agent._resolve_ema_momentum_endpoints()
    assert start == pytest.approx(0.996**2)
    assert end == pytest.approx(0.9999**2)


def test_agent_resolves_global_batch_96_ema_endpoints():
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.batch_size = 6
    agent.num_gpus = 16
    agent.ema_config = SimpleNamespace(
        reference_global_batch=16,
        scale_momentum_by_global_batch=True,
        start_momentum_reference=0.996,
        end_momentum_reference=0.9999,
    )
    start, end = agent._resolve_ema_momentum_endpoints()
    assert start == pytest.approx(0.996**6)
    assert end == pytest.approx(0.9999**6)


def test_ema_scaling_rejects_invalid_batches():
    with pytest.raises(ValueError, match="global batches"):
        scale_ema_momentum_for_global_batch(0.996, 0, 16)
