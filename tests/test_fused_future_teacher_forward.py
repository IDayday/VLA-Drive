from types import SimpleNamespace

import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent


class _CountingEMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0
        self.last_tile_counts = None

    def forward(self, pixel_values, tile_counts, tile_metadata=None):
        self.calls += 1
        self.last_tile_counts = list(tile_counts)
        groups = pixel_values.split(tile_counts)
        values = torch.stack([group.float().mean() for group in groups])
        return values[:, None, None].expand(-1, 16, 32)


def test_current_and_three_futures_use_one_ema_vision_call(monkeypatch):
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.future_mode = "correct"
    agent.ema_register_target = _CountingEMA()
    monkeypatch.setattr(
        "navsim.agents.EpisodeDrive.drivevla_base_agent.load_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Worker-preprocessed images should avoid path decode")
        ),
    )
    metadata = torch.tensor([[0.5, 0.5, 1.0, 1.0, 1.0]])
    current = torch.stack(
        [torch.full((1, 3, 2, 2), 1.0), torch.full((1, 3, 2, 2), 2.0)]
    )
    futures = [
        [torch.full((1, 3, 2, 2), float(10 * batch + horizon + 3)) for horizon in range(3)]
        for batch in range(2)
    ]
    features = {
        "pixel_values": current,
        "tile_metadata": torch.stack([metadata, metadata]),
        "future_pixel_values": futures,
        "future_tile_metadata": [[metadata, metadata, metadata] for _ in range(2)],
    }
    targets = {
        "future_image_paths": torch.zeros(2, 3, 1024, dtype=torch.uint8),
        "future_image_path_lengths": torch.zeros(2, 3, dtype=torch.long),
        "future_valid_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    current_target, future_target, valid = agent._encode_ema_register_targets(
        features, targets, batch_size=2
    )
    assert agent.ema_register_target.calls == 1
    assert agent.ema_register_target.last_tile_counts == [1] * 8
    assert current_target.shape == (2, 16, 32)
    assert future_target.shape == (2, 3, 16, 32)
    assert valid.all()
