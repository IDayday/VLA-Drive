import torch
from torch import nn
from torch.nn import functional as F


class SpatialReadout(nn.Module):
    """Restricted token-to-native-context transformation, with no visual input."""
    def __init__(self, hidden_size, contract, grid_hw=(16, 16), seed=4202):
        super().__init__()
        self.contract, self.grid_hw = contract.validate(), tuple(grid_hw)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.project = nn.Linear(hidden_size, contract.channels)
            self.refine = nn.Sequential(nn.Conv2d(contract.channels, contract.channels, 3, padding=1),
                                        nn.GELU(), nn.Conv2d(contract.channels, contract.channels, 1))

    def forward(self, hidden):
        b, n, _ = hidden.shape
        h, w = self.grid_hw
        if n != h*w:
            raise ValueError("Spatial query/grid mismatch")
        x = self.project(hidden).transpose(1, 2).reshape(b, self.contract.channels, h, w)
        x = self.refine(x)
        return F.interpolate(x, (self.contract.height, self.contract.width), mode="bilinear", align_corners=False)


class ContextBottleneck(nn.Module):
    """Train-only compression upper bound using the identical reconstruction map."""
    def __init__(self, hidden_size, contract, grid_hw=(16,16)):
        super().__init__()
        self.grid_hw = grid_hw
        self.encode = nn.Linear(contract.channels, hidden_size)
        self.readout = SpatialReadout(hidden_size, contract, grid_hw)

    def forward(self, teacher_context):
        pooled = F.adaptive_avg_pool2d(teacher_context, self.grid_hw).flatten(2).transpose(1, 2)
        return self.readout(self.encode(pooled))
