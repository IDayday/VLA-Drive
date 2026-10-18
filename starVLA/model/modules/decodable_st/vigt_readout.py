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


class PatchSpatialReadout(nn.Module):
    """Second bounded capacity variant: retain sub-cell feature bandwidth.

    Each Qwen token predicts one native feature patch. This shallow projection
    does not inspect images, teacher features or another visual backbone.
    """
    def __init__(self, hidden_size, contract, grid_hw=(32,32), seed=4202):
        super().__init__();self.contract=contract;self.grid_hw=tuple(grid_hw)
        if contract.height%grid_hw[0] or contract.width%grid_hw[1]:raise ValueError("Patch/grid divisibility")
        self.patch=(contract.height//grid_hw[0],contract.width//grid_hw[1])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.project=nn.Linear(hidden_size,contract.channels*self.patch[0]*self.patch[1])

    def forward(self,hidden):
        if hidden.shape[1]!=self.grid_hw[0]*self.grid_hw[1]:raise ValueError("Patch query count")
        return F.fold(self.project(hidden).transpose(1,2),(self.contract.height,self.contract.width),
                      kernel_size=self.patch,stride=self.patch)


class ContextPatchBottleneck(nn.Module):
    def __init__(self,hidden_size,contract,grid_hw=(32,32),seed=4202):
        super().__init__();self.grid_hw=tuple(grid_hw)
        self.readout=PatchSpatialReadout(hidden_size,contract,grid_hw,seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1)
            self.encode=nn.Linear(contract.channels*self.readout.patch[0]*self.readout.patch[1],hidden_size)

    def forward(self,teacher_context):
        patches=F.unfold(teacher_context,kernel_size=self.readout.patch,stride=self.readout.patch).transpose(1,2)
        return self.readout(self.encode(patches))
