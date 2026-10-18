"""Plan-conditioned world readout. Sources are decoded student representations."""
import torch
from torch import nn
from torch.nn import functional as F
from .student_queries import grid_positions


class WorldReader(nn.Module):
    def __init__(self, action_dim, memory_dim, hidden=512):
        super().__init__()
        self.query = nn.Linear(action_dim, hidden)
        self.memory = nn.Linear(memory_dim, hidden)
        self.position = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, 8, dropout=0., batch_first=True)
        self.output = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, action, memory, positions):
        m = self.memory(memory) + self.position(positions)[None]
        read = self.attention(self.query(action), m, m, need_weights=False)[0]
        return self.output(read)


class WorldToActionBridge(nn.Module):
    def __init__(self, action_dim, spatial_contract=None, temporal_contract=None, seed=4204):
        super().__init__()
        self.spatial_contract, self.temporal_contract = spatial_contract, temporal_contract
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if spatial_contract:
                self.spatial_reader = WorldReader(action_dim, spatial_contract.channels)
            if temporal_contract:
                self.temporal_reader = WorldReader(action_dim, temporal_contract.hidden_size)
        self.register_buffer("xy64", grid_positions(8, 8), persistent=False)

    def forward(self, representation, intervention=None):
        action = representation.original_action_hidden
        result = action
        for source, contract, value in (
            ("spatial", self.spatial_contract, representation.predicted_spatial_context),
            ("temporal", self.temporal_contract, representation.predicted_future_latent)):
            if contract is None:
                continue
            if value is None:
                raise ValueError(f"Missing student {source} memory")
            if source == "spatial":
                memory = F.adaptive_avg_pool2d(value, (8,8)).flatten(2).transpose(1,2)
                lo, hi = value.new_tensor(contract.roi_min[:2]), value.new_tensor(contract.roi_max[:2])
                xy = self.xy64*(hi-lo)+lo
                pos = torch.cat((xy, xy.new_zeros((64,2))), -1)
            else:
                side = int(contract.num_queries**.5)
                if side*side != contract.num_queries:
                    raise ValueError("Native temporal grid must be square")
                memory = F.adaptive_avg_pool2d(value.transpose(1,2).reshape(len(value), -1, side, side), (8,8)).flatten(2).transpose(1,2)
                if contract.horizon_seconds is None:
                    raise ValueError("Unverified temporal horizon cannot enter a driving reader")
                # Normalized native task grid is explicitly distinguished from metric BEV.
                pos = torch.cat((self.xy64, self.xy64.new_full((64,1),contract.horizon_seconds), self.xy64.new_ones((64,1))), -1)
            if intervention == "zero": memory = memory*0
            elif intervention == "shuffle": memory = memory.roll(1,0)
            elif intervention not in (None, "zero", "shuffle"): raise ValueError("Unknown memory intervention")
            result = result + getattr(self, source+"_reader")(action, memory, pos)
        return result
