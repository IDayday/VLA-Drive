"""DDP-proposal FGTR: the pinned deformable reading operator, one residual only."""
from contextlib import contextmanager
import torch
from torch import nn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from .future import mature_decoder, physical_time_features
from .grid import GridSpec


@contextmanager
def deterministic_action_head(head):
    # Preserve every submodule's prior state, including intentionally frozen ones.
    modes = [(module, module.training) for module in head.modules()]
    head.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def sample_proposal(head, H_A, *, generator=None, initial_noise=None):
    """One native ten-step sample, independent of all FM training randomness."""
    if head.num_inference_timesteps != 10:
        raise ValueError('Proposal must use the original ten Euler steps')
    if initial_noise is None:
        if generator is None:
            raise ValueError('Explicit independent proposal RNG required')
        shape = (len(H_A), head.config.action_horizon, head.config.action_dim)
        initial_noise = torch.randn(shape, generator=generator, device=H_A.device, dtype=torch.float32)
    with torch.no_grad(), deterministic_action_head(head):
        # H_A itself stays live outside this no_grad block for the refine query.
        return head.predict_action(H_A.detach(), initial_noise=initial_noise).float().detach()


class DDPProposalFGTR(nn.Module):
    def __init__(self, hidden_dim, steps, *, channels=256, grid=None):
        super().__init__()
        if steps not in (6, 8):
            raise ValueError('Registered horizon required')
        self.steps, self.channels = steps, channels
        self.grid = grid or GridSpec()
        self.action_projection = nn.Linear(hidden_dim, channels)
        self.way_decoder = mature_decoder(channels, layers=1, cross=True)
        self.time = nn.Sequential(nn.Linear(4, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.position = nn.Sequential(nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.out_of_range_embedding = nn.Embedding(2, channels)
        self.col_attn = MultiScaleDeformableAttention(channels, num_points=8, num_levels=1)
        self.coordinate_decoder = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                                nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 4))
        nn.init.zeros_(self.coordinate_decoder[-1].weight)
        nn.init.zeros_(self.coordinate_decoder[-1].bias)
        self.register_buffer('times_s', .5*torch.arange(1, steps+1), persistent=True)

    def forward(self, H_A, Bt, q0_encoded, q0_physical):
        batch = len(H_A)
        if Bt.shape != (batch, self.steps, self.channels, self.grid.height, self.grid.width) or q0_encoded.shape != (batch, self.steps, 4):
            raise ValueError('FGTR horizon/grid mismatch')
        if q0_encoded.requires_grad or q0_physical.requires_grad:
            raise ValueError('Proposal positions and conditions must be detached')
        memory = self.action_projection(H_A.float()).transpose(0, 1)
        temporal = self.time(physical_time_features(self.times_s))[:, None].expand(-1, batch, -1)
        queries = self.way_decoder(query=temporal, key=memory, value=memory, query_pos=temporal)
        uv, out_of_range = self.grid.normalized_reference(q0_physical[..., :2])
        positions = self.position(q0_physical.float()/40.)+self.out_of_range_embedding(out_of_range.long())
        shapes = torch.tensor([[self.grid.height, self.grid.width]], device=Bt.device, dtype=torch.long)
        start = torch.zeros(1, device=Bt.device, dtype=torch.long)
        residuals = []
        for t in range(self.steps):
            read = self.col_attn(query=queries[t:t+1]+positions[:, t].unsqueeze(0),
                                 value=Bt[:, t].flatten(2).permute(2, 0, 1),
                                 reference_points=uv[:, t:t+1].unsqueeze(2),
                                 spatial_shapes=shapes, level_start_index=start)
            residuals.append(self.coordinate_decoder(read.squeeze(0)))
        residual = torch.stack(residuals, 1)
        final = q0_encoded+residual
        # Decode uses atan2(sin, cos); avoid a new trajectory/control representation.
        return {'q_final_encoded': final, 'residual': residual, 'out_of_range': out_of_range,
                'reference_points': uv}
