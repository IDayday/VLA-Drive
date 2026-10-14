"""Single-frame physical-time future adapter; this is not original ResWorld.

TokenLearner, MLN, latent decoder and TokenFuser are pinned upstream modules.
New time and VLM conditioning wrappers generate one spatial residual per time.
"""
import torch
from torch import nn
from third_party.resworld.ported.tokenlearner import TokenLearner, TokenFuser
from third_party.resworld.ported.mln import MLN
from third_party.resworld.ported.custom_module import CustomTransformerDecoder
from .grid import GridSpec


def mature_decoder(channels=256, *, layers=3, cross=False):
    return CustomTransformerDecoder(
        num_layers=layers, return_intermediate=False,
        transformerlayers={
            'type': 'BaseTransformerLayer',
            'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': channels, 'num_heads': 8}],
            'ffn_cfgs': {'type': 'FFN', 'embed_dims': channels, 'feedforward_channels': 2*channels, 'num_fcs': 2},
            'operation_order': ('cross_attn' if cross else 'self_attn', 'norm', 'ffn', 'norm')})


def physical_time_features(seconds):
    return torch.stack((seconds/4., torch.sin(seconds), torch.cos(seconds), torch.ones_like(seconds)), -1)


class SingleFrameFuture(nn.Module):
    def __init__(self, hidden_dim, steps, *, channels=256, scene_tokens=16, grid=None):
        super().__init__()
        if steps not in (6, 8):
            raise ValueError('Only the registered 3s/4s horizons are supported')
        self.steps, self.channels = steps, channels
        self.grid = grid or GridSpec()
        self.position = nn.Sequential(nn.Linear(2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.tokenlearner = TokenLearner(scene_tokens, 2*channels)
        self.latent_decoder = mature_decoder(channels)
        self.res_latent_decoder = mature_decoder(channels)
        self.action_mln = MLN(steps*2, channels)
        self.semantic = nn.Linear(hidden_dim, channels)
        self.time = nn.Sequential(nn.Linear(4, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.tokenfuser = TokenFuser(scene_tokens, channels)
        self.register_buffer('times_s', .5*torch.arange(1, steps+1), persistent=True)

    def forward(self, B0, action_hidden, q0_physical):
        if B0.shape[1:] != (self.channels, self.grid.height, self.grid.width) or q0_physical.shape != (len(B0), self.steps, 3):
            raise ValueError('Current geometry/proposal horizon mismatch')
        if q0_physical.requires_grad:
            raise ValueError('Future conditional proposal must be detached')
        batch = len(B0)
        # W is never reshaped: the spatial feature and position originate in GeoBEV.
        xy = self.grid.centers(device=B0.device)
        uv, _ = self.grid.normalized_reference(xy)
        position = self.position((2*uv-1).flatten(0, 1)).unsqueeze(0).expand(batch, -1, -1)
        current = B0.flatten(2).transpose(1, 2)
        learned, _ = self.tokenlearner(torch.cat((current, position), -1))
        content, latent_pos = learned.chunk(2, -1)
        latent = self.latent_decoder(query=content.transpose(0, 1), key=content.transpose(0, 1),
                                     value=content.transpose(0, 1), query_pos=latent_pos.transpose(0, 1),
                                     key_pos=latent_pos.transpose(0, 1)).transpose(0, 1)
        semantic = self.semantic(action_hidden.float()).mean(1, keepdim=True)
        condition = q0_physical[..., :2].flatten(1).unsqueeze(1)/40.
        time = self.time(physical_time_features(self.times_s))
        future = []
        for t in range(self.steps):
            residual_tokens = self.action_mln(latent, condition)+semantic+time[t]
            residual_tokens = self.res_latent_decoder(
                query=residual_tokens.transpose(0, 1), key=residual_tokens.transpose(0, 1),
                value=residual_tokens.transpose(0, 1), query_pos=latent_pos.transpose(0, 1),
                key_pos=latent_pos.transpose(0, 1)).transpose(0, 1)
            delta = self.tokenfuser(residual_tokens, B0)
            future.append(B0+delta)
        return torch.stack(future, 1)
