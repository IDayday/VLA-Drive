"""Single-frame physical-time future adapter; this is not original ResWorld.

TokenLearner, MLN, latent decoder and TokenFuser are pinned upstream modules.
New time and VLM conditioning wrappers generate one spatial residual per time.
"""
from numbers import Real
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


def can_batch_independent(module):
    """Batching must not change random draws or batch-dependent normalization.

    In particular, native MultiheadAttention stores dropout as a float rather
    than an nn.Dropout child. No training flags or probabilities are changed.
    """
    for child in module.modules():
        # BN also interprets tensor axes differently from Linear/LN, even with
        # frozen running statistics. Native decoders have no BN; be conservative
        # if a caller substitutes one later rather than changing its layout.
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            return False
        if child.training:
            if isinstance(child, nn.modules.dropout._DropoutNd) and child.p > 0:
                return False
            dropout = getattr(child, 'dropout', None)
            if isinstance(dropout, Real) and dropout > 0:
                return False
    return True


def time_chunk_size(value, steps):
    """Zero selects the whole horizon; positive integers bound temporaries."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Time chunk size must be a nonnegative integer')
    return min(value or steps, steps)


class SingleFrameFuture(nn.Module):
    def __init__(self, hidden_dim, steps, *, channels=256, scene_tokens=16, grid=None):
        super().__init__()
        if steps not in (6, 8):
            raise ValueError('Only the registered 3s/4s horizons are supported')
        self.steps, self.channels = steps, channels
        # Opt in through configure_execution; reference remains the default.
        # Plain attributes add no parameters or persistent checkpoint entries.
        self.execution_fast = False
        # P1 is separately opt-in, including when P0 is enabled by its caller.
        self.future_time_vectorized = False
        self.future_time_chunk_size = 0
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

    def _decode_times(self, conditioned, semantic, time, latent_pos):
        batch, tokens, channels = conditioned.shape
        chunk = time_chunk_size(self.future_time_chunk_size, self.steps)
        decoded = []
        for first in range(0, self.steps, chunk):
            count = min(chunk, self.steps-first)
            # Time joins the batch axis, never the attention sequence axis.
            queries = (conditioned[:, None]+semantic[:, None])+time[None, first:first+count, None]
            queries = queries.reshape(batch*count, tokens, channels).transpose(0, 1)
            positions = latent_pos[:, None].expand(-1, count, -1, -1)
            positions = positions.reshape(batch*count, tokens, channels).transpose(0, 1)
            output = self.res_latent_decoder(query=queries, key=queries, value=queries,
                                            query_pos=positions, key_pos=positions)
            decoded.append(output.transpose(0, 1).reshape(batch, count, tokens, channels))
        return torch.cat(decoded, 1)

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
        vectorized = (self.future_time_vectorized and can_batch_independent(self.res_latent_decoder)
                      and can_batch_independent(self.action_mln)
                      # A substituted BN fuser can amplify tiny decoder batch
                      # roundoff. Keep its entire original per-time path.
                      and not any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                                  for m in self.tokenfuser.modules()))
        fusion_vectorized = vectorized and self.tokenfuser.can_batch_time_fusion()
        conditioned = self.action_mln(latent, condition) if self.execution_fast or vectorized else None
        spatial_mix = (self.tokenfuser.prepare_spatial_mix(B0)
                       if fusion_vectorized or (self.execution_fast and self.tokenfuser.can_reuse_spatial_mix()) else None)
        decoded = self._decode_times(conditioned, semantic, time, latent_pos) if vectorized else None
        if fusion_vectorized:
            # GN remains independent for every scene/time, with exactly its
            # original C,N and C,D axes. No B0/spatial_mix repeat or Bt stack.
            tokens = decoded.reshape(batch*self.steps, decoded.shape[2], self.channels)
            values = self.tokenfuser.prepare_token_values(tokens)
            values = values.reshape(batch, self.steps, self.channels, -1)
            delta = self.tokenfuser.fuse_time(values, spatial_mix, B0)
            return B0[:, None]+delta
        future = []
        for t in range(self.steps):
            if decoded is None:
                residual_tokens = (conditioned if self.execution_fast else self.action_mln(latent, condition))+semantic+time[t]
                residual_tokens = self.res_latent_decoder(
                    query=residual_tokens.transpose(0, 1), key=residual_tokens.transpose(0, 1),
                    value=residual_tokens.transpose(0, 1), query_pos=latent_pos.transpose(0, 1),
                    key_pos=latent_pos.transpose(0, 1)).transpose(0, 1)
            else:
                residual_tokens = decoded[:, t]
            if spatial_mix is None:
                delta = self.tokenfuser(residual_tokens, B0)
            else:
                token_values = self.tokenfuser.prepare_token_values(residual_tokens)
                delta = self.tokenfuser.fuse(token_values, spatial_mix, B0)
            future.append(B0+delta)
        return torch.stack(future, 1)
