"""Change the spatial queries, retaining the original S11 parameterization.

The global control is the original StudyScorer. The conditioned variant uses
the same parameter names/shapes and random initialization; candidate time
tokens additionally condition the private spatial queries. Neither variant
trains the frozen visual source or performs cross-candidate attention.
"""
import torch
from torch import nn
import torch.nn.functional as F
from iqe.contracts import require
from iqe.internal_baseline.scorer import InternalScorer
from iqe.specialist_study.models import SpatialReader, StudyScorer


def interpolate_time_tokens(tokens, output_steps):
    """Half-pixel linear interpolation with deterministic GEMM backward.

    CUDA upsample_linear1d_backward is rejected by the locked deterministic
    runtime. The fixed interpolation matrix implements the same time weights;
    no learned parameter, position change or precision workaround is introduced.
    """
    steps=tokens.shape[-2]
    require(steps>0 and output_steps>0, 'positive time dimensions required')
    position=((torch.arange(output_steps,device=tokens.device,dtype=tokens.dtype)+.5)
              *(steps/output_steps)-.5).clamp(0,steps-1)
    left=position.floor().long();right=(left+1).clamp_max(steps-1)
    fraction=(position-left.to(position.dtype))[:,None]
    weights=(F.one_hot(left,steps).to(tokens.dtype)*(1-fraction)
             +F.one_hot(right,steps).to(tokens.dtype)*fraction)
    return torch.matmul(weights,tokens)


class ConditionedSpatialReader(SpatialReader):
    def forward(self, visual, valid, xy, trajectory_tokens=None):
        if trajectory_tokens is None:
            return super().forward(visual, valid, xy)
        require(visual.ndim == 4 and visual.shape[1] == self.views
                and valid.shape == visual.shape[:3], 'spatial view/mask contract')
        require(xy.shape == (*visual.shape[:3], 2) and valid.any(-1).all().item(),
                'spatial 2D layout/empty view')
        b, views, n, _ = visual.shape
        require(trajectory_tokens.ndim == 4 and trajectory_tokens.shape[0] == b
                and trajectory_tokens.shape[-1] == self.hidden, 'candidate condition schema')
        k = trajectory_tokens.shape[1]
        require(k > 0, 'empty candidate set')
        p = xy.detach().float()*torch.pi
        # Both expensive input projection and fixed position/camera encoding
        # are evaluated once per scene, before the candidate axis is expanded.
        memory = (self.projection(visual.detach().float())
                  + self.position(torch.cat((p.sin(), p.cos()), -1))
                  + self.camera[None, :, None])
        query_count = self.queries.shape[1]
        condition = interpolate_time_tokens(trajectory_tokens,query_count)
        query = self.queries[None, None] + condition[:, :, None]
        # The batch axes are scene, candidate, view. Attention inside this
        # layer sees only the queries of that one candidate and that one view.
        query = query.reshape(b*k*views, query_count, self.hidden)
        memory = memory[:, None].expand(-1, k, -1, -1, -1).reshape(b*k*views, n, self.hidden)
        mask = (~valid[:, None].expand(-1, k, -1, -1)).reshape(b*k*views, n)
        out = self.layer(query, memory, memory_key_padding_mask=mask)
        return self.norm(out).reshape(b, k, views*query_count, self.hidden)


class ConditionalReadoutScorer(StudyScorer):
    def __init__(self, world_dim, action_dim, trajectory_contract, *, visual_dim,
                 views, conditioned=True):
        # Same constructor order and parameter initialization as StudyScorer.
        InternalScorer.__init__(self, world_dim, action_dim, trajectory_contract, dropout=0.)
        self.spatial = True
        self.conditioned = conditioned
        self.spatial_reader = ConditionedSpatialReader(visual_dim, views)

    def forward(self, context, trajectories):
        if not self.conditioned:
            return super().forward(context, trajectories)
        standard_keys = ('world', 'action', 'current_ego', 'world_valid', 'action_valid')
        require(set(context) == set(standard_keys) | {'visual', 'visual_valid', 'visual_xy'},
                'conditioned readout deployment whitelist')
        memory, mask = InternalScorer.project_context(self, {key: context[key] for key in standard_keys})
        t = trajectories.detach().float()
        require(t.ndim == 4 and t.shape[-2:] == (len(self.times), 3), 'physical candidate contract')
        valid = torch.isfinite(t).all((-1, -2))
        safe = torch.where(valid[..., None, None], t, torch.zeros_like(t))
        b, k, steps, _ = t.shape
        xy = (safe[..., :2]-self.xy_mean)/self.xy_scale
        times = self.times.expand(b, k, steps)[..., None]
        tokens = self.trajectory_encoder(torch.cat((xy, safe[..., 2:3].sin(), safe[..., 2:3].cos(), times), -1))
        spatial = self.spatial_reader(context['visual'], context['visual_valid'], context['visual_xy'], tokens)
        common = memory[:, None].expand(-1, k, -1, -1)
        combined = torch.cat((common, spatial), 2).flatten(0, 1)
        combined_mask = torch.cat((mask[:, None].expand(-1, k, -1),
                                  torch.ones(spatial.shape[:3], device=mask.device, dtype=torch.bool)), 2)
        z = self.decoder(tokens.reshape(b*k, steps, -1), combined,
                         memory_key_padding_mask=~combined_mask.flatten(0, 1))
        z = z.mean(1).reshape(b, k, -1)
        logits = self.value_head(z).squeeze(-1)
        return dict(values=logits.sigmoid(), component_logits=self.component_head(z), valid=valid)
