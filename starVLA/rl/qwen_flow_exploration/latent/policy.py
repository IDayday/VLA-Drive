"""Small observation-conditional or global root distribution, no generator copy."""
import math
import torch
from torch import nn
from .gaussian import GaussianParameters


def _masked_mean(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Pooled condition must carry its actual padding mask')
    if not mask.any(dim=1).all():
        raise ValueError('No valid condition token for a scene')
    return (hidden.float()*mask[..., None]).sum(dim=1)/mask.sum(dim=1, keepdim=True)


@torch.no_grad()
def condition_features(condition):
    """Only legal, already-frozen observation memory; no scene IDs or feedback.

    Context includes source text/state projection; visual V is the last mapped
    Qwen layer after the frozen observation provider. Pooling is a small policy
    readout only and does not replace full visual memory used by the generator.
    """
    keys = {'context', 'context_mask', 'visual_kv', 'visual_mask'}
    if not keys.issubset(condition) or not condition['visual_kv']:
        raise ValueError('Materialized native observation condition required')
    context = _masked_mean(condition['context'], condition['context_mask'])
    visual = _masked_mean(condition['visual_kv'][-1]['v'], condition['visual_mask'])
    result = torch.cat((context, visual), dim=-1).detach()
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite legal observation features')
    return result


class LatentNoisePolicy(nn.Module):
    def __init__(self, feature_dim, *, conditional=True, hidden_dims=(128, 64),
                 log_std_bounds=(-math.log(2), math.log(2))):
        super().__init__()
        low, high = map(float, log_std_bounds)
        if feature_dim <= 0 or not low < 0 < high or not math.isclose(low, -high):
            raise ValueError('The locked symmetric log_std range must contain initial log_std=0')
        self.feature_dim = int(feature_dim)
        self.conditional = bool(conditional)
        self.log_std_bounds = (low, high)
        if conditional:
            layers = [nn.LayerNorm(feature_dim)]
            width = feature_dim
            for hidden in hidden_dims:
                layers.extend([nn.Linear(width, hidden), nn.Tanh()])
                width = hidden
            last = nn.Linear(width, 48)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers.append(last)
            self.network = nn.Sequential(*layers)
        else:
            self.global_output = nn.Parameter(torch.zeros(48))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim or not torch.isfinite(features).all():
            raise ValueError('Latent policy requires finite [scene,feature] observation readout')
        # Detachment documents the frozen-generator contract; only this small
        # policy is an actor. There is no reward path through the generator.
        features = features.detach().float()
        # Rollout obtains one scene at a time. Keep the SAME tiny GEMM layout
        # when updating a collected B-scene batch, rather than fabricating old
        # densities with a different batch kernel. The large frozen generator
        # is not re-executed here.
        raw = (torch.cat([self.network(row[None].contiguous()) for row in features], dim=0)
               if self.conditional else self.global_output.expand(len(features), -1))
        mean, raw_log_std = raw.split(24, dim=-1)
        low, high = self.log_std_bounds
        log_std = (low+high)/2+(high-low)/2*torch.tanh(raw_log_std)
        return GaussianParameters(mean.reshape(-1, 8, 3), log_std.reshape(-1, 8, 3)).validate()

    def identity(self):
        return dict(schema='native_conditional_root_diagonal_Gaussian_v1',
                    conditional=self.conditional, feature_dim=self.feature_dim, root_shape=[8, 3],
                    log_std_bounds=list(self.log_std_bounds), initial_prior='N(0,I)',
                    log_probability='joint sum over all24 root coordinates',
                    network_execution_layout='scene_batch1 for rollout and fixed-root recomputation',
                    feature_readout='masked mean text/state context + last visual V; full generator memory retained',
                    deterministic_readout='conditional mean root; separately reported from sampled root')
