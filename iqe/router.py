"""Scene-only efficiency control. Stable expert IDs bind output rows."""
from __future__ import annotations
import torch
from torch import nn
from .contracts import require


class SceneRouter(nn.Module):
    def __init__(self, scene_dim, ego_dim, expert_ids, hidden=256):
        super().__init__()
        require(expert_ids[0] == "expert_0" and len(set(expert_ids)) == len(expert_ids), "Router registry order")
        self.expert_ids = tuple(expert_ids)
        self.scene_dim, self.ego_dim = scene_dim, ego_dim
        self.encoder = nn.Sequential(nn.Linear(scene_dim + ego_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, len(expert_ids))

    def forward(self, features):
        f = features.detached()
        scene = (f.scene * f.valid_tokens[..., None]).sum(1) / f.valid_tokens.sum(1, keepdim=True)
        ego = f.ego.mean(1) if f.ego.ndim == 3 else f.ego
        return self.head(self.encoder(torch.cat((scene, ego), -1)))

    def expanded(self, expert_ids):
        require(tuple(expert_ids[:len(self.expert_ids)]) == self.expert_ids, "Router rows cannot be reordered")
        new = SceneRouter(self.scene_dim, self.ego_dim, expert_ids, self.head.in_features)
        new.encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        new.to(self.head.weight)
        with torch.no_grad():
            new.head.weight[:len(self.expert_ids)].copy_(self.head.weight)
            new.head.bias[:len(self.expert_ids)].copy_(self.head.bias)
        return new


def soft_targets(scores, valid, temperature=0.1):
    require(temperature > 0, "Router temperature")
    valid = valid & torch.isfinite(scores)
    safe = scores.detach().float().masked_fill(~valid, -torch.inf)
    any_valid = valid.any(-1)
    safe = torch.where(any_valid[:, None], safe, torch.zeros_like(safe))
    p = ((safe - safe.max(-1, keepdim=True).values) / temperature).softmax(-1)
    return torch.where(valid, p, torch.zeros_like(p)), any_valid
