"""Shared trajectory-conditioned value function, independent of candidate count/identity."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from .contracts import FeatureBundle, require


@dataclass
class ScorePrediction:
    values: torch.Tensor
    component_logits: dict[str, torch.Tensor]
    valid: torch.Tensor

    @property
    def components(self):
        return {k: v.sigmoid() for k, v in self.component_logits.items()}


class TrajectoryScorer(nn.Module):
    def __init__(self, scene_dim, ego_dim, component_names, hidden=256, layers=2, heads=4, dropout=0.0):
        super().__init__()
        require(hidden % heads == 0 and hidden % 2 == 0, "Scorer width/head mismatch")
        require(len(set(component_names)) == len(component_names), "duplicate metric names")
        self.scene_projection = nn.Linear(scene_dim, hidden)
        self.ego_projection = nn.Linear(ego_dim, hidden)
        self.trajectory_encoder = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            hidden, heads, hidden * 4, dropout, batch_first=True, activation="gelu"), layers)
        self.value_head = nn.Linear(hidden, 1)
        self.component_heads = nn.ModuleDict({c: nn.Linear(hidden, 1) for c in component_names})
        self.hidden = hidden
        self.architecture = {"scene_dim": scene_dim, "ego_dim": ego_dim, "component_names": list(component_names),
                             "hidden": hidden, "layers": layers, "heads": heads, "dropout": dropout}

    def forward(self, features: FeatureBundle, trajectories, valid=None):
        features = features.detached()
        trajectories = trajectories.detach()
        require(trajectories.ndim == 4 and trajectories.shape[-1] == 3, "Scorer requires [B,K,T,3] physical trajectories")
        b, k, t, _ = trajectories.shape
        require(b == len(features.scene) and k > 0 and t > 0, "candidate batch mismatch")
        finite = torch.isfinite(trajectories).all((-1, -2))
        valid = finite if valid is None else finite & valid.bool()
        # Invalid inputs must never propagate NaNs into otherwise valid candidates.
        tau = torch.where(valid[..., None, None], trajectories, torch.zeros_like(trajectories)).float()
        enc = torch.cat((tau[..., :2], tau[..., 2:3].sin(), tau[..., 2:3].cos()), -1)
        dtype = self.trajectory_encoder[0].weight.dtype
        tokens = self.trajectory_encoder(enc.to(dtype)).reshape(b * k, t, self.hidden)
        time = torch.arange(t, device=tau.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, self.hidden, 2, device=tau.device).float() * (-math.log(10000) / self.hidden))
        pos = torch.stack(((time * freq).sin(), (time * freq).cos()), -1).flatten(-2)
        tokens = tokens + pos.to(dtype)[None]
        ego = features.ego[:, None] if features.ego.ndim == 2 else features.ego
        memory = torch.cat((self.scene_projection(features.scene.to(dtype)), self.ego_projection(ego.to(dtype))), 1)
        memory_mask = torch.cat((features.valid_tokens, torch.ones(ego.shape[:2], device=ego.device, dtype=torch.bool)), 1)
        # No cross-candidate attention: each expanded batch item has one candidate's time sequence.
        mem = memory[:, None].expand(b, k, *memory.shape[1:]).reshape(b * k, *memory.shape[1:])
        mask = memory_mask[:, None].expand(b, k, -1).reshape(b * k, -1)
        pooled = self.decoder(tokens, mem, memory_key_padding_mask=~mask).mean(1)
        values = self.value_head(pooled).reshape(b, k).float().sigmoid()
        logits = {c: h(pooled).reshape(b, k).float() for c, h in self.component_heads.items()}
        valid = valid & torch.isfinite(values)
        for l in logits.values():
            valid = valid & torch.isfinite(l)
        return ScorePrediction(values, logits, valid)


@torch.no_grad()
def geometry_diagnostics(scorer, features, candidates):
    scorer.eval()
    original = scorer(features, candidates).values
    shuffled = scorer(features, candidates.roll(1, dims=2)).values
    mismatch = FeatureBundle(features.scene.roll(1, 0), features.ego.roll(1, 0), features.valid_tokens.roll(1, 0),
                             features.contract_hash, features.scene_ids)
    return {"original": original.cpu().tolist(), "time_shuffle": shuffled.cpu().tolist(),
            "scene_mismatch": scorer(mismatch, candidates).values.cpu().tolist(),
            "candidate_spread": (original.max(1).values - original.min(1).values).cpu().tolist()}
