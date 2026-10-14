"""The same independent temporal decoder, with explicit reward estimands.

The hurdle is the *joint* event R > 0. It does not multiply independent NC
and DAC marginals, or claim those correlated outcomes are independent.
All projection/decoder/readout parameters are initialized locally from scratch.
"""
import torch
from torch import nn

from iqe.contracts import require
from iqe.internal_baseline.scorer import InternalScorer


class RobustScorer(InternalScorer):
    def __init__(self, world_dim, action_dim, trajectory_contract, *, objective, dropout):
        require(objective in ("mean_bce", "zero_hurdle"), "unsupported reward estimand")
        super().__init__(world_dim, action_dim, trajectory_contract, dropout=dropout)
        self.objective = objective
        self.spatial = False
        if objective == "zero_hurdle":
            self.nonzero_head = nn.Linear(self.type_embeddings.shape[1], 1)

    def score_projected(self, memory, memory_valid, trajectories):
        t = trajectories.detach().float()
        require(t.ndim == 4 and t.shape[-2:] == (len(self.times), 3), "physical candidate contract")
        valid = torch.isfinite(t).all((-1, -2))
        safe = torch.where(valid[..., None, None], t, torch.zeros_like(t))
        b, k, steps, _ = t.shape
        xy = (safe[..., :2]-self.xy_mean)/self.xy_scale
        times = self.times.expand(b, k, steps)[..., None]
        tokens = self.trajectory_encoder(torch.cat((xy, safe[..., 2:3].sin(), safe[..., 2:3].cos(), times), -1))
        z = self.decoder(tokens.reshape(b*k, steps, -1),
                         memory[:, None].expand(-1, k, -1, -1).reshape(b*k, memory.shape[1], -1),
                         memory_key_padding_mask=(~memory_valid[:, None].expand(-1, k, -1)).reshape(b*k, -1))
        z = z.mean(1).reshape(b, k, -1)
        logits = self.value_head(z).squeeze(-1)
        result = {"value_logits": logits, "values": logits.sigmoid(),
                  "component_logits": self.component_head(z), "valid": valid}
        if self.objective == "zero_hurdle":
            gate = self.nonzero_head(z).squeeze(-1)
            result.update(nonzero_logits=gate, conditional_values=logits.sigmoid(),
                          values=gate.sigmoid()*logits.sigmoid())
        return result
