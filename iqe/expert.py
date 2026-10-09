"""Private action branches copied from the actual S0; no shared trainable storage."""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from .contracts import FeatureBundle, TrajectoryBundle, TrajectoryContract, require


class IndependentExpert(nn.Module):
    def __init__(self, query, decoder, heads, trajectory_contract: TrajectoryContract):
        super().__init__()
        self.query = deepcopy(query)
        self.decoder = deepcopy(decoder)
        self.heads = deepcopy(heads)
        self.trajectory_contract = trajectory_contract
        require(self.query.weight.shape[0] == 1, "IQE must copy an actual single Query S0, never slice a 64-query model")
        require(len(self.heads) > 1, "S0 action intermediate and final heads required")

    def forward(self, features: FeatureBundle):
        # The audited legacy decoder has no padding API: its Q-Former outputs all valid tokens.
        require(bool(features.valid_tokens.all()), "S0 decoder cannot silently ignore a padded scene mask")
        ego = features.ego[:, None] if features.ego.ndim == 2 else features.ego
        query = ego + self.query.weight[None]
        tokens = self.decoder(query, features.scene)
        require(len(tokens) + 1 == len(self.heads), "S0 decoder/head count mismatch")
        outputs = [self.heads[0](query)]
        outputs.extend(head(tok) for head, tok in zip(self.heads[1:], tokens))
        horizon = self.trajectory_contract.horizon
        trajectories = tuple(t.reshape(len(t), horizon, self.trajectory_contract.raw_dim) for t in outputs)
        return TrajectoryBundle(trajectories[-1], self.trajectory_contract.physical(trajectories[-1]),
                                trajectories, self.trajectory_contract.dt)


class AdapterExpert(IndependentExpert):
    """Private, zero-initialized bottleneck at decoder input, with frozen copied decoder."""
    def __init__(self, source: IndependentExpert, bottleneck: int):
        super().__init__(source.query, source.decoder, source.heads, source.trajectory_contract)
        width = source.query.weight.shape[-1]
        self.adapter = nn.Sequential(nn.Linear(width, bottleneck), nn.GELU(), nn.Linear(bottleneck, width))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, features):
        ego = features.ego[:, None] if features.ego.ndim == 2 else features.ego
        query = ego + self.query.weight[None]
        query = query + self.adapter(query)
        require(bool(features.valid_tokens.all()), "S0 decoder padding unsupported")
        tokens = self.decoder(query, features.scene)
        out = [self.heads[0](query)] + [h(t) for h, t in zip(self.heads[1:], tokens)]
        trajectories = tuple(t.reshape(len(t), self.trajectory_contract.horizon, self.trajectory_contract.raw_dim) for t in out)
        return TrajectoryBundle(trajectories[-1], self.trajectory_contract.physical(trajectories[-1]),
                                trajectories, self.trajectory_contract.dt)


class ResidualExpert(nn.Module):
    """Unconstrained normalized-domain residual with exactly matching initial outputs.

    The frozen initial branch is subtracted from the private trainable branch.
    residual = private - initial, output = base + residual; no bounded scale.
    """
    def __init__(self, source):
        super().__init__()
        self.private = deepcopy(source)
        self.initial = deepcopy(source).eval().requires_grad_(False)
        self.trajectory_contract = source.trajectory_contract

    def train(self, mode=True):
        super().train(mode)
        self.initial.eval()
        return self

    def forward(self, features):
        new = self.private(features)
        with torch.no_grad():
            base = self.initial(features)
        # Equivalent unrestricted parameterization, no yaw wrap discontinuity in the S0 raw IL space.
        out = tuple(b + (n - b) for n, b in zip(new.intermediates, base.intermediates))
        return TrajectoryBundle(out[-1], self.trajectory_contract.physical(out[-1]), out, new.dt)


def assert_no_alias(modules):
    seen = {}
    for name, module in modules.items():
        for key, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if not tensor.numel():
                continue
            storage = (tensor.device, tensor.untyped_storage().data_ptr())
            require(storage not in seen or seen[storage][0] == name,
                    f"cross-expert storage alias: {name}.{key} and {seen.get(storage)}")
            seen[storage] = (name, key)
