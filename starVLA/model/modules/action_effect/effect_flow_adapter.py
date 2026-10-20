"""Small private residual velocity field; never receives simulator feedback."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class EffectFlowAdapterConfig:
    condition_dim: int = 2048
    hidden_dim: int = 256
    heads: int = 4
    layers: int = 2
    horizon: int = 8
    action_dim: int = 4
    dropout: float = 0.0
    alpha_cap: float = 1.0
    init_seed: int = 42

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.condition_dim, self.hidden_dim, self.heads, self.layers, self.horizon, self.action_dim)):
            raise ValueError('Positive integer dimensions required')
        if self.hidden_dim % self.heads or (self.horizon, self.action_dim) != (8, 4):
            raise ValueError('Requires divisible attention width and original raw8x4')
        if type(self.init_seed) is not int or self.init_seed < 0:
            raise ValueError('An explicit nonnegative initialization seed is required')
        if not 0 <= self.dropout < 1 or not math.isfinite(self.alpha_cap) or not 0 < self.alpha_cap <= 1:
            raise ValueError('Invalid explicit adapter dropout/alpha bound')
        return self


class EffectFlowAdapter(nn.Module):
    """Predict g(F_plan,x_u,u) inside every existing Euler velocity evaluation.

    Raw XY/sin/cos channels stay in the original S0 representation. The output
    is not clipped to a small displacement; alpha is an explicit fixed bound,
    not a second trajectory refiner. All input condition states are detached.
    """
    def __init__(self, config=EffectFlowAdapterConfig()):
        super().__init__()
        self.config = config.validate()
        # CPU construction preserves external training RNG. Device transfer is
        # performed only after this separately seeded initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed)
            self._build()

    def _build(self):
        config = self.config
        d = config.hidden_dim
        self.context = nn.Sequential(nn.LayerNorm(config.condition_dim), nn.Linear(config.condition_dim, d))
        self.state = nn.Linear(config.action_dim, d)
        self.time = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.randn(config.horizon, d) * .02)
        layer = nn.TransformerDecoderLayer(d, config.heads, d * 4, dropout=config.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, config.layers)
        self.output = nn.Linear(d, config.action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition, state, times):
        cfg = self.config
        if (condition.ndim != 3 or condition.shape[0] != state.shape[0]
                or condition.shape[-1] != cfg.condition_dim or condition.shape[1] < 1
                or state.shape[1:] != (cfg.horizon, cfg.action_dim) or times.shape != (len(state),)):
            raise ValueError('Adapter requires finite F_plan[B,N,H], x[B,8,4], u[B]')
        if not all(x.is_floating_point() and torch.isfinite(x).all() for x in (condition, state, times)):
            raise ValueError('Nonfinite or nonfloating adapter input')
        t = times.to(state.dtype)
        time = torch.stack((t, t*t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)), -1)
        memory = self.context(condition.detach())
        query = self.state(state) + self.position[None] + self.time(time)[:, None]
        return self.output(self.decoder(query, memory)) * cfg.alpha_cap

    def identity_config(self):
        return {'schema': 'effect_flow_adapter_v1', **asdict(self.config),
                'initialization': 'random_body_zero_final_layer', 'input': 'frozen_F_plan,x_raw,u',
                'alpha_schedule': 'fixed_cap', 'external_weights_loaded': False}
