"""Small private residual velocity field; never receives simulator feedback."""
from dataclasses import asdict, dataclass, replace
import math
import torch
from torch import nn
from .effect_model import module_initialization


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
    effect_dim: int = 0

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.condition_dim, self.hidden_dim, self.heads, self.layers, self.horizon, self.action_dim)):
            raise ValueError('Positive integer dimensions required')
        if self.hidden_dim % self.heads or (self.horizon, self.action_dim) != (8, 4):
            raise ValueError('Requires divisible attention width and original raw8x4')
        if type(self.init_seed) is not int or self.init_seed < 0:
            raise ValueError('An explicit nonnegative initialization seed is required')
        if type(self.effect_dim) is not int or self.effect_dim<0:
            raise ValueError('Explicit nonnegative optional physical-effect dimension required')
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
        with module_initialization(config.init_seed):
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
        # Appended after the complete legacy velocity branch: identical seed
        # leaves every shared velocity parameter bitwise unchanged.
        self.effect_projection=nn.Linear(d,d) if config.effect_dim else None
        self.effect_head=nn.Linear(d,config.effect_dim) if config.effect_dim else None

    def forward(self, condition, state, times, *, return_effect=False):
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
        hidden=self.decoder(query,memory)
        velocity=self.output(hidden)*cfg.alpha_cap
        if not return_effect:return velocity
        if not cfg.effect_dim:raise ValueError('Legacy velocity-only adapter has no effect-training heads')
        pooled=hidden.mean(1)
        return {'velocity':velocity,'effect_delta':self.effect_head(pooled),
                'effect_embedding':self.effect_projection(pooled)}

    def velocity_export(self):
        """Strip training-only effect modules without changing a velocity value."""
        return {'config':asdict(replace(self.config,effect_dim=0)),
            'state_dict':{k:v for k,v in self.state_dict().items()
                          if not k.startswith(('effect_projection.','effect_head.'))}}

    def identity_config(self):
        return {'schema': 'effect_flow_adapter_v1', **asdict(self.config),
                'initialization': 'random_body_zero_final_layer', 'input': 'frozen_F_plan,x_raw,u',
                'alpha_schedule': 'fixed_cap', 'external_weights_loaded': False}
