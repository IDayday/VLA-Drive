"""Small independently initialized action-effect models, without a VLM/scorer.

Inputs are frozen current-observation tokens and physical trajectories. Labels,
metric contexts, IDs and future state never enter this API. The D/E/F mean is
antisymmetric by construction; this is an inductive bias, not a causal proof.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


MODES = ("absolute_score", "terminal_pdms", "absolute_feedback", "delta_feedback",
         "trajectory_only", "shuffled_context")


@contextmanager
def module_initialization(seed: int):
    """CPU construction changes neither the caller's CPU nor CUDA RNG state."""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(int(seed))
        yield


def unit_scale(unit: str) -> float:
    # Preregistered physical scales, never estimated from DEV/test outcomes.
    scales = {"0..1": 1., "m": 20., "s": 4., "m/s": 20., "m/s^2": 10.,
              "m/s2": 10., "m/s^3": 50., "m/s3": 50., "rad/s": 1.5,
              "radps": 1.5, "radian": 3.141592653589793}
    if unit not in scales:
        raise ValueError(f"No registered effect-label scale for unit {unit!r}")
    return scales[unit]


@dataclass(frozen=True)
class FeedbackLabelSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]
    scales: tuple[float, ...]

    @classmethod
    def from_label_arrays(cls, labels: Mapping):
        """Bind the ACTUAL adapter label_arrays output, not a guessed width."""
        names, units = tuple(labels["names"]), tuple(labels["units"])
        result = cls(names, units, tuple(unit_scale(u) for u in units))
        result.validate()
        return result

    def validate(self):
        if (not self.names or len(set(self.names)) != len(self.names)
                or len(self.units) != len(self.names) or len(self.scales) != len(self.names)
                or "score" not in self.names
                or any(not torch.isfinite(torch.tensor(s)) or s <= 0 for s in self.scales)):
            raise ValueError("Invalid identity-bound label names, units or fixed scales")
        if any(unit_scale(u) != s for u, s in zip(self.units, self.scales)):
            raise ValueError("Label scale does not match the registered unit contract")
        return self


@dataclass(frozen=True)
class EffectModelConfig:
    mode: str = "delta_feedback"
    context_dim: int = 2048
    world_tokens: int = 144
    action_tokens: int = 8
    ego_dim: int = 4
    hidden_dim: int = 256
    layers: int = 2
    heads: int = 4
    horizon: int = 8
    dt: float = .5
    temporal_steps: int = 41
    temporal_dt: float = .1
    xy_scale_m: float = 20.
    dropout: float = 0.
    init_seed: int = 20261020

    def validate(self):
        if self.mode == "privileged":
            raise ValueError("BLOCKED_PRIVILEGED_ASSET: no verified legal privileged context is bound")
        if self.mode not in MODES:
            raise ValueError(f"Unsupported action-effect mode {self.mode!r}")
        if (self.heads < 1 or self.hidden_dim % self.heads or self.hidden_dim < 4 or self.layers < 1
                or self.world_tokens < 0 or self.action_tokens < 1 or self.context_dim < 1
                or self.ego_dim < 1 or self.horizon < 3 or self.dt <= 0
                or self.xy_scale_m <= 0 or self.temporal_steps < 1 or self.temporal_dt <= 0):
            raise ValueError("Invalid bound model dimensions/time/physical scale")
        if not all(torch.isfinite(torch.tensor(x)) for x in (self.dt, self.temporal_dt, self.xy_scale_m)):
            raise ValueError("Model physical units/time must be finite")
        if abs((self.temporal_steps - 1) * self.temporal_dt - self.horizon * self.dt) > 1e-9:
            raise ValueError("Native temporal feedback grid must cover the same physical horizon")
        if self.dropout != 0.:
            raise ValueError("Paired antisymmetry requires this registered dropout=0 model")
        return self


def _finite(value, name):
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite floating {name}")


class ContextTrajectoryBackbone(nn.Module):
    """Per-scene context projected once; each ordered pair is independent."""
    def __init__(self, config: EffectModelConfig):
        super().__init__()
        self.config = config.validate()
        h = config.hidden_dim
        self.context_projection = nn.Sequential(nn.LayerNorm(config.context_dim), nn.Linear(config.context_dim, h))
        self.ego_projection = nn.Sequential(nn.LayerNorm(config.ego_dim), nn.Linear(config.ego_dim, h))
        self.type_embedding = nn.Parameter(torch.randn(3, h) * .02)
        self.trajectory_projection = nn.Sequential(nn.Linear(14, h), nn.GELU(), nn.Linear(h, h))
        self.time_embedding = nn.Parameter(torch.randn(config.horizon, h) * .02)
        layer = nn.TransformerDecoderLayer(h, config.heads, h * 4, dropout=0., batch_first=True,
                                           norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, config.layers, norm=nn.LayerNorm(h))

    def context_memory(self, context: Mapping, *, permutation=None):
        if not isinstance(context, Mapping) or set(context) != {"memory", "ego", "mask"}:
            raise ValueError("Context must contain only frozen current memory, ego and valid mask")
        memory, ego, mask = context["memory"], context["ego"], context["mask"]
        cfg = self.config
        if memory.ndim != 3 or memory.shape[1:] != (cfg.world_tokens + cfg.action_tokens, cfg.context_dim):
            raise ValueError("Current W/A context shape does not match the bound schema")
        if ego.shape == (len(memory), 1, cfg.ego_dim):
            ego = ego[:, 0]
        if ego.shape != (len(memory), cfg.ego_dim):
            raise ValueError("Expected bound current-ego fields")
        if mask.dtype != torch.bool or mask.shape != memory.shape[:2]:
            raise ValueError("Context mask must be bool True=valid with the exact token layout")
        _finite(memory, "current W/A")
        _finite(ego, "current ego")
        if permutation is not None:
            if (permutation.dtype != torch.long or permutation.shape != (len(memory),)
                    or not torch.equal(permutation.sort().values, torch.arange(len(memory), device=permutation.device))):
                raise ValueError("Scene shuffle requires an explicit full batch permutation")
            memory, ego, mask = memory[permutation], ego[permutation], mask[permutation]
        if cfg.mode == "trajectory_only":
            # Keep exactly the same parameters and context tensor shape/cost.
            memory, ego, mask = torch.zeros_like(memory), torch.zeros_like(ego), torch.zeros_like(mask)
        memory, ego = memory.detach().float(), ego.detach().float()
        # Padding can never affect valid predictions or poison attention with NaN.
        memory = torch.where(mask[..., None], memory, torch.zeros_like(memory))
        types = torch.cat((self.type_embedding[0].expand(cfg.world_tokens, -1),
                           self.type_embedding[1].expand(cfg.action_tokens, -1)), 0)
        encoded = self.context_projection(memory) + types[None]
        ego_encoded = self.ego_projection(ego) + self.type_embedding[2]
        encoded = torch.cat((encoded, ego_encoded[:, None]), 1)
        valid = torch.cat((mask, torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device)), 1)
        return encoded, valid

    def pair_features(self, before, after):
        cfg = self.config
        if before.shape != after.shape or before.ndim != 3 or before.shape[1:] != (cfg.horizon, 3):
            raise ValueError("Expected physical rear-axle [B,horizon,XY,yaw] trajectories")
        _finite(before, "base physical trajectory")
        _finite(after, "modified physical trajectory")
        t = torch.arange(1, cfg.horizon + 1, device=before.device, dtype=before.dtype) * cfg.dt
        t = (t / (cfg.horizon * cfg.dt))[None, :, None].expand(len(before), -1, -1)
        def encode(a):
            return torch.cat((a[..., :2] / cfg.xy_scale_m, a[..., 2:3].sin(), a[..., 2:3].cos(), t), -1)
        yaw_delta = after[..., 2:3] - before[..., 2:3]
        return torch.cat((encode(before), encode(after), (after[..., :2] - before[..., :2]) / cfg.xy_scale_m,
                          yaw_delta.sin(), yaw_delta.cos()), -1).float()

    def ordered(self, memory, valid, before, after):
        tokens = self.trajectory_projection(self.pair_features(before, after)) + self.time_embedding[None]
        return self.decoder(tokens, memory, memory_key_padding_mask=~valid).mean(1)


class ActionEffectModel(nn.Module):
    """A/B/C absolute controls and D/E/F delta models have identical parameters.

    ``shuffled_context`` requires an explicit permutation, including in eval;
    this prevents a nominal negative control silently using the correct scene.
    D/E/F receive two physical trajectories; intervention is their signed change.
    Predicted repair/event probabilities are uncalibrated diagnostics.
    """
    event_names = ("collision", "road", "ttc")

    def __init__(self, config: EffectModelConfig, label_spec: FeedbackLabelSpec):
        super().__init__()
        self.config, self.label_spec = config.validate(), label_spec.validate()
        with module_initialization(config.init_seed):
            self.backbone = ContextTrajectoryBackbone(config)
            self.mean_head = nn.Linear(config.hidden_dim, len(label_spec.names))
            self.logvar_head = nn.Linear(config.hidden_dim, len(label_spec.names))
            self.event_head = nn.Linear(config.hidden_dim, len(self.event_names) * 3)
            self.temporal_head = nn.Linear(config.hidden_dim, config.temporal_steps * len(self.event_names))
            self.repair_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, context, base_physical, modified_physical, *, context_permutation=None):
        mode = self.config.mode
        if mode == "shuffled_context" and context_permutation is None:
            raise ValueError("Shuffled-context negative control requires a recorded permutation")
        if mode != "shuffled_context" and context_permutation is not None:
            raise ValueError("Context permutation is only legal in the explicit shuffle control")
        memory, valid = self.backbone.context_memory(context, permutation=context_permutation)
        b = len(base_physical)
        # No gradient can update generators or frozen upstream representations.
        # Trajectories may require gradients for A1 guidance; do NOT detach them.
        if b != len(memory):
            raise ValueError("Scene/trajectory batch identity mismatch")
        mm, vv = torch.cat((memory, memory), 0), torch.cat((valid, valid), 0)
        absolute = mode in ("absolute_score", "terminal_pdms", "absolute_feedback")
        if absolute:
            before, after = torch.cat((base_physical, modified_physical)), torch.cat((base_physical, modified_physical))
        else:
            before, after = torch.cat((base_physical, modified_physical)), torch.cat((modified_physical, base_physical))
        features = self.backbone.ordered(mm, vv, before, after)
        means, logvar = self.mean_head(features), self.logvar_head(features).clamp(-8., 6.)
        first, second = means[:b], means[b:]
        if absolute:
            delta, absolute_before, absolute_after = second - first, first, second
            delta_logvar = torch.logaddexp(logvar[:b], logvar[b:])
        else:
            delta, absolute_before, absolute_after = (first - second) * .5, None, None
            delta_logvar = (logvar[:b] + logvar[b:]) * .5
        events = self.event_head(features).reshape(2 * b, 3, 3)
        # Classes: repaired(-1), unchanged(0), introduced(+1).
        events = .5 * (events[:b] + events[b:].flip(-1))
        temporal = self.temporal_head(features).reshape(2 * b, self.config.temporal_steps, 3)
        temporal = (temporal[b:] - temporal[:b]) if absolute else .5 * (temporal[:b] - temporal[b:])
        return {"mean_delta": delta, "absolute_before": absolute_before, "absolute_after": absolute_after,
                "logvar_delta": delta_logvar, "event_logits": events,
                "temporal_delta": temporal, "repair_logit": self.repair_head(features[:b]).squeeze(-1)}


# Short alias for thin campaign trainers; no architecture selection at runtime.
EffectModel = ActionEffectModel
