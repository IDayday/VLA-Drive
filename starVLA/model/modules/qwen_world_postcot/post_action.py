"""Training-only metric reasoning over the actor's exact clean-candidate tokens.

There is no scene encoder, candidate identifier, trajectory generator or action
output projection in this head. Gradients intentionally flow through ``hidden``
to the shared actor and its current-observation memory.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


METRIC_NAMES = ("NC", "DAC", "TTC", "EP", "Comfort", "DDC")
OFFICIAL_METRIC_NAMES = (
    "no_at_fault_collisions", "drivable_area_compliance",
    "time_to_collision_within_bound", "ego_progress", "comfort",
    "driving_direction_compliance",
)


class PostActionHead(nn.Module):
    """Six shared metric queries cross-attend only to one candidate's 8 tokens."""

    def __init__(self, input_dim: int = 1024, width: int = 512,
                 num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        if width % num_heads or num_layers < 1:
            raise ValueError("Invalid post-action head dimensions")
        self.input_dim, self.width = input_dim, width
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, width)
        self.metric_queries = nn.Parameter(torch.empty(1, len(METRIC_NAMES), width))
        nn.init.normal_(self.metric_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=num_heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers,
                                             norm=nn.LayerNorm(width))
        self.metric_output = nn.Linear(width, 1)

    def forward(self, hidden: torch.Tensor,
                action_valid: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (8, self.input_dim):
            raise ValueError("Post head expects one candidate's [N,8,actor_width] tokens")
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Nonfinite actor candidate features")
        padding = None
        if action_valid is not None:
            if action_valid.shape != hidden.shape[:2] or action_valid.dtype != torch.bool:
                raise ValueError("action_valid must be bool[N,8]")
            if not action_valid.any(dim=1).all():
                raise ValueError("A candidate cannot have zero valid trajectory tokens")
            padding = ~action_valid
        memory = self.input_projection(self.input_norm(hidden))
        queries = self.metric_queries.expand(hidden.shape[0], -1, -1)
        states = self.decoder(queries, memory, memory_key_padding_mask=padding)
        # Each flattened row is an independent candidate. No candidate-axis
        # attention is performed; expansion changes only batch placement.
        logits = self.metric_output(states).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite post-action metric logits")
        return logits


def _scene_mean(values: torch.Tensor, valid: torch.Tensor, *,
                distributed: bool,
                global_denominator: torch.Tensor | float | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDP's averaged gradients represent one global valid-scene mean.

    Only the denominator is detached/communicated. An empty local rank retains
    the graph and participates in the same collectives as nonempty ranks.
    """
    count = valid.sum(dtype=torch.float32)
    denominator = count.detach().clone()
    world = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        if global_denominator is None:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    if global_denominator is not None:
        denominator = torch.as_tensor(global_denominator, device=values.device,
                                      dtype=torch.float32).detach()
        if denominator.ndim or not torch.isfinite(denominator) or denominator < count:
            raise ValueError("Full-update denominator cannot be smaller than local valid scenes")
    numerator = torch.where(valid, values.float(), torch.zeros_like(values.float())).sum()
    return numerator * world / denominator.clamp_min(1.0), denominator, numerator


def post_action_loss(logits: torch.Tensor, labels: torch.Tensor,
                     valid: torch.Tensor, pair_mask: torch.Tensor | None = None,
                     *, margin: float = 1.0, pair_weight: float = 0.25,
                     distributed: bool = True,
                     scene_denominators: Mapping[str, torch.Tensor | float] | None = None
                     ) -> dict[str, torch.Tensor]:
    """Factor BCE plus same-scene hard NC/DAC boundary-pair softplus.

    ``logits/labels/valid`` have shape [scenes,2,6]. ``pair_mask`` optionally
    enables NC/DAC independently with shape [scenes,2]. NC=0.5 remains a soft
    BCE target and never becomes a hard safety pair. Each objective first
    averages its own valid labels within a scene, then averages valid scenes.
    For gradient accumulation, pass full-update globally summed valid scene
    counts as ``scene_denominators={'factor': ..., 'pair': ...}``; sum the
    resulting microbatch losses without another division by accumulation.
    """
    if logits.ndim != 3 or logits.shape[1:] != (2, len(METRIC_NAMES)):
        raise ValueError("Expected two same-scene candidates and six metric logits")
    if labels.shape != logits.shape or valid.shape != logits.shape or valid.dtype != torch.bool:
        raise ValueError("Metric labels and bool validity must match logits")
    if not math.isfinite(margin) or margin < 0 or not math.isfinite(pair_weight) or pair_weight < 0:
        raise ValueError("Invalid post-action loss weights")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Valid candidate logits must never be silently masked")
    labels, valid = labels.detach().float(), valid.detach()
    selected = labels[valid]
    if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
        raise ValueError("Valid canonical metric labels must be finite and in [0,1]")
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    raw_bce = F.binary_cross_entropy_with_logits(logits.float(), safe_labels, reduction="none")
    factor_values = torch.where(valid, raw_bce, torch.zeros_like(raw_bce)).sum((1, 2))
    factor_elements = valid.sum((1, 2))
    factor_per_scene = factor_values / factor_elements.clamp_min(1)
    factor, factor_scene_count, factor_numerator = _scene_mean(
        factor_per_scene, factor_elements > 0, distributed=distributed,
        global_denominator=None if scene_denominators is None else scene_denominators['factor'])

    safety = labels[:, :, :2]
    safety_valid = valid[:, :, :2].all(dim=1)
    hard = ((safety == 0) | (safety == 1)).all(dim=1)
    opposite = safety[:, 0] != safety[:, 1]
    eligible = safety_valid & hard & opposite
    if pair_mask is not None:
        if pair_mask.shape != eligible.shape or pair_mask.dtype != torch.bool:
            raise ValueError("pair_mask must be bool[scenes,NC/DAC]")
        eligible &= pair_mask.detach()
    # The label sign is attached to the exact replayed coordinates; it never
    # supervises the separately predicted clean action from the main FM pass.
    direction = torch.where(eligible, safety[:, 0] - safety[:, 1],
                            torch.zeros_like(safety[:, 0]))
    gap_logits = logits[:, 0, :2].float() - logits[:, 1, :2].float()
    raw_pair = F.softplus(margin - direction * gap_logits)
    pair_elements = eligible.sum(dim=1)
    pair_per_scene = torch.where(eligible, raw_pair, torch.zeros_like(raw_pair)).sum(1)
    pair_per_scene = pair_per_scene / pair_elements.clamp_min(1)
    pair, pair_scene_count, pair_numerator = _scene_mean(
        pair_per_scene, pair_elements > 0, distributed=distributed,
        global_denominator=None if scene_denominators is None else scene_denominators['pair'])
    return dict(loss=factor + pair_weight * pair, factor=factor, pair=pair,
                factor_count=factor_elements.sum().detach(),
                pair_count=pair_elements.sum().detach(),
                factor_scene_count=factor_scene_count, pair_scene_count=pair_scene_count,
                factor_numerator=factor_numerator, pair_numerator=pair_numerator,
                pair_eligible=eligible.detach())


def choose_aux_scenes(scene_ids: Sequence[str], fraction: float = 0.5, *,
                      generator: torch.Generator) -> torch.Tensor:
    """Uniform scene positions using a separately checkpointed CPU RNG.

    Ceil preserves at least one replay on tiny microbatches. Repeated scene IDs
    remain distinct sampler exposures; no position is permanently privileged.
    """
    if not 0 < fraction <= 1 or generator.device.type != "cpu":
        raise ValueError("Aux selection requires fraction in (0,1] and a CPU generator")
    count = len(scene_ids)
    if not count:
        return torch.empty(0, dtype=torch.long)
    selected = max(1, math.ceil(count * fraction))
    return torch.randperm(count, generator=generator)[:selected]


def diagnostic_predicted_pdms(logits: torch.Tensor) -> torch.Tensor:
    """Fixed canonical5:5:2 diagnostic, never an actor maximization objective."""
    if logits.shape[-1] != len(METRIC_NAMES):
        raise ValueError("Expected six metric logits")
    p = logits.float().sigmoid()
    return p[..., 0] * p[..., 1] * (5 * p[..., 3] + 5 * p[..., 2] + 2 * p[..., 4]) / 12


def replay_post_action(actor, post_head: PostActionHead, prepared_context,
                       scene_indices: torch.Tensor, pair_batch, *, codec=None,
                       **loss_options) -> dict[str, torch.Tensor]:
    """One batched encode-only actor pass over the exact labelled candidates.

    ``prepared_context`` must be the same differentiable context already used
    by the main clean-x pass. It is indexed, never recomputed or detached.
    Neither ODE sampling nor the final action output projection is called.
    """
    if codec is None:
        from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
        codec = ActionCodec
    if pair_batch.trajectory.shape != (len(scene_indices), 2, 8, 3) or not len(scene_indices):
        raise ValueError("Replay requires two exact trajectories per selected scene")
    device = next(post_head.parameters()).device
    coordinates = pair_batch.trajectory.detach().to(device=device, dtype=torch.float32)
    normalized = codec.norm_odo(coordinates).reshape(-1, 8, 3)
    mapping = scene_indices.detach().to(device=device, dtype=torch.long).repeat_interleave(2)
    context = prepared_context.index_select(mapping)
    hidden = actor.forward_features(normalized,
        torch.zeros(len(normalized), device=device, dtype=torch.float32), context)
    logits = post_head(hidden).reshape(len(scene_indices), 2, len(METRIC_NAMES))
    result = post_action_loss(logits, pair_batch.labels.to(device).detach(),
        pair_batch.valid.to(device).detach(), pair_batch.pair_mask.to(device).detach(),
        **loss_options)
    result['logits'] = logits
    return result
