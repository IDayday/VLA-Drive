"""Training-only current/future readouts from current student memory.

No target is an input to a readout. All reductions are FP32, first over valid
elements within each scene, then over scenes with supervision across ranks.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F


def _scene_reduce(values: Tensor, valid: Tensor, *, return_stats: bool = False,
                  scene_denominator=None):
    if valid.dtype != torch.bool:
        raise TypeError('Supervision masks must be boolean')
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(values)
    if not torch.isfinite(values[valid]).all():
        raise FloatingPointError('Nonfinite value in valid supervision')
    count = valid.flatten(1).sum(1)
    sums = torch.where(valid, values.float(), 0.).flatten(1).sum(1)
    has_target = count > 0
    numerator = (sums / count.clamp_min(1)).sum()
    scenes = has_target.sum().detach().double()
    elements = count.sum().detach().double()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if scene_denominator is not None:
        scenes = torch.as_tensor(scene_denominator, device=values.device, dtype=torch.float64).detach()
        if scenes.ndim or not torch.isfinite(scenes) or scenes < 0:
            raise ValueError('Full-update scene denominator must be a finite nonnegative scalar')
        if scenes == 0 and has_target.any():
            raise ValueError('Nonempty supervision with a zero full-update denominator')
    elif world > 1:
        dist.all_reduce(scenes)
        dist.all_reduce(elements)
    loss = numerator * world / scenes.clamp_min(1).to(numerator.dtype)
    if return_stats:
        return loss, dict(valid_scenes=scenes, valid_elements=elements,
                          local_scene_sum=numerator.detach())
    return loss


def valid_scene_mse(prediction: Tensor, target: Tensor, valid: Tensor, *, return_stats=False,
                    scene_denominator=None):
    if prediction.shape != target.shape:
        raise ValueError(f'Target shape mismatch: {prediction.shape} versus {target.shape}')
    # Mask before arithmetic: NaN invalid markers must not poison the graph.
    mask = valid
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(prediction)
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise FloatingPointError('Nonfinite valid feature prediction/target')
    error = (torch.where(mask, prediction.float(), 0.) -
             torch.where(mask, target.detach().float(), 0.)).square()
    return _scene_reduce(error, mask, return_stats=return_stats, scene_denominator=scene_denominator)


class CrossReadoutBlock(nn.Module):
    def __init__(self, width=512, heads=8, *, self_attention=False):
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.self_norm = nn.LayerNorm(width)
            self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width*4), nn.GELU(approximate='tanh'),
                                 nn.Linear(width*4, width))

    def forward(self, query, memory, valid):
        if not valid.any(1).all():
            raise ValueError('Readout requires at least one valid current memory token per scene')
        if self.self_attention:
            q = self.self_norm(query)
            query = query + self.self_attn(q, q, q, need_weights=False)[0]
        q, kv = self.query_norm(query), self.memory_norm(memory)
        query = query + self.cross_attn(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0]
        return query + self.ffn(self.ffn_norm(query))


class DINOReadout(nn.Module):
    def __init__(self, memory_dim=1024, width=512, layers=2, heads=8,
                 grid=(9, 16), channels=1024):
        super().__init__()
        self.grid = tuple(grid)
        self.camera = nn.Embedding(3, width)
        self.y = nn.Embedding(self.grid[0], width)
        self.x = nn.Embedding(self.grid[1], width)
        self.memory_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, width))
        self.blocks = nn.ModuleList([CrossReadoutBlock(width, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, channels))

    def forward(self, memory):
        source = self.memory_projection(memory.spatial)
        c, y, x = torch.meshgrid(torch.arange(3, device=source.device),
                                 torch.arange(self.grid[0], device=source.device),
                                 torch.arange(self.grid[1], device=source.device), indexing='ij')
        query = (self.camera(c) + self.y(y) + self.x(x)).flatten(0, 2)
        query = query[None].expand(len(source), -1, -1)
        for block in self.blocks:
            query = block(query, source, memory.spatial_valid)
        return self.output(query).reshape(len(source), 3, *self.grid, -1)


def sample_future_queries(batch_size: int, count: int = 128, *, generator: torch.Generator,
                          grid=(8, 16), device=None):
    """No-replacement camera/time-stratified sampling; independent CPU RNG."""
    if generator.device.type != 'cpu' or not 12 <= count <= 12*grid[0]*grid[1]:
        raise ValueError('Require CPU generator and12..all future patches')
    rows = []
    for _ in range(batch_size):
        order = torch.randperm(12, generator=generator)
        allocation = torch.full((12,), count//12, dtype=torch.long)
        allocation[order[:count % 12]] += 1
        items = []
        for bucket, n in enumerate(allocation.tolist()):
            spatial = torch.randperm(grid[0]*grid[1], generator=generator)[:n]
            items.append(torch.stack([torch.full_like(spatial, bucket//4),
                                      torch.full_like(spatial, bucket % 4),
                                      spatial//grid[1], spatial % grid[1]], dim=-1))
        rows.append(torch.cat(items)[torch.randperm(count, generator=generator)])
    return torch.stack(rows).to(device=device)


class FutureReadout(nn.Module):
    def __init__(self, memory_dim=1024, width=512, layers=4, heads=8,
                 grid=(8, 16), channels=512):
        super().__init__()
        self.grid = tuple(grid)
        self.camera = nn.Embedding(3, width)
        self.time = nn.Embedding(4, width)
        self.y = nn.Embedding(self.grid[0], width)
        self.x = nn.Embedding(self.grid[1], width)
        self.memory_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, width))
        self.predictive_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, width))
        self.blocks = nn.ModuleList([CrossReadoutBlock(width, heads, self_attention=True)
                                    for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, channels))

    def forward(self, memory, query_indices: Tensor | None = None):
        source = self.memory_projection(memory.actor_memory)
        full = query_indices is None
        if full:
            axes = [torch.arange(n, device=source.device) for n in (3, 4, *self.grid)]
            query_indices = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(1, -1, 4)
            query_indices = query_indices.expand(len(source), -1, -1)
        if query_indices.dtype != torch.long or query_indices.shape[0] != len(source) or query_indices.shape[-1] != 4:
            raise ValueError('Future query indices must be[B,N,4] int64')
        for column, bound in enumerate((3, 4, *self.grid)):
            if (query_indices[..., column] < 0).any() or (query_indices[..., column] >= bound).any():
                raise ValueError('Future query outside actual camera/tubelet/grid contract')
        c, t, y, x = query_indices.unbind(-1)
        if memory.predictive.shape[1] != 32:
            raise ValueError('Future readout expects four groups of eight predictive tokens')
        dynamics = self.predictive_projection(memory.predictive.reshape(len(source), 4, 8, -1).mean(2))
        selected = dynamics.gather(1, t[..., None].expand(-1, -1, dynamics.shape[-1]))
        query = self.camera(c) + self.time(t) + self.y(y) + self.x(x) + selected
        for block in self.blocks:
            query = block(query, source, memory.actor_valid)
        result = self.output(query)
        return result.reshape(len(source), 3, 4, *self.grid, -1) if full else result


def gather_future_targets(features: Tensor, valid: Tensor, queries: Tensor):
    if features.ndim != 6 or tuple(features.shape[1:5]) != (3, 4, 8, 16):
        raise ValueError('Expected actual projected WA target[B,3,4,8,16,512]')
    if tuple(valid.shape) != tuple(features.shape[:-1]) or valid.dtype != torch.bool:
        raise ValueError('WA target requires patch validity')
    b = torch.arange(len(features), device=features.device)[:, None]
    c, t, y, x = queries.unbind(-1)
    return features[b, c, t, y, x], valid[b, c, t, y, x]


class CurrentObjectReadout(nn.Module):
    def __init__(self, memory_dim=1024, width=256, layers=2, heads=4,
                 num_queries=32, num_classes=7):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.empty(num_queries, width))
        nn.init.normal_(self.query, std=.02)
        self.memory_projection = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, width))
        self.blocks = nn.ModuleList([CrossReadoutBlock(width, heads, self_attention=True)
                                    for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.existence = nn.Linear(width, 1)
        self.classes = nn.Linear(width, num_classes)
        self.boxes = nn.Linear(width, 8)

    def forward(self, memory):
        source = self.memory_projection(torch.cat([memory.spatial, memory.global_memory], dim=1))
        valid = torch.cat([memory.spatial_valid, memory.global_valid], dim=1)
        query = self.query[None].expand(len(source), -1, -1)
        for block in self.blocks:
            query = block(query, source, valid)
        hidden = self.norm(query)
        raw_boxes = self.boxes(hidden)
        boxes = torch.cat([raw_boxes[..., :3], F.softplus(raw_boxes[..., 3:6]), raw_boxes[..., 6:]], dim=-1)
        return dict(existence_logits=self.existence(hidden).squeeze(-1),
                    class_logits=self.classes(hidden), boxes=boxes)


def current_object_loss(prediction: dict[str, Tensor], target: dict[str, Tensor], *, return_terms=False,
                        scene_denominators=None):
    from scipy.optimize import linear_sum_assignment
    existence = prediction['existence_logits']
    classes = prediction['class_logits']
    boxes = prediction['boxes']
    truth, truth_class, valid = target['boxes'].detach(), target['classes'].detach(), target['valid'].detach()
    coverage = target['coverage_valid'].detach()
    if (truth.ndim != 3 or truth.shape[-1] != 8 or valid.dtype != torch.bool or coverage.dtype != torch.bool
            or valid.shape != truth.shape[:-1] or truth_class.shape != valid.shape
            or truth_class.dtype != torch.long or coverage.shape != (len(truth),)):
        raise ValueError('Current objects require physical8D boxes and explicit visibility/coverage masks')
    if len(truth) != len(boxes) or classes.shape[:2] != existence.shape or boxes.shape[:2] != existence.shape:
        raise ValueError('Object batch/query mismatch')
    if (valid.sum(1) > existence.shape[1]).any():
        raise ValueError('Dataset must explicitly select at most32 visible objects before matching')
    if not torch.isfinite(truth[valid]).all():
        raise FloatingPointError('Nonfinite visible current annotation')
    if not all(torch.isfinite(value).all() for value in (existence, classes, boxes)):
        raise FloatingPointError('Nonfinite current-object prediction')
    scale = boxes.new_tensor([50., 50., 5., 10., 5., 5., 1., 1.])
    existence_target = torch.zeros_like(existence)
    existence_valid = coverage[:, None].expand_as(existence).clone()
    box_target = torch.zeros_like(boxes)
    class_target = torch.zeros_like(existence, dtype=torch.long)
    match_valid = torch.zeros_like(existence, dtype=torch.bool)
    for batch in range(len(boxes)):
        selected = valid[batch].nonzero().flatten()
        if not len(selected):
            continue
        labels = truth_class[batch, selected]
        if (labels < 0).any() or (labels >= classes.shape[-1]).any():
            raise ValueError('Current object class not covered by the fixed class map')
        cost_geometry = torch.cdist(boxes[batch].detach().float()/scale, truth[batch, selected].float()/scale, p=1)
        cost_class = -classes[batch].detach().float().softmax(-1)[:, labels]
        pred_ids, target_ids = linear_sum_assignment((cost_geometry + cost_class).cpu().numpy())
        pred_ids = torch.as_tensor(pred_ids, device=boxes.device)
        target_ids = selected[torch.as_tensor(target_ids, device=selected.device)]
        existence_target[batch, pred_ids] = 1.
        existence_valid[batch, pred_ids] = True
        match_valid[batch, pred_ids] = True
        box_target[batch, pred_ids] = truth[batch, target_ids]
        class_target[batch, pred_ids] = truth_class[batch, target_ids]
    denominators = scene_denominators or {}
    loss_exists = _scene_reduce(F.binary_cross_entropy_with_logits(existence.float(), existence_target, reduction='none'),
                                existence_valid, scene_denominator=denominators.get('existence'))
    loss_class = _scene_reduce(F.cross_entropy(classes.float().transpose(1, 2), class_target, reduction='none'),
                               match_valid, scene_denominator=denominators.get('classification'))
    loss_box = _scene_reduce(F.smooth_l1_loss(boxes.float()/scale, box_target.float()/scale, reduction='none'),
                             match_valid, scene_denominator=denominators.get('box'))
    loss = loss_exists + loss_class + loss_box
    if return_terms:
        return loss, dict(existence=loss_exists.detach(), classification=loss_class.detach(),
                          box=loss_box.detach(), matched=match_valid.sum().detach())
    return loss
