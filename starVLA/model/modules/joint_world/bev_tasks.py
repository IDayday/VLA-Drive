"""Current-camera BEV tasks: covered occupancy and track-consistent motion, not RGB."""
import torch
from torch import nn
from torch.nn import functional as F
from starVLA.model.modules.structured_world.rehab import reference_support


class TaskBEVEncoder(nn.Module):
    def __init__(self, input_dim=1024, dim=128, grid_shape=(49, 40), steps=8):
        super().__init__()
        self.grid_shape, self.steps = tuple(grid_shape), steps
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.position = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
                                     nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.occupancy = nn.Linear(dim, 1)
        self.motion = nn.Linear(dim, steps * 2)

    def forward(self, features, coordinates, support):
        b, n, _ = features.shape
        if n != self.grid_shape[0] * self.grid_shape[1] or coordinates.shape != (b, n, 3) or support.shape != (b, n):
            raise ValueError('BEV geometry/feature shape mismatch')
        if support.dtype != torch.bool:
            raise ValueError('BEV observation support must be boolean')
        h = self.project(features.float()) + self.position(coordinates.float() / 50.)
        h = h.masked_fill(~support[..., None], 0.)
        h = self.spatial(h.transpose(1, 2).reshape(b, -1, *self.grid_shape)).flatten(2).transpose(1, 2)
        return {'memory': h.masked_fill(~support[..., None], 0.), 'occupancy_logits': self.occupancy(h).squeeze(-1),
                'future_displacement': self.motion(h).reshape(b, n, self.steps, 2) * 5.}


class BEVGraphFusion(nn.Module):
    """Task-trained spatial memory is actually consumed by trajectory graph actors."""
    def __init__(self, condition_dim, bev_dim=128, heads=4):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, bev_dim))
        self.location = nn.Linear(2, bev_dim)
        self.attention = nn.MultiheadAttention(bev_dim, heads, dropout=0., batch_first=True)
        self.output = nn.Linear(bev_dim, condition_dim)
        self.null = nn.Parameter(torch.zeros(1, 1, bev_dim))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, actors, centres, memory, support):
        query = self.query(actors.float()) + self.location(centres.float() / 50.)
        # Null memory keeps empty-camera-support batches finite. It is not observed space.
        keys = torch.cat([memory, self.null.expand(len(actors), -1, -1)], 1)
        mask = torch.cat([~support, torch.zeros(len(actors), 1, dtype=torch.bool, device=support.device)], 1)
        features, _ = self.attention(query, keys, keys, key_padding_mask=mask, need_weights=False)
        return actors + self.gate.tanh() * self.output(features), features


class InteractionGeometryHead(nn.Module):
    """Auxiliary pair separation regression, never a candidate scorer or planning cost."""
    def __init__(self, dim=128, hidden=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, actor_features):
        left, right = actor_features[:, :, None], actor_features[:, None, :]
        pair = torch.cat([left + right, (left - right).abs()], -1)
        return F.softplus(self.head(pair).squeeze(-1)) * 20.


@torch.no_grad()
def raster_targets(target, coordinates, observation_support, steps=8):
    """GT is used only here. Future absence never creates free-space negatives."""
    xy = coordinates[:, :2]
    n = len(xy)
    covered = reference_support(xy, target) & observation_support
    if not bool(target.annotation_valid_mask): covered[:] = False
    occupied = torch.zeros(n, dtype=torch.bool, device=xy.device)
    displacement = xy.new_zeros(n, steps, 2)
    valid = torch.zeros(n, steps, dtype=torch.bool, device=xy.device)
    boxes = target.current_boxes
    if not len(boxes): return occupied, covered, displacement, valid
    box_valid = target.box_valid_mask.bool()
    geometry_valid = box_valid[:, [0, 1, 3, 4, 6, 7]].all(-1)
    # Unknown shape/orientation must not be silently treated as free space.
    if not geometry_valid.all():
        # Without a reliable footprint there is no safe negative-cell assignment.
        # Conservatively omit current free-space loss for this scene, retain known positives.
        covered[:] = False
    boxes = boxes.masked_fill(~box_valid, 0.)
    delta = xy[:, None] - boxes[None, :, :2]
    sin, cos = boxes[:, 6], boxes[:, 7]
    longitudinal = delta[..., 0] * cos + delta[..., 1] * sin
    lateral = -delta[..., 0] * sin + delta[..., 1] * cos
    inside = (longitudinal.abs() <= boxes[None, :, 3] / 2) & (lateral.abs() <= boxes[None, :, 4] / 2)
    inside &= geometry_valid[None]
    ignored = ~target.current_supervision_mask.bool()
    covered &= ~(inside & ignored[None]).any(-1)
    inside &= ~ignored[None]
    occupied = inside.any(-1)
    # Known positive footprints stay supervised when unrelated box geometry is incomplete.
    covered |= occupied & reference_support(xy, target) & observation_support & bool(target.annotation_valid_mask)
    distances = delta.square().sum(-1).masked_fill(~inside, float('inf'))
    assignment = distances.argmin(-1)
    # Overlapping labels are ambiguous: don't stitch different tracks' motion.
    unique = inside.sum(-1) == 1
    ids = torch.where(unique & covered)[0]
    if len(ids):
        tracks = assignment[ids]
        valid[ids] = target.future_valid_mask[tracks]
        offsets = target.future_xy_in_ego_t0[tracks] - target.current_boxes[tracks, None, :2]
        displacement[ids] = offsets.masked_fill(~valid[ids, :, None], 0.)
    return occupied, covered, displacement, valid


def bev_loss_sums(prediction, targets, coordinates, support):
    zero = prediction['occupancy_logits'].sum() * 0 + prediction['future_displacement'].sum() * 0
    sums = {k: zero for k in ['occupied', 'free', 'motion']}; counts = {k: 0 for k in sums}
    for i, target in enumerate(targets):
        occupied, covered, future, valid = raster_targets(target, coordinates[i], support[i], prediction['future_displacement'].shape[-2])
        bce = F.binary_cross_entropy_with_logits(prediction['occupancy_logits'][i], occupied.float(), reduction='none')
        for key, selected in [('occupied', covered & occupied), ('free', covered & ~occupied)]:
            sums[key] = sums[key] + bce[selected].sum(); counts[key] += int(selected.sum())
        error = F.smooth_l1_loss(prediction['future_displacement'][i] / 5., future / 5., reduction='none').sum(-1)
        sums['motion'] = sums['motion'] + error[valid].sum(); counts['motion'] += int(valid.sum()) * 2
    return sums, counts


def interaction_loss_sum(predicted_distance, future_xy, valid):
    # One undirected pair per scene; separation is observational geometry, not risk probability.
    clean = future_xy.masked_fill(~valid[..., None], 0.)
    delta = clean[:, :, None] - clean[:, None, :]
    distance = delta.norm(dim=-1)
    common = valid[:, :, None] & valid[:, None, :]
    nearest = distance.masked_fill(~common, float('inf')).amin(-1)
    pair = common.any(-1) & torch.triu(torch.ones_like(nearest, dtype=torch.bool), diagonal=1)
    label = nearest.masked_fill(~pair, 0.)
    error = F.smooth_l1_loss(predicted_distance / 20., label / 20., reduction='none')
    return error[pair].sum(), int(pair.sum())
