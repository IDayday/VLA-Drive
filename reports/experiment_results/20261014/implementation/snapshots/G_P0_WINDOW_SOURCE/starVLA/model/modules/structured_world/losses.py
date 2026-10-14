"""Scene-normalized structured losses; FM is left to the original action head."""
import torch
from torch.nn import functional as F
from .grid import GridSpec
from .queries import read_time_field, body_relations, _TimeReadPlan, _body_footprints, ForwardFieldFiniteCache


def scene_means(values, valid):
    valid = valid.detach().bool()
    if valid.shape != values.shape:
        valid = valid.expand_as(values)
    sums = (torch.where(valid, values, torch.zeros_like(values))).flatten(1).sum(1)
    counts = valid.flatten(1).sum(1)
    return sums/counts.clamp_min(1), counts


def scene_reduce(values, *, global_scenes=None):
    from torch import distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    denominator = float(global_scenes if global_scenes is not None else len(values)*world)
    if denominator <= 0:
        raise ValueError('Positive global effective scene count required')
    return values.sum()*world/denominator


def binary_loss(logits, labels, valid, *, positive_weight=1.):
    if positive_weight <= 0:
        raise ValueError('Class weight must be derived from training statistics')
    raw = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none',
                                              pos_weight=logits.new_tensor(positive_weight))
    return scene_means(raw, valid)


def compute_structured_losses(prediction, targets, queries, bodies, *, semantic_mode,
                               statistics, global_scenes=None, grid=None, execution_fast=False):
    grid = grid or GridSpec()
    xy, time = queries['poses'][..., :2], queries['time_index']
    prepared = grid.prepare_sample(xy) if execution_fast else None
    plan = _TimeReadPlan(xy, time, targets['occupancy'].shape[1]-1, grid) if execution_fast else None
    finite_cache = ForwardFieldFiniteCache() if execution_fast else None
    # The registered labels are uint8 and bool. Their float casts cannot create
    # nonfinite values; unusual floating labels still take the checked path.
    occupancy_finite = not targets['occupancy'].is_floating_point() and not targets['occupancy'].is_complex()
    valid_finite = not targets['occupancy_valid'].is_floating_point() and not targets['occupancy_valid'].is_complex()
    road, in_grid = grid.sample(prediction['road_distance_m'], xy, _prepared=prepared)
    road_gt, _ = grid.sample(targets['road_distance'][:, None].float(), xy, _prepared=prepared)
    road_valid, _ = grid.sample(targets['road_valid'][:, None].float(), xy, mode='nearest', _prepared=prepared)
    current, _ = grid.sample(prediction['current_logits'], xy, _prepared=prepared)
    O0, _ = grid.sample(targets['occupancy'][:, :1].float(), xy, mode='nearest', _prepared=prepared)
    valid0, _ = grid.sample(targets['occupancy_valid'][:, :1].float(), xy, mode='nearest', _prepared=prepared)
    Ot, in_future = read_time_field(targets['occupancy'][:, 1:, None].float(), xy, time, grid, mode='nearest',
        execution_fast=execution_fast, known_finite=occupancy_finite, finite_cache=finite_cache, _plan=plan)
    validt, _ = read_time_field(targets['occupancy_valid'][:, 1:, None].float(), xy, time, grid, mode='nearest',
        execution_fast=execution_fast, known_finite=valid_finite, finite_cache=finite_cache, _plan=plan)
    valid0 = (valid0[..., 0] > .5) & in_grid & (O0[..., 0] != 255)
    pair_valid = valid0 & (validt[..., 0] > .5) & in_future & (Ot[..., 0] != 255)
    raw, counts = {}, {}
    values, count = scene_means(F.smooth_l1_loss(road[..., 0]/10., road_gt[..., 0]/10., reduction='none'),
                                (road_valid[..., 0] > .5) & in_grid)
    raw['road'] = scene_reduce(values, global_scenes=global_scenes); counts['road'] = count.detach()
    values, count = binary_loss(current[..., 0], (O0[..., 0] == 1), valid0,
                                positive_weight=statistics['current_positive_weight'])
    raw['current_occupancy'] = scene_reduce(values, global_scenes=global_scenes); counts['current_occupancy'] = count.detach()
    if semantic_mode != 'none':
        future_logits, _ = read_time_field(prediction['future_logits'], xy, time, grid,
            execution_fast=execution_fast, finite_cache=finite_cache, _plan=plan)
        if semantic_mode == 'full':
            weight = future_logits.new_tensor([1., statistics['future_positive_weight']])
            labels = (Ot[..., 0] == 1).long()
            values, count = scene_means(F.cross_entropy(future_logits.transpose(1, 2), labels,
                                                       reduction='none', weight=weight), pair_valid)
            raw['future_semantics'] = scene_reduce(values, global_scenes=global_scenes)
            counts['future_semantics'] = count.detach()
        elif semantic_mode == 'event':
            arrival_mask, release_mask = pair_valid & (O0[..., 0] == 0), pair_valid & (O0[..., 0] == 1)
            arrival, acount = binary_loss(future_logits[..., 0], (Ot[..., 0] == 1), arrival_mask,
                                          positive_weight=statistics['arrival_positive_weight'])
            release, rcount = binary_loss(future_logits[..., 1], (Ot[..., 0] == 0), release_mask,
                                          positive_weight=statistics['release_positive_weight'])
            # Separate conditional Bernoulli means, explicitly different from
            # the full-future softmax objective's normalization.
            raw['future_semantics'] = scene_reduce(.5*(arrival+release), global_scenes=global_scenes)
            counts['arrival'], counts['release'] = acount.detach(), rcount.detach()
        else:
            raise ValueError('Unknown future semantic objective')
    # Local relations are reads of these same prediction fields, not a risk head.
    footprints = _body_footprints(queries['poses'], time, bodies, grid, targets['occupancy'].shape[1]-1) if execution_fast else None
    predicted = body_relations(prediction['road_distance_m'], prediction['pt'],
                                queries['poses'], time, bodies, grid=grid, execution_fast=execution_fast,
                                finite_cache=finite_cache, _footprints=footprints)
    truth = body_relations(targets['road_distance'][:, None].float(), (targets['occupancy'][:, 1:, None] == 1).float(),
                            queries['poses'], time, bodies, grid=grid, execution_fast=execution_fast,
                            known_finite=True, finite_cache=finite_cache, _footprints=footprints)
    if execution_fast:
        road_margins, full_valid = [], []
        for row, footprint in enumerate(footprints):
            count_k, points_k = footprint['points'].shape[:2]
            support, _ = grid.sample(targets['road_valid'][row:row+1, None].float(), footprint['xy'],
                                      _prepared=footprint['prepared'])
            # Same bilinear reads and amin tie rule as body_relations. Its
            # discarded future-valid overlap/in_range are not needed here.
            road_margins.append(support.reshape(count_k, points_k).amin(1))
            if semantic_mode != 'none':
                sampled, supported = read_time_field(targets['occupancy_valid'][row:row+1, 1:, None].float(),
                    footprint['xy'], footprint['time'], grid, execution_fast=True, known_finite=valid_finite,
                    finite_cache=finite_cache, _plan=footprint['plan'])
                full_valid.append(((sampled[..., 0] >= 1.-1e-5) & supported).reshape(count_k, points_k).all(1))
        road_support = {'road_margin_m': torch.stack(road_margins)}
    else:
        road_support = body_relations(targets['road_valid'][:, None].float(),
                            targets['occupancy_valid'][:, 1:, None].float(), queries['poses'], time, bodies, grid=grid,
                            execution_fast=execution_fast, known_finite=valid_finite, finite_cache=finite_cache, _footprints=footprints)
    # The minimum over all body points requires all road labels to be known.
    road_mask = (road_support['road_margin_m'] >= 1.-1e-5) & predicted['in_range']
    values, count = scene_means(F.smooth_l1_loss(predicted['road_margin_m']/10., truth['road_margin_m']/10., reduction='none'), road_mask)
    relation = scene_reduce(values, global_scenes=global_scenes)
    counts['body_road'] = count.detach()
    if semantic_mode != 'none':
        # Verify that every future footprint cell is labeled.
        from .queries import body_points
        if not execution_fast:
            full_valid = []
            for row, body in enumerate(bodies):
                footprint = footprints[row] if footprints is not None else None
                points = footprint['points'] if footprint is not None else body_points(queries['poses'][row], body)
                count_k, points_k = points.shape[:2]
                t = footprint['time'] if footprint is not None else time[row, :, None].expand(-1, points_k).reshape(1, -1)
                body_xy = footprint['xy'] if footprint is not None else points.reshape(1, -1, 2)
                sampled, supported = read_time_field(targets['occupancy_valid'][row:row+1, 1:, None].float(),
                                                      body_xy, t, grid, execution_fast=execution_fast,
                                                      known_finite=valid_finite, finite_cache=finite_cache,
                                                      _plan=footprint['plan'] if footprint is not None else None)
                full_valid.append(((sampled[..., 0] >= 1.-1e-5) & supported).reshape(count_k, points_k).all(1))
        mask = torch.stack(full_valid) & predicted['in_range']
        values, count = scene_means((predicted['occupancy_overlap_proxy']-truth['occupancy_overlap_proxy']).square(), mask)
        relation = relation+scene_reduce(values, global_scenes=global_scenes)
        counts['body_occupancy'] = count.detach()
    raw['query_relations'] = relation
    return raw, counts


def final_trajectory_loss(encoded, ego_gt, future_valid, *, global_scenes=None):
    heading = F.normalize(encoded[..., 2:4], dim=-1, eps=1e-6)
    final = torch.cat((encoded[..., :2], heading), -1)
    values, count = scene_means((final-ego_gt).square(), future_valid[..., None])
    return scene_reduce(values, global_scenes=global_scenes), count
