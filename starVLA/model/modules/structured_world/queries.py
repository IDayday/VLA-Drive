"""Registered K pose distributions and complete-body field reads."""
import torch
from .grid import GridSpec


def body_points(poses, body, *, spacing_m=.5, lateral_offset_m=0.):
    """Cover the rectangle, including every edge/corner and rear-axle offset."""
    length, width = float(body['length']), float(body['width'])
    xs = torch.linspace(-length/2, length/2, int(length/spacing_m)+2, device=poses.device)
    ys = torch.linspace(-width/2, width/2, int(width/spacing_m)+2, device=poses.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    local = torch.stack((xx.flatten()+float(body['rear_axle_to_center']), yy.flatten()+lateral_offset_m), -1)
    angle = poses[..., 2:3]
    c, s = angle.cos(), angle.sin()
    x = c*local[:, 0]-s*local[:, 1]
    y = s*local[:, 0]+c*local[:, 1]
    return torch.stack((x, y), -1)+poses[..., None, :2]


def boundary_candidates(road, occupancy, valid):
    current = occupancy[0]
    changed = (occupancy[1:] != current[None]) & valid[1:] & valid[:1]
    binary = (occupancy[1:] == 1).float().unsqueeze(1)
    edge = (torch.nn.functional.max_pool2d(binary, 3, stride=1, padding=1) -
            (-torch.nn.functional.max_pool2d(-binary, 3, stride=1, padding=1)))[:, 0] > 0
    return valid[1:] & ((road.abs() <= 1.)[None] | changed | edge)


def sample_queries(q0_physical, targets, *, mode, generator, count=1024, grid=None):
    """50% proposal-body neighborhood, 25% boundaries/events, 25% global.

    Poses and labels are detached. Empty strata use global valid cells with
    replacement. Out-of-domain proposals remain in planner/evaluation outputs;
    auxiliary pose fallback counts are explicitly reported.
    """
    if mode not in ('uniform', 'local') or count % 4:
        raise ValueError('Registered distribution and K multiple of four required')
    grid = grid or GridSpec()
    result, times, records = [], [], []
    valid = targets['occupancy_valid'].detach()
    occ = targets['occupancy'].detach()
    for row in range(len(q0_physical)):
        candidates = torch.nonzero(valid[row, 1:], as_tuple=False)
        if len(candidates) == 0:
            # Preserve scene/trajectory population, return invalid auxiliary
            # positions explicitly rather than hallucinating free labels.
            result.append(q0_physical.new_zeros(count, 3)); times.append(torch.zeros(count, dtype=torch.long, device=q0_physical.device))
            records.append({'global_valid_cells': 0, 'fallback': count, 'invalid_scene': True})
            continue
        def draw(pool, number):
            chosen = pool[torch.randint(len(pool), (number,), generator=generator, device=pool.device)]
            position = q0_physical.new_empty(number, 3)
            position[:, 0] = grid.xmin+(chosen[:, 2]+.5)*grid.dx
            position[:, 1] = grid.ymin+(chosen[:, 1]+.5)*grid.dy
            position[:, 2] = (torch.rand(number, generator=generator, device=pool.device)*2-1)*torch.pi
            return position, chosen[:, 0]
        position, selected_t = draw(candidates, count)
        fallback = 0
        if mode == 'local':
            local_count, boundary_count = count//2, count//4
            anchor_t = torch.randint(q0_physical.shape[1], (local_count,), generator=generator, device=q0_physical.device)
            neighbor = torch.randint(3, (local_count,), generator=generator, device=q0_physical.device)-1
            t = (anchor_t+neighbor).clamp(0, q0_physical.shape[1]-1)
            local = q0_physical[row, anchor_t].detach().clone()
            perturbation = (torch.rand(local_count, 3, generator=generator, device=local.device)*2-1)*local.new_tensor([3., 3., .2])
            local += perturbation
            _, out = grid.normalized_reference(local[:, :2])
            x = torch.floor((local[:, 0]-grid.xmin)/grid.dx).long().clamp(0, grid.width-1)
            y = torch.floor((local[:, 1]-grid.ymin)/grid.dy).long().clamp(0, grid.height-1)
            eligible = ~out & valid[row, t+1, y, x]
            position[:local_count] = torch.where(eligible[:, None], local, position[:local_count])
            selected_t[:local_count] = torch.where(eligible, t, selected_t[:local_count])
            fallback += int((~eligible).sum())
            boundary = torch.nonzero(boundary_candidates(targets['road_distance'][row], occ[row], valid[row]), as_tuple=False)
            if len(boundary):
                position[local_count:local_count+boundary_count], selected_t[local_count:local_count+boundary_count] = draw(boundary, boundary_count)
            else:
                fallback += boundary_count
        result.append(position.detach()); times.append(selected_t.detach())
        records.append({'global_valid_cells': len(candidates), 'fallback': fallback, 'invalid_scene': False,
                        'registered_pose_strata': {'proposal_body_neighborhood': count//2 if mode == 'local' else 0,
                            'road_occupancy_event_boundary': count//4 if mode == 'local' else 0,
                            'global_uniform': count//4 if mode == 'local' else count},
                        'neighbor_time_queries': int(((t != anchor_t) & eligible).sum()) if mode == 'local' else 0})
    return {'poses': torch.stack(result), 'time_index': torch.stack(times), 'strata_records': records}


def read_time_field(field, xy, time_index, grid=None, *, mode='bilinear'):
    """Read B,T,C,H,W at each pose's real physical time, without time averaging."""
    grid = grid or GridSpec()
    batch, count = time_index.shape
    result = field.new_zeros(batch, count, field.shape[2])
    supported = torch.zeros(batch, count, dtype=torch.bool, device=field.device)
    for t in range(field.shape[1]):
        value, in_grid = grid.sample(field[:, t], xy, mode=mode)
        selected = time_index == t
        result = result+value*selected[..., None]
        supported |= selected & in_grid
    return result, supported


def body_relations(road, future_occupancy, poses, time_index, bodies, *, grid=None):
    grid = grid or GridSpec()
    road_margins, overlaps, supported = [], [], []
    for row, body in enumerate(bodies):
        points = body_points(poses[row], body)
        count, coverage = points.shape[:2]
        xy = points.reshape(1, count*coverage, 2)
        distance, in_road = grid.sample(road[row:row+1], xy)
        time = time_index[row, :, None].expand(-1, coverage).reshape(1, -1)
        occupancy, in_occ = read_time_field(future_occupancy[row:row+1], xy, time, grid)
        road_margins.append(distance.reshape(count, coverage).amin(1))
        overlaps.append(occupancy.reshape(count, coverage).amax(1))
        supported.append((in_road & in_occ).reshape(count, coverage).all(1))
    return {'road_margin_m': torch.stack(road_margins), 'occupancy_overlap_proxy': torch.stack(overlaps),
            'in_range': torch.stack(supported)}
