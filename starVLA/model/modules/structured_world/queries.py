"""Registered K pose distributions and complete-body field reads."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import weakref
import torch
from .grid import GridSpec


def _body_template(length, width, rear, spacing_m, lateral_offset_m, device, dtype):
    xs = torch.linspace(-length/2, length/2, int(length/spacing_m)+2, device=device, dtype=dtype)
    ys = torch.linspace(-width/2, width/2, int(width/spacing_m)+2, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((xx.flatten()+rear, yy.flatten()+lateral_offset_m), -1)


_cached_body_template = lru_cache(maxsize=32)(_body_template)


def body_points(poses, body, *, spacing_m=.5, lateral_offset_m=0., execution_fast=False):
    """Cover the rectangle, including every edge/corner and rear-axle offset."""
    template = _cached_body_template if execution_fast else _body_template
    # The reference linspace uses the default dtype, even for double poses.
    local = template(float(body['length']), float(body['width']), float(body['rear_axle_to_center']),
                     spacing_m, lateral_offset_m, poses.device, torch.get_default_dtype())
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


def _label_identity(targets, row, *, required=False):
    hashes, schemas = targets.get('label_sha256'), targets.get('label_schema')
    if hashes is None or schemas is None:
        if required:
            raise ValueError('CPU candidate preparation requires verified label_sha256/label_schema')
        return None
    sha = hashes[row]
    schema = schemas if isinstance(schemas, str) else schemas[row]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or not isinstance(schema, str) or not schema:
        raise ValueError('Verified per-sample label SHA256 and schema required')
    return sha, schema


def bind_query_label_identities(targets, tokens, *, cache_root, expected_cache_identity, grid=None):
    """Bind already-verified Dataset labels to their original read-only records.

    The Dataset must first verify/load the payloads, as in the reference path.
    This hook only binds those batch rows to metadata; it is not a replacement
    payload verifier. Returns a new shallow targets dict, leaving labels intact.
    """
    grid = grid or GridSpec()
    root = Path(cache_root)
    identity = json.loads((root/'identity.json').read_text())
    if identity['identity'] != expected_cache_identity or GridSpec(**identity['grid']) != grid:
        raise ValueError('Original label cache identity/grid mismatch')
    if len(tokens) != len(targets['occupancy']):
        raise ValueError('Label tokens must follow collated batch row order')
    hashes = []
    for token in tokens:
        if not isinstance(token, str) or not token or Path(token).name != token:
            raise ValueError('Expected an original scene token')
        record = json.loads((root/'records'/(token+'.json')).read_text())
        if record['cache_identity'] != expected_cache_identity or record['token'] != token:
            raise ValueError('Foreign label record or scene token')
        hashes.append(record['label_sha256'])
    bound = dict(targets, label_sha256=hashes, label_schema=[identity['schema']]*len(tokens))
    for row in range(len(tokens)):
        _label_identity(bound, row, required=True)
    return bound


@dataclass(frozen=True)
class PreparedQueryCandidates:
    """Picklable CPU-prepared row; slice the list with the other batch labels.

    Indices retain torch.nonzero's original t/y/x ordering. The caller verifies
    the payload SHA and keeps the identified labels immutable after preparing.
    No RNG is used. Only index tensors move to the device or pinned memory.
    """
    global_indices: torch.Tensor
    boundary_indices: torch.Tensor
    label_sha256: str
    label_schema: str
    grid: GridSpec
    label_shapes: tuple

    def to(self, device, *, non_blocking=False):
        return replace(self, global_indices=self.global_indices.to(device=device, non_blocking=non_blocking),
                       boundary_indices=self.boundary_indices.to(device=device, non_blocking=non_blocking))

    def pin_memory(self):
        return replace(self, global_indices=self.global_indices.pin_memory(),
                       boundary_indices=self.boundary_indices.pin_memory())

    def validate(self, targets, row, grid):
        if _label_identity(targets, row, required=True) != (self.label_sha256, self.label_schema) or self.grid != grid:
            raise ValueError('Prepared candidate label hash/schema/grid mismatch')
        if self.label_shapes != tuple(tuple(targets[key].shape[1:]) for key in ('occupancy_valid', 'occupancy', 'road_distance')):
            raise ValueError('Prepared candidate label shape mismatch')
        for indices in (self.global_indices, self.boundary_indices):
            if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype != torch.long:
                raise ValueError('Expected original nonzero t/y/x indices')


def prepare_query_candidates(targets, *, grid=None, execution_cache=None):
    """CPU collator hook; attach its row list as targets['query_candidates'].

    Call after stacking labels, before pinning/transferring or microbatch slicing.
    Both strata are prepared without RNG so the same result supports G0--G3.
    Pass verified per-row payload hashes/schema alongside the labels. There is
    no persistent cache or label-file modification.
    """
    grid = grid or GridSpec()
    keys = ('occupancy_valid', 'occupancy', 'road_distance')
    if any(targets[key].device.type != 'cpu' for key in keys):
        raise ValueError('Prepare query indices on CPU before optimizer-batch transfer')
    cache = execution_cache if execution_cache is not None else QueryExecutionCache()
    prepared = []
    for row in range(len(targets['occupancy'])):
        sha, schema = _label_identity(targets, row, required=True)
        candidates, boundary = cache.candidates(targets, row, grid, local=True)
        if boundary is None:
            boundary = candidates.new_empty(0, 3)
        prepared.append(PreparedQueryCandidates(candidates, boundary, sha, schema, grid,
            tuple(tuple(targets[key].shape[1:]) for key in keys)))
    return prepared


class QueryExecutionCache:
    """Bounded, memory-only indices for an unchanged, identified label sample.

    Pass collated ``label_sha256`` (one verified payload SHA per row) and
    ``label_schema`` (one schema per row, or a shared string) in targets.
    Missing identities disable index caching. New tensors, in-place changes,
    grid/schema/hash changes invalidate a hit; no label files are read/written.
    """
    def __init__(self, max_samples=8):
        if max_samples < 1:
            raise ValueError('Positive cache capacity required')
        self.max_samples = max_samples
        self._samples = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self._samples.clear()

    def candidates(self, targets, row, grid, *, local):
        tensors = [targets[key] for key in ('occupancy_valid', 'occupancy', 'road_distance')]
        key, snapshot = None, None
        identity = _label_identity(targets, row)
        if identity is not None:
            sha, schema = identity
            try:
                snapshot = [(weakref.ref(t), t._version) for t in tensors]
            except RuntimeError:  # Inference tensors have no mutation counter.
                pass
            if snapshot is not None:
                key = (sha, schema, grid, row, tensors[0].device)
        entry = self._samples.get(key) if key is not None else None
        if entry is not None and all(ref() is t and version == t._version
                                     for (ref, version), t in zip(entry['snapshot'], tensors)):
            self.hits += 1
            self._samples.move_to_end(key)
        else:
            self.misses += 1
            entry = {'snapshot': snapshot, 'global': torch.nonzero(tensors[0][row, 1:].detach(), as_tuple=False),
                     'boundary': None}
            if key is not None:
                self._samples[key] = entry
                self._samples.move_to_end(key)
                while len(self._samples) > self.max_samples:
                    self._samples.popitem(last=False)
        if local and len(entry['global']) and entry['boundary'] is None:
            entry['boundary'] = torch.nonzero(boundary_candidates(tensors[2][row].detach(),
                tensors[1][row].detach(), tensors[0][row].detach()), as_tuple=False)
        return entry['global'], entry['boundary']


def sample_queries(q0_physical, targets, *, mode, generator, count=1024, grid=None,
                   execution_fast=False, execution_cache=None):
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
    cache = (execution_cache if execution_cache is not None else QueryExecutionCache()) if execution_fast else None
    prepared = targets.get('query_candidates') if execution_fast else None
    if prepared is not None and len(prepared) != len(q0_physical):
        raise ValueError('Prepared candidate rows must follow optimizer-batch slicing')
    for row in range(len(q0_physical)):
        if prepared is not None:
            item = prepared[row]
            item.validate(targets, row, grid)
            item = item.to(q0_physical.device, non_blocking=True)
            candidates, boundary = item.global_indices, item.boundary_indices
        elif execution_fast:
            candidates, boundary = cache.candidates(targets, row, grid, local=mode == 'local')
        else:
            candidates = torch.nonzero(valid[row, 1:], as_tuple=False)
        if len(candidates) == 0:
            # Preserve scene/trajectory population, return invalid auxiliary
            # positions explicitly rather than hallucinating free labels.
            result.append(q0_physical.new_zeros(count, 3)); times.append(torch.zeros(count, dtype=torch.long, device=q0_physical.device))
            records.append({'global_valid_cells': 0,
                            'fallback': torch.full((), count, dtype=torch.long, device=q0_physical.device) if execution_fast else count,
                            'invalid_scene': True})
            continue
        def draw(pool, number):
            chosen = pool[torch.randint(len(pool), (number,), generator=generator, device=pool.device)]
            position = q0_physical.new_empty(number, 3)
            position[:, 0] = grid.xmin+(chosen[:, 2]+.5)*grid.dx
            position[:, 1] = grid.ymin+(chosen[:, 1]+.5)*grid.dy
            position[:, 2] = (torch.rand(number, generator=generator, device=pool.device)*2-1)*torch.pi
            return position, chosen[:, 0]
        position, selected_t = draw(candidates, count)
        fallback = selected_t.new_zeros(()) if execution_fast else 0
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
            local_fallback = (~eligible).sum().detach()
            fallback += local_fallback if execution_fast else int(local_fallback)
            if not execution_fast:
                boundary = torch.nonzero(boundary_candidates(targets['road_distance'][row], occ[row], valid[row]), as_tuple=False)
            if len(boundary):
                position[local_count:local_count+boundary_count], selected_t[local_count:local_count+boundary_count] = draw(boundary, boundary_count)
            else:
                fallback += boundary_count
        result.append(position.detach()); times.append(selected_t.detach())
        neighbor_queries = ((t != anchor_t) & eligible).sum().detach() if mode == 'local' else selected_t.new_zeros(()) if execution_fast else 0
        if not execution_fast and mode == 'local':
            neighbor_queries = int(neighbor_queries)
        records.append({'global_valid_cells': len(candidates), 'fallback': fallback, 'invalid_scene': False,
                        'registered_pose_strata': {'proposal_body_neighborhood': count//2 if mode == 'local' else 0,
                            'road_occupancy_event_boundary': count//4 if mode == 'local' else 0,
                            'global_uniform': count//4 if mode == 'local' else count},
                        'neighbor_time_queries': neighbor_queries})
    return {'poses': torch.stack(result), 'time_index': torch.stack(times), 'strata_records': records}


class _TimeReadPlan:
    """Pack each (batch,time) into a 2D grid_sample batch, preserving K order."""
    def __init__(self, xy, time_index, steps, grid):
        self.xy, self.time_index, self.steps, self.grid = xy, time_index, steps, grid
        batch, count = time_index.shape
        integral_time = time_index.long()
        selected = torch.nonzero((time_index >= 0) & (time_index < steps) & (time_index == integral_time), as_tuple=False)
        group = selected[:, 0]*steps+integral_time[selected[:, 0], selected[:, 1]]
        order = torch.argsort(group, stable=True)
        self.selected, self.group = selected[order], group[order]
        counts = torch.bincount(group, minlength=batch*steps)
        starts = counts.cumsum(0)-counts
        self.slot = torch.arange(len(group), device=xy.device)-starts[self.group]
        width = max(1, int(counts.max()))
        self.packed_xy = xy.new_zeros(batch*steps, width, 2)
        self.packed_xy[self.group, self.slot] = xy[self.selected[:, 0], self.selected[:, 1]]
        self.prepared = grid.prepare_sample(self.packed_xy)

    def read(self, field, mode):
        value, supported = self.grid.sample(field.flatten(0, 1), self.packed_xy,
                                            mode=mode, _prepared=self.prepared)
        batch, count = self.time_index.shape
        # Keep zero gradients (rather than None) even with no selected times.
        result = field.new_zeros(batch, count, field.shape[2])+value.reshape(-1, field.shape[2])[:0].sum(0)
        result = result+self.xy[:, :0].sum()
        result[self.selected[:, 0], self.selected[:, 1]] = value[self.group, self.slot]
        valid = torch.zeros(batch, count, dtype=torch.bool, device=field.device)
        valid[self.selected[:, 0], self.selected[:, 1]] = supported[self.group, self.slot]
        return result, valid


class ForwardFieldFiniteCache:
    """One forward only: check a floating field once across all its row views.

    Never retain this cache across optimizer updates. In-place mutations change
    the version and force a new check. Integer/bool sources are finite by type.
    """
    def __init__(self):
        self._fields = {}
        self.checks = 0

    def is_finite(self, field):
        if not field.is_floating_point() and not field.is_complex():
            return True
        base = field
        while base._base is not None:
            base = base._base
        try:
            version = base._version
        except RuntimeError:  # No safe mutation tracking for inference tensors.
            self.checks += 1
            return bool(torch.isfinite(field).all())
        entry = self._fields.get(id(base))
        if entry is None or entry[0] is not base or entry[1] != version:
            self.checks += 1
            entry = (base, version, bool(torch.isfinite(base).all()))
            self._fields[id(base)] = entry
        return entry[2]


def read_time_field(field, xy, time_index, grid=None, *, mode='bilinear', execution_fast=False,
                    known_finite=False, finite_cache=None, _plan=None):
    """Read B,T,C,H,W at each pose's real physical time, without time averaging.

    ``known_finite`` is a caller-provided proof (e.g. byte/bool label casts),
    not a requested relaxation of NaN semantics. Otherwise a forward-scoped
    ``finite_cache`` checks floating sources once and shares results over views.
    """
    grid = grid or GridSpec()
    # In the reference, NaN/Inf at an unselected time still enters through
    # value*0. Sparse reads must retain that behavior, including its gradients.
    finite = known_finite
    if execution_fast and not finite and len(field) and field.shape[1]:
        finite = finite_cache.is_finite(field) if finite_cache is not None else bool(torch.isfinite(field).all())
    if execution_fast and len(field) and field.shape[1] and finite:
        if _plan is None or _plan.steps != field.shape[1]:
            _plan = _TimeReadPlan(xy, time_index, field.shape[1], grid)
        if _plan.xy is not xy or _plan.time_index is not time_index or _plan.grid != grid:
            raise ValueError('Time read plans are scoped to the same forward/XY/time/grid')
        return _plan.read(field, mode)
    batch, count = time_index.shape
    result = field.new_zeros(batch, count, field.shape[2])
    supported = torch.zeros(batch, count, dtype=torch.bool, device=field.device)
    for t in range(field.shape[1]):
        value, in_grid = grid.sample(field[:, t], xy, mode=mode)
        selected = time_index == t
        result = result+value*selected[..., None]
        supported |= selected & in_grid
    return result, supported


def _body_footprints(poses, time_index, bodies, grid, steps):
    """Reuse poses' geometry/coordinates only inside one loss forward."""
    footprints = []
    for row, body in enumerate(bodies):
        points = body_points(poses[row], body, execution_fast=True)
        count, coverage = points.shape[:2]
        xy = points.reshape(1, count*coverage, 2)
        time = time_index[row, :, None].expand(-1, coverage).reshape(1, -1)
        footprints.append({'points': points, 'xy': xy, 'time': time,
                           'prepared': grid.prepare_sample(xy), 'plan': _TimeReadPlan(xy, time, steps, grid)})
    return footprints


def body_relations(road, future_occupancy, poses, time_index, bodies, *, grid=None,
                   execution_fast=False, known_finite=False, finite_cache=None, _footprints=None):
    grid = grid or GridSpec()
    if execution_fast and finite_cache is None:
        finite_cache = ForwardFieldFiniteCache()
    road_margins, overlaps, supported = [], [], []
    for row, body in enumerate(bodies):
        footprint = _footprints[row] if execution_fast and _footprints is not None else None
        points = footprint['points'] if footprint is not None else body_points(poses[row], body, execution_fast=execution_fast)
        count, coverage = points.shape[:2]
        xy = footprint['xy'] if footprint is not None else points.reshape(1, count*coverage, 2)
        distance, in_road = grid.sample(road[row:row+1], xy, _prepared=footprint['prepared'] if footprint is not None else None)
        time = footprint['time'] if footprint is not None else time_index[row, :, None].expand(-1, coverage).reshape(1, -1)
        occupancy, in_occ = read_time_field(future_occupancy[row:row+1], xy, time, grid,
            execution_fast=execution_fast, known_finite=known_finite, finite_cache=finite_cache,
            _plan=footprint['plan'] if footprint is not None else None)
        road_margins.append(distance.reshape(count, coverage).amin(1))
        overlaps.append(occupancy.reshape(count, coverage).amax(1))
        supported.append((in_road & in_occ).reshape(count, coverage).all(1))
    return {'road_margin_m': torch.stack(road_margins), 'occupancy_overlap_proxy': torch.stack(overlaps),
            'in_range': torch.stack(supported)}
