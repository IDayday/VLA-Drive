"""One fixed command-space road diagnostic; no policy, reward, or tracker gradient.

The canonical scorer remains authoritative. This surrogate differentiates only
raw decode -> commanded rear axle -> four corners -> a fixed drivable union.
Nearest-segment, contains and angular unwrap choices are piecewise discrete.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Point
from shapely.ops import unary_union

from starVLA.dataloader.foresight_dataset import decode_ego


class DiagnosticUndefined(ValueError):
    """Preserve the fixed scene and denominator; do not invent a direction."""


@dataclass(frozen=True)
class RoadDiagnosticSpec:
    margin_m: float = 0.1
    scale_m: float = 0.1
    raw_l2_step: float = 0.005
    sin_cos_min_radius: float = 0.25
    min_gradient_norm: float = 1e-12
    max_corner_shift_m: float = 0.15
    max_yaw_shift_rad: float = 0.025
    finite_difference_steps: tuple = (1e-4, 1e-5)
    finite_difference_relative: float = 1e-4
    finite_difference_absolute: float = 1e-6

    def validate(self):
        # This is a registered single diagnostic, not a parameter search API.
        if asdict(self) != asdict(RoadDiagnosticSpec()):
            raise ValueError('The fixed road_direction_v1 constants cannot be changed')


@dataclass(frozen=True)
class Footprint:
    half_length: float
    half_width: float
    rear_axle_to_center: float
    wheel_base: float

    @classmethod
    def from_vehicle(cls, value):
        result = cls(*(float(getattr(value, key)) for key in cls.__dataclass_fields__))
        if not all(math.isfinite(v) for v in asdict(result).values()) or min(
                result.half_length, result.half_width, result.wheel_base) <= 0:
            raise ValueError('Invalid actual vehicle parameters')
        return result


def _tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float64:
        raise ValueError(f'{name} must be CPU float64 (no implicit precision conversion)')
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f'Invalid {name} shape/finite values')


def validate_raw(raw):
    _tensor(raw, (8, 4), 'raw command')
    if bool((torch.linalg.vector_norm(raw[:, 2:], dim=-1) < .25).any()):
        raise DiagnosticUndefined('SIN_COS_RADIUS_BELOW_0_25')


def periodic_difference(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


class AnchoredDecode:
    """Explicit constant origin correction, pending independent scientific review.

    Saved original GPU physical command is the value at the original raw point.
    The local derivative is the unchanged decode_ego derivative. This is NOT a
    bitwise CPU/GPU decoder equivalence claim and never alters the saved command.
    """
    def __init__(self, original_raw, original_physical):
        validate_raw(original_raw)
        _tensor(original_physical, (8, 3), 'original physical command')
        if bool((original_physical[:, 2].abs() > math.pi + 1e-8).any()):
            raise ValueError('Original yaw must already be principal radians')
        self.raw = original_raw.detach().clone()
        self.physical = original_physical.detach().clone()
        self.reference = decode_ego(self.raw)
        self.decode_delta = self.physical - self.reference
        self.decode_delta[:, 2] = periodic_difference(self.physical[:, 2], self.reference[:, 2])

    def __call__(self, raw):
        validate_raw(raw)
        decoded = decode_ego(raw)
        xy = self.physical[:, :2] + (decoded[:, :2] - self.reference[:, :2])
        yaw_delta = periodic_difference(decoded[:, 2], self.reference[:, 2])
        # Keep the exact stored original values at zero offset; only a crossing
        # requires periodic representation. The wrap is not sin/cos normalization.
        yaw = self.physical[:, 2] + yaw_delta
        wrapped = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
        yaw = torch.where((yaw < -math.pi) | (yaw > math.pi), wrapped, yaw)
        return torch.cat((xy, yaw[:, None]), -1)

    def audit(self):
        return {'mode': 'stored_gpu_origin_plus_cpu_float64_decode_delta',
                'cpu_decode_max_xy_difference_m': float(self.decode_delta[:, :2].abs().max()),
                'cpu_decode_max_yaw_difference_rad': float(self.decode_delta[:, 2].abs().max()),
                'original_raw_float64': self.raw.tolist(),
                'original_physical_as_saved': self.physical.tolist(),
                'unanchored_cpu_float64_decode': self.reference.tolist(),
                'constant_origin_correction': self.decode_delta.tolist(),
                'raw_decoder': 'starVLA.dataloader.foresight_dataset.decode_ego',
                'bitwise_cpu_gpu_equivalence_claimed': False}


def command_poses(physical, initial_pose, initial_time_us):
    """Differentiable counterpart of official transform/interpolation, 41 poses.

    Timestamp truncation and angular unwrap use precisely the original numpy
    convention. The constant integer/unwrap choices are not differentiated.
    """
    _tensor(physical, (8, 3), 'physical command')
    _tensor(initial_pose, (3,), 'initial rear axle pose')
    if type(initial_time_us) is not int or initial_time_us < 0:
        raise ValueError('Expected the actual nonnegative integer initial timestamp')
    if bool((physical[:, 2].abs() > math.pi + 1e-8).any()):
        raise ValueError('Command yaw must already be principal')
    c, s = torch.cos(initial_pose[2]), torch.sin(initial_pose[2])
    x, y, h = physical.unbind(-1)
    xy = torch.stack((c*x - s*y + initial_pose[0], s*x + c*y + initial_pose[1]), -1)
    # Match relative_to_absolute_poses matrix product, including periodic yaw.
    ch, sh = torch.cos(h), torch.sin(h)
    yaw = torch.atan2(s*ch + c*sh, c*ch - s*sh)
    knots = torch.cat((initial_pose[None], torch.cat((xy, yaw[:, None]), -1)), 0)
    angular = knots[:, 2].detach().numpy()
    unwrap_offset = np.unwrap(angular) - angular
    unwrapped = knots[:, 2] + torch.from_numpy(unwrap_offset)
    start_s = initial_time_us / 1e6
    source_us = np.array([initial_time_us] + [int(t*1e6) for t in
        (np.arange(0., 4., .5) + .5 + start_s)], dtype=np.int64)
    target_us = np.clip(np.array([int(t*1e6) for t in
        np.arange(0., 4.+.1, .1) + start_s], dtype=np.int64), source_us[0], source_us[-1])
    indices = np.clip(np.searchsorted(source_us, target_us, side='right') - 1, 0, 7)
    weights = torch.from_numpy((target_us - source_us[indices]) /
                              (source_us[indices+1] - source_us[indices]))
    positions = knots[indices, :2] + weights[:, None]*(knots[indices+1, :2]-knots[indices, :2])
    angles = unwrapped[indices] + weights*(unwrapped[indices+1]-unwrapped[indices])
    angles = torch.remainder(angles + math.pi, 2*math.pi) - math.pi
    poses = torch.cat((positions, angles[:, None]), -1)
    return poses, np.rint(unwrap_offset/(2*math.pi)).astype(np.int64)


def footprint_corners(poses, vehicle):
    _tensor(poses, (41, 3), '41 rear axle poses')
    c, s = torch.cos(poses[:, 2]), torch.sin(poses[:, 2])
    center = poses[:, :2] + vehicle.rear_axle_to_center*torch.stack((c, s), -1)
    # Canonical BBCoordsIndex: front-left, rear-left, rear-right, front-right.
    offsets = poses.new_tensor([[vehicle.half_length, vehicle.half_width],
                               [-vehicle.half_length, vehicle.half_width],
                               [-vehicle.half_length, -vehicle.half_width],
                               [vehicle.half_length, -vehicle.half_width]])
    return center[:, None] + torch.stack((c[:, None]*offsets[:, 0]-s[:, None]*offsets[:, 1],
                                          s[:, None]*offsets[:, 0]+c[:, None]*offsets[:, 1]), -1)


class RoadUnion:
    """Exact fixed polygon union; holes retained; no buffer/repair/simplification."""
    def __init__(self, polygons):
        self.polygons = tuple(polygons)
        if not self.polygons or any(p.is_empty or not p.is_valid or
                p.geom_type not in ('Polygon', 'MultiPolygon') for p in self.polygons):
            raise ValueError('Empty/invalid/non-polygon drivable geometry')
        self.geometry = unary_union(self.polygons)
        if self.geometry.is_empty or not self.geometry.is_valid or self.geometry.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Invalid polygon union; never repair geometry')
        shapes = [self.geometry] if self.geometry.geom_type == 'Polygon' else list(self.geometry.geoms)
        segments = []
        for shape in shapes:
            for ring in [shape.exterior, *shape.interiors]:
                coords = np.asarray(ring.coords, dtype=np.float64)
                if coords.shape[1] != 2 or not np.isfinite(coords).all():
                    raise ValueError('Geometry must have finite XY coordinates')
                edges = np.stack((coords[:-1], coords[1:]), axis=1)
                if np.any(np.linalg.norm(edges[:, 1]-edges[:, 0], axis=1) == 0):
                    raise ValueError('Degenerate ring segment')
                segments.extend(edges)
        if len(segments) > 200000:
            raise ValueError('Geometry exceeds bounded segment capacity; not silently subsampled')
        self.segments = torch.tensor(np.asarray(segments), dtype=torch.float64)

    @classmethod
    def from_map(cls, area):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        indices = area.get_indices_of_map_type([L.ROADBLOCK, L.INTERSECTION, L.DRIVABLE_AREA, L.CARPARK_AREA])
        return cls([area._geometries[i] for i in indices])

    def distances(self, corners):
        _tensor(corners, (41, 4, 2), 'four corners')
        points = corners.reshape(-1, 2)
        # Bounded detached search, then differentiate the chosen exact segment.
        detached = points.detach()
        best = torch.full((len(points),), torch.inf, dtype=torch.float64)
        second = best.clone()
        chosen = torch.zeros(len(points), dtype=torch.long)
        for start in range(0, len(self.segments), 1024):
            seg = self.segments[start:start+1024]
            a, v = seg[:, 0], seg[:, 1]-seg[:, 0]
            t = ((detached[:, None]-a)*v).sum(-1)/v.square().sum(-1)
            dist2 = (detached[:, None]-a-t.clamp(0, 1)[..., None]*v).square().sum(-1)
            vals, ids = torch.topk(dist2, min(2, len(seg)), largest=False)
            update = vals[:, 0] < best
            chosen = torch.where(update, ids[:, 0]+start, chosen)
            candidates = torch.stack((best, second, vals[:, 0], vals[:, -1]), -1)
            # A one-segment chunk must not count the same segment twice.
            if len(seg) == 1: candidates[:, -1] = torch.inf
            best, second = torch.topk(candidates, 2, largest=False).values.unbind(-1)
        selected = self.segments[chosen]
        a, v = selected[:, 0], selected[:, 1]-selected[:, 0]
        t = ((points-a)*v).sum(-1)/v.square().sum(-1)
        distance = torch.linalg.vector_norm(points-a-t.clamp(0, 1)[:, None]*v, dim=-1)
        np_points = detached.numpy()
        inside = np.array([self.geometry.contains(Point(p)) for p in np_points])
        any_contains = np.array([any(g.contains(Point(p)) for g in self.polygons) for p in np_points])
        signs = torch.from_numpy(np.where(inside, 1., -1.))
        signed = signs*distance
        td = t.detach().numpy()
        regime = np.where(td <= 0, -1, np.where(td >= 1, 1, 0))
        return signed.reshape(41, 4), {
            'nearest_segment': chosen.numpy(), 'inside': inside, 'projection_regime': regime,
            'nearest_tie': ((second-best).abs() <= 1e-12).numpy(),
            'boundary': (distance.detach() <= 1e-12).numpy(),
            'union_vs_any_polygon_contains': inside != any_contains}

    def cost(self, corners):
        distances, regime = self.distances(corners)
        # Fixed 164 denominator including t0. No safety/margin-dependent masking.
        cost = (.1*F.softplus((.1-distances)/.1)).mean()
        return cost, distances, regime


def regime_equal(left, right):
    return all(np.array_equal(left[key], right[key]) for key in
               ('nearest_segment', 'inside', 'projection_regime', 'unwrap_turns'))


class CommandRoadCost:
    def __init__(self, decoder, road, vehicle, initial_pose, initial_time_us):
        self.decoder, self.road, self.vehicle = decoder, road, vehicle
        self.initial_pose, self.initial_time_us = initial_pose, initial_time_us

    def __call__(self, raw):
        physical = self.decoder(raw)
        poses, turns = command_poses(physical, self.initial_pose, self.initial_time_us)
        corners = footprint_corners(poses, self.vehicle)
        cost, distance, regime = self.road.cost(corners)
        regime['unwrap_turns'] = turns
        return cost, {'physical': physical, 'poses': poses, 'corners': corners,
                      'signed_distances': distance, 'regime': regime}


def finite_difference_check(objective, raw, gradient):
    checks = []
    _, base = objective(raw)
    for step in (1e-4, 1e-5):
        fd = torch.zeros_like(raw)
        stable = True
        for i in range(raw.numel()):
            delta = torch.zeros_like(raw); delta.reshape(-1)[i] = step
            plus, p = objective(raw+delta); minus, m = objective(raw-delta)
            stable &= regime_equal(base['regime'], p['regime']) and regime_equal(base['regime'], m['regime'])
            fd.reshape(-1)[i] = (plus-minus)/(2*step)
        absolute = float((gradient-fd).abs().max())
        relative = float(torch.linalg.vector_norm(gradient-fd)/max(
            float(torch.linalg.vector_norm(gradient)), float(torch.linalg.vector_norm(fd)), 1e-12))
        checks.append({'step': step, 'stable_discrete_regime': bool(stable),
                       'absolute_max': absolute, 'relative_l2': relative,
                       'passed': bool(stable and (relative <= 1e-4 or absolute <= 1e-6))})
    return checks


def direction_variants(objective, raw, scene_id, selection_hash):
    validate_raw(raw)
    original = raw.detach().clone().requires_grad_(True)
    loss, aux = objective(original)
    gradient, = torch.autograd.grad(loss, original)
    norm = float(torch.linalg.vector_norm(gradient))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise DiagnosticUndefined('ZERO_OR_NONFINITE_GRADIENT')
    # t0 has no action dependence. Record its ties, but do not reject a derivative
    # of a constant for that reason; moving corners have no smoothness guarantee.
    regime = aux['regime']
    if (regime['boundary'][4:] | regime['nearest_tie'][4:]).any():
        raise DiagnosticUndefined('MOVING_CORNER_BOUNDARY_OR_NEAREST_TIE')
    with torch.no_grad():
        checks = finite_difference_check(objective, raw, gradient)
    if not all(c['passed'] for c in checks):
        raise DiagnosticUndefined('FINITE_DIFFERENCE_OR_REGIME_FAILURE: '+repr(checks))
    seed = int.from_bytes(hashlib.sha256((scene_id+'\0'+selection_hash+'\0road_direction_v1').encode()).digest()[:8], 'big')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    random = torch.randn((8, 4), generator=rng, dtype=torch.float64)
    random = random/torch.linalg.vector_norm(random)
    unit = gradient.detach()/norm
    variants = {'original': raw.detach().clone(), 'negative': raw.detach()-.005*unit,
                'positive': raw.detach()+.005*unit, 'random': raw.detach()+.005*random}
    rows = {}
    with torch.no_grad():
        for name, variant in variants.items():
            cost, details = objective(variant)
            shift = float(torch.linalg.vector_norm(details['corners']-aux['corners'], dim=-1).max())
            yaw_shift = float(periodic_difference(details['poses'][:, 2], aux['poses'][:, 2]).abs().max())
            if shift > .15 or yaw_shift > .025:
                raise DiagnosticUndefined('GEOMETRIC_PERTURBATION_CAP: '+name)
            if not torch.equal(details['poses'][0], aux['poses'][0]):
                raise DiagnosticUndefined('INITIAL_STATE_CHANGED')
            rows[name] = {'raw': variant, 'physical': details['physical'],
                'command_poses': details['poses'], 'command_cost_m': float(cost),
                'min_signed_corner_distance_m': float(details['signed_distances'].min()),
                'outside_corner_times': int((details['signed_distances'] <= 0).sum()),
                'raw_l2_offset': float(torch.linalg.vector_norm(variant-raw)),
                'max_corner_shift_m': shift, 'max_yaw_shift_rad': yaw_shift,
                'union_vs_any_polygon_contains_count': int(details['regime']['union_vs_any_polygon_contains'].sum())}
    return {'gradient': gradient.detach(), 'gradient_norm': norm, 'random_seed': seed,
            'finite_difference': checks, 'variants': rows, 'decoder': objective.decoder.audit()}


def executed_geometry(road, vehicle, states):
    """The SAME custom cost after real simulation; no executed-state derivative."""
    value = np.asarray(states, dtype=np.float64)
    if value.shape != (41, 11) or not np.isfinite(value).all():
        raise ValueError('Invalid canonical executed state trace')
    with torch.no_grad():
        corners = footprint_corners(torch.from_numpy(value[:, :3].copy()), vehicle)
        cost, distances, regime = road.cost(corners)
    return {'custom_executed_cost_m':float(cost),
            'minimum_signed_corner_distance_m':float(distances.min()),
            'outside_corner_times':int((distances <= 0).sum()),
            'union_vs_any_polygon_contains_count':int(regime['union_vs_any_polygon_contains'].sum())}


def score_diagnostic_variants(engine, original_request, original_record, directions,
                              road, vehicle, generator_hash, admit_call, save_result):
    """Use the canonical backend only inside a separately admitted caller.

    `admit_call` must durably charge EACH invocation before it happens and enforce
    the shared <=128 hard cap. This function neither creates an admission nor
    retries. Caller owns process/IO/deadline limits and fixed32 population.
    `save_result` must retain failures and each exact request/record. Originals
    are reused only after full canonical identity validation; never re-scored
    under a silently changed float64 CPU original decode.
    """
    from dataclasses import replace
    from starVLA.model.modules.st_world.consequence import (
        array_sha256, validated_reference_input)
    if not engine.spec.allow_engineering_controls:
        raise ValueError('Canonical backend has not admitted engineering controls')
    if not callable(admit_call) or not callable(save_result):
        raise ValueError('Explicit budget accounting and result persistence required')
    if original_record.get('identity') != original_request.identity() or not original_record.get('valid'):
        raise ValueError('Original candidate/context identity mismatch')
    original_context = validated_reference_input(original_record, original_request)
    if array_sha256(original_record['requested_trajectory']) != original_request.trajectory_sha256:
        raise ValueError('Original physical payload mismatch')
    rows = {}
    for name in ('original','negative','positive','random'):
        values = directions['variants'][name]
        physical = values['physical'].detach().cpu().numpy().astype(np.float64, copy=True)
        if name == 'original':
            if array_sha256(physical) != original_request.trajectory_sha256:
                raise ValueError('Anchored original is not exactly the stored request')
            request, record = original_request, original_record
        else:
            request = replace(original_request,
                candidate_id=original_request.candidate_id+':road_direction_v1:'+name,
                generator_kind='engineering_control', generator_checkpoint_sha256=None,
                generator_identity=None, generator_source_hash=generator_hash,
                trajectory=physical, trajectory_sha256=array_sha256(physical))
            admit_call(request.identity())
            record = engine.score(request, expected_policy_version=original_request.policy_version)
        result = {'variant':name,'request':request.identity(),'record':record,
                  'original_reused':name=='original','geometry':None}
        if record.get('valid'):
            if validated_reference_input(record, request) != original_context:
                raise ValueError('Reference/context changed between diagnostic variants')
            result['geometry'] = executed_geometry(road, vehicle, record['executed_states'])
        save_result(result)
        rows[name] = result
    return rows
