"""Bounded CEM search primitives; no training gradients or actor reward inputs.

Elite retention and smoothed mean/std updates follow martius-lab/iCEM
98c1c1fe2bfc94a87e4658b6c793f4e81bc30203, icem/controllers/icem.py.
This is a contextual trajectory search, not a port of its MPC environment.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import chi2

from .contracts import SAFETY_METRICS


def safety_order(metrics):
    """Predeclared search order, leaving the true reward completely unchanged."""
    cost = np.array([1. - metrics[k] for k in SAFETY_METRICS])
    if not np.isfinite(cost).all() or not np.isfinite(metrics['score']):
        raise ValueError('Invalid official score is an infrastructure failure')
    return (int(np.any(cost > 1e-12)), float(cost.max()), float(cost.sum()), -float(metrics['score']))


@dataclass(frozen=True)
class CEMSpec:
    rounds: int = 4
    elite_count: int = 4
    smoothing_old_fraction: float = .25
    latent_dimensions: int = 32
    latent_mean_norm_max: float = 2.
    latent_std_min: float = .5
    latent_std_max: float = 1.
    latent_coordinate_abs_max: float = 3.5
    latent_prior_radius_quantile: float = .995
    behavior_lower: tuple = (-.5, .25, -.6, -.35)
    behavior_upper: tuple = (.15, 2.5, .6, .35)
    behavior_std_fraction: float = .25
    behavior_std_floor_fraction: float = .05


class BoundedCEM:
    def __init__(self, kind, rng, spec=CEMSpec()):
        if kind not in ('latent_search', 'behavior_search'):
            raise ValueError('Unknown bounded search')
        self.kind, self.rng, self.spec = kind, rng, spec
        self.elites = []
        if kind == 'latent_search':
            self.mean = np.zeros(spec.latent_dimensions)
            self.std = np.ones(spec.latent_dimensions)
            self.lower = np.full(spec.latent_dimensions, -spec.latent_coordinate_abs_max)
            self.upper = -self.lower
            self.floor = np.full(spec.latent_dimensions, spec.latent_std_min)
        else:
            self.lower, self.upper = np.array(spec.behavior_lower), np.array(spec.behavior_upper)
            self.mean = np.array([0., 1., 0., 0.])
            self.std = (self.upper-self.lower) * spec.behavior_std_fraction
            self.floor = (self.upper-self.lower) * spec.behavior_std_floor_fraction
        self.round = 0

    def draw(self, count):
        if self.round >= self.spec.rounds or count < 1:
            raise ValueError('CEM exceeded its fixed budget')
        raw = self.mean + self.rng.standard_normal((count, len(self.mean))) * self.std
        values = np.clip(raw, self.lower, self.upper)
        clipped = np.any(raw != values, axis=1)
        if self.kind == 'latent_search':
            radius = float(np.sqrt(chi2.ppf(self.spec.latent_prior_radius_quantile, self.spec.latent_dimensions)))
            norms = np.linalg.norm(values, axis=1)
            scale = np.minimum(1., radius / np.maximum(norms, 1e-12))
            values *= scale[:, None]
            clipped |= scale < 1
            prior = [{'standard_gaussian_squared_norm': float(x @ x),
                      'chi2_prior_radius_cdf': float(chi2.cdf(x @ x, len(x))),
                      'log_standard_gaussian_density': float(-.5 * (x @ x + len(x)*np.log(2*np.pi))),
                      'was_bounded': bool(c)} for x, c in zip(values, clipped)]
        else:
            prior = [{'was_bounded': bool(c)} for c in clipped]
        return values, {'round': self.round, 'mean': self.mean.tolist(), 'std': self.std.tolist(),
                        'new_draws': count, 'retained_elites_not_rescored': len(self.elites),
                        'proposal_is_bounded_not_original_gaussian': True, 'draws': prior}

    def update(self, values, metrics):
        if len(values) != len(metrics):
            raise ValueError('Search parameters and scores misaligned')
        candidates = self.elites + [(np.array(x, copy=True), safety_order(m)) for x, m in zip(values, metrics) if m is not None]
        # Elites are reused only in this selection step, never appended to the
        # bank or billed as new scores/discoveries.
        candidates.sort(key=lambda row: row[1])
        self.elites = candidates[:self.spec.elite_count]
        if self.elites:
            selected = np.stack([row[0] for row in self.elites])
            alpha = self.spec.smoothing_old_fraction
            self.mean = alpha*self.mean + (1-alpha)*selected.mean(0)
            self.std = np.maximum(self.floor, alpha*self.std+(1-alpha)*selected.std(0))
            self.mean = np.clip(self.mean, self.lower, self.upper)
            if self.kind == 'latent_search':
                norm = np.linalg.norm(self.mean)
                self.mean *= min(1., self.spec.latent_mean_norm_max/max(norm, 1e-12))
                self.std = np.minimum(self.std, self.spec.latent_std_max)
            else:
                self.std = np.minimum(self.std, self.upper-self.lower)
        self.round += 1


def smooth_behavior_request(base_request, initial_velocity_xy, parameters):
    """Four continuous controls on one original ODE path.

    Parameters: terminal clock-rate change, onset seconds, lateral offset m,
    lateral bending amplitude m. Integrating a quintic smoothstep changes
    arrival time without changing t=0 position/heading/velocity. XY is one
    continuous Hermite path, not eight independent point displacements.
    """
    base = np.asarray(base_request, dtype=np.float64)
    velocity0 = np.asarray(initial_velocity_xy, dtype=np.float64)
    pars = np.asarray(parameters, dtype=np.float64)
    if base.shape != (8, 3) or velocity0.shape != (2,) or pars.shape != (4,) or not np.isfinite(np.concatenate([base.ravel(), velocity0, pars])).all():
        raise ValueError('Invalid behavior-search inputs')
    rate, onset, lateral, bending = pars
    spec = CEMSpec()
    if np.any(pars < spec.behavior_lower) or np.any(pars > spec.behavior_upper):
        raise ValueError('Behavior parameters outside the registered bounds')
    grid = np.arange(9)*.5
    xy = np.concatenate([np.zeros((1, 2)), base[:, :2]])
    slopes = np.gradient(xy, grid, axis=0, edge_order=2)
    slopes[0] = velocity0
    path = CubicHermiteSpline(grid, xy, slopes, extrapolate=False)

    def curve(t):
        t = np.asarray(t)
        u = np.clip((t-onset)/(4-onset), 0., 1.)
        # Integral of 6u^5-15u^4+10u^3.
        warped = t + rate*(4-onset)*(u**6 - 3*u**5 + 2.5*u**4)
        clipped = np.clip(warped, 0., 4.)
        position = path(clipped) + np.maximum(warped-4., 0.)[:, None]*slopes[-1]
        tangent = path(clipped, 1)
        tangent[warped > 4.] = slopes[-1]
        speed = np.linalg.norm(tangent, axis=1)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1) / np.maximum(speed[:, None], .05)
        q = t/4.
        smooth = 10*q**3 - 15*q**4 + 6*q**5
        offset = smooth*(lateral+bending*np.sin(np.pi*q))
        return position + normal*offset[:, None]

    # A dense physical-time check records invalid proposals before expensive
    # simulation. Invalid draws consume search budget and are never reward=0.
    t = np.linspace(0., 4., 161)
    positions = curve(t)
    velocities = np.gradient(positions, t, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, t, axis=0, edge_order=2)
    speed = np.linalg.norm(velocities, axis=-1)
    cross = velocities[:, 0]*accelerations[:, 1]-velocities[:, 1]*accelerations[:, 0]
    curvature = np.abs(cross)/np.maximum(speed, .5)**3
    yaw_rate = np.abs(cross)/np.maximum(speed, .5)**2
    limits = {'maximum_speed_mps': max(30., float(np.linalg.norm(velocity0))+8.),
              'maximum_acceleration_mps2': 8., 'maximum_curvature_per_m': .3,
              'maximum_yaw_rate_rps': .8, 'maximum_lateral_acceleration_mps2': 5.}
    measurements = {'maximum_speed_mps': float(speed.max()),
                    'maximum_acceleration_mps2': float(np.linalg.norm(accelerations, axis=-1).max()),
                    'maximum_curvature_per_m': float(curvature.max()),
                    'maximum_yaw_rate_rps': float(yaw_rate.max()),
                    'maximum_lateral_acceleration_mps2': float((np.abs(cross)/np.maximum(speed, .5)).max())}
    valid = all(measurements[k] <= bound for k, bound in limits.items())
    requested_t = np.arange(1, 9)*.5
    requested_xy = curve(requested_t)
    eps = 1e-4
    derivative = (curve(requested_t+eps)-curve(requested_t-eps))/(2*eps)
    heading = np.arctan2(derivative[:, 1], derivative[:, 0])
    stopped = np.linalg.norm(derivative, axis=-1) < .05
    heading[stopped] = base[stopped, 2]
    # At t=0 the original ego heading is zero. The continuous interpolated
    # heading is defined by a smooth connection to the first requested yaw;
    # the official tracker receives only the original eight pose interface.
    request = np.concatenate([requested_xy, heading[:, None]], axis=1).astype(np.float32)
    return request, {'status': 'PHYSICAL_PROPOSAL_VALID' if valid else 'INVALID_PHYSICAL_PROPOSAL',
                     'limits': limits, 'measurements': measurements,
                     'initial_position_xy': [0., 0.], 'initial_heading': 0.,
                     'initial_velocity_xy': velocity0.tolist(), 'parameters': pars.tolist(),
                     'time_axis': 'physical_driving_seconds', 'requires_official_scoring': valid}
