"""Reuse the project's bounded four-control physical-time Hermite proposal.

This is a search proposal, never an on-policy Flow sample. The coordinate and
heading encoding remain native source absolute local x/y/heading.
"""
from dataclasses import asdict

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from starVLA.rl.safe_flow_improvement.candidate_search import CEMSpec, BoundedCEM, smooth_behavior_request


PARAMETER_NAMES = ('terminal_clock_rate_change', 'onset_seconds', 'lateral_offset_m', 'lateral_bending_m')


def structured_proposal(base_request, initial_velocity_xy, parameters):
    """Same mature XY curve, with a zero-control heading identity correction.

The old helper recomputes heading from the XY tangent even at zero controls.
Here the original heading-minus-base-tangent residual is transported along
the same clock warp. Thus zero controls preserve all 8x3 requested values.
Initial position/velocity are preserved by the original quintic onset; initial
acceleration is not additionally constrained and is reported explicitly.
"""
    base = np.asarray(base_request, dtype=np.float64)
    velocity0 = np.asarray(initial_velocity_xy, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    requested, diagnostic = smooth_behavior_request(base, velocity0, parameters)
    if np.array_equal(parameters, np.array([0., 1., 0., 0.])):
        requested = base.astype(np.float32).copy()
    else:
        grid = np.arange(9, dtype=np.float64)*.5
        xy = np.concatenate([np.zeros((1, 2)), base[:, :2]])
        slopes = np.gradient(xy, grid, axis=0, edge_order=2)
        slopes[0] = velocity0
        path = CubicHermiteSpline(grid, xy, slopes, extrapolate=False)
        t = grid[1:]
        rate, onset, _, _ = parameters
        u = np.clip((t-onset)/(4-onset), 0., 1.)
        warped = t+rate*(4-onset)*(u**6-3*u**5+2.5*u**4)
        bounded = np.clip(warped, 0., 4.)
        tangent = path(bounded, 1)
        tangent[warped > 4] = slopes[-1]
        angle = np.arctan2(tangent[:, 1], tangent[:, 0])
        headings = np.unwrap(np.concatenate([[0.], base[:, 2]]))
        old_heading = np.interp(bounded, grid, headings)
        correction = np.arctan2(np.sin(old_heading-angle), np.cos(old_heading-angle))
        requested[:, 2] = np.unwrap(np.concatenate([[0.], requested[:, 2]+correction]))[1:]
    if not np.isfinite(requested).all():
        raise FloatingPointError('Nonfinite proposal is not an official zero score')
    # The mature native requests are eight pose targets followed by the locked
    # tracker, not a directly executed Hermite curve. Its interpolation can
    # have high acceleration even when the actual official simulation is safe
    # and comfortable. Keep those proxy values, but do not discard finite
    # bounded requests before the required real controller evaluation.
    diagnostic['requested_curve_proxy_status'] = diagnostic['status']
    diagnostic['status'] = 'PHYSICAL_PROPOSAL_VALID'
    diagnostic['requires_official_scoring'] = True
    diagnostic.update(parameter_names=list(PARAMETER_NAMES),
                      original_heading_residual_preserved=True,
                      initial_acceleration_additionally_constrained=False,
                      requested_curve_limits_are_diagnostics_only=True,
                      dynamic_validation='actual_official_simulated_states_required',
                      external_proposal=True, on_policy_probability_available=False)
    return requested, diagnostic


class StructuredCEM(BoundedCEM):
    """Reuse the old draw/bounds; ranking is injected without reading metrics."""
    def __init__(self, rng):
        super().__init__('behavior_search', rng, CEMSpec(rounds=3))

    def update_ranked(self, entries):
        candidates = self.elites+[(np.asarray(x, dtype=np.float64).copy(), rank) for x, rank in entries]
        candidates.sort(key=lambda row: row[1])
        self.elites = candidates[:self.spec.elite_count]
        if self.elites:
            selected = np.stack([row[0] for row in self.elites])
            alpha = self.spec.smoothing_old_fraction
            self.mean = np.clip(alpha*self.mean+(1-alpha)*selected.mean(0), self.lower, self.upper)
            self.std = np.clip(alpha*self.std+(1-alpha)*selected.std(0), self.floor, self.upper-self.lower)
        self.round += 1

    def identity(self):
        return dict(spec=asdict(self.spec), kind=self.kind, parameters=list(PARAMETER_NAMES),
                    coordinate='ego_rear_axle_local_x_forward_y_left_heading_radians',
                    physical_time_seconds=[.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.])
