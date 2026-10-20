"""Low-dimensional interventions in the bound rear-axle trajectory frame.

These are proposed controlled interventions, not a causal-gradient oracle.
Quintic smoothstep has zero first/second derivatives at each support boundary.
Physical executability is decided separately from the real controller replay.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def smoothstep(value):
    x = np.clip(np.asarray(value, dtype=np.float64), 0., 1.)
    return x ** 3 * (10. - 15. * x + 6. * x ** 2)


@dataclass(frozen=True)
class BasisSpec:
    horizon: int = 8
    dt: float = .5
    windows: tuple = ((0., 2.), (1., 3.), (2., 4.), (0., 4.))
    schema: str = "action_effect_quintic_intervention_v1"

    def validate(self):
        if self.horizon < 3 or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid bound physical time contract")
        if not self.windows or any(not (0 <= a < b <= self.horizon*self.dt)
                                   for a, b in self.windows):
            raise ValueError("Intervention windows must lie inside the bound horizon")
        return self

    @property
    def size(self):
        return 2 * len(self.windows)

    @property
    def sha256(self):
        self.validate()
        return identity(asdict(self))


def validate_trajectory(poses, spec):
    spec.validate()
    x = np.asarray(poses, dtype=np.float64)
    if x.shape != (spec.horizon, 3) or not np.isfinite(x).all():
        raise ValueError("Expected finite physical [horizon,XY,yaw] trajectory")
    return x


def basis_matrix(base, spec):
    """[T,XY,K]; coefficient units are metres, axes follow the base heading.

    Each intervention ramps smoothly inside its window then holds its offset.
    Early/later windows therefore test different intervention onset times.
    This does not label the onset as responsibility for a subsequent event.
    """
    a = validate_trajectory(base, spec)
    times = np.arange(1, spec.horizon+1, dtype=np.float64) * spec.dt
    along = np.stack((np.cos(a[:, 2]), np.sin(a[:, 2])), -1)
    lateral = np.stack((-along[:, 1], along[:, 0]), -1)
    columns = []
    for direction in (along, lateral):
        for begin, end in spec.windows:
            weight = smoothstep((times-begin)/(end-begin))
            columns.append(direction * weight[:, None])
    return np.stack(columns, -1)


def reconstruct(base, coefficients, spec):
    """Apply one intervention. Exact zero returns original poses including yaw.

    The heading adjustment is the change in sampled tangent, added to the
    original heading. This preserves any existing base heading/tangent relation.
    A low-speed undefined tangent retains the original heading and is disclosed.
    """
    a = validate_trajectory(base, spec)
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (spec.size,) or not np.isfinite(c).all():
        raise ValueError("Invalid intervention coefficients")
    if not np.any(c):
        return a.copy()
    xy = a[:, :2] + np.einsum("tdk,k->td", basis_matrix(a, spec), c)
    old_d = np.diff(np.concatenate((np.zeros((1, 2)), a[:, :2])), axis=0)
    new_d = np.diff(np.concatenate((np.zeros((1, 2)), xy)), axis=0)
    defined = (np.linalg.norm(old_d, axis=-1) > 1e-6) & (np.linalg.norm(new_d, axis=-1) > 1e-6)
    delta_yaw = wrap_angle(np.arctan2(new_d[:, 1], new_d[:, 0])-
                           np.arctan2(old_d[:, 1], old_d[:, 0]))
    yaw = wrap_angle(a[:, 2] + np.where(defined, delta_yaw, 0.))
    return np.column_stack((xy, yaw))


def intervention_grid(spec, *, seed=20261020):
    """Fixed ordered24 controlled proposals; no scoring-based parameter search.

    zero; early/late longitudinal and lateral small/medium controls; joint
    controls; larger-support controls. Additional policy samples are separate.
    Duplicate numerical endpoints are removed only after actual reconstruction.
    """
    spec.validate()
    if len(spec.windows) != 4:
        raise ValueError("This registered grid requires its explicit four-window schema")
    rows = [{"kind": "identity", "coefficients": [0.]*spec.size}]
    for axis, radii in ((0, (.5, 1.)), (1, (.15, .30))):
        for window in (0, 2):
            for radius in radii:
                for sign in (-1., 1.):
                    c = np.zeros(spec.size); c[axis*4+window] = sign*radius
                    rows.append({"kind": "longitudinal" if axis == 0 else "lateral",
                                 "coefficients": c.tolist()})
    for longitudinal, lateral in ((-.5, -.15), (-.5, .15), (.5, -.15), (.5, .15)):
        c = np.zeros(spec.size);c[0] = longitudinal;c[4] = lateral
        rows.append({"kind": "joint", "coefficients": c.tolist()})
    for axis, radius in ((0, -2.), (0, 2.), (1, .6)):
        c = np.zeros(spec.size);c[axis*4+3] = radius
        rows.append({"kind": "larger_support", "coefficients": c.tolist()})
    assert len(rows) == 24
    for i, row in enumerate(rows):
        row.update(intervention_id=f"intervention_{i:02d}", basis_sha256=spec.sha256,
                   seed=int(seed), ramp_intervals_seconds=[list(spec.windows[k % 4])
                   for k, value in enumerate(row["coefficients"]) if value != 0],
                   active_intervals_seconds=[[spec.windows[k % 4][0], spec.horizon*spec.dt]
                   for k, value in enumerate(row["coefficients"]) if value != 0])
    return rows
