"""Seven complete, current-plan-only trajectories, with smooth physical warps.

SciPy CubicHermiteSpline supplies the interpolation. We add the current rear
axle origin, use legal current speed for its derivative, and never read GT.
These are field queries, not seven planner samples or candidate selection.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import CubicHermiteSpline


@dataclass(frozen=True)
class QueryConfig:
    dt: float = .5
    lateral_m: tuple = (-.5, .5, -1., 1.)
    time_shift_s: tuple = (-.5, .5)
    stationary_speed_mps: float = .15
    dense_dt: float = .025
    # Initial engineering limits; freeze the training-only calibration before
    # formal label production. These do not remove scenes or modify q0.
    max_speed_mps: float = 40.
    max_acceleration_mps2: float = 12.
    max_yaw_rate_rps: float = 1.5

    def to_dict(self):
        return asdict(self)


def time_warp(t, horizon, amplitude):
    """phi(0)=0, phi(T)=T; phi'=1+a*pi/T*sin(2*pi*t/T)>0."""
    if abs(amplitude)*np.pi >= horizon:
        raise ValueError('Requested warp is not strictly monotone')
    t = np.asarray(t, dtype=np.float64)
    return t+amplitude*np.sin(np.pi*t/horizon)**2


def smooth_offset(u):
    return u**3*(10.-15.*u+6.*u**2)


def trajectory_queries(q0, current_velocity, config=QueryConfig()):
    q0 = np.asarray(q0, dtype=np.float64)
    velocity = np.asarray(current_velocity, dtype=np.float64)
    if q0.shape not in ((8, 3), (6, 3)) or velocity.shape != (2,) or not np.isfinite(q0).all() or not np.isfinite(velocity).all():
        raise ValueError('Finite model q0 and legal current velocity are required')
    T = len(q0)*config.dt
    ts = np.arange(len(q0)+1)*config.dt
    points = np.concatenate((np.zeros((1, 2)), q0[:, :2]))
    derivatives = np.gradient(points, config.dt, axis=0, edge_order=2)
    # Current vehicle heading is zero in ego(t0). Do not estimate initial speed
    # from the first GT future displacement or from a privileged future label.
    derivatives[0] = [np.linalg.norm(velocity), 0.]
    curve = CubicHermiteSpline(ts, points, derivatives, axis=0)
    fallback_yaw = np.unwrap(np.r_[0., q0[:, 2]])

    def base_yaw(t):
        tangent = curve(t, 1)
        moving = np.linalg.norm(tangent, axis=-1) >= config.stationary_speed_mps
        return np.where(moving, np.arctan2(tangent[..., 1], tangent[..., 0]), np.interp(t, ts, fallback_yaw))

    def lateral(t, amount):
        yaw = base_yaw(t)
        normal = np.stack((-np.sin(yaw), np.cos(yaw)), -1)
        return curve(t)+amount*smooth_offset(np.asarray(t)/T)[..., None]*normal

    functions = [lambda t: curve(t)]
    functions += [lambda t, a=a: lateral(t, a) for a in config.lateral_m]
    functions += [lambda t, a=a: curve(time_warp(t, T, a)) for a in config.time_shift_s]
    if len(functions) != 7:
        raise ValueError('Exactly one anchor, four lateral and two temporal queries required')
    dense = np.linspace(0., T, int(np.ceil(T/config.dense_dt))+1)
    result, checks = [], []
    for k, function in enumerate(functions):
        xy = function(ts[1:])
        epsilon = 1e-3
        lo, hi = np.maximum(ts[1:]-epsilon, 0.), np.minimum(ts[1:]+epsilon, T)
        v = (function(hi)-function(lo))/(hi-lo)[:, None]
        heading = np.where(np.linalg.norm(v, axis=-1) >= config.stationary_speed_mps,
                           np.arctan2(v[:, 1], v[:, 0]), np.interp(ts[1:], ts, fallback_yaw))
        poses = np.column_stack((xy, heading))
        if k == 0:
            poses = q0.copy()  # exact original query, including its own heading
        dense_xy = function(dense)
        dv = np.gradient(dense_xy, dense, axis=0, edge_order=2)
        da = np.gradient(dv, dense, axis=0, edge_order=2)
        speed = np.linalg.norm(dv, axis=-1)
        rate = np.abs(dv[:, 0]*da[:, 1]-dv[:, 1]*da[:, 0])/np.maximum(speed**2, config.stationary_speed_mps**2)
        stats = dict(speed_max_mps=float(speed.max()), acceleration_max_mps2=float(np.linalg.norm(da, axis=-1).max()),
                     yaw_rate_max_rps=float(rate.max()), initial_position_error_m=float(np.linalg.norm(dense_xy[0])))
        stats['valid'] = (stats['speed_max_mps'] <= config.max_speed_mps and
                          stats['acceleration_max_mps2'] <= config.max_acceleration_mps2 and
                          stats['yaw_rate_max_rps'] <= config.max_yaw_rate_rps)
        result.append(poses); checks.append(stats)
    return {'poses': np.asarray(result, dtype=np.float32), 'times': ts[1:].astype(np.float32),
            'kinematic_valid': np.asarray([v['valid'] for v in checks], dtype=bool),
            'kinematics': checks, 'environment_time_is_unwarped': True}
