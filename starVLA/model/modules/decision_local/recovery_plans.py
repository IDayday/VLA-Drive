"""Actual repair inputs, distinct from the seven field queries around each input."""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from .trajectory_queries import smooth_offset, QueryConfig, trajectory_queries, initial_derivative


@dataclass(frozen=True)
class RecoveryConfig:
    dt: float = .5
    lateral_amplitudes_m: tuple = (.25, .75, 1.25)
    lateral_size_probabilities: tuple = (.5, .4, .1)
    time_amplitude_s: float = .5
    selection_seed: int = 17015
    version: str = 'actual_plan_pool_v1'
    velocity_boundary: str = 'body_aligned_speed_v1'

    def to_dict(self):
        values=asdict(self)
        if self.velocity_boundary=='body_aligned_speed_v1':values.pop('velocity_boundary')
        return values


class PlanCurve:
    """Local four-coefficient family, with current pose and first derivative fixed.

    Two smooth normal offsets, an endpoint-preserving time warp, and a limited
    endpoint delay. No environment or logged future ego is accepted here.
    """
    def __init__(self, plan, velocity, dt=.5, *, velocity_boundary='body_aligned_speed_v1'):
        self.plan = np.asarray(plan, np.float64).copy()
        self.velocity = np.asarray(velocity, np.float64)
        self.velocity_boundary=velocity_boundary
        if self.plan.shape not in ((8, 3), (6, 3)) or self.velocity.shape != (2,) or not np.isfinite(self.plan).all():
            raise ValueError('Finite actual plan and legal current velocity required')
        self.times = np.arange(len(plan)+1)*dt
        self.horizon = self.times[-1]
        xy = np.vstack((np.zeros(2), self.plan[:, :2]))
        derivative = np.gradient(xy, dt, axis=0, edge_order=2)
        derivative[0] = initial_derivative(self.velocity,velocity_boundary)
        self.spline = CubicHermiteSpline(self.times, xy, derivative, axis=0)
        self.yaws = np.unwrap(np.r_[0., self.plan[:, 2]])

    def warp(self, times, coefficients):
        _, _, middle, delay = np.asarray(coefficients, np.float64)
        u = np.asarray(times)/self.horizon
        phi = np.asarray(times)+middle*np.sin(np.pi*u)**2-delay*u**3
        derivative = 1+middle*np.pi/self.horizon*np.sin(2*np.pi*u)-3*delay/self.horizon*u*u
        return phi, derivative

    def xy(self, times, coefficients):
        end, middle, _, _ = np.asarray(coefficients, np.float64)
        ts = np.asarray(times, np.float64)
        phi, derivative = self.warp(ts, coefficients)
        if np.any(phi < -1e-8) or np.any(phi > self.horizon+1e-8) or np.any(derivative <= 0):
            raise ValueError('Nonmonotone/out-of-domain time map; no extrapolation')
        v = self.spline(phi, 1)
        speed = np.linalg.norm(v, axis=-1)
        yaw = np.where(speed >= .15, np.arctan2(v[..., 1], v[..., 0]), np.interp(phi, self.times, self.yaws))
        u = ts/self.horizon
        offset = end*smooth_offset(u)+middle*64*u**3*(1-u)**3
        return self.spline(phi)+offset[..., None]*np.stack((-np.sin(yaw), np.cos(yaw)), -1)

    def poses(self, times, coefficients=None):
        t = np.asarray(times, np.float64)
        c = np.zeros(4) if coefficients is None else np.asarray(coefficients, np.float64)
        xy = self.xy(t, c)
        phi,_=self.warp(t,c)
        lo, hi = np.maximum(t-1e-4, 0), np.minimum(t+1e-4, self.horizon)
        v = (self.xy(hi, c)-self.xy(lo, c))/(hi-lo)[..., None]
        base_v=self.spline(phi,1)
        delta=np.arctan2(v[...,1],v[...,0])-np.arctan2(base_v[...,1],base_v[...,0])
        delta=np.arctan2(np.sin(delta),np.cos(delta))
        moving=(np.linalg.norm(v,axis=-1)>=.15)&(np.linalg.norm(base_v,axis=-1)>=.15)
        yaw=np.interp(phi,self.times,self.yaws)+np.where(moving,delta,0.)
        if not np.any(c):yaw=np.interp(t,self.times,self.yaws)
        if self.velocity_boundary=='measured_vector_v2':yaw=np.where(t==0.,0.,yaw)
        return np.column_stack((xy, yaw))


def stable_number(*values):
    return int(hashlib.sha256(':'.join(map(str, values)).encode()).hexdigest()[:16], 16)


def build_plan_pool(q0, velocity, token, config=RecoveryConfig()):
    # Amplitude and direction never inspect a label, GT, score, or valid mask.
    u = stable_number(token, config.selection_seed, 'size')/2**64
    size = min(int(np.searchsorted(np.cumsum(config.lateral_size_probabilities), u)), 2)
    amplitude = config.lateral_amplitudes_m[size]
    curve = PlanCurve(q0, velocity, config.dt,velocity_boundary=config.velocity_boundary)
    coefficients = np.array([[0,0,0,0],[-amplitude,0,0,0],[amplitude,0,0,0],
                             [0,0,-config.time_amplitude_s,0],[0,0,config.time_amplitude_s,0]], np.float64)
    plans = np.stack([curve.poses(curve.times[1:], c) for c in coefficients]).astype(np.float32)
    plans[0] = np.asarray(q0, np.float32)  # no resmoothing of native plan
    return {'plans': plans, 'coefficients': coefficients.astype(np.float32), 'size': size,
            'times': curve.times[1:].astype(np.float32), 'environment_times_warped': False}


def select_plan(token, epoch, nanchors=2, config=RecoveryConfig()):
    # Exactly one original scene per epoch. Every eight epochs: 4 native,
    # one of each signed lateral and temporal plan. Offset is arm independent.
    phase = stable_number(token, config.selection_seed, 'plan') % 8
    plan = (0, 0, 0, 0, 1, 2, 3, 4)[(phase+epoch) % 8]
    anchor = (stable_number(token, 'anchor')+epoch+epoch//8) % nanchors
    return int(anchor), int(plan)
