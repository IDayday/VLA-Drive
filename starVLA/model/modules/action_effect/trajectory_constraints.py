"""Declared geometric screens and independent real controller tracking checks.

    The screens are research bounds, not a general driving safety certificate.
    Collision/road/TTC failures remain valid negative effect-learning examples.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .intervention_basis import validate_trajectory, wrap_angle


@dataclass(frozen=True)
class ConstraintSpec:
    max_speed_mps: float = 45.
    max_acceleration_mps2: float = 10.
    max_jerk_mps3: float = 50.
    max_yaw_rate_rps: float = 1.5
    max_curvature_inv_m: float = .5
    max_lateral_acceleration_mps2: float = 10.
    max_tracking_xy_m: float = 2.
    max_tracking_yaw_rad: float = .5

    def validate(self):
        if any(not np.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("Constraints require explicit finite positive bounds")
        return self


def geometric_screen(poses, basis_spec, limits, *, initial_velocity_xy=None):
    a = validate_trajectory(poses, basis_spec);limits.validate();dt=basis_spec.dt
    velocity=np.diff(np.concatenate((np.zeros((1,2)),a[:,:2])),axis=0)/dt
    if initial_velocity_xy is not None:
        v=np.asarray(initial_velocity_xy,dtype=np.float64)
        if v.shape!=(2,) or not np.isfinite(v).all():raise ValueError("Bad current velocity")
        acceleration=np.diff(np.concatenate((v[None],velocity)),axis=0)/dt
    else:
        acceleration=np.diff(velocity,axis=0)/dt
    jerk=np.diff(acceleration,axis=0)/dt
    speed=np.linalg.norm(velocity,axis=-1)
    yaw_rate=wrap_angle(np.diff(np.r_[0.,a[:,2]]))/dt
    translating = speed > 1e-6
    undefined_turn = (~translating) & (np.abs(yaw_rate) > 1e-6)
    curvature=np.divide(np.abs(yaw_rate),speed,out=np.zeros_like(speed),where=translating)
    values={"speed_mps":float(speed.max()),"acceleration_mps2":float(np.linalg.norm(acceleration,axis=-1).max()),
            "jerk_mps3":float(np.linalg.norm(jerk,axis=-1).max()),"yaw_rate_rps":float(np.abs(yaw_rate).max()),
            "curvature_inv_m":float(curvature.max()),"lateral_acceleration_mps2":float((curvature*speed**2).max())}
    failures=[k for k,v in values.items() if v>getattr(limits,'max_'+k)+1e-12]
    if undefined_turn.any():
        failures.append("yaw_change_without_translation")
    return {"geometrically_admissible":not failures,"failures":failures,"maxima":values,
            "curvature_defined_steps":int(translating.sum()),
            "undefined_stationary_turn_steps":int(undefined_turn.sum()),
            "initial_velocity_observed":initial_velocity_xy is not None,
            "executable":"REQUIRES_REAL_CONTROLLER_REPLAY","is_safety_label":False}


def tracking_screen(submitted, executed_states, basis_spec, limits, *, simulation_dt=.1):
    """Native global states are converted back to ego(t0) before comparison."""
    a=validate_trajectory(submitted,basis_spec);limits.validate()
    states=np.asarray(executed_states,dtype=np.float64)
    if states.ndim!=2 or states.shape[1]<3 or not np.isfinite(states).all():
        raise ValueError("Invalid actual native execution")
    if not np.isfinite(simulation_dt) or simulation_dt<=0:raise ValueError("Invalid simulation dt")
    times=np.arange(1,basis_spec.horizon+1)*basis_spec.dt
    ids=np.rint(times/simulation_dt).astype(int)
    if ids[-1]>=len(states) or not np.allclose(ids*simulation_dt,times,rtol=0,atol=1e-10):
        raise ValueError("Submitted and native time grids do not align")
    initial=states[0,:3];yaw=initial[2];c,s=np.cos(yaw),np.sin(yaw)
    xy=(states[ids,:2]-initial[:2]) @ np.array([[c,-s],[s,c]])
    headings=wrap_angle(states[ids,2]-yaw)
    xy_error=np.linalg.norm(xy-a[:,:2],axis=-1);yaw_error=np.abs(wrap_angle(headings-a[:,2]))
    return {"controller_tracking_pass":bool(xy_error.max()<=limits.max_tracking_xy_m and yaw_error.max()<=limits.max_tracking_yaw_rad),
            "max_xy_error_m":float(xy_error.max()),"mean_xy_error_m":float(xy_error.mean()),
            "max_yaw_error_rad":float(yaw_error.max()),"sampled_state_indices":ids.tolist(),
            "submitted_frame":"ego_t0_rear_axle","native_frame":"global_rear_axle",
            "is_official_metric":False}
