"""Offline labels only. T_A_from_B acts on homogeneous column vectors."""
import numpy as np


def transform_points(T_A_from_B,points_B):
    t=np.asarray(T_A_from_B,dtype=np.float64);p=np.asarray(points_B,dtype=np.float64)
    if t.shape!=(4,4) or p.ndim!=2 or p.shape[1]!=3:raise ValueError("Transform/point shape")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):raise ValueError("Nonfinite geometry")
    if not np.allclose(t[3],[0,0,0,1],atol=1e-9):raise ValueError("Not a rigid homogeneous transform")
    return p@t[:3,:3].T+t[:3,3]


def future_lidar_to_current(T_W_from_E0,T_W_from_Etau,T_Etau_from_Ltau,points_Ltau):
    return transform_points(np.linalg.inv(T_W_from_E0)@T_W_from_Etau@T_Etau_from_Ltau,points_Ltau)


def match_timestamp(current_us,candidates_us,horizon_s,tolerance_s):
    times=np.asarray(candidates_us,dtype=np.int64)
    if times.ndim!=1 or not len(times):return {"valid":False,"reason":"missing_future"}
    delta=(times-int(current_us))/1e6
    index=int(np.argmin(np.abs(delta-horizon_s)))
    error=float(abs(delta[index]-horizon_s))
    return {"index":index,"requested_horizon_s":float(horizon_s),"actual_horizon_s":float(delta[index]),
            "absolute_error_s":error,"valid":bool(error<=tolerance_s and delta[index]>0)}


def ray_evidence(origins,first_returns,*,fractions=(.25,.5,.75),surface_clearance=.5):
    """Sparse beam-supported queries, never a dense unobserved-free map.

    The caller must verify these are first valid returns from a single sweep.
    No-return rays, occluded space, and all unsampled voxels remain unknown.
    """
    o=np.asarray(origins,dtype=np.float64);p=np.asarray(first_returns,dtype=np.float64)
    if o.shape!=p.shape or p.ndim!=2 or p.shape[1]!=3:raise ValueError("Ray shape")
    d=p-o;length=np.linalg.norm(d,axis=-1)
    valid=np.isfinite(p).all(-1)&np.isfinite(o).all(-1)&(length>surface_clearance)
    points=[p];labels=[np.ones(len(p),dtype=np.int64)];masks=[valid]
    for fraction in fractions:
        if not 0<fraction<1:raise ValueError("Free sample must precede first return")
        points.append(o+d*fraction);labels.append(np.zeros(len(p),dtype=np.int64))
        masks.append(valid&(length*(1-fraction)>=surface_clearance))
    return np.concatenate(points),np.concatenate(labels),np.concatenate(masks)
