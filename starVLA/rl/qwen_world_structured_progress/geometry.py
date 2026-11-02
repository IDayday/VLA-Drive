"""Exact CPU geometry used in the sealed112-query TRAIN diagnostic.

These are synthetic proposals, not logged future labels or policy samples.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline,CubicHermiteSpline,PchipInterpolator
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec

DELTAS=(.5,1.,1.5,2.,3.,4.,5.,6.)
TIMES=np.arange(1,9,dtype=np.float64)*.5


class InvalidGeometry(ValueError):pass


def path_and_time(base,current_vx):
    base=np.asarray(base,dtype=np.float64)
    if base.shape!=(8,3) or not np.isfinite(base).all() or not np.isfinite(current_vx):
        raise InvalidGeometry('invalid_base_or_current_vx')
    points=np.vstack((np.zeros((1,2)),base[:,:2]));s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-6]
    if keep.sum()<3:raise InvalidGeometry('insufficient_distinct_path_geometry')
    xy=points[keep];ss=s[keep];yaw=float(base[-1,2])
    spatial=CubicSpline(ss,xy,axis=0,bc_type=((1,np.array([1.,0.])),(1,np.array([np.cos(yaw),np.sin(yaw)]))))
    knots=np.arange(9)*.5;derivative=PchipInterpolator(knots,s).derivative()(knots);derivative[0]=float(current_vx)
    temporal=CubicHermiteSpline(knots,s,derivative);dense=np.linspace(0,4.,801)
    if current_vx<0 or temporal(dense,1).min() < -1e-6:raise InvalidGeometry('nonmonotone_forward_profile')
    length=float(ss[-1]);endpoint=spatial(length);tangent=spatial(length,1);second=spatial(length,2)
    theta=float(np.arctan2(tangent[1],tangent[0]));curvature=float((tangent[0]*second[1]-tangent[1]*second[0])/np.linalg.norm(tangent)**3)
    def geometry(progress):
        progress=np.asarray(progress);inside=progress<=length
        xy=spatial(np.minimum(progress,length));direction=spatial(np.minimum(progress,length),1)
        angle=np.arctan2(direction[:,1],direction[:,0]);u=progress[~inside]-length
        if len(u):
            a=theta+curvature*u
            if abs(curvature)<1e-9:xy[~inside]=endpoint+u[:,None]*np.array([np.cos(theta),np.sin(theta)])
            else:xy[~inside]=endpoint+np.column_stack(((np.sin(a)-np.sin(theta))/curvature,(np.cos(theta)-np.cos(a))/curvature))
            angle[~inside]=a
        return xy,np.arctan2(np.sin(angle),np.cos(angle))
    initial=spatial(0.,1)*temporal(0.,1)
    if not np.allclose(initial,[current_vx,0.],rtol=0,atol=1e-9):raise InvalidGeometry('initial_velocity_continuity')
    return temporal,geometry,dict(path_length_parameter=length,terminal_heading=theta,terminal_curvature_per_m=curvature,
        initial_velocity_xy=initial.tolist(),baseline_initial_path_acceleration=float(temporal(0.,2)))


def propose_family(base,vx,family):
    temporal,geometry,meta=path_and_time(base,vx);requests=[];clean=[];descriptors=[]
    for delta in DELTAS:
        progress=temporal(TIMES)+delta*(TIMES/4.)**3;xy,yaw=geometry(progress)
        physical=np.column_stack((xy,yaw)).astype(np.float32)
        normalized=ActionCodec.norm_odo(torch.from_numpy(physical));actual=ActionCodec.denorm_odo(normalized).numpy()
        if not np.isfinite(actual).all():raise InvalidGeometry('nonfinite_proposal')
        requests.append(actual);clean.append(normalized)
        descriptors.append(dict(family=family,synthetic=True,not_logged_GT=True,delta_progress_at4_m=delta,
            synthetic_extension_past_last_known_path_m=float(max(0,progress[-1]-meta['path_length_parameter'])),**meta))
    return np.stack(requests),torch.stack(clean),descriptors
