"""Offline annotated-scene constraints, separate from free-space evidence.

No score or ego future GT enters this checker. Polygon collision is a labelled
geometric relation, not responsibility, safety certification, or traffic-rule GT.
"""
from dataclasses import dataclass, asdict
import numpy as np
import shapely
from shapely.geometry import Polygon, GeometryCollection
from .recovery_plans import PlanCurve


@dataclass(frozen=True)
class ConstraintConfig:
    horizon_s: float = 4.
    sample_dt: float = .1
    validation_dt: float = .05
    road_margin_m: float = .02
    obstacle_clearance_m: float = .05
    uncertain_boundary_m: float = .15
    max_speed_mps: float = 40.
    max_acceleration_mps2: float = 12.
    max_yaw_rate_rps: float = 1.5
    max_heading_tangent_error_rad: float = .35
    max_plan_change_m: float = 2.
    max_heading_change_rad: float = .4
    min_progress_fraction: float = .65
    distance_clip_m: float = 10.
    version: str = 'annotated_local_constraints_v1'

    def to_dict(self):
        return asdict(self)


def body_polygons(poses, body):
    """Complete oriented rectangle at rear axle; mature GEOS polygon operators."""
    poses = np.asarray(poses, np.float64)
    length, width, offset = [float(body[k]) for k in ('length','width','rear_axle_to_center')]
    corners = np.array([[-length/2,-width/2],[length/2,-width/2],
                        [length/2,width/2],[-length/2,width/2]])
    corners[:, 0] += offset
    c,s = np.cos(poses[...,2]),np.sin(poses[...,2])
    rot = np.stack((c,-s,s,c),-1).reshape(*c.shape,2,2)
    xy = np.einsum('...ij,pj->...pi',rot,corners)+poses[...,:2,None].swapaxes(-1,-2)
    return shapely.polygons(xy)


def polygon_relations(road, obstacles, poses, body, clip=10.):
    footprint = body_polygons(poses,body)
    covered = shapely.covers(road,footprint)
    clearance = shapely.distance(footprint,road.boundary)
    # Outside depth is explicitly a polygon Hausdorff proxy, not exact
    # signed penetration. Covers/intersects are independently exact GEOS tests.
    inside = shapely.intersection(footprint,road)
    outside_proxy = shapely.hausdorff_distance(footprint,inside)
    outside_proxy = np.where(np.isfinite(outside_proxy),outside_proxy,clip)
    road_distance = np.where(covered,clearance,-np.maximum(outside_proxy,1e-6))
    sep = shapely.distance(footprint,obstacles)
    sep = np.where(shapely.is_empty(obstacles),clip,sep)
    overlap = shapely.intersects(footprint,obstacles)
    penetration = np.sqrt(np.maximum(0,shapely.area(shapely.intersection(footprint,obstacles))))
    signed_sep = np.where(overlap,-np.maximum(penetration,1e-6),sep)
    return np.stack((np.clip(road_distance,-clip,clip),np.clip(signed_sep,-clip,clip),overlap.astype(float)),-1),footprint


class LocalConstraintChecker:
    def __init__(self, scene, config=ConstraintConfig()):
        self.scene,self.config=scene,config

    def relations(self, poses, times, source='FUTURE'):
        poses=np.asarray(poses,np.float64); times=np.asarray(times,np.float64)
        obstacles,unknown,frame_valid=self.scene.at_times(times,source)
        values,footprint=polygon_relations(self.scene.road,obstacles,poses,self.scene.body,self.config.distance_clip_m)
        # An unknown-height or disappearing object's possible footprint can
        # invalidate a negative assertion. No absent annotation becomes free.
        unknown_distance=shapely.distance(footprint,unknown)
        unknown_distance=np.where(shapely.is_empty(unknown),np.inf,unknown_distance)
        known_object_relation=(unknown_distance>np.maximum(values[...,1],0)+1e-6)&frame_valid
        bounds=shapely.bounds(footprint)
        in_grid=(bounds[...,0]>=-20)&(bounds[...,1]>=-40)&(bounds[...,2]<=80)&(bounds[...,3]<=40)
        height_known=bool(self.scene.road_height_known)
        overlap_known=((values[...,2]>.5)|(unknown_distance>1e-6))&frame_valid
        valid=np.stack((in_grid&height_known&(~shapely.is_empty(self.scene.road)),
                        in_grid&known_object_relation,in_grid&overlap_known),-1)
        support=self.scene.observed_support(poses,times,source)
        return values,valid,{'in_grid':in_grid,'observed_free_support_fraction':support,
                             'constraint_object_support':frame_valid&(unknown_distance>self.config.obstacle_clearance_m),
                             'unknown_object_nearby':~known_object_relation}

    def check(self, plan, velocity, reference=None, *, dt=None):
        c=self.config;dt=c.validation_dt if dt is None else dt
        curve=PlanCurve(plan,velocity)
        times=np.linspace(0,curve.horizon,int(round(curve.horizon/dt))+1)
        poses=curve.poses(times)
        relations,valid,support=self.relations(poses,times)
        v=curve.spline(times,1);a=curve.spline(times,2)
        speed=np.linalg.norm(v,axis=-1)
        yawrate=np.abs(np.gradient(np.unwrap(poses[:,2]),times,edge_order=2))
        tangent=np.arctan2(v[:,1],v[:,0])
        yaw_error=np.abs(np.arctan2(np.sin(poses[:,2]-tangent),np.cos(poses[:,2]-tangent)))
        yaw_error=np.where(speed>=.5,yaw_error,0.)
        physical_ok=bool(speed.max()<=c.max_speed_mps and np.linalg.norm(a,axis=-1).max()<=c.max_acceleration_mps2
                         and yawrate.max()<=c.max_yaw_rate_rps and yaw_error.max()<=c.max_heading_tangent_error_rad)
        road_violation=bool(((relations[:,0]<-c.uncertain_boundary_m)&valid[:,0]).any())
        overlap_violation=bool(((relations[:,2]>.5)&valid[:,2]).any())
        clear=bool((relations[:,0]>=c.road_margin_m).all() and (relations[:,1]>=c.obstacle_clearance_m).all())
        complete=bool(valid[:,0].all() and support['constraint_object_support'].all())
        route_ok=True;change=heading_change=0.;progress_ratio=1.
        if reference is not None:
            reference=np.asarray(reference);delta=np.asarray(plan)[:,:2]-reference[:,:2]
            change=float(np.linalg.norm(delta,axis=-1).max())
            heading_change=float(np.abs(np.arctan2(np.sin(np.asarray(plan)[:,2]-reference[:,2]),np.cos(np.asarray(plan)[:,2]-reference[:,2]))).max())
            # Stay on the input path's local intent, not fixed ego x or GT route.
            progress=np.linalg.norm(reference[-1,:2])
            direction=reference[-1,:2]/max(progress,1e-6)
            progress_ratio=float(np.dot(np.asarray(plan)[-1,:2],direction)/max(progress,1e-6)) if progress>.5 else 1.
            route_ok=change<=c.max_plan_change_m and heading_change<=c.max_heading_change_rad and progress_ratio>=c.min_progress_fraction
        known_violation=road_violation or overlap_violation or not physical_ok
        return {'feasible':complete and clear and physical_ok and route_ok,
            'complete_annotations':complete,'clear_geometric_constraints':clear,'known_violation':known_violation,
            'road_violation':road_violation,'annotated_overlap':overlap_violation,'kinematic_ok':physical_ok,
            'route_scope_ok':route_ok,'road_margin_min_m':float(relations[:,0].min()),
            'annotated_clearance_min_m':float(relations[:,1].min()),
            'speed_max_mps':float(speed.max()),'acceleration_max_mps2':float(np.linalg.norm(a,axis=-1).max()),
            'yaw_rate_max_rps':float(yawrate.max()),'heading_tangent_error_max_rad':float(yaw_error.max()),
            'maximum_change_m':change,'maximum_heading_change_rad':heading_change,'progress_ratio':progress_ratio,
            'observed_free_support_fraction':float(support['observed_free_support_fraction'].mean()),
            'out_of_range':bool((~support['in_grid']).any()),'sample_dt':dt,
            'scope':'local planar map and reliably annotated objects; not complete free-space or traffic-rule certification'}
