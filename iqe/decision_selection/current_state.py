"""Privileged diagnostic extraction: static map and timestamp-attested t0 objects only.

Metric-cache deserialization happens in an isolated NAVSIM process. Neither its
future reference nor future occupancy is read by extract(). The original cache
can contain one-frame future objects repeated at t0; timestamp filtering is
therefore mandatory even for observation[0].
"""
from pathlib import Path
import argparse
import io
import json
import lzma
import pickle
import sys
import numpy as np
from iqe.contracts import require
from iqe.io import read_json,atomic_json,atomic_bytes,file_hash,digest,complete,reusable

AGENTS=64
MAP_POINTS=256
RADIUS=50.


def local_xy(xy,pose):
    c,s=np.cos(pose[2]),np.sin(pose[2])
    return (np.asarray(xy)-np.asarray(pose[:2]))@np.array([[c,-s],[s,c]])


def in_frustum(xyz,calibration):
    xyz=np.asarray(xyz);result=np.zeros(xyz.shape[:-1],dtype=bool)
    for k,e in zip(calibration['intrinsics'],calibration['extrinsics']):
        k,e=np.asarray(k),np.asarray(e)
        cam=(xyz-e[:3,3])@e[:3,:3]
        pixel=cam@k.T
        uv=pixel[...,:2]/np.maximum(pixel[...,2:],1e-3)
        result|=(cam[...,2]>1e-3)&(uv[...,0]>0)&(uv[...,0]<1023)&(uv[...,1]>0)&(uv[...,1]<575)
    return result


def extract(cache,observation):
    t0=cache.ego_state.time_point.time_us
    require(t0==observation['timestamp'],'current timestamp does not match cache ego')
    rear=cache.ego_state.rear_axle;pose=np.array([rear.x,rear.y,rear.heading],dtype=float)
    require(np.max(np.abs(pose-np.asarray(observation['global_pose_history'][-1])))<1e-3,'cache/current rear-axle frame mismatch')
    calibration=observation['current_calibration']
    require(len(calibration['intrinsics'])==len(calibration['extrinsics'])==3,'locked three current views')
    occupancy=cache.observation[0]  # Never index a future occupancy frame.
    candidates=[];future_only=0;outside=0;far=0
    for token in occupancy.tokens:
        obj=cache.observation.unique_objects.get(token)
        if obj is None:continue
        if obj.metadata.timestamp_us!=t0:
            future_only+=1;continue
        xy=local_xy([[obj.center.x,obj.center.y]],pose)[0]
        if np.linalg.norm(xy)>RADIUS:far+=1;continue
        yaw=float(obj.center.heading-pose[2]);box=obj.box
        # Check center plus current footprint vertices, at explicit height hypotheses.
        coords=np.asarray(occupancy[token].exterior.coords)[:-1]
        current_coords=np.asarray(box.geometry.exterior.coords)[:-1]
        require(coords.shape==current_coords.shape and np.max(np.abs(coords-current_coords))<1e-5,'t0 object/occupancy mismatch')
        anchors=local_xy(np.concatenate(([[obj.center.x,obj.center.y]],coords)),pose)
        points=np.concatenate((np.repeat(anchors,2,axis=0),np.tile([0.,float(box.height)/2],len(anchors))[:,None]),1)
        if not in_frustum(points,calibration).any():outside+=1;continue
        velocity=getattr(obj,'velocity',None)
        # Original t0 annotation velocity, never differenced from future frames.
        vel=np.zeros(2) if velocity is None else local_xy([[velocity.x,velocity.y]],np.array([0.,0.,pose[2]]))[0]
        category=int(obj.tracked_object_type.value);require(0<=category<10,'unregistered tracked object category')
        onehot=np.eye(10,dtype=np.float32)[category]
        feature=np.concatenate((xy,[np.sin(yaw),np.cos(yaw),box.length,box.width,box.height],vel,onehot))
        candidates.append((float(np.dot(xy,xy)),str(token),feature))
    candidates.sort(key=lambda r:(r[0],r[1]));agent=np.zeros((AGENTS,19),np.float32);am=np.zeros(AGENTS,bool)
    for i,(_,_,f) in enumerate(candidates[:AGENTS]):agent[i]=f;am[i]=True
    points=[];maps=cache.drivable_area_map
    for token,layer,geometry in zip(maps.tokens,maps.map_types,maps._geometries):
        polygons=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
        for polygon_index,polygon in enumerate(polygons):
            boundary=polygon.exterior
            # Static boundary samples are independent of candidates, scores and future reference.
            length=boundary.length
            for j,distance in enumerate(np.arange(0.,length,2.)):
                p=boundary.interpolate(float(distance));q=boundary.interpolate(float(min(distance+.25,length)))
                xy=local_xy([[p.x,p.y]],pose)[0]
                if np.linalg.norm(xy)>RADIUS:continue
                if not in_frustum(np.array([[xy[0],xy[1],0.]]),calibration)[0]:continue
                tangent=local_xy([[q.x-p.x,q.y-p.y]],np.array([0.,0.,pose[2]]))[0]
                tangent/=max(np.linalg.norm(tangent),1e-9)
                # Only static semantic layer, no route/reference lane identity.
                code=int(layer.value);require(0<=code<20,'map category exceeds fixed schema')
                f=np.concatenate((xy,tangent,np.eye(20,dtype=np.float32)[code]))
                points.append((float(np.dot(xy,xy)),str(token),polygon_index,j,f))
    points.sort(key=lambda r:r[:4]);map_features=np.zeros((MAP_POINTS,24),np.float32);mm=np.zeros(MAP_POINTS,bool)
    for i,row in enumerate(points[:MAP_POINTS]):map_features[i]=row[-1];mm[i]=True
    dynamic=cache.ego_state.dynamic_car_state
    v=dynamic.rear_axle_velocity_2d;a=dynamic.rear_axle_acceleration_2d
    ego=np.array([v.x,v.y,a.x,a.y],np.float32)
    values=dict(agents=agent,agent_valid=am,map_points=map_features,map_valid=mm,current_ego=ego)
    require(all(np.isfinite(v).all() for v in values.values()),'nonfinite current structured input')
    return values,dict(t0_us=t0,agent_timestamp_rule='metadata.timestamp_us == ego t0; later first appearances excluded',
        agents_before_truncation=len(candidates),agents_kept=int(am.sum()),map_points_before_truncation=len(points),map_points_kept=int(mm.sum()),
        future_only_objects_rejected=future_only,outside_current_frusta_rejected=outside,outside_radius_rejected=far,
        future_occupancy_indices_read=[],reference_or_GT_trajectory_read=False,current_occupancy_index=0,
        velocity_source='provided t0 annotation velocity; no future differencing',
        occlusion_verified=False,frustum_membership_is_not_visibility=True,static_map_extra_information=True,
        map_z_assumption='flat ground z=0, not an estimated road surface',range_m=RADIUS,
        spatial_rules='nearest squared distance, stable token tie; current three-view frustum only; no route filtering')


def worker(request):
    r=read_json(request);sys.path.insert(0,r['navsim_root'])
    from navsim.planning.metric_caching.metric_cache import MetricCache
    import navsim.planning.metric_caching.metric_cache as loaded
    require(Path(loaded.__file__).resolve().is_relative_to(Path(r['navsim_root']).resolve()),'wrong NAVSIM import')
    root=Path(r['output']);rows=[]
    for scene in r['scenes']:
        require(scene['role'] in ('internal_fit','internal_val'),'probe role whitelist')
        identity=dict(scene_id=scene['scene_id'],observation_hash=scene['observation_hash'],metric_context_hash=scene['metric_context_hash'],
                      plan_hash=r['plan_hash'],extractor=file_hash(__file__),schema='privileged_current_only_v1')
        folder=root/'records'/scene['scene_id'];receipt=folder/'COMPLETE.json';out=folder/'current.npz'
        if reusable(receipt,identity):rows.append(dict(scene_id=scene['scene_id'],reused=True));continue
        require(file_hash(scene['metric_context_ref'])==scene['metric_context_hash'],'metric context changed')
        observation=read_json(scene['observation_ref'])
        require(observation['token']==scene['scene_id'],'current scene mismatch')
        require(digest(dict(record=observation,images={p:file_hash(p) for p in observation['image_paths']}))==scene['observation_hash'],'current image/metadata changed')
        with lzma.open(scene['metric_context_ref'],'rb') as f:cache=pickle.load(f)
        require(isinstance(cache,MetricCache),'unexpected metric cache')
        values,evidence=extract(cache,observation)
        # Remove all future containers/reference and verify no model input changes.
        cache.trajectory=None;cache.centerline=None;cache.route_lane_ids=[]
        cache.observation._occupancy_maps=cache.observation._occupancy_maps[:1]
        again,_=extract(cache,observation)
        require(all(np.array_equal(v,again[k]) for k,v in values.items()),'future/reference stripping changed current features')
        b=io.BytesIO();np.savez(b,**values);atomic_bytes(out,b.getvalue(),immutable=True)
        atomic_json(folder/'AUDIT.json',evidence|dict(future_reference_removal_equal=True),immutable=True)
        complete(receipt,identity,[out,folder/'AUDIT.json'])
        rows.append(dict(scene_id=scene['scene_id'],reused=False))
    atomic_json(r['receipt'],dict(status='COMPLETE',plan_hash=r['plan_hash'],records=rows),immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True)
    worker(p.parse_args().request)
