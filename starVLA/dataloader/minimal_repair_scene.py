"""Read-only NAVSIM map and track geometry; no image/model construction.

Uses the original centered-box transform and map extractor. Existing height
validity is retained per object; annotated-object relations and measured ground
support are separate. Intermediate log states are interpolation, not new GT.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import cv2
import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from starVLA.dataloader.structured_world.adapters import NAV_CLASSES, pose_matrix, box_container, transform_boxes_to_ego
from starVLA.dataloader.structured_world.labels import RoadMap
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.decision_local.local_constraint_checker import body_polygons


@lru_cache(maxsize=2)
def raw_log(root,log):
    with (Path(root)/(log+'.pkl')).open('rb') as f:frames=pickle.load(f)
    return frames,{v['token']:i for i,v in enumerate(frames)}


@lru_cache(maxsize=1)
def road_maps(): return RoadMap('navsim','/mnt/navsim/maps')


class AnnotatedScene:
    def __init__(self,*,road,body,frames,observed,road_height_known,quality=None):
        self.road,self.body,self.frames=road,body,frames
        self.observed=observed;self.road_height_known=road_height_known
        self.quality=quality or {};self._time_cache={}
        self.track_history={}
        for i,frame in enumerate(frames):
            for track,obj in (frame or {}).items():self.track_history.setdefault(track,[]).append((i,obj))

    @staticmethod
    def unknown_reachable(obj,delta):
        # Legal observed velocity centres the reachable tube; acceleration and
        # heading change have explicit finite bounds. No future velocity is
        # ever used by CURRENT/CV. This is annotation support, not a policy.
        polygon=shapely.polygons(obj['corners']+obj['velocity']*delta)
        if obj.get('stationary',False):return polygon
        radius=np.linalg.norm(obj['corners']-obj['corners'].mean(0),axis=-1).max()
        padding=.25+6*delta*delta+2*radius*np.sin(min(1.5*abs(delta),np.pi)/2)
        return polygon.buffer(padding)

    def at_times(self,times,source='FUTURE'):
        if source not in ('CURRENT','CV','FUTURE'):raise ValueError('Unknown annotation source')
        key=(tuple(np.round(times,7)),source)
        if key in self._time_cache:return self._time_cache[key]
        known,unknown,valid=[],[],[]
        for t in times:
            if source in ('CURRENT','CV'):
                frame=self.frames[0];yes=[];no=[]
                if frame is not None:
                    for obj in frame.values():
                        corners=obj['corners']+(obj['velocity']*t if source=='CV' else 0)
                        (yes if obj['reliable'] else no).append(shapely.polygons(corners))
                valid.append(frame is not None)
            else:
                position=float(t)/.5;lo=int(np.floor(position+1e-7));hi=int(np.ceil(position-1e-7))
                if hi>=len(self.frames):raise ValueError('Future outside logged horizon')
                a,b=self.frames[lo],self.frames[hi];yes=[];no=[]
                valid.append(a is not None and b is not None)
                if a is not None and b is not None:
                    for track,history in self.track_history.items():
                        if track in a and track in b:
                            obj0,obj1=a[track],b[track];alpha=position-lo
                            polygon=shapely.polygons((1-alpha)*obj0['corners']+alpha*obj1['corners'])
                            (yes if obj0['reliable'] and obj1['reliable'] else no).append(polygon)
                        else:
                            previous=[(i,obj) for i,obj in history if i<=position]
                            if previous:
                                at,obj=previous[-1]
                                no.append(self.unknown_reachable(obj,float(t)-at*.5))
                            elif track in b and hi!=lo:
                                no.append(self.unknown_reachable(b[track],float(t)-hi*.5))
            known.append(unary_union(yes) if yes else GeometryCollection())
            unknown.append(unary_union(no) if no else GeometryCollection())
        out=(np.asarray(known,dtype=object),np.asarray(unknown,dtype=object),np.asarray(valid,bool))
        self._time_cache[key]=out
        return out

    def observed_support(self,poses,times,source):
        # Descriptive sensor-supported occupancy/free coverage, NEVER merged
        # into an assertion that no annotated object means observed free.
        footprints=body_polygons(poses,self.body)
        corners=np.asarray(shapely.get_coordinates(footprints)).reshape(len(poses),5,2)[:,:4]
        points=np.concatenate((corners,np.asarray(poses)[:,:2,None].swapaxes(-1,-2)),1)
        cells=np.floor((points-[-20.,-40.])/.5).astype(int)
        inside=(cells[...,0]>=0)&(cells[...,0]<200)&(cells[...,1]>=0)&(cells[...,1]<160)
        phase=np.zeros(len(times),int) if source!='FUTURE' else np.clip(np.rint(np.asarray(times)/.5).astype(int),0,len(self.observed)-1)
        x=np.clip(cells[...,0],0,199);y=np.clip(cells[...,1],0,159)
        return (self.observed[phase[:,None],y,x]&inside).mean(-1)


def load_annotated_scene(anchor,split,row):
    root=Path(anchor['data'][split]['label_root']);token=row['token']
    meta=json.loads((root/'records'/(token+'.json')).read_text())
    if meta['cache_identity']!=anchor['data'][split]['label_identity']:raise ValueError('Foreign original geometry')
    with np.load(root/'labels'/(token+'.npz'),allow_pickle=False) as f:
        masks=f['occupancy_valid'].copy();occ=f['occupancy'].copy()
    frames,lookup=raw_log('/mnt/navsim/trainval_navsim_logs/trainval',row['log'])
    index=lookup[token];current=frames[index]
    reference=pose_matrix(current['ego2global_translation'],current['ego2global_rotation'],planar=True)
    inverse=np.linalg.inv(reference);annotations=[];velocity_fallback=0
    for offset in range(9):
        raw=frames[index+offset] if index+offset<len(frames) else None
        if raw is None or abs(raw['timestamp']-current['timestamp']-offset*500000)>60000:
            annotations.append(None);continue
        names=np.asarray(raw['anns']['gt_names']);keep=np.array([n in NAV_CLASSES for n in names],bool)
        matrix=inverse@pose_matrix(raw['ego2global_translation'],raw['ego2global_rotation'])@pose_matrix(raw['lidar2ego_translation'],raw['lidar2ego_rotation'])
        boxes=transform_boxes_to_ego(box_container(raw['anns']['gt_boxes'][keep]),matrix)
        corners=boxes.corners[:,[0,3,7,4],:2].numpy() if len(boxes) else np.empty((0,4,2),np.float32)
        velocity=np.asarray(raw['anns'].get('gt_velocity_3d',np.full((len(keep),3),np.nan)))[keep]
        good=np.isfinite(velocity).all(-1);velocity=np.where(good[:,None],velocity,0.)@matrix[:3,:3].T
        stationary=np.isin(names[keep],['traffic_cone','barrier','generic_object']);velocity[stationary]=0
        if offset==0:velocity_fallback=int((~good&~stationary).sum())
        tracks=np.asarray(raw['anns']['track_tokens'])[keep]
        result={}
        for track,polygon,v,static in zip(tracks,corners,velocity,stationary):
            raster=np.zeros(masks.shape[1:],np.uint8)
            cv2.fillPoly(raster,[np.round((polygon-[-20.,-40.])/.5).astype(np.int32)],1)
            pixels=raster.astype(bool)
            reliable=bool(pixels.any() and masks[offset,pixels].all())
            result[str(track)]={'corners':polygon.astype(np.float64),'velocity':v[:2], 'reliable':reliable,'stationary':bool(static)}
        annotations.append(result)
    road=road_maps().polygon(SimpleNamespace(inverse_transform=reference,map_name=current['map_location']),GridSpec())
    height=meta['height_quality'];levels=[h.get('ground_level') for h in height]
    height_known=all(v is not None and 'reason' not in h for v,h in zip(levels,height)) and np.ptp(levels)<=1.
    return AnnotatedScene(road=road,body=meta['ego_body'],frames=annotations,
        observed=masks&(occ==0),road_height_known=bool(height_known),quality={
            'source_label_sha256':meta['label_sha256'],'current_velocity_zero_fallback':velocity_fallback,
            'height_rule':'retain original per-object full-footprint height validity; planar road with valid near-ego ground at all times and <=1m level range',
            'road_layer_limit':'planar vector map; no complete multilevel-road certification',
            'intermediate_time_rule':'linear matched-box corner interpolation; after disappearance preserve unknown reachable tube from last annotated velocity, acceleration<=12m/s2/yawrate<=1.5rad/s plus0.25m; never empty',
            'free_support':'original sensor-supported free cells kept separately, never inferred from empty object set'})
