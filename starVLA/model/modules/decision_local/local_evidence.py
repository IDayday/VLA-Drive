"""Loss-side local facts, independent of strict whole-plan repair certification.

Geometry comes from the existing GEOS checker/box interpolation. A distance to
reliable annotated objects is conditional on that set, not a free-space claim.
Known positive overlap survives unrelated unknown objects. A scene-level
negative overlap requires full footprint observed-free support and no unknown
reachable object. No fields in this module are camera-only model inputs.
"""
from dataclasses import dataclass
from enum import IntFlag
import numpy as np
import shapely
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from .local_constraint_checker import polygon_relations

SCHEMA = 'natural_error_local_evidence_v1'
CHANNELS = ('road_body_margin_m', 'annotated_object_distance_m', 'known_geometric_overlap')


class UnknownReason(IntFlag):
    ROAD_HEIGHT = 1
    MAP_RANGE = 2
    CURRENT_OBJECT_SUPPORT = 4
    FUTURE_TRACK_MISSING = 8
    UNKNOWN_REACHABILITY = 16
    BOUNDARY_AMBIGUITY = 32
    OTHER = 64
    OBSERVED_FREE_SUPPORT = 128
    FUTURE_FRAME_MISSING = 256


@dataclass(frozen=True)
class LocalEvidence:
    constraint_type: str
    physical_time: float
    time_window: tuple
    actor_or_map_identity: str
    value: float
    positive_known: bool
    negative_known: bool
    label_valid: bool
    coverage_scope: str
    unknown_reason: int

    def __post_init__(self):
        if self.positive_known and self.negative_known:
            raise ValueError('One event cannot be simultaneously known positive and negative')
        if (self.positive_known or self.negative_known) and not self.label_valid:
            raise ValueError('Known facts must have valid labels')


@dataclass(frozen=True)
class LocalEvidenceBatch:
    values: np.ndarray
    label_valid: np.ndarray
    positive_known: np.ndarray
    negative_known: np.ndarray
    unknown_reason: np.ndarray
    physical_times: np.ndarray
    actor_or_map_identity: np.ndarray
    local_constraint_support: np.ndarray
    observed_free_support: np.ndarray
    events: tuple
    source: str
    schema: str = SCHEMA

    def __post_init__(self):
        shape=(len(self.physical_times),3)
        for name in ('values','label_valid','positive_known','negative_known','unknown_reason','actor_or_map_identity'):
            if getattr(self,name).shape != shape: raise ValueError('Local evidence shape: '+name)
        if np.any(self.positive_known & self.negative_known): raise ValueError('Contradictory event truth')
        if np.any((self.positive_known|self.negative_known)&~self.label_valid): raise ValueError('Known invalid fact')
        if not np.isfinite(self.values[self.label_valid]).all(): raise ValueError('Nonfinite known fact')


def objects_at_time(scene, t, source):
    """Per-actor view of exactly the existing current/CV/interpolated boxes.

    Unknown tubes keep their actor identity and reason. No geometry is used to
    select a plan or change its order. Cache lifetime is this immutable scene.
    """
    if source not in ('CURRENT','CV','FUTURE'): raise ValueError('Unknown relation source')
    if not hasattr(scene,'_local_evidence_objects'): scene._local_evidence_objects={}
    key=(round(float(t),7),source)
    if key in scene._local_evidence_objects: return scene._local_evidence_objects[key]
    known={};unknown={}
    if source in ('CURRENT','CV'):
        frame=scene.frames[0];valid=frame is not None
        for track,obj in (frame or {}).items():
            polygon=shapely.polygons(obj['corners']+(obj['velocity']*t if source=='CV' else 0))
            if obj['reliable']: known[track]=polygon
            else: unknown[track]=(polygon,int(UnknownReason.CURRENT_OBJECT_SUPPORT))
    else:
        position=float(t)/.5; lo=int(np.floor(position+1e-7));hi=int(np.ceil(position-1e-7))
        if lo<0 or hi>=len(scene.frames): raise ValueError('Physical time outside labelled horizon')
        a,b=scene.frames[lo],scene.frames[hi];valid=a is not None and b is not None
        if valid:
            for track,history in scene.track_history.items():
                if track in a and track in b:
                    first,second=a[track],b[track]
                    poly=scene.interpolate_rectangle(first['corners'],second['corners'],position-lo)
                    if first['reliable'] and second['reliable']:known[track]=poly
                    else:unknown[track]=(poly,int(UnknownReason.CURRENT_OBJECT_SUPPORT))
                else:
                    previous=[(i,obj) for i,obj in history if i<=position]
                    if previous:
                        at,obj=previous[-1]
                        unknown[track]=(scene.unknown_reachable(obj,float(t)-at*.5),
                                        int(UnknownReason.FUTURE_TRACK_MISSING|UnknownReason.UNKNOWN_REACHABILITY))
                    elif track in b and hi!=lo:
                        unknown[track]=(scene.unknown_reachable(b[track],float(t)-hi*.5),
                                        int(UnknownReason.FUTURE_TRACK_MISSING|UnknownReason.UNKNOWN_REACHABILITY))
    result=(known,unknown,valid);scene._local_evidence_objects[key]=result
    return result


def local_road_support(scene,times):
    # New metadata is per-frame. The old strict checker still uses its original
    # complete-horizon road_height_known; changing this does not relax KEEP.
    flags=getattr(scene,'local_road_height_known',None)
    if flags is None:return np.full(len(times),bool(scene.road_height_known))
    p=np.asarray(times)/.5
    lo=np.floor(p+1e-7).astype(int);hi=np.ceil(p-1e-7).astype(int)
    return np.asarray(flags,bool)[lo]&np.asarray(flags,bool)[hi]


def footprint_observed_free(scene,footprints,times,source):
    """Conservative all intersected 0.5m cells, not just corners/centre.

    Test intersecting cell polygons against the complete oriented body. Exact
    GEOS intersection identifies boundary cells; no-point pixels stay unknown.
    """
    answer=np.zeros(len(footprints),bool)
    for i,(foot,t) in enumerate(zip(footprints,times)):
        bounds=shapely.bounds(foot)
        if bounds[0]<-20 or bounds[1]<-40 or bounds[2]>=80 or bounds[3]>=40:continue
        begin=np.floor((bounds[:2]-[-20,-40])/.5).astype(int)
        end=np.floor((bounds[2:]-[-20,-40])/.5).astype(int)
        yy,xx=np.meshgrid(np.arange(begin[1],end[1]+1),np.arange(begin[0],end[0]+1),indexing='ij')
        cells=shapely.box(-20+xx*.5,-40+yy*.5,-20+(xx+1)*.5,-40+(yy+1)*.5)
        selected=shapely.intersects(cells,foot)
        p=float(t)/.5 if source=='FUTURE' else 0.
        lo=int(np.floor(p+1e-7));hi=int(np.ceil(p-1e-7))
        answer[i]=bool(selected.any() and np.all(scene.observed[lo,yy[selected],xx[selected]])
                       and np.all(scene.observed[hi,yy[selected],xx[selected]]))
    return answer


def extract_local_evidence(scene, poses, times, source='FUTURE', *, clip_m=10., boundary_m=.15):
    poses=np.asarray(poses,np.float64);times=np.asarray(times,np.float64)
    if poses.shape!=(len(times),3): raise ValueError('Physical pose/time alignment required')
    details=[]
    for t in times:
        actors,tubes,valid=objects_at_time(scene,t,source);details.append((actors,tubes))
    # Reuse the original scene's per-time union cache, not one union per query.
    known,unknown,frame_valid=scene.at_times(times,source)
    values,footprints=polygon_relations(scene.road,np.asarray(known,object),poses,scene.body,clip_m)
    bounds=shapely.bounds(footprints)
    in_grid=(bounds[:,0]>=-20)&(bounds[:,1]>=-40)&(bounds[:,2]<80)&(bounds[:,3]<40)
    road_valid=in_grid&local_road_support(scene,times)&(not bool(shapely.is_empty(scene.road)))
    has_known=~shapely.is_empty(np.asarray(known,object))
    unknown_distance=shapely.distance(footprints,np.asarray(unknown,object))
    unknown_distance=np.where(shapely.is_empty(np.asarray(unknown,object)),np.inf,unknown_distance)
    observed_free=footprint_observed_free(scene,footprints,times,source)
    overlap_positive=in_grid&frame_valid&(values[:,2]>.5)
    overlap_negative=in_grid&frame_valid&(values[:,2]<.5)&(unknown_distance>1e-6)&observed_free
    valid=np.stack((road_valid,in_grid&frame_valid&has_known,overlap_positive|overlap_negative),-1)
    positive=np.stack((road_valid&(values[:,0]<-boundary_m),valid[:,1]&(values[:,1]<0),overlap_positive),-1)
    negative=np.stack((road_valid&(values[:,0]>=boundary_m),valid[:,1]&(values[:,1]>0),overlap_negative),-1)
    reasons=np.zeros_like(values,dtype=np.uint16)
    reasons[~in_grid,:]|=int(UnknownReason.MAP_RANGE)
    reasons[~local_road_support(scene,times),0]|=int(UnknownReason.ROAD_HEIGHT)
    reasons[road_valid&(abs(values[:,0])<boundary_m),0]|=int(UnknownReason.BOUNDARY_AMBIGUITY)
    reasons[~frame_valid,1:]|=int(UnknownReason.CURRENT_OBJECT_SUPPORT if source!='FUTURE' else UnknownReason.FUTURE_FRAME_MISSING)
    reasons[~observed_free,2]|=int(UnknownReason.OBSERVED_FREE_SUPPORT)
    reasons[~has_known,1]|=int(UnknownReason.CURRENT_OBJECT_SUPPORT)
    if shapely.is_empty(scene.road):reasons[:,0]|=int(UnknownReason.OTHER)
    identities=np.full(values.shape,'',dtype=object);identities[:,0]=getattr(scene,'map_identity','planar_drivable_map')
    events=[];current,_,current_valid=objects_at_time(scene,0.,'CURRENT')
    for i,(foot,(actors,tubes)) in enumerate(zip(footprints,details)):
        if tubes:
            tube_values=list(tubes.values())
            near=shapely.distance(foot,np.asarray([x[0] for x in tube_values],object))<=max(values[i,1],.05)+1e-6
            if near.any():reasons[i,1:]|=np.bitwise_or.reduce(np.asarray([x[1] for x in tube_values],np.uint16)[near])
        if actors:
            names=list(actors);polys=np.asarray([actors[a] for a in names],object)
            distances=shapely.distance(foot,polys);identities[i,1]=names[int(np.argmin(distances))]
            overlaps=shapely.intersects(foot,polys)
            identities[i,2]='|'.join(n for n,yes in zip(names,overlaps) if yes)
            for actor,yes in zip(names,overlaps):
                if yes and overlap_positive[i]:events.append({'constraint_type':'known_overlap','time':float(times[i]),'actor':actor,'query_index':i})
                if source=='FUTURE' and current_valid and actor in current and frame_valid[i] and in_grid[i]:
                    before=bool(shapely.intersects(foot,current[actor]))
                    # Endpoint event for the SAME actor at the SAME query pose;
                    # a disappeared track cannot become a known release.
                    if before!=bool(yes):events.append({'constraint_type':'enter' if yes else 'release',
                        'time':float(times[i]),'time_window':[0.,float(times[i])],'actor':actor,'query_index':i})
        if positive[i,0]:events.append({'constraint_type':'road_violation','time':float(times[i]),
                                      'actor':identities[i,0],'query_index':i})
    # Original certificate support is descriptive only; never promoted from
    # the local positive/conditional-distance masks.
    strict_support=road_valid&frame_valid&(unknown_distance>.05)
    return LocalEvidenceBatch(values.astype(np.float32),valid,positive,negative,reasons,times,identities,
                              strict_support,observed_free,tuple(events),source)
