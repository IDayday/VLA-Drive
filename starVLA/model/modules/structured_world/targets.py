"""Offline raw-log target construction. This module is never imported by providers."""
import numpy as np
import torch
from .contracts import WorldTargets
from .geometry import future_track_positions

CLASSES = ('vehicle','pedestrian','bicycle','traffic_cone','barrier','czone_sign','generic_object')


def make_targets(current, future, capacity=32, bounds=(1.,-20.,50.,20.), steps=8, current_eligibility=None):
    anns = current.get('anns')
    present = anns is not None
    raw = np.asarray(anns['gt_boxes'] if present else np.empty((0,7)),dtype=np.float32)
    tracks = list(anns['track_tokens']) if present else []
    names = list(anns['gt_names']) if present else []
    if len(tracks) != len(set(tracks)) or not len(raw) == len(tracks) == len(names):
        raise ValueError('Invalid current track identity contract')
    bounds = np.asarray(bounds,dtype=np.float32)
    valid = np.isfinite(raw[:,:2]).all(-1)
    valid &= ((raw[:,:2] >= bounds[:2]) & (raw[:,:2] <= bounds[2:])).all(-1)
    unknown = set(names) - set(CLASSES)
    if unknown:
        raise ValueError(f'Unknown annotation classes: {unknown}')
    if current_eligibility is not None:
        valid &= np.asarray(current_eligibility,dtype=bool)
    selected = np.where(valid)[0]
    # Current range then track ID: deterministic overflow policy, no future information.
    selected = sorted(selected,key=lambda i:(float(np.linalg.norm(raw[i,:2])),tracks[i]))
    overflow = max(0,len(selected)-capacity)
    selected = np.asarray(selected[:capacity],dtype=np.int64)
    boxes = raw[selected]
    encoded = np.concatenate([boxes[:,:6],np.sin(boxes[:,6:7]),np.cos(boxes[:,6:7])],-1)
    mask = np.isfinite(encoded)
    mask[:,3:6] &= encoded[:,3:6] > 0
    encoded = np.where(mask,encoded,0.)
    chosen_tracks = tuple(tracks[i] for i in selected)
    # Require regular 0.5 s action supervision, with a tolerance for log timestamp jitter.
    timed_frames = []
    for t in range(steps):
        frame = future[t] if t < len(future) else None
        if frame is not None and abs((frame['timestamp']-current['timestamp'])/1e6 - .5*(t+1)) > .05:
            frame = None
        timed_frames.append(frame)
    xy, future_mask = future_track_positions(chosen_tracks,timed_frames,current['ego2global'],steps)
    return WorldTargets(torch.from_numpy(encoded),torch.tensor([CLASSES.index(names[i]) for i in selected],dtype=torch.long),chosen_tracks,
                        torch.from_numpy(xy),torch.from_numpy(future_mask),torch.ones(len(selected),dtype=torch.bool),
                        torch.tensor(present),torch.from_numpy(mask),torch.from_numpy(bounds),overflow)
