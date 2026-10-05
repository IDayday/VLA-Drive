"""Label-side diagnostics only. RGB colours are a display, never a decoder."""
import hashlib
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


def clean_features(value, valid, normalize=False):
    if valid.dtype != torch.bool or valid.shape != value.shape[:-1]:
        raise ValueError('Expected channel-last features and Boolean patch mask')
    if not torch.isfinite(value[valid]).all():
        raise ValueError('Nonfinite valid feature')
    value = torch.where(valid[..., None], value.float(), 0.)
    return F.layer_norm(value, (value.shape[-1],)) if normalize else value


def patch_error(prediction, target, valid):
    p, t = [clean_features(x, valid) for x in (prediction, target)]
    return (p-t).square().mean(-1)


def reference_metrics(prediction, target, valid):
    error = patch_error(prediction, target, valid)
    n = int(valid.sum())
    return {'squared_channel_mean_sum': float(error[valid].sum()), 'patches': n,
            'mse': float(error[valid].mean()) if n else None}


def affinity_map(features, anchor):
    """One target patch anchors both teacher and student maps; no GT model input."""
    if not torch.isfinite(features).all() or not torch.isfinite(anchor).all():
        raise ValueError('Nonfinite semantic anchor')
    return F.cosine_similarity(features.float(), anchor.float(), dim=-1)


def select_representatives(rows, limit=32):
    """Round-robin attribute strata and logs. Never inspect model errors."""
    if limit < 1 or len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Invalid representative population')
    groups = defaultdict(list)
    order = lambda r: hashlib.sha256(('visualization-v1:'+r['token']).encode()).hexdigest()
    for r in rows:
        groups[(r['navigation'], r['ego_motion'], r['peer_motion'], r['clip_valid'])].append(r)
    for group in groups.values():
        group.sort(key=order)
    chosen, logs = [], defaultdict(int)
    while groups and len(chosen) < min(limit, len(rows)):
        for key in sorted(list(groups)):
            group = groups[key]
            # Prefer logs not yet represented within each attribute stratum.
            j = min(range(len(group)), key=lambda i: (logs[group[i]['log']], order(group[i])))
            row = group.pop(j); chosen.append(row); logs[row['log']] += 1
            if not group: del groups[key]
            if len(chosen) == limit: break
    return chosen


def projected_vehicle_centres(record, current, active, grid=(6, 8)):
    """Distorted calibrated GT centres, for coarse diagnostic anchors only.

    Calibration is the project's 1024x576 current ROI. A point is not an
    occlusion label or a segmentation mask; only selected GT vehicles appear.
    """
    cal = record['current_calibration']; result = []
    for view, (k, ext, distortion) in enumerate(zip(cal['intrinsics'], cal['extrinsics'], cal['distortion'])):
        k, ext = np.asarray(k), np.asarray(ext)
        for slot in range(1, len(current)):
            if not active[slot]: continue
            xyz = np.linalg.inv(ext) @ np.r_[current[slot, :3], 1.]
            if xyz[2] <= .1: continue
            x, y = xyz[:2]/xyz[2]; r2 = x*x+y*y
            k1,k2,p1,p2,k3 = distortion
            radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
            derivative = 1+3*k1*r2+5*k2*r2*r2+7*k3*r2*r2*r2
            xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
            yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
            uv = k @ np.array([xd, yd, 1.]); u, v = uv[:2]/[1024.,576.]
            if radial>0 and derivative>0 and 0<=u<1 and 0<=v<1:
                result.append({'view':view,'slot':slot,'row':int(v*grid[0]),'column':int(u*grid[1]),
                               'u':float(u),'v':float(v),'distance_m':float(np.linalg.norm(current[slot,:2]))})
    return sorted(result, key=lambda r: (r['distance_m'], r['view'], r['slot']))


def relative_gain(model, reference):
    return 1-model/reference if reference is not None and reference>1e-8 else None
