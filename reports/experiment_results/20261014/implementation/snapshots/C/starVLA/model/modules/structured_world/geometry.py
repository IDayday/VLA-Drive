"""Geometry in metres, x forward, y left, z up. No future ego-relative labels."""
import numpy as np


def transform_points(points, source_to_target):
    points = np.asarray(points, dtype=np.float64)
    return points @ source_to_target[:3, :3].T + source_to_target[:3, 3]


def crop_resize_intrinsics(intrinsics, original_size, output_size):
    """Match dataset centre crop to 16:9 followed by resize; sizes are W,H."""
    w, h = original_size
    ow, oh = output_size
    if w / h > 16 / 9:
        cw, ch = int(h * 16 / 9), h
    else:
        cw, ch = w, int(w * 9 / 16)
    left, top = (w - cw) // 2, (h - ch) // 2
    affine = np.array([[ow/cw,0,-left*ow/cw], [0,oh/ch,-top*oh/ch], [0,0,1.]])
    return affine @ intrinsics, affine


def future_track_positions(current_tracks, frames, ego_t0_to_global, steps=8):
    """frames are future annotations in each frame's ego; missing tracks stay masked."""
    xy = np.zeros((len(current_tracks), steps, 2), dtype=np.float32)
    valid = np.zeros((len(current_tracks), steps), dtype=bool)
    global_to_t0 = np.linalg.inv(ego_t0_to_global)
    for t, frame in enumerate(frames[:steps]):
        if frame is None or frame.get('anns') is None:
            continue
        anns = frame['anns']
        tracks = list(anns['track_tokens'])
        if len(tracks) != len(set(tracks)):
            raise ValueError('Duplicate track ID in future frame')
        by_track = {track: i for i, track in enumerate(tracks)}
        centres = transform_points(anns['gt_boxes'][:, :3], global_to_t0 @ frame['ego2global'] @ frame.get('lidar2ego',np.eye(4)))
        for n, track in enumerate(current_tracks):
            if track in by_track and np.isfinite(centres[by_track[track], :2]).all():
                xy[n,t] = centres[by_track[track], :2]
                valid[n,t] = True
    return xy, valid


def geometric_fov(points, intrinsics, camera_to_ego, distortion, image_size=(1024,576)):
    """Union of calibrated camera frusta, Brown-Conrady distortion; not occlusion."""
    points = np.asarray(points,dtype=np.float64)
    visible = np.zeros(points.shape[:-1],dtype=bool)
    for k,ext,d in zip(intrinsics,camera_to_ego,distortion):
        camera = transform_points(points,np.linalg.inv(ext))
        z = camera[...,2]
        x,y = (camera[...,:2]/np.maximum(z[...,None],1e-6)).T
        r2 = np.minimum(x*x+y*y,1e4)
        k1,k2,p1,p2,k3 = d
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
        yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
        uv = np.stack([xd,yd,np.ones_like(xd)],-1) @ k.T
        derivative = 1+3*k1*r2+5*k2*r2*r2+7*k3*r2*r2*r2
        visible |= (z>.1)&(radial>0)&(derivative>0)&(uv[...,0]>=0)&(uv[...,0]<image_size[0])&(uv[...,1]>=0)&(uv[...,1]<image_size[1])
    return visible
