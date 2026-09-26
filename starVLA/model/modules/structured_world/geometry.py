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
        centres = transform_points(anns['gt_boxes'][:, :3], global_to_t0 @ frame['ego2global'])
        for n, track in enumerate(current_tracks):
            if track in by_track and np.isfinite(centres[by_track[track], :2]).all():
                xy[n,t] = centres[by_track[track], :2]
                valid[n,t] = True
    return xy, valid
