"""Task-local I/O, explicit source identities, and original trajectory conventions."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
MEAN = np.array([10.172484, .360762])
STD = np.array([8.805105, 2.277741])


def read(path):
    for attempt in range(8):
        try:
            return json.loads(Path(path).read_text())
        except OSError as error:
            if error.errno != 116 or attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if not replace and path.exists():
        tmp.unlink()
        raise FileExistsError(path)
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def poses_valid(poses):
    a = np.asarray(poses, dtype=np.float64)
    if a.shape != (8, 3) or not np.isfinite(a).all():
        raise ValueError('Expected finite 8x3 ego rear-axle trajectory')
    if np.any(np.abs(a[:, 2]) > np.pi + 1e-5):
        raise ValueError('Heading not in radians wrapped to [-pi,pi]')
    return a


def decode(raw):
    a = np.asarray(raw, dtype=np.float64)
    if a.shape != (8, 4) or not np.isfinite(a).all():
        raise ValueError('Expected finite raw 8x4 prediction')
    return poses_valid(np.column_stack((a[:, :2] * STD + MEAN, np.arctan2(a[:, 2], a[:, 3]))))


def raw_scene(row):
    # Metadata is a local team asset; no arbitrary remote pickle input.
    with Path(row['raw_meta']).open('rb') as f:
        raw = pickle.load(f)
    global_poses = np.asarray(raw['glo_status']['global_poses'], dtype=np.float64)[:12]
    if global_poses.shape != (12, 3) or not np.isfinite(global_poses).all():
        raise ValueError('Invalid 4-history + 8-future global poses')
    origin = global_poses[3]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    xy = (global_poses[:, :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    heading = wrap(global_poses[:, 2] - origin[2])
    relative = np.column_stack((xy, heading))
    gt = poses_valid(relative[4:12])
    previous_delta = relative[3] - relative[2]
    state = np.array([*((previous_delta[:2] - MEAN) / STD),
                      np.sin(previous_delta[2]), np.cos(previous_delta[2])], dtype=np.float32)[None]
    command = np.asarray(raw['glo_status']['commands'][3]).ravel()
    if command.shape != (4,) or not np.isfinite(command).all():
        raise ValueError('Invalid navigation command')
    nav = ['turn left', 'keep straight', 'turn right', 'unknown'][int(command.argmax())]
    instruction = f'You are an autonomous driving agent. The navigation command for the current timestep is {nav}. Your task is to plan future actions based on the understanding of the driving scene.'
    return dict(gt=gt, origin=origin, state=state, navigation=nav, instruction=instruction,
                image_paths=[raw['glo_images'][v]['image_paths'][3] for v in ('cam_f0', 'cam_l0', 'cam_r0')])
