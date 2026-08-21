"""NAVSIM trajectory loading and coordinate conversion without model inputs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses


NAVSIM_NUM_FUTURE_POSES = 8
NAVSIM_INTERVAL_LENGTH = 0.5
NAVSIM_TIME_HORIZON = 4.0


def wrap_to_pi(value: np.ndarray | float) -> np.ndarray:
    """Wrap radians into ``[-pi, pi)``."""

    array = np.asarray(value, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def absolute_poses_to_current_ego(
    global_poses: np.ndarray,
    *,
    current_index: int = 3,
    num_future: int = NAVSIM_NUM_FUTURE_POSES,
) -> np.ndarray:
    """Convert global ego poses into NAVSIM's current rear-axle frame.

    Only the returned future poses are privileged targets. The current pose is
    used solely as the coordinate origin and is never augmented with a future
    actor or future image.
    """

    poses = np.asarray(global_poses, dtype=np.float64)
    required = current_index + num_future + 1
    if poses.ndim != 2 or poses.shape[1] < 3 or poses.shape[0] < required:
        raise ValueError(f"expected at least [{required},3] global poses, got {poses.shape}")
    se2 = [StateSE2(float(x), float(y), float(h)) for x, y, h in poses[:required, :3]]
    relative = absolute_to_relative_poses(se2[current_index:required])[1:]
    result = np.asarray([[pose.x, pose.y, wrap_to_pi(pose.heading)] for pose in relative], dtype=np.float64)
    return result


def load_processed_record(data_root: Path, split: str, scene_id: str) -> Mapping[str, Any]:
    """Load one processed record by logical dataset root and token."""

    path = data_root / "meta" / split / f"{scene_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"processed NAVSIM record is missing: {path}")
    with path.open("rb") as stream:
        record = pickle.load(stream)
    if not isinstance(record, Mapping):
        raise TypeError(f"processed record must be a mapping, got {type(record)!r}: {path}")
    return record


def load_expert_anchor(data_root: Path, split: str, scene_id: str) -> np.ndarray:
    """Load the eight-pose expert anchor from processed ego-only metadata."""

    record = load_processed_record(data_root, split, scene_id)
    try:
        global_poses = record["glo_status"]["global_poses"]
    except KeyError as error:
        raise KeyError(f"processed record lacks glo_status.global_poses: {scene_id}") from error
    return absolute_poses_to_current_ego(global_poses)


def resolve_policy_prediction(prediction_root: Path, split: str, scene_id: str) -> Path:
    """Resolve either ``root/split/token.npy`` or ``root/token.npy``."""

    candidates = [prediction_root / split / f"{scene_id}.npy", prediction_root / f"{scene_id}.npy"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"policy anchor is missing for {scene_id}; tried {candidates}")


def load_policy_anchor(prediction_root: Path, split: str, scene_id: str) -> np.ndarray:
    """Load a deterministic baseline prediction in physical NAVSIM pose format."""

    path = resolve_policy_prediction(prediction_root, split, scene_id)
    trajectory = np.asarray(np.load(path), dtype=np.float64)
    if trajectory.shape != (NAVSIM_NUM_FUTURE_POSES, 3):
        raise ValueError(f"policy anchor must be [8,3], got {trajectory.shape}: {path}")
    trajectory[:, 2] = wrap_to_pi(trajectory[:, 2])
    return trajectory
