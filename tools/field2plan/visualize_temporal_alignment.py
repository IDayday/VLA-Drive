#!/usr/bin/env python3
"""Render and validate Phase 3 ego-motion/teacher time alignment.

The input is a safe ``.npz`` bundle containing the nested ``sample["temporal"]``
arrays flattened by key.  All plotted ego origins are expressed in the current
planning frame (x forward, y left); future poses are diagnostic/supervision
metadata and are never inputs to the action-free dynamics writer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"temporal bundle is missing {name!r}")
    return np.asarray(arrays[name])


def _indices(
    arrays: Mapping[str, np.ndarray], name: str, time_count: int
) -> np.ndarray:
    values = _array(arrays, name)
    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a non-empty integer [N] array")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= time_count):
        raise ValueError(f"{name} contains an out-of-range frame")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def render_temporal_alignment(
    arrays: Mapping[str, np.ndarray], output_path: Path
) -> dict:
    """Validate a temporal bundle, render it, and return a JSON-safe summary.

    Required tensors are ``current_from_ego/ego_from_current [T,4,4]``,
    ``frame_times_s/valid_mask [T]`` and integer history/future frame arrays.
    Optional teacher arrays must be supplied together and exactly match the
    future frames/times selected by the dataset contract.
    """

    current_from_ego = _array(arrays, "current_from_ego").astype(
        np.float32, copy=False
    )
    ego_from_current = _array(arrays, "ego_from_current").astype(
        np.float32, copy=False
    )
    if current_from_ego.ndim != 3 or current_from_ego.shape[1:] != (4, 4):
        raise ValueError("current_from_ego must have shape [T,4,4]")
    if ego_from_current.shape != current_from_ego.shape:
        raise ValueError("ego_from_current must match current_from_ego")
    if not np.isfinite(current_from_ego).all() or not np.isfinite(ego_from_current).all():
        raise ValueError("temporal transforms contain non-finite values")
    time_count = current_from_ego.shape[0]

    current_value = _array(arrays, "current_frame_index")
    if current_value.size != 1:
        raise ValueError("current_frame_index must be scalar")
    current_index = int(current_value.reshape(-1)[0])
    if not 0 <= current_index < time_count:
        raise ValueError("current_frame_index is outside [0,T)")
    history_indices = _indices(arrays, "history_frame_indices", time_count)
    future_indices = _indices(arrays, "future_frame_indices", time_count)
    if np.any(history_indices > current_index):
        raise ValueError("history frame indices cannot be after the current frame")
    if np.any(future_indices <= current_index):
        raise ValueError("future frame indices must be after the current frame")

    frame_times = _array(arrays, "frame_times_s").astype(np.float32, copy=False)
    valid_mask = _array(arrays, "valid_mask").astype(np.bool_, copy=False)
    if frame_times.shape != (time_count,) or valid_mask.shape != (time_count,):
        raise ValueError("frame_times_s and valid_mask must have shape [T]")
    if not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame_times_s must be finite and strictly increasing")

    identity = np.eye(4, dtype=np.float32)[None]
    inverse_error = float(
        np.max(np.abs(current_from_ego @ ego_from_current - identity))
    )
    if inverse_error > 1e-3:
        raise ValueError(
            f"temporal transform inverse error {inverse_error:.6g} exceeds 1e-3"
        )
    current_error = float(
        np.max(np.abs(current_from_ego[current_index] - identity[0]))
    )
    if current_error > 1e-3:
        raise ValueError("the current-frame transform is not identity")

    teacher_indices_present = "teacher_frame_indices" in arrays
    teacher_times_present = "teacher_frame_times_s" in arrays
    if teacher_indices_present != teacher_times_present:
        raise ValueError("teacher frame indices and times must be supplied together")
    teacher_time_error = None
    if teacher_indices_present:
        teacher_indices = _array(arrays, "teacher_frame_indices")
        teacher_times = _array(arrays, "teacher_frame_times_s").astype(
            np.float32, copy=False
        )
        if teacher_indices.shape != future_indices.shape or not np.array_equal(
            teacher_indices.astype(np.int64, copy=False), future_indices
        ):
            raise ValueError("teacher frame indices differ from future_frame_indices")
        if teacher_times.shape != future_indices.shape:
            raise ValueError("teacher_frame_times_s must have shape [H]")
        teacher_time_error = float(
            np.max(np.abs(teacher_times - frame_times[future_indices]))
        )
        if teacher_time_error > 1e-4:
            raise ValueError(
                "teacher frame times differ from the dataset temporal contract"
            )

    origins = current_from_ego[:, :2, 3]
    headings = current_from_ego[:, :2, 0]
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    groups = (
        (history_indices, "history", "tab:blue"),
        (np.asarray([current_index]), "current", "black"),
        (future_indices, "future (target alignment only)", "tab:orange"),
    )
    for indices, label, color in groups:
        selected = origins[indices]
        axis.plot(
            selected[:, 1],
            selected[:, 0],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axis.quiver(
            selected[:, 1],
            selected[:, 0],
            headings[indices, 1],
            headings[indices, 0],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            color=color,
        )
        for frame_index, origin in zip(indices, selected):
            axis.annotate(
                f"t{int(frame_index)}\n{float(frame_times[frame_index]):+.1f}s",
                (float(origin[1]), float(origin[0])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    invalid = np.flatnonzero(~valid_mask)
    if invalid.size:
        axis.scatter(
            origins[invalid, 1],
            origins[invalid, 0],
            marker="x",
            s=90,
            color="red",
            label="invalid",
        )
    axis.set_xlabel("y left in current ego [m]")
    axis.set_ylabel("x forward in current ego [m]")
    axis.set_title("Field2Plan temporal / ego-motion alignment")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, output_path)

    def selected_origins(indices: np.ndarray) -> list[list[float]]:
        return origins[indices].astype(float).tolist()

    return {
        "coordinate_frame": "current_planning_ego_x_forward_y_left",
        "current_frame_index": current_index,
        "current_origin_xy_m": origins[current_index].astype(float).tolist(),
        "history_frame_indices": history_indices.tolist(),
        "future_frame_indices": future_indices.tolist(),
        "history_origins_xy_m": selected_origins(history_indices),
        "future_origins_xy_m": selected_origins(future_indices),
        "valid_frames": int(valid_mask.sum()),
        "total_frames": int(time_count),
        "transform_inverse_max_abs_error": inverse_error,
        "teacher_alignment_max_abs_time_error_s": teacher_time_error,
        "future_usage": "offline_supervision_alignment_only",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Temporal .npz bundle")
    parser.add_argument("--output", required=True, help="Output PNG")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON summary; defaults to <output>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.input)
    if source.suffix != ".npz":
        raise ValueError("--input must be a .npz file")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    output = Path(args.output)
    summary = render_temporal_alignment(arrays, output)
    summary_path = Path(args.summary_json) if args.summary_json else output.with_suffix(
        ".json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, summary_path)
    print(f"[field2plan] wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
