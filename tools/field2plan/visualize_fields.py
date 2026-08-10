#!/usr/bin/env python3
"""Render Field2Plan field/readout diagnostics from a safe ``.npz`` bundle.

The tool accepts both the lightweight inference diagnostics written by
``infer.py --save_diagnostics`` and richer debug bundles containing field or
teacher maps.  BEV convention is x-forward (vertical) and y-left (horizontal).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Diagnostic .npz bundle")
    parser.add_argument("--output", required=True, help="Output PNG")
    parser.add_argument("--x-range-m", nargs=2, type=float, default=(-8.0, 56.0))
    parser.add_argument("--y-range-m", nargs=2, type=float, default=(-32.0, 32.0))
    parser.add_argument("--candidate-index", type=int, default=0)
    return parser.parse_args()


def _physical_trajectory(
    arrays: Mapping[str, np.ndarray], prefix: str, candidate_index: int
) -> Optional[np.ndarray]:
    physical_key = f"{prefix}_physical"
    action_key = f"{prefix}_action"
    if physical_key in arrays:
        trajectory = np.asarray(arrays[physical_key], dtype=np.float32)
    elif action_key in arrays:
        action = np.asarray(arrays[action_key], dtype=np.float32)
        if action.shape[-1] != 4:
            raise ValueError(f"{action_key} must end in [x,y,sin,cos]")
        decoded = TrajectoryCodec().decode_action(torch.from_numpy(action))
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("TrajectoryCodec returned a non-tensor")
        trajectory = decoded.numpy()
    else:
        return None
    if trajectory.ndim == 3:
        if not 0 <= candidate_index < trajectory.shape[0]:
            raise IndexError(
                f"candidate index {candidate_index} outside M={trajectory.shape[0]}"
            )
        trajectory = trajectory[candidate_index]
    if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
        raise ValueError(f"{prefix} trajectory must have shape [H,3] or [M,H,3]")
    return trajectory


def _spatial_map(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if name == "geometry_field":
        if array.ndim != 3:
            raise ValueError("geometry_field must have shape [C,Ny,Nx]")
        return np.linalg.norm(array.astype(np.float32), axis=0)
    if array.ndim == 4:
        # Teacher/prediction maps use [V,Z,Ny,Nx].  A mean is only a visual
        # summary; quantitative probes use the unreduced tensors in training.
        return array.astype(np.float32).mean(axis=(0, 1))
    if array.ndim == 3:
        return array.astype(np.float32).mean(axis=0)
    if array.ndim == 2:
        return array.astype(np.float32)
    raise ValueError(f"{name} must be a 2D map or have leading V/Z dimensions")


def _plot_trajectory(
    axis,
    trajectory: Optional[np.ndarray],
    *,
    label: str,
    color: str,
) -> None:
    if trajectory is None:
        return
    axis.plot(
        trajectory[:, 1],
        trajectory[:, 0],
        marker="o",
        markersize=3,
        linewidth=2,
        label=label,
        color=color,
    )


def render_field_diagnostics(
    arrays: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    x_range_m: Sequence[float] = (-8.0, 56.0),
    y_range_m: Sequence[float] = (-32.0, 32.0),
    candidate_index: int = 0,
) -> Tuple[str, ...]:
    """Render diagnostic arrays and return the emitted panel names.

    Supported field tensors are ``geometry_field [C,Ny,Nx]``, masks
    ``[Ny,Nx]``, and target/prediction maps ``[V,Z,Ny,Nx]``.  Trajectories are
    physical ``[H,3]``/``[M,H,3]`` or normalized actions ``[...,H,4]``.
    """

    x_range = (float(x_range_m[0]), float(x_range_m[1]))
    y_range = (float(y_range_m[0]), float(y_range_m[1]))
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError("BEV ranges must be strictly increasing")
    draft = _physical_trajectory(arrays, "draft", candidate_index)
    final = _physical_trajectory(arrays, "final", candidate_index)

    map_keys = (
        "geometry_field",
        "field_valid_mask",
        "geometry_target_depth",
        "geometry_pred_depth",
        "geometry_target_occupancy",
        "geometry_pred_occupancy",
        "geometry_target_free_space",
        "geometry_pred_free_space",
        "geometry_target_relative",
        "geometry_pred_relative",
    )
    panels = [(name, _spatial_map(arrays[name], name)) for name in map_keys if name in arrays]
    if draft is not None or final is not None or "tube_points" in arrays:
        panels.append(("trajectory_readout", None))
    if not panels:
        raise ValueError(
            "bundle contains no recognized field, trajectory, or tube diagnostics"
        )

    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 5.0 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    extent = (y_range[0], y_range[1], x_range[0], x_range[1])
    for axis, (name, image) in zip(axes_array, panels):
        if image is not None:
            plot = axis.imshow(image, origin="lower", extent=extent, aspect="equal")
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
        _plot_trajectory(axis, draft, label="draft", color="tab:orange")
        _plot_trajectory(axis, final, label="final", color="tab:blue")
        if name == "trajectory_readout" and "tube_points" in arrays:
            tube = np.asarray(arrays["tube_points"], dtype=np.float32)
            if tube.ndim == 4:
                tube = tube[candidate_index]
            if tube.ndim != 3 or tube.shape[-1] != 3:
                raise ValueError("tube_points must have shape [H,P,3] or [M,H,P,3]")
            axis.scatter(
                tube[..., 1].reshape(-1),
                tube[..., 0].reshape(-1),
                s=7,
                alpha=0.55,
                label="tube",
                color="tab:green",
            )
        axis.set_title(name)
        axis.set_xlabel("y left [m]")
        axis.set_ylabel("x forward [m]")
        axis.set_xlim(y_range)
        axis.set_ylim(x_range)
        axis.grid(alpha=0.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
    for axis in axes_array[len(panels) :]:
        axis.set_visible(False)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    figure.savefig(temporary, format="png", dpi=140)
    plt.close(figure)
    os.replace(temporary, output_path)
    return tuple(name for name, _ in panels)


def main() -> None:
    args = _parse_args()
    source = Path(args.input)
    if source.suffix != ".npz":
        raise ValueError("--input must be a .npz file")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    panels = render_field_diagnostics(
        arrays,
        Path(args.output),
        x_range_m=args.x_range_m,
        y_range_m=args.y_range_m,
        candidate_index=args.candidate_index,
    )
    print(f"[field2plan] wrote {args.output}; panels={','.join(panels)}")


if __name__ == "__main__":
    main()
