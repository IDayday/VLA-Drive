#!/usr/bin/env python3
"""Validate NAVSIM camera/planning coordinates and write projection overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import ImageDraw

from starVLA.dataloader.field2plan_cache import atomic_write_json
from starVLA.dataloader.navsim_dataset import NavSimDataset
from starVLA.model.modules.field2plan.camera_geometry import project_ego_points
from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project an ego BEV grid and future trajectory into NAVSIM cameras."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datalist-path", required=True)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-front-valid-ratio", type=float, default=0.05)
    return parser.parse_args()


def _make_dataset(cfg, args: argparse.Namespace) -> NavSimDataset:
    if os.environ.get("NAVSIM_FEATURE_CACHE_ROOT", "").strip():
        raise RuntimeError(
            "coordinate validation requires source images; unset NAVSIM_FEATURE_CACHE_ROOT"
        )
    cfg.datasets.vla_data.data_root = args.data_root
    cfg.datasets.vla_data.datalist_path = args.datalist_path
    cfg.datasets.vla_data.split = args.split
    cfg.field2plan.proposal.source = "online_debug"
    cfg.field2plan.proposal.cache_dir = None
    cfg.field2plan.proposal.cache_splits = []
    return NavSimDataset(
        datalist_path=args.datalist_path,
        split=args.split,
        video_data_cfg=cfg.datasets.video_data,
        gs_data_cfg=cfg.datasets.gs_data,
        reward_data_cfg=cfg.datasets.reward_data,
        ver_1225=cfg.ver_1225,
        dataset_cfg=cfg.datasets.vla_data,
        all_cfg=cfg,
        data_root=args.data_root,
    )


def _project(points: torch.Tensor, camera: dict):
    intrinsics = torch.from_numpy(camera["intrinsics"])[None].float()
    ego_to_camera = torch.from_numpy(camera["ego_to_camera"])[None].float()
    image_hw = torch.from_numpy(camera["image_hw"])[None].float()
    return project_ego_points(
        points[None].float(), intrinsics, ego_to_camera, image_hw
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    dataset = _make_dataset(cfg, args)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} is outside dataset size {len(dataset)}")
    sample = dataset[args.index]
    camera = sample["camera"]
    if camera["ego_to_camera"] is None:
        raise RuntimeError(
            f"unresolved camera transform: {camera['transform_status']}"
        )

    x = torch.linspace(2.0, 50.0, 13)
    y = torch.linspace(-20.0, 20.0, 11)
    grid_xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)
    grid_points = torch.cat((grid_xy, torch.zeros(len(grid_xy), 1)), dim=-1)
    grid_pixels, grid_valid, grid_depth = _project(grid_points, camera)

    action = torch.as_tensor(sample["action"], dtype=torch.float32)
    route = TrajectoryCodec().decode_action(action)
    if not isinstance(route, torch.Tensor):
        raise TypeError("trajectory codec returned a non-tensor")
    route_points = torch.cat((route[:, :2], torch.zeros(len(route), 1)), dim=-1)
    route_pixels, route_valid, _ = _project(route_points, camera)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = camera["view_names"]
    for view_index, (name, image) in enumerate(zip(names, sample["image"])):
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for pixel, valid, depth in zip(
            grid_pixels[0, view_index],
            grid_valid[0, view_index],
            grid_depth[0, view_index],
        ):
            if bool(valid):
                radius = 3 if float(depth) < 20.0 else 2
                u, v = (float(value) for value in pixel)
                draw.ellipse(
                    (u - radius, v - radius, u + radius, v + radius),
                    fill=(0, 220, 0),
                )
        previous = None
        for pixel, valid in zip(
            route_pixels[0, view_index], route_valid[0, view_index]
        ):
            if bool(valid):
                current = tuple(float(value) for value in pixel)
                if previous is not None:
                    draw.line((previous, current), fill=(255, 40, 40), width=4)
                draw.ellipse(
                    (
                        current[0] - 4,
                        current[1] - 4,
                        current[0] + 4,
                        current[1] + 4,
                    ),
                    fill=(255, 40, 40),
                )
                previous = current
        temporary = output_dir / f"{name}.png.tmp-{os.getpid()}"
        overlay.save(temporary, format="PNG")
        os.replace(temporary, output_dir / f"{name}.png")

    grid_ratios = grid_valid[0].float().mean(dim=-1)
    route_ratios = route_valid[0].float().mean(dim=-1)
    summary = {
        "token": sample["token"],
        "split": args.split,
        "index": args.index,
        "frame_index": int(camera["frame_index"]),
        "coordinate_frame": "planning_ego_at_t3_x_forward_y_left_z_up",
        "transform_status": camera["transform_status"],
        "view_names": names,
        "grid_valid_ratio": {
            name: float(grid_ratios[index]) for index, name in enumerate(names)
        },
        "trajectory_valid_ratio": {
            name: float(route_ratios[index]) for index, name in enumerate(names)
        },
        "action_shape": list(action.shape),
        "overlay_legend": {"green": "ego BEV anchors", "red": "GT route sanity only"},
    }
    atomic_write_json(output_dir / "coordinate_summary.json", summary)
    front_ratio = summary["grid_valid_ratio"]["cam_f0"]
    print(OmegaConf.to_yaml(OmegaConf.create(summary), resolve=True))
    if front_ratio < args.min_front_valid_ratio:
        raise RuntimeError(
            f"front projection valid ratio {front_ratio:.4f} is below "
            f"{args.min_front_valid_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
