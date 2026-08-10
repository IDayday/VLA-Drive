#!/usr/bin/env python3
"""Visualize categorized BEV diagnostics for zero-score NAVSIM scenes.

Example command:

    python tools/visualize_zero_score_bev.py \
      --eval-csv /mnt/workspace/VLA-Drive/navsim_exp/eval_v1.1/drivedreamer-policy/test_minimal_prompt2_88.89/2026.08.06.20.15.18.csv \
      --pred-dir /mnt/workspace/VLA-Drive/navsim_planning_results/pytorch_model.pt/test_minimal_prompt2 \
      --critical-agents-dir /mnt/workspace/VLA-Drive/navsim_exp/eval_v1.1/drivedreamer-policy/test_minimal_prompt2_88.89/critical_agents_zero_score/test \
      --output-dir /mnt/workspace/VLA-Drive/navsim_exp/eval_v1.1/drivedreamer-policy/test_minimal_prompt2_88.89/zero_score_bev_diagnostics_v2/test \
      --split test \
      --data-root /mnt/data/navsim \
      --log-dir /mnt/data/navsim/test_navsim_logs/test \
      --sensor-dir /mnt/data/navsim/test_sensor_blobs/test \
      --tokens-file /mnt/workspace/VLA-Drive/navsim_exp/eval_v1.1/drivedreamer-policy/test_minimal_prompt2_88.89/zero_score_tokens.txt \
      --overwrite

Important arguments:
    --eval-csv: NAVSIM eval CSV. Rows with valid=True and score=0 are selected.
    --pred-dir: Directory containing per-token predicted trajectory .npy files.
        Each file is expected to be named <token>.npy and store [x, y, heading].
    --critical-agents-dir: Optional directory with critical-agent JSON sidecars from
        mine_critical_agents_navsim.py. Collision labels prefer these agents.
    --output-dir: Destination root. Images are written under collision/ and
        off_drivable/ subdirectories. A scene can appear in both categories.
    --split: NAVSIM split name, for example test or mini.
    --data-root: Root containing logs, sensor blobs, and maps.
    --log-dir / --sensor-dir: Explicit log and sensor directories when defaults do
        not match the local data layout.
    --map-root: Optional nuPlan/NAVSIM maps root. Defaults to <data-root>/maps, then
        /mnt/data/navsim/maps, then /mnt/data/nuplan/maps.
    --tokens-file: Optional newline-separated token allowlist, commonly the
        zero_score_tokens.txt produced from the eval CSV.
    --max-samples: Optional cap for debugging.
    --current-frame-index: History frame used as the current BEV frame. Defaults to 3.
    --x-min/--x-max/--y-min/--y-max: BEV axis limits in ego-frame meters.
    --overwrite: Regenerate existing images.

Outputs:
    collision/<token>.png marks the predicted collision/TTC point and labels the
    nearest attributed agent with rank/class/track token when available.
    off_drivable/<token>.png marks the first predicted point outside drivable-area
    polygons when map data is available, otherwise a geometric approximation.
    The BEV background uses NAVSIM's map/annotation visualization API.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NAVSIM_PROCESS_ROOT = REPO_ROOT / "navsim_data_process"
NAVSIM_ROOT = REPO_ROOT / "navsim"
for module_root in (NAVSIM_PROCESS_ROOT, NAVSIM_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

CURRENT_FRAME_INDEX = 3


def rotation_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def box_corners_xy(box: Iterable[float]) -> np.ndarray:
    box = np.asarray(list(box), dtype=np.float64)
    x, y, _, length, width, _, yaw = box[:7]
    local = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    return local @ rotation_matrix(float(yaw)).T + np.array([x, y], dtype=np.float64)


def ann_get(annotations: Any, key: str, default: Any = None) -> Any:
    if isinstance(annotations, dict):
        return annotations.get(key, default)
    return getattr(annotations, key, default)


def ego_pose(status: Any) -> np.ndarray:
    if isinstance(status, dict):
        value = status["ego_pose"]
    else:
        value = getattr(status, "ego_pose")
    return np.asarray(value, dtype=np.float64)


def global_to_local(points_xy: np.ndarray, ego_pose_value: np.ndarray) -> np.ndarray:
    return (points_xy - ego_pose_value[:2]) @ rotation_matrix(float(ego_pose_value[2]))


def ego_future_path_current_frame(frame_data: List[Dict[str, Any]], current_idx: int) -> np.ndarray:
    poses = np.stack([ego_pose(frame["ego_status"]) for frame in frame_data], axis=0)
    current_pose = poses[current_idx]
    future_global = poses[current_idx + 1 :, :2]
    if len(future_global) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return global_to_local(future_global, current_pose)


def resolve_split_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = Path(args.data_root).resolve()
    split = args.split
    log_dir = Path(args.log_dir).resolve() if args.log_dir else data_root / f"{split}_navsim_logs" / split
    sensor_dir = Path(args.sensor_dir).resolve() if args.sensor_dir else data_root / f"{split}_sensor_blobs" / split
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Missing log dir: {log_dir}")
    if not sensor_dir.is_dir():
        raise FileNotFoundError(f"Missing sensor dir: {sensor_dir}")
    return log_dir, sensor_dir


def prepare_vlmnavsim_root(args: argparse.Namespace) -> Path:
    log_dir, sensor_dir = resolve_split_dirs(args)
    loader_split = "trainval" if args.split == "train" else args.split
    runtime_root = Path("/tmp") / "vla_drive_navsim_bev" / args.split
    expected_log = runtime_root / "navsim_logs" / loader_split
    expected_sensor = runtime_root / "sensor_blobs" / loader_split
    expected_log.parent.mkdir(parents=True, exist_ok=True)
    expected_sensor.parent.mkdir(parents=True, exist_ok=True)
    for link, target in ((expected_log, log_dir), (expected_sensor, sensor_dir)):
        if link.is_symlink() or link.exists():
            if link.resolve() == target:
                continue
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target, target_is_directory=True)
    print(f"[zero-bev] log_dir={log_dir}")
    print(f"[zero-bev] sensor_dir={sensor_dir}")
    return runtime_root


def patch_hydra_instantiate() -> None:
    import hydra.utils

    original_instantiate = hydra.utils.instantiate

    def instantiate_with_scene_filter_defaults(*inst_args, **inst_kwargs):
        obj = original_instantiate(*inst_args, **inst_kwargs)
        if obj.__class__.__name__ == "SceneFilter":
            if not hasattr(obj, "include_synthetic_scenes"):
                setattr(obj, "include_synthetic_scenes", False)
            if not hasattr(obj, "synthetic_scene_tokens"):
                setattr(obj, "synthetic_scene_tokens", None)
        return obj

    hydra.utils.instantiate = instantiate_with_scene_filter_defaults


def is_collision_row(row: pd.Series) -> bool:
    return any(col in row and float(row[col]) < 0.999 for col in ("no_at_fault_collisions", "time_to_collision_within_bound"))


def is_off_drivable_row(row: pd.Series) -> bool:
    return "drivable_area_compliance" in row and float(row["drivable_area_compliance"]) < 0.999


def failure_reasons(row: pd.Series, diagnostic_type: str) -> List[str]:
    if diagnostic_type == "collision":
        checks = [("collision", "no_at_fault_collisions"), ("ttc", "time_to_collision_within_bound")]
    elif diagnostic_type == "off_drivable":
        checks = [("off_drivable", "drivable_area_compliance")]
    else:
        checks = []
    reasons = []
    for name, col in checks:
        if col in row and float(row[col]) < 0.999:
            reasons.append(f"{name}={float(row[col]):.2f}")
    return reasons


def annotation_agents(annotations: Any) -> List[Dict[str, Any]]:
    boxes = np.asarray(ann_get(annotations, "boxes", []), dtype=np.float64)
    names = list(ann_get(annotations, "names", []))
    track_tokens = list(ann_get(annotations, "track_tokens", []))
    agents: List[Dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        class_name = str(names[idx]) if idx < len(names) else "agent"
        track_token = str(track_tokens[idx]) if idx < len(track_tokens) else ""
        agents.append(
            {
                "valid": True,
                "rank": idx,
                "class_name": class_name,
                "track_token": track_token,
                "box_ego": box.tolist(),
                "source": "annotation",
            }
        )
    return agents


def collision_agent_candidates(annotations: Any, critical_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    critical_agents = [agent for agent in (critical_payload or {}).get("critical_agents", []) if agent.get("valid", False)]
    if critical_agents:
        return critical_agents
    return annotation_agents(annotations)


def nearest_pred_to_agents(pred_xy: np.ndarray, agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    for agent in agents:
        if not agent.get("valid", False):
            continue
        center = np.asarray(agent["box_ego"][:2], dtype=np.float64)
        if len(pred_xy) == 0:
            continue
        distances = np.linalg.norm(pred_xy - center[None, :], axis=1)
        idx = int(np.argmin(distances))
        record = {"point": pred_xy[idx], "agent": agent, "distance": float(distances[idx]), "step": idx + 1}
        if best is None or record["distance"] < best["distance"]:
            best = record
    return best


def local_to_global(points_xy: np.ndarray, ego_pose_value: np.ndarray) -> np.ndarray:
    return points_xy @ rotation_matrix(float(ego_pose_value[2])).T + ego_pose_value[:2]


def first_off_drivable_point(scene: Any, frame_idx: int, pred_xy: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(pred_xy) == 0 or scene is None:
        return None
    try:
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point

        origin_pose = np.asarray(scene.frames[frame_idx].ego_status.ego_pose, dtype=np.float64)
        pred_global = local_to_global(pred_xy, origin_pose)
        map_objects = scene.map_api.get_proximal_map_objects(
            point=StateSE2(*origin_pose).point,
            radius=90.0,
            layers=[SemanticMapLayer.DRIVABLE_AREA],
        )
        polygons = [obj.polygon for obj in map_objects.get(SemanticMapLayer.DRIVABLE_AREA, [])]
        if not polygons:
            return None
        for idx, point_xy in enumerate(pred_global):
            point = Point(float(point_xy[0]), float(point_xy[1]))
            if not any(poly.contains(point) or poly.touches(point) for poly in polygons):
                return {"point": pred_xy[idx], "step": idx + 1, "method": "map"}
    except Exception:
        return None
    return None


def diagnostic_points(
    row: pd.Series,
    pred_xy: np.ndarray,
    annotations: Any,
    critical_payload: Optional[Dict[str, Any]],
    scene: Any,
    frame_idx: int,
    diagnostic_type: str,
) -> List[Dict[str, Any]]:
    points = []
    if diagnostic_type == "collision" and is_collision_row(row):
        nearest = nearest_pred_to_agents(pred_xy, collision_agent_candidates(annotations, critical_payload))
        if nearest is not None:
            agent = nearest["agent"]
            track_token = str(agent.get("track_token", ""))
            track_short = f" track={track_token[:8]}" if track_token else ""
            rank_label = f"#{agent.get('rank', '?')}" if agent.get("source") != "annotation" else f"ann[{agent.get('rank', '?')}]"
            label = (
                f"collision/TTC with {rank_label} {agent.get('class_name', 'agent')}"
                f"{track_short} d={nearest['distance']:.1f}m t+{nearest['step'] * 0.5:.1f}s"
            )
            points.append({"xy": nearest["point"], "label": label, "color": "#d62728"})
    elif diagnostic_type == "off_drivable" and is_off_drivable_row(row) and len(pred_xy):
        off_point = first_off_drivable_point(scene, frame_idx, pred_xy)
        if off_point is None:
            idx = int(np.argmax(np.abs(pred_xy[:, 1])))
            off_point = {"point": pred_xy[idx], "step": idx + 1, "method": "approx"}
        label = f"off-drivable t+{off_point['step'] * 0.5:.1f}s ({off_point['method']})"
        points.append({"xy": off_point["point"], "label": label, "color": "#9467bd"})
    return points


def load_lightweight_scene(dataset: Any, token: str) -> Optional[Any]:
    """Load only map, ego status, and annotations for BEV rendering."""
    try:
        import navsim.common.dataclasses as navsim_dataclasses
        if "NUPLAN_MAPS_ROOT" in os.environ:
            navsim_dataclasses.NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
        from navsim.common.dataclasses import Camera, Cameras, Frame, Lidar, Scene, SceneMetadata

        loader = dataset.navsim._scene_loader
        if token in loader.synthetic_scenes:
            return loader.get_scene_from_token(token)
        scene_dict_list = loader.scene_frames_dicts[token]
        num_history_frames = loader._scene_filter.num_history_frames
        num_future_frames = loader._scene_filter.num_future_frames
        current_frame = scene_dict_list[num_history_frames - 1]
        scene_metadata = SceneMetadata(
            log_name=current_frame["log_name"],
            scene_token=current_frame["scene_token"],
            map_name=current_frame["map_location"],
            initial_token=current_frame["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        empty_cameras = Cameras(
            cam_f0=Camera(),
            cam_l0=Camera(),
            cam_l1=Camera(),
            cam_l2=Camera(),
            cam_r0=Camera(),
            cam_r1=Camera(),
            cam_r2=Camera(),
            cam_b0=Camera(),
        )
        frames = []
        for scene_frame in scene_dict_list:
            frames.append(
                Frame(
                    token=scene_frame["token"],
                    timestamp=scene_frame["timestamp"],
                    roadblock_ids=scene_frame["roadblock_ids"],
                    traffic_lights=scene_frame["traffic_lights"],
                    annotations=Scene._build_annotations(scene_frame),
                    ego_status=Scene._build_ego_status(scene_frame),
                    lidar=Lidar(),
                    cameras=empty_cameras,
                )
            )
        return Scene(scene_metadata=scene_metadata, map_api=Scene._build_map_api(scene_metadata.map_name), frames=frames)
    except Exception as exc:
        print(f"[zero-bev] could not build lightweight scene for {token}: {exc}")
        return None


def draw_bev(
    scene: Any,
    container: Dict[str, Any],
    pred: np.ndarray,
    row: pd.Series,
    critical_payload: Optional[Dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
    diagnostic_type: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_idx = min(args.current_frame_index, len(container["frame_data"]) - 1)
    frame = container["frame_data"][frame_idx]
    annotations = frame["annotations"]
    gt_xy = ego_future_path_current_frame(container["frame_data"], frame_idx)
    pred_xy = np.asarray(pred[:, :2], dtype=np.float64)
    agents = (critical_payload or {}).get("critical_agents", [])

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.set_facecolor("white")

    used_navsim_bev = False
    if scene is not None:
        try:
            from navsim.visualization.bev import add_configured_bev_on_ax

            add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
            used_navsim_bev = True
        except Exception as exc:
            print(f"[zero-bev] fallback without map for {row['token']}: {exc}")

    if not used_navsim_bev:
        boxes = np.asarray(ann_get(annotations, "boxes", []), dtype=np.float64)
        names = list(ann_get(annotations, "names", []))
        for idx, box in enumerate(boxes):
            corners = box_corners_xy(box)
            closed = np.vstack([corners, corners[0]])
            cls = str(names[idx]) if idx < len(names) else "agent"
            color = "#2ca02c" if cls == "pedestrian" else "#17becf" if cls == "bicycle" else "#9e9e9e"
            ax.fill(closed[:, 1], closed[:, 0], color=color, alpha=0.12, zorder=2)
            ax.plot(closed[:, 1], closed[:, 0], color=color, linewidth=0.7, alpha=0.75, zorder=3)
        ego = np.array([[2.2, 1.0], [2.2, -1.0], [-2.2, -1.0], [-2.2, 1.0], [2.2, 1.0]])
        ax.plot(ego[:, 1], ego[:, 0], color="black", linewidth=2.0, zorder=6, label="ego")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.set_xlabel("y left (m)")
    ax.set_ylabel("x forward (m)")
    ax.set_xlim(args.y_max, args.y_min)
    ax.set_ylim(args.x_min, args.x_max)

    if len(gt_xy):
        gt = np.vstack([np.zeros((1, 2)), gt_xy])
        ax.plot(gt[:, 1], gt[:, 0], color="#1f77b4", linewidth=2.2, marker="o", markersize=3, label="GT", zorder=7)
    if len(pred_xy):
        pred_plot = np.vstack([np.zeros((1, 2)), pred_xy])
        ax.plot(pred_plot[:, 1], pred_plot[:, 0], color="#d62728", linewidth=2.2, marker="x", markersize=4, label="prediction", zorder=8)

    colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for agent in agents:
        if not agent.get("valid", False):
            continue
        rank = int(agent["rank"])
        color = colors[rank % len(colors)]
        corners = box_corners_xy(agent["box_ego"])
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 1], closed[:, 0], color=color, linewidth=2.0, zorder=9)
        ax.fill(corners[:, 1], corners[:, 0], color=color, alpha=0.18, zorder=4)
        cx, cy = float(agent["box_ego"][0]), float(agent["box_ego"][1])
        ax.text(cy, cx, f"#{rank} {agent['class_name']} {agent['score']:.2f}", color=color, fontsize=7, weight="bold", zorder=10)

    for point in diagnostic_points(row, pred_xy, annotations, critical_payload, scene, frame_idx, diagnostic_type):
        xy = np.asarray(point["xy"], dtype=np.float64)
        marker = "X" if diagnostic_type == "collision" else "*"
        ax.scatter([xy[1]], [xy[0]], marker=marker, s=190, color=point["color"], edgecolor="black", linewidth=0.8, zorder=12)
        ax.text(xy[1], xy[0] + 1.2, point["label"], color=point["color"], fontsize=8, weight="bold", zorder=13)

    reasons = failure_reasons(row, diagnostic_type)
    title = f"{diagnostic_type} | {row['token']} score={float(row['score']):.2f}"
    if reasons:
        title += " | " + "; ".join(reasons)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--critical-agents-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="/mnt/data/navsim")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--sensor-dir", default=None)
    parser.add_argument("--map-root", default=None, help="nuPlan/NAVSIM maps root; defaults to <data-root>/maps if present")
    parser.add_argument("--tokens-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--current-frame-index", type=int, default=CURRENT_FRAME_INDEX)
    parser.add_argument("--x-min", type=float, default=-15.0)
    parser.add_argument("--x-max", type=float, default=65.0)
    parser.add_argument("--y-min", type=float, default=-35.0)
    parser.add_argument("--y-max", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_map_root(args: argparse.Namespace) -> Optional[Path]:
    candidates = []
    if args.map_root:
        candidates.append(Path(args.map_root))
    candidates.extend([Path(args.data_root) / "maps", Path("/mnt/data/navsim/maps"), Path("/mnt/data/nuplan/maps")])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def main() -> None:
    args = parse_args()
    map_root = resolve_map_root(args)
    if map_root is not None:
        os.environ["NUPLAN_MAPS_ROOT"] = str(map_root)
        os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
        print(f"[zero-bev] map_root={map_root}")
    else:
        print("[zero-bev] map_root not found; BEV map layer may fall back to annotations only")
    os.environ["OPENSCENE_DATA_ROOT"] = str(prepare_vlmnavsim_root(args))
    patch_hydra_instantiate()
    from data_engine.datasets.navsim import dataset_navsim as dataset_navsim_module

    # Ensure dataset_navsim uses the patched instantiate imported at module scope.
    import hydra.utils

    dataset_navsim_module.instantiate = hydra.utils.instantiate
    dataset = dataset_navsim_module.VLMNavsim(mode=args.split)
    token_to_index = {token: idx for idx, token in enumerate(dataset.navsim._scene_loader.tokens)}

    df = pd.read_csv(args.eval_csv)
    df = df[(df["valid"] == True) & (df["score"].astype(float) == 0.0)].copy()
    if args.tokens_file:
        tokens = [line.strip() for line in Path(args.tokens_file).read_text().splitlines() if line.strip()]
        df = df[df["token"].astype(str).isin(tokens)]
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    pred_dir = Path(args.pred_dir)
    critical_dir = Path(args.critical_agents_dir) if args.critical_agents_dir else None
    output_dir = Path(args.output_dir)

    written = 0
    missing = []
    from tqdm import tqdm

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering zero-score BEV"):
        token = str(row["token"])
        if token not in token_to_index:
            missing.append(token)
            continue
        pred_path = pred_dir / f"{token}.npy"
        if not pred_path.is_file():
            missing.append(token)
            continue
        diagnostic_types = []
        if is_collision_row(row):
            diagnostic_types.append("collision")
        if is_off_drivable_row(row):
            diagnostic_types.append("off_drivable")
        if not diagnostic_types:
            continue

        output_paths = [output_dir / diagnostic_type / f"{token}.png" for diagnostic_type in diagnostic_types]
        if all(output_path.exists() for output_path in output_paths) and not args.overwrite:
            continue
        container = dataset.get_container_in(token_to_index[token])
        scene = load_lightweight_scene(dataset, token)
        pred = np.load(pred_path)
        critical_payload = None
        if critical_dir is not None:
            critical_path = critical_dir / f"{token}.json"
            if critical_path.is_file():
                with critical_path.open("r", encoding="utf-8") as stream:
                    critical_payload = json.load(stream)
        for diagnostic_type, output_path in zip(diagnostic_types, output_paths):
            if output_path.exists() and not args.overwrite:
                continue
            draw_bev(scene, container, pred, row, critical_payload, output_path, args, diagnostic_type)
            written += 1

    print(f"wrote BEV diagnostics: {written} -> {output_dir} (subdirs: collision, off_drivable)")
    if missing:
        missing_path = output_dir / "missing_tokens.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"missing tokens/files: {len(missing)} written to {missing_path}")


if __name__ == "__main__":
    main()
