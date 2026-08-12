#!/usr/bin/env python3
"""Count all visible NAVSIM agents for navtrain sample windows.

The visibility test reuses projection and occlusion helpers from
``tools/mine_critical_agents_navsim.py``. Unlike that mining script, this script
keeps every visible traffic participant and does not score or top-k filter
critical agents.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from mine_critical_agents_navsim import (  # noqa: E402
    ann_get,
    estimate_occlusion_ratios,
    normalize_class,
    project_box_to_camera,
)

DEFAULT_AGENT_CLASSES = ("vehicle", "pedestrian", "bicycle")
DEFAULT_VIEWS = ("CAM_L0", "CAM_F0", "CAM_R0")
DEFAULT_IMAGE_HW = (1120, 1920)

SELECTED_TOKENS: Optional[set[str]] = None
HISTORY_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/workspace/Public_Space/navsim")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--token-file", default="train_meta.json", help="JSON list of navtrain anchor frame tokens.")
    parser.add_argument("--output-dir", default="navsim_dataset/stats")
    parser.add_argument("--history-frames", type=int, default=4, help="Frames per sample window, ending at the navtrain anchor token.")
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--agent-classes", nargs="+", default=list(DEFAULT_AGENT_CLASSES))
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HW[0])
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_HW[1])
    parser.add_argument("--min-bbox-area", type=float, default=256.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.25)
    parser.add_argument("--min-visible-corners", type=int, default=1)
    parser.add_argument("--max-occlusion-ratio", type=float, default=0.6)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_token_file(path: str, max_samples: Optional[int]) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list of tokens")
    if max_samples is not None:
        data = data[:max_samples]
    return {str(item) for item in data}


def init_worker(selected_tokens: Optional[set[str]], history_frames: int) -> None:
    global SELECTED_TOKENS, HISTORY_FRAMES
    SELECTED_TOKENS = selected_tokens
    HISTORY_FRAMES = history_frames


def camera_from_raw(raw_camera: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sensor2lidar_rotation": raw_camera["sensor2lidar_rotation"],
        "sensor2lidar_translation": raw_camera["sensor2lidar_translation"],
        "intrinsics": raw_camera.get("intrinsics", raw_camera.get("cam_intrinsic")),
    }


def agent_key_for_index(anns: Any, idx: int) -> str:
    boxes = ann_get(anns, "gt_boxes", ann_get(anns, "boxes", []))
    track_tokens = list(ann_get(anns, "track_tokens", [None] * len(boxes)))
    instance_tokens = list(ann_get(anns, "instance_tokens", [None] * len(boxes)))
    track_token = track_tokens[idx] if idx < len(track_tokens) else None
    instance_token = instance_tokens[idx] if idx < len(instance_tokens) else None
    return str(track_token or instance_token or f"idx_{idx}")


def count_visible_agents_in_frame(
    frame: Dict[str, Any],
    views: Tuple[str, ...],
    agent_classes: set[str],
    image_hw: Tuple[int, int],
    min_bbox_area: float,
    min_visible_ratio: float,
    min_visible_corners: int,
    max_occlusion_ratio: float,
) -> Dict[str, Any]:
    anns = frame.get("anns", frame.get("annotations", {}))
    boxes = np.asarray(ann_get(anns, "gt_boxes", ann_get(anns, "boxes", [])), dtype=np.float64)
    names = list(ann_get(anns, "gt_names", ann_get(anns, "names", [])))
    raw_cameras = frame.get("cams", frame.get("cameras", {}))

    projections: List[Dict[str, Any]] = []
    class_by_agent: Dict[str, str] = {}
    for idx, box in enumerate(boxes):
        class_name = normalize_class(names[idx] if idx < len(names) else "generic_object")
        if class_name not in agent_classes:
            continue
        if box.shape[0] < 7 or np.any(np.asarray(box[3:6]) <= 0):
            continue
        agent_key = agent_key_for_index(anns, idx)
        class_by_agent[agent_key] = class_name
        for view in views:
            raw_camera = raw_cameras.get(view) or raw_cameras.get(view.lower())
            if raw_camera is None:
                continue
            projection = project_box_to_camera(
                box=box,
                camera=camera_from_raw(raw_camera),
                image_hw=image_hw,
                min_visible_corners=min_visible_corners,
            )
            if projection is None:
                continue
            projection["view"] = view
            projection["agent_key"] = agent_key
            projection["class_name"] = class_name
            projections.append(projection)

    estimate_occlusion_ratios(projections)

    visible_by_view = {view: set() for view in views}
    class_counts_by_view = {view: Counter() for view in views}
    for projection in projections:
        if projection["bbox_area"] < min_bbox_area:
            continue
        if projection["visible_ratio"] < min_visible_ratio:
            continue
        if projection.get("occlusion_ratio", 0.0) > max_occlusion_ratio:
            continue
        view = str(projection["view"])
        agent_key = str(projection["agent_key"])
        visible_by_view[view].add(agent_key)
        class_counts_by_view[view][str(projection["class_name"])] += 1

    visible_union = set().union(*(visible_by_view[view] for view in views))
    class_counts_union = Counter(class_by_agent[key] for key in visible_union if key in class_by_agent)
    return {
        "visible_union": visible_union,
        "visible_by_view": visible_by_view,
        "class_counts_union": class_counts_union,
        "class_counts_by_view": class_counts_by_view,
    }


def selected_anchor_indices(frames: List[Dict[str, Any]]) -> List[int]:
    if SELECTED_TOKENS is None:
        return list(range(len(frames)))
    anchors: List[int] = []
    for idx, frame in enumerate(frames):
        if str(frame.get("token", "")) in SELECTED_TOKENS:
            anchors.append(idx)
    return anchors


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def new_scene_stats(frame: Dict[str, Any], views: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        "scene_name": str(frame.get("scene_name", "")),
        "scene_token": str(frame.get("scene_token", "")),
        "log_name": str(frame.get("log_name", "")),
        "map_location": str(frame.get("map_location", "")),
        "sample_frames": 0,
        "union_counts": [],
        "view_counts": {view: [] for view in views},
        "unique_union": set(),
        "unique_by_view": {view: set() for view in views},
        "class_counts_union": Counter(),
        "class_counts_by_view": {view: Counter() for view in views},
    }


def merge_scene_stats(dst: Dict[str, Dict[str, Any]], src: Dict[str, Dict[str, Any]], views: Tuple[str, ...]) -> None:
    for scene_name, src_stats in src.items():
        if scene_name not in dst:
            dst[scene_name] = {
                **{k: src_stats[k] for k in ("scene_name", "scene_token", "log_name", "map_location")},
                "sample_frames": 0,
                "union_counts": [],
                "view_counts": {view: [] for view in views},
                "unique_union": set(),
                "unique_by_view": {view: set() for view in views},
                "class_counts_union": Counter(),
                "class_counts_by_view": {view: Counter() for view in views},
            }
        dst_stats = dst[scene_name]
        dst_stats["sample_frames"] += src_stats["sample_frames"]
        dst_stats["union_counts"].extend(src_stats["union_counts"])
        dst_stats["unique_union"].update(src_stats["unique_union"])
        dst_stats["class_counts_union"].update(src_stats["class_counts_union"])
        for view in views:
            dst_stats["view_counts"][view].extend(src_stats["view_counts"][view])
            dst_stats["unique_by_view"][view].update(src_stats["unique_by_view"][view])
            dst_stats["class_counts_by_view"][view].update(src_stats["class_counts_by_view"][view])


def process_pkl_file(task: Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[int, int], float, float, int, float]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    pkl_path_str, views, agent_classes_tuple, image_hw, min_bbox_area, min_visible_ratio, min_visible_corners, max_occlusion_ratio = task
    agent_classes = set(agent_classes_tuple)
    with Path(pkl_path_str).open("rb") as f:
        frames = pickle.load(f)

    anchors = selected_anchor_indices(frames)
    if SELECTED_TOKENS is not None:
        anchors = [idx for idx in anchors if idx - HISTORY_FRAMES + 1 >= 0]

    needed_frame_indices = sorted({frame_idx for anchor_idx in anchors for frame_idx in range(max(0, anchor_idx - HISTORY_FRAMES + 1), anchor_idx + 1)})
    visible_cache: Dict[int, Dict[str, Any]] = {}
    for frame_idx in needed_frame_indices:
        visible_cache[frame_idx] = count_visible_agents_in_frame(
            frame=frames[frame_idx],
            views=views,
            agent_classes=agent_classes,
            image_hw=image_hw,
            min_bbox_area=min_bbox_area,
            min_visible_ratio=min_visible_ratio,
            min_visible_corners=min_visible_corners,
            max_occlusion_ratio=max_occlusion_ratio,
        )

    scene_stats: Dict[str, Dict[str, Any]] = {}
    sample_rows: List[Dict[str, Any]] = []
    sample_frame_rows: List[Dict[str, Any]] = []
    skipped_samples = 0

    for anchor_idx in anchors:
        start_idx = anchor_idx - HISTORY_FRAMES + 1
        if start_idx < 0:
            skipped_samples += 1
            continue
        anchor = frames[anchor_idx]
        sample_token = str(anchor.get("token", ""))
        sample_union_counts: List[int] = []
        per_offset_counts: Dict[str, int] = {}
        per_view_observations = Counter()
        sample_unique_union: set[str] = set()

        for offset, frame_idx in enumerate(range(start_idx, anchor_idx + 1)):
            frame = frames[frame_idx]
            visible = visible_cache[frame_idx]
            visible_union = visible["visible_union"]
            sample_union_counts.append(len(visible_union))
            per_offset_counts[f"visible_t{offset}"] = len(visible_union)
            sample_unique_union.update(visible_union)

            scene_name = str(frame.get("scene_name", ""))
            stats = scene_stats.setdefault(scene_name, new_scene_stats(frame, views))
            stats["sample_frames"] += 1
            stats["union_counts"].append(len(visible_union))
            stats["unique_union"].update(visible_union)
            stats["class_counts_union"].update(visible["class_counts_union"])
            row = {
                "sample_token": sample_token,
                "sample_frame_offset": offset,
                "scene_name": scene_name,
                "frame_token": str(frame.get("token", "")),
                "frame_idx": int(frame.get("frame_idx", frame_idx)),
                "raw_frame_index": frame_idx,
                "visible_union": len(visible_union),
            }
            for view in views:
                view_count = len(visible["visible_by_view"][view])
                row[f"visible_{view}"] = view_count
                stats["view_counts"][view].append(view_count)
                stats["unique_by_view"][view].update(visible["visible_by_view"][view])
                stats["class_counts_by_view"][view].update(visible["class_counts_by_view"][view])
                per_view_observations[view] += view_count
            sample_frame_rows.append(row)

        sample_row = {
            "sample_token": sample_token,
            "anchor_scene_name": str(anchor.get("scene_name", "")),
            "anchor_frame_token": str(anchor.get("token", "")),
            "anchor_frame_idx": int(anchor.get("frame_idx", anchor_idx)),
            "history_frames": HISTORY_FRAMES,
            "visible_union_observations": int(sum(sample_union_counts)),
            "visible_union_mean_per_frame": f"{float(np.mean(sample_union_counts)) if sample_union_counts else 0.0:.6f}",
            "visible_union_max_per_frame": int(max(sample_union_counts) if sample_union_counts else 0),
            "visible_union_unique_tracks": len(sample_unique_union),
        }
        sample_row.update(per_offset_counts)
        for view in views:
            sample_row[f"{view}_observations"] = int(per_view_observations[view])
        sample_rows.append(sample_row)

    return scene_stats, sample_rows, sample_frame_rows, len(anchors) - skipped_samples, skipped_samples


def write_outputs(
    output_dir: Path,
    split_name: str,
    history_frames: int,
    views: Tuple[str, ...],
    agent_classes: Tuple[str, ...],
    scene_stats: Dict[str, Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
    sample_frame_rows: List[Dict[str, Any]],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"navsim_{split_name}_front_visible_agents_first{history_frames}"
    scene_path = output_dir / f"{prefix}_by_scene.csv"
    sample_path = output_dir / f"{prefix}_by_sample.csv"
    sample_frame_path = output_dir / f"{prefix}_by_sample_frame.csv"

    scene_fields = [
        "scene_name",
        "scene_token",
        "log_name",
        "map_location",
        "sample_frames",
        "visible_union_observations",
        "visible_union_mean_per_frame",
        "visible_union_p50_per_frame",
        "visible_union_p95_per_frame",
        "visible_union_max_per_frame",
        "visible_union_unique_tracks",
    ]
    for view in views:
        scene_fields.extend([f"{view}_observations", f"{view}_mean_per_frame", f"{view}_max_per_frame", f"{view}_unique_tracks"])
    for class_name in sorted(agent_classes):
        scene_fields.append(f"union_{class_name}_observations")

    with scene_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scene_fields)
        writer.writeheader()
        for scene_name in sorted(scene_stats):
            stats = scene_stats[scene_name]
            union_counts = stats["union_counts"]
            row = {
                "scene_name": stats["scene_name"],
                "scene_token": stats["scene_token"],
                "log_name": stats["log_name"],
                "map_location": stats["map_location"],
                "sample_frames": stats["sample_frames"],
                "visible_union_observations": int(sum(union_counts)),
                "visible_union_mean_per_frame": f"{float(np.mean(union_counts)) if union_counts else 0.0:.6f}",
                "visible_union_p50_per_frame": f"{percentile(union_counts, 50):.6f}",
                "visible_union_p95_per_frame": f"{percentile(union_counts, 95):.6f}",
                "visible_union_max_per_frame": int(max(union_counts) if union_counts else 0),
                "visible_union_unique_tracks": len(stats["unique_union"]),
            }
            for view in views:
                counts = stats["view_counts"][view]
                row[f"{view}_observations"] = int(sum(counts))
                row[f"{view}_mean_per_frame"] = f"{float(np.mean(counts)) if counts else 0.0:.6f}"
                row[f"{view}_max_per_frame"] = int(max(counts) if counts else 0)
                row[f"{view}_unique_tracks"] = len(stats["unique_by_view"][view])
            for class_name in sorted(agent_classes):
                row[f"union_{class_name}_observations"] = int(stats["class_counts_union"][class_name])
            writer.writerow(row)

    sample_fields = [
        "sample_token",
        "anchor_scene_name",
        "anchor_frame_token",
        "anchor_frame_idx",
        "history_frames",
        "visible_union_observations",
        "visible_union_mean_per_frame",
        "visible_union_max_per_frame",
        "visible_union_unique_tracks",
    ] + [f"visible_t{i}" for i in range(history_frames)] + [f"{view}_observations" for view in views]
    with sample_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    sample_frame_fields = ["sample_token", "sample_frame_offset", "scene_name", "frame_token", "frame_idx", "raw_frame_index", "visible_union"] + [f"visible_{view}" for view in views]
    with sample_frame_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_frame_fields)
        writer.writeheader()
        writer.writerows(sample_frame_rows)

    return scene_path, sample_path, sample_frame_path


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    log_dir = Path(args.log_dir) if args.log_dir else data_root / "trainval_navsim_logs" / "trainval"
    pkl_files = sorted(log_dir.glob("*.pkl"))
    if args.max_files is not None:
        pkl_files = pkl_files[: args.max_files]
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found under {log_dir}")

    selected_tokens = load_token_file(args.token_file, args.max_samples) if args.token_file else set()
    init_worker(selected_tokens if selected_tokens else None, args.history_frames)
    views = tuple(str(view) for view in args.views)
    agent_classes = tuple(str(name) for name in args.agent_classes)
    image_hw = (int(args.image_height), int(args.image_width))
    tasks = [
        (
            str(pkl_path),
            views,
            agent_classes,
            image_hw,
            args.min_bbox_area,
            args.min_visible_ratio,
            args.min_visible_corners,
            args.max_occlusion_ratio,
        )
        for pkl_path in pkl_files
    ]

    print(f"log_dir={log_dir}")
    print(f"token_file={args.token_file or '<all raw frames>'} tokens={len(selected_tokens) if selected_tokens else 'all'} history_frames={args.history_frames}")

    scene_stats: Dict[str, Dict[str, Any]] = {}
    sample_rows: List[Dict[str, Any]] = []
    sample_frame_rows: List[Dict[str, Any]] = []
    matched_samples = 0
    skipped_samples = 0

    if args.workers <= 1:
        iterator = map(process_pkl_file, tasks)
        for partial_stats, partial_samples, partial_frames, partial_matched, partial_skipped in tqdm(iterator, total=len(tasks), desc="Counting visible agents"):
            merge_scene_stats(scene_stats, partial_stats, views)
            sample_rows.extend(partial_samples)
            sample_frame_rows.extend(partial_frames)
            matched_samples += partial_matched
            skipped_samples += partial_skipped
    else:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(selected_tokens if selected_tokens else None, args.history_frames)) as pool:
            iterator = pool.imap_unordered(process_pkl_file, tasks)
            for partial_stats, partial_samples, partial_frames, partial_matched, partial_skipped in tqdm(iterator, total=len(tasks), desc="Counting visible agents"):
                merge_scene_stats(scene_stats, partial_stats, views)
                sample_rows.extend(partial_samples)
                sample_frame_rows.extend(partial_frames)
                matched_samples += partial_matched
                skipped_samples += partial_skipped

    sample_rows.sort(key=lambda row: row["sample_token"])
    sample_frame_rows.sort(key=lambda row: (row["sample_token"], int(row["sample_frame_offset"])))
    split_name = Path(args.token_file).stem if args.token_file else "trainval_all"
    scene_path, sample_path, sample_frame_path = write_outputs(
        output_dir=Path(args.output_dir),
        split_name=split_name,
        history_frames=args.history_frames,
        views=views,
        agent_classes=agent_classes,
        scene_stats=scene_stats,
        sample_rows=sample_rows,
        sample_frame_rows=sample_frame_rows,
    )
    print(f"scene_csv={scene_path}")
    print(f"sample_csv={sample_path}")
    print(f"sample_frame_csv={sample_frame_path}")
    print(
        "scenes="
        f"{len(scene_stats)} samples={len(sample_rows)} sample_frame_observations={len(sample_frame_rows)} "
        f"matched_samples={matched_samples} skipped_samples={skipped_samples} files={len(pkl_files)}"
    )


if __name__ == "__main__":
    main()
