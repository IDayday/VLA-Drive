#!/usr/bin/env python3
"""Evaluate whether top query-DINO response regions recall teacher agent boxes.

The default metric first matches unordered agent queries to teacher agents with
Hungarian assignment on predicted-vs-teacher DINO features, then evaluates if the
query heatmap response box recalls the matched teacher bbox.

Example:
  BASE_VLM=/mnt/workspace/VLA-Drive/weights/derived/Qwen3-VL-2B-WorldAction \
  AGENT_QUERY_COUNT=4 \
  python3 tools/eval_agent_query_heatmap_recall.py \
    --ckpt-dir /mnt/workspace/VLA-Drive/navsim_exp/0808_11-agent-action-lr1e5-16g-bz_2-ga_1-train \
    --datalist-path /mnt/workspace/VLA-Drive/test_meta.json \
    --split test \
    --agent-cache-root /mnt/workspace/VLA-Drive/navsim_feature_cache/agent_dino_vits14_test_top4_multiview \
    --sensor-dir /mnt/data/navsim/trainval_sensor_blobs/trainval \
    --output-json /mnt/workspace/VLA-Drive/navsim_exp/0808_11-agent-action-lr1e5-16g-bz_2-ga_1-train/agent_query_heatmap_recall_test_top_component.json \
    --max-samples 64
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import VLAAgent, to_device
from starVLA.dataloader.navsim_dataset import collate_fn
from tools.visualize_agent_query_dino_heatmap import (
    DinoPatchHeatmap,
    build_dataset,
    decode_metadata,
    extract_agent_query_predictions,
    resolve_image_path,
    select_projections,
    top_response_component_bbox,
)


def linear_sum_assignment(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    try:
        from scipy.optimize import linear_sum_assignment as scipy_lsa

        rows, cols = scipy_lsa(cost)
        return list(map(int, rows)), list(map(int, cols))
    except Exception:
        rows_n, cols_n = cost.shape
        if rows_n == 0 or cols_n == 0:
            return [], []
        if rows_n <= cols_n:
            best = None
            best_cols = None
            for cols in itertools.permutations(range(cols_n), rows_n):
                score = sum(float(cost[row, col]) for row, col in enumerate(cols))
                if best is None or score < best:
                    best = score
                    best_cols = cols
            return list(range(rows_n)), list(best_cols or [])
        best = None
        best_rows = None
        for rows in itertools.permutations(range(rows_n), cols_n):
            score = sum(float(cost[row, col]) for col, row in enumerate(rows))
            if best is None or score < best:
                best = score
                best_rows = rows
        return list(best_rows or []), list(range(cols_n))


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area([ix1, iy1, ix2, iy2])
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_in_box(center_box: Iterable[float], target_box: Iterable[float]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in center_box]
    tx1, ty1, tx2, ty2 = [float(v) for v in target_box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return tx1 <= cx <= tx2 and ty1 <= cy <= ty2


def valid_teacher_indices(payload: dict, agents: Sequence[dict]) -> List[int]:
    feature_count = int(payload["agent_features"].shape[0]) if "agent_features" in payload else len(agents)
    count = min(feature_count, len(agents))
    mask = payload.get("agent_valid_mask")
    if mask is None:
        return list(range(count))
    mask = mask[:count].detach().cpu().bool().tolist() if torch.is_tensor(mask) else list(mask)[:count]
    return [idx for idx, is_valid in enumerate(mask) if bool(is_valid)]


def compute_response_metrics(dino: DinoPatchHeatmap, image_path: str, query_feature: torch.Tensor, bbox: Iterable[float], top_patch_percent: float) -> Tuple[Optional[List[int]], float, bool]:
    image = Image.open(image_path).convert("RGB")
    heatmap = dino.heatmap(image, query_feature)
    response_bbox = top_response_component_bbox(heatmap, top_patch_percent)
    if response_bbox is None:
        return None, 0.0, False
    iou = box_iou(response_bbox, bbox)
    return response_bbox, iou, center_in_box(response_bbox, bbox)


def feature_match_pairs(pred_features: torch.Tensor, payload: dict, query_indices: Sequence[int], teacher_indices: Sequence[int]) -> List[Tuple[int, int, float]]:
    teacher_features = payload["agent_features"].detach().cpu().float()
    q = pred_features[list(query_indices)].float()
    t = teacher_features[list(teacher_indices)].float()
    cost = F.smooth_l1_loss(q[:, None, :].expand(-1, t.shape[0], -1), t[None, :, :].expand(q.shape[0], -1, -1), reduction="none").mean(dim=-1)
    rows, cols = linear_sum_assignment(cost.numpy())
    return [(int(query_indices[row]), int(teacher_indices[col]), float(cost[row, col])) for row, col in zip(rows, cols)]


def ordered_match_pairs(query_indices: Sequence[int], teacher_indices: Sequence[int]) -> List[Tuple[int, int, float]]:
    count = min(len(query_indices), len(teacher_indices))
    return [(int(query_indices[i]), int(teacher_indices[i]), 0.0) for i in range(count)]


def oracle_iou_match_pairs(dino: DinoPatchHeatmap, pred_features: torch.Tensor, agents: Sequence[dict], query_indices: Sequence[int], teacher_indices: Sequence[int], sensor_dir: Optional[str], views: str, top_patch_percent: float) -> Tuple[List[Tuple[int, int, float]], dict]:
    cost = np.ones((len(query_indices), len(teacher_indices)), dtype=np.float32)
    cache = {}
    for q_pos, query_index in enumerate(query_indices):
        for t_pos, teacher_index in enumerate(teacher_indices):
            agent_meta = agents[teacher_index]
            projections = select_projections(agent_meta, views)
            best = (None, 0.0, False, "")
            for projection_index, projection in enumerate(projections):
                bbox = projection.get("bbox_xyxy") or agent_meta.get("bbox_xyxy")
                if bbox is None:
                    continue
                view = str(projection.get("view", agent_meta.get("view", f"view{projection_index}")))
                image_path = resolve_image_path(str(projection.get("image_path", agent_meta.get("image_path", ""))), sensor_dir)
                response_bbox, iou, center_hit = compute_response_metrics(dino, image_path, pred_features[query_index], bbox, top_patch_percent)
                if iou >= best[1]:
                    best = (response_bbox, iou, center_hit, view)
            cache[(int(query_index), int(teacher_index))] = best
            cost[q_pos, t_pos] = -float(best[1])
    rows, cols = linear_sum_assignment(cost)
    return [(int(query_indices[row]), int(teacher_indices[col]), float(cost[row, col])) for row, col in zip(rows, cols)], cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--datalist-path", default="train_meta.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default="navsim_dataset")
    parser.add_argument("--agent-cache-root", required=True)
    parser.add_argument("--sensor-dir", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-forward-mode", default="auto", choices=("auto", "legacy", "optimized"))
    parser.add_argument("--dino-backbone", default="dinov2_vits14")
    parser.add_argument("--dino-cache-dir", default="/mnt/workspace/VLA-Drive/weights/derived")
    parser.add_argument("--query-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--views", choices=("best", "all"), default="best")
    parser.add_argument("--top-patch-percent", type=float, default=5.0)
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--match-strategy", choices=("feature", "ordered", "oracle_iou"), default="feature")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["NAVSIM_AGENT_DINO_CACHE_ROOT"] = args.agent_cache_root
    os.environ.setdefault("NAVSIM_AGENT_DINO_CACHE_STRICT", "1")
    os.environ["NAVSIM_USE_FEATURE_CACHE"] = "0"
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)

    agent = VLAAgent(args.ckpt_dir, device=args.device, qwen_forward_mode=args.qwen_forward_mode)
    dataset = build_dataset(agent, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False)
    dino = DinoPatchHeatmap(args.dino_backbone, args.dino_cache_dir, agent.device)

    records: List[dict] = []
    totals = {f"iou@{thr:g}": 0 for thr in args.iou_thresholds}
    center_hits = 0
    valid = 0
    missing_response = 0

    for batch in tqdm(loader, desc="agent-query heatmap recall"):
        examples = to_device(batch, agent.device)
        pred_features, _ = extract_agent_query_predictions(agent.model, examples)
        pred_features = pred_features.detach().cpu()
        for batch_index, example in enumerate(examples):
            payload = example.get("agent_dino_feature_cache")
            if payload is None or "agent_features" not in payload:
                continue
            metadata = decode_metadata(payload)
            token = str(example.get("token", metadata.get("token", f"sample_{valid}")))
            agents = metadata.get("agents", [])
            q_indices = [idx for idx in args.query_indices if idx < pred_features.shape[1]]
            t_indices = valid_teacher_indices(payload, agents)
            if not q_indices or not t_indices:
                continue

            oracle_cache = {}
            if args.match_strategy == "ordered":
                pairs = ordered_match_pairs(q_indices, t_indices)
            elif args.match_strategy == "oracle_iou":
                pairs, oracle_cache = oracle_iou_match_pairs(dino, pred_features[batch_index], agents, q_indices, t_indices, args.sensor_dir, args.views, args.top_patch_percent)
            else:
                pairs = feature_match_pairs(pred_features[batch_index], payload, q_indices, t_indices)

            for query_index, teacher_index, match_cost in pairs:
                agent_meta = agents[teacher_index]
                for projection_index, projection in enumerate(select_projections(agent_meta, args.views)):
                    bbox = projection.get("bbox_xyxy") or agent_meta.get("bbox_xyxy")
                    if bbox is None:
                        continue
                    view = str(projection.get("view", agent_meta.get("view", f"view{projection_index}")))
                    if args.match_strategy == "oracle_iou" and (query_index, teacher_index) in oracle_cache:
                        response_bbox, iou, center_hit, cached_view = oracle_cache[(query_index, teacher_index)]
                        if cached_view and cached_view != view:
                            continue
                    else:
                        image_path = resolve_image_path(str(projection.get("image_path", agent_meta.get("image_path", ""))), args.sensor_dir)
                        response_bbox, iou, center_hit = compute_response_metrics(dino, image_path, pred_features[batch_index, query_index], bbox, args.top_patch_percent)

                    valid += 1
                    if response_bbox is None:
                        missing_response += 1
                    center_hits += int(center_hit)
                    for thr in args.iou_thresholds:
                        totals[f"iou@{thr:g}"] += int(iou >= thr)
                    records.append({
                        "token": token,
                        "query_index": int(query_index),
                        "teacher_index": int(teacher_index),
                        "match_strategy": args.match_strategy,
                        "match_cost": float(match_cost),
                        "view": view,
                        "rank": agent_meta.get("rank", teacher_index),
                        "class_name": agent_meta.get("class_name", ""),
                        "gt_bbox": list(map(float, bbox)),
                        "response_bbox": list(map(float, response_bbox)) if response_bbox is not None else None,
                        "iou": float(iou),
                        "center_in_gt": bool(center_hit),
                    })
        if args.max_samples is not None and len(records) >= args.max_samples * max(1, len(args.query_indices)):
            break

    summary = {
        "ckpt_dir": args.ckpt_dir,
        "agent_cache_root": args.agent_cache_root,
        "samples_limit": args.max_samples,
        "views": args.views,
        "top_patch_percent": args.top_patch_percent,
        "match_strategy": args.match_strategy,
        "num_eval_items": valid,
        "missing_response": missing_response,
        "center_recall": center_hits / valid if valid else 0.0,
    }
    for key, value in totals.items():
        summary[f"recall_{key}"] = value / valid if valid else 0.0

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["token"])
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
