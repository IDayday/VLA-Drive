#!/usr/bin/env python3
"""Evaluate key-agent bbox/visibility recall on NAVSIM."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from infer import VLAAgent, to_device
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    size = boxes[..., 2:].clamp(min=1e-4)
    return torch.cat([center - size * 0.5, center + size * 0.5], dim=-1).clamp(0.0, 1.0)


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(pred[:, None, :2], target[None, :, :2])
    rb = torch.minimum(pred[:, None, 2:], target[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = ((pred[:, 2] - pred[:, 0]).clamp(min=0.0) * (pred[:, 3] - pred[:, 1]).clamp(min=0.0))[:, None]
    area_t = ((target[:, 2] - target[:, 0]).clamp(min=0.0) * (target[:, 3] - target[:, 1]).clamp(min=0.0))[None, :]
    return inter / (area_p + area_t - inter).clamp(min=1e-8)


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def build_dataset(agent: VLAAgent, args: argparse.Namespace) -> NavSimDataset:
    cfg = agent.model_config
    data_cfg = copy.deepcopy(cfg)
    data_cfg.datasets.video_data.load_2d_data = 0
    data_cfg.datasets.video_data.load_3d_data = 0
    data_cfg.datasets.gs_data.load_3d_data = 0
    data_cfg.datasets.reward_data.load_reward_data = 0
    data_cfg.datasets.vla_data.w_neg_traj = None
    data_cfg.w_depth = 0
    data_cfg.enable_image_aug = 0

    return NavSimDataset(
        datalist_path=args.datalist_path,
        split=args.split,
        video_data_cfg=data_cfg.datasets.video_data,
        gs_data_cfg=data_cfg.datasets.gs_data,
        reward_data_cfg=data_cfg.datasets.reward_data,
        ver_1225=OmegaConf.select(cfg, "ver_1225", default=False),
        dataset_cfg=data_cfg.datasets.vla_data,
        all_cfg=data_cfg,
        max_samples=args.max_samples,
        data_root=args.data_root,
    )


def repair_mine_agent_token_ids(agent: VLAAgent) -> None:
    """Keep only mine-agent special tokens that exist in the selected tokenizer."""
    tokenizer = agent.model.qwen_vl_interface.processor.tokenizer
    pairs = [
        (token, tokenizer.convert_tokens_to_ids(token))
        for token in agent.model.mine_agent_query_tokens
    ]
    valid_pairs = [(token, token_id) for token, token_id in pairs if token_id is not None]
    if not valid_pairs:
        raise RuntimeError("tokenizer has no mine_agent special tokens")
    agent.model.mine_agent_query_tokens = [token for token, _token_id in valid_pairs]
    agent.model._special_token_ids["mine_agent"] = tuple(int(token_id) for _token, token_id in valid_pairs)

class AgentHeadCapture:
    def __init__(self) -> None:
        self.bbox_raw: torch.Tensor | None = None
        self.vis_logits: torch.Tensor | None = None

    def clear(self) -> None:
        self.bbox_raw = None
        self.vis_logits = None

    def bbox_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.bbox_raw = output.detach()

    def vis_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.vis_logits = output.detach()


def selected_gt(payload: dict[str, Any], view_ids: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bboxes = payload["bbox_norm_per_view"].to(device=device, dtype=torch.float32)
    visible = payload["visible_per_view"].to(device=device, dtype=torch.float32)
    view_mask = payload["view_mask"].to(device=device, dtype=torch.bool)
    valid_count = min(
        int(payload["agent_features"].shape[0]),
        int(bboxes.shape[0]),
        int(visible.shape[0]),
        int(view_mask.shape[0]),
    )
    if valid_count <= 0:
        return bboxes[:0, : len(view_ids)], torch.zeros((0, len(view_ids)), device=device, dtype=torch.bool)

    bboxes = bboxes[:valid_count]
    visible = visible[:valid_count]
    view_mask = view_mask[:valid_count]

    agent_valid_mask = payload.get("agent_valid_mask")
    if agent_valid_mask is not None:
        agent_valid_mask = agent_valid_mask[:valid_count].to(device=device, dtype=torch.bool)
        valid_indices = torch.nonzero(agent_valid_mask, as_tuple=False).squeeze(1)
        bboxes = bboxes.index_select(dim=0, index=valid_indices)
        visible = visible.index_select(dim=0, index=valid_indices)
        view_mask = view_mask.index_select(dim=0, index=valid_indices)

    view_index = torch.as_tensor(view_ids, device=device, dtype=torch.long)
    bboxes = bboxes.index_select(dim=1, index=view_index)
    visible = visible.index_select(dim=1, index=view_index)
    view_mask = view_mask.index_select(dim=1, index=view_index)
    target_mask = view_mask & (visible >= 0.5)
    return bboxes, target_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--datalist-path", default="test_meta.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="navsim_dataset")
    parser.add_argument("--agent-cache-root", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--qwen-forward-mode", default="auto", choices=("auto", "legacy", "optimized"))
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument("--iou-thresholds", default="0.3,0.5")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--per-batch-csv", default=None)
    args = parser.parse_args()

    os.environ["NAVSIM_AGENT_DINO_CACHE_ROOT"] = args.agent_cache_root
    os.environ.setdefault("NAVSIM_AGENT_DINO_CACHE_STRICT", "1")
    os.environ["NAVSIM_USE_FEATURE_CACHE"] = "0"
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)

    iou_thresholds = tuple(float(x) for x in args.iou_thresholds.split(",") if x.strip())
    agent = VLAAgent(args.ckpt_dir, device=args.device, qwen_forward_mode=args.qwen_forward_mode)
    repair_mine_agent_token_ids(agent)
    view_ids = tuple(int(v) for v in agent.model.agent_view_ids)
    dataset = build_dataset(agent, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    capture = AgentHeadCapture()
    bbox_handle = agent.model.agent_bbox_head.register_forward_hook(capture.bbox_hook)
    vis_handle = agent.model.agent_vis_head.register_forward_hook(capture.vis_hook)

    totals = {
        thr: {
            "agent_targets": 0,
            "agent_recalled": 0,
            "view_targets": 0,
            "view_recalled": 0,
        }
        for thr in iou_thresholds
    }
    loss_sums = {key: 0.0 for key in ("agent_dino_loss", "agent_bbox_loss", "agent_vis_loss", "agent_match_count")}
    count = 0
    rows: list[dict[str, Any]] = []
    use_cuda_amp = agent.device.type == "cuda"

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(loader, desc="key-agent recall")):
                capture.clear()
                batch = to_device(batch, agent.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
                    output = agent.model.forward(batch)

                if capture.bbox_raw is None or capture.vis_logits is None:
                    raise RuntimeError("agent bbox/vis head did not run; check action_prompt_mode and cache availability")

                if isinstance(batch, dict):
                    examples = batch.get("examples", [])
                else:
                    examples = batch
                bs = len(examples)
                pred_bbox = xywh_to_xyxy(capture.bbox_raw.float().view(bs, -1, len(view_ids), 4))
                pred_vis = torch.sigmoid(capture.vis_logits.float().view(bs, -1, len(view_ids)))
                pred_pos = pred_vis >= args.vis_threshold

                row: dict[str, Any] = {"batch_index": batch_index, "batch_size": bs}
                for key in loss_sums:
                    value = scalar(output.get(key, 0.0))
                    loss_sums[key] += value * bs
                    row[key] = value

                batch_totals = {thr: {"agent_targets": 0, "agent_recalled": 0, "view_targets": 0, "view_recalled": 0} for thr in iou_thresholds}
                for sample_index, example in enumerate(examples):
                    payload = example.get("agent_dino_feature_cache")
                    if payload is None:
                        continue
                    gt_bbox, gt_mask = selected_gt(payload, view_ids, agent.device)
                    if gt_bbox.numel() == 0:
                        continue

                    for view_index in range(len(view_ids)):
                        gt_indices = torch.nonzero(gt_mask[:, view_index], as_tuple=False).squeeze(1)
                        if gt_indices.numel() == 0:
                            continue
                        pos_indices = torch.nonzero(pred_pos[sample_index, :, view_index], as_tuple=False).squeeze(1)
                        if pos_indices.numel() > 0:
                            ious = box_iou(pred_bbox[sample_index, pos_indices, view_index], gt_bbox[gt_indices, view_index])
                            best_iou = ious.max(dim=0).values
                        else:
                            best_iou = torch.zeros((gt_indices.numel(),), device=agent.device)

                        for thr in iou_thresholds:
                            recalled = best_iou >= thr
                            batch_totals[thr]["view_targets"] += int(gt_indices.numel())
                            batch_totals[thr]["view_recalled"] += int(recalled.sum().item())

                    agent_visible = gt_mask.any(dim=1)
                    for thr in iou_thresholds:
                        recalled_agent = torch.zeros_like(agent_visible)
                        for view_index in range(len(view_ids)):
                            gt_indices = torch.nonzero(gt_mask[:, view_index], as_tuple=False).squeeze(1)
                            if gt_indices.numel() == 0:
                                continue
                            pos_indices = torch.nonzero(pred_pos[sample_index, :, view_index], as_tuple=False).squeeze(1)
                            if pos_indices.numel() == 0:
                                continue
                            ious = box_iou(pred_bbox[sample_index, pos_indices, view_index], gt_bbox[gt_indices, view_index])
                            recalled_agent[gt_indices] |= ious.max(dim=0).values >= thr
                        batch_totals[thr]["agent_targets"] += int(agent_visible.sum().item())
                        batch_totals[thr]["agent_recalled"] += int((recalled_agent & agent_visible).sum().item())

                for thr in iou_thresholds:
                    for key, value in batch_totals[thr].items():
                        totals[thr][key] += value
                    row[f"agent_recall@{thr:g}"] = (
                        batch_totals[thr]["agent_recalled"] / max(batch_totals[thr]["agent_targets"], 1)
                    )
                    row[f"view_recall@{thr:g}"] = (
                        batch_totals[thr]["view_recalled"] / max(batch_totals[thr]["view_targets"], 1)
                    )
                rows.append(row)
                count += bs
    finally:
        bbox_handle.remove()
        vis_handle.remove()

    recall = {}
    for thr in iou_thresholds:
        data = totals[thr]
        recall[f"agent_recall@{thr:g}"] = data["agent_recalled"] / max(data["agent_targets"], 1)
        recall[f"view_recall@{thr:g}"] = data["view_recalled"] / max(data["view_targets"], 1)
        recall[f"agent_targets@{thr:g}"] = data["agent_targets"]
        recall[f"view_targets@{thr:g}"] = data["view_targets"]

    summary = {
        "checkpoint_dir": str(Path(args.ckpt_dir).resolve()),
        "agent_cache_root": str(Path(args.agent_cache_root).resolve()),
        "split": args.split,
        "num_samples": count,
        "batch_size": args.batch_size,
        "view_ids": view_ids,
        "vis_threshold": args.vis_threshold,
        "recall": recall,
        "loss_metrics_mean": {key: loss_sums[key] / max(count, 1) for key in loss_sums},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_batch_csv:
        csv_path = Path(args.per_batch_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
