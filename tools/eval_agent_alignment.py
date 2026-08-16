#!/usr/bin/env python3
"""Evaluate agent-alignment losses on NAVSIM without running PDM scoring."""

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from infer import VLAAgent, to_device
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn


METRIC_KEYS = (
    "action_loss",
    "agent_dino_loss",
    "agent_bbox_loss",
    "agent_vis_loss",
    "agent_match_count",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--datalist-path", default="test_meta.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="navsim_dataset")
    parser.add_argument("--agent-cache-root", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--qwen-forward-mode", default="auto", choices=("auto", "legacy", "optimized"))
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--per-batch-csv", default=None)
    args = parser.parse_args()

    os.environ["NAVSIM_AGENT_DINO_CACHE_ROOT"] = args.agent_cache_root
    os.environ.setdefault("NAVSIM_AGENT_DINO_CACHE_STRICT", "1")
    os.environ["NAVSIM_USE_FEATURE_CACHE"] = "0"
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)

    agent = VLAAgent(
        args.ckpt_dir,
        device=args.device,
        qwen_forward_mode=args.qwen_forward_mode,
    )
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

    sums = {key: 0.0 for key in METRIC_KEYS}
    count = 0
    rows = []
    agent.model.eval()
    use_cuda_amp = agent.device.type == "cuda"

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="agent-align eval")):
            batch = to_device(batch, agent.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
                output = agent.model.forward(batch)
            bs = len(batch.get("examples", [])) if isinstance(batch, dict) else args.batch_size
            row = {"batch_index": batch_index, "batch_size": bs}
            for key in METRIC_KEYS:
                value = scalar(output.get(key, 0.0))
                row[key] = value
                sums[key] += value * bs
            rows.append(row)
            count += bs

    summary = {
        "checkpoint_dir": str(Path(args.ckpt_dir).resolve()),
        "agent_cache_root": str(Path(args.agent_cache_root).resolve()),
        "split": args.split,
        "num_samples": count,
        "batch_size": args.batch_size,
        "metrics_mean": {key: sums[key] / max(count, 1) for key in METRIC_KEYS},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_batch_csv:
        csv_path = Path(args.per_batch_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["batch_index", "batch_size", *METRIC_KEYS])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
