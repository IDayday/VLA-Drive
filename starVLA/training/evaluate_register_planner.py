#!/usr/bin/env python3
"""Evaluate an integrated Register planner without training-time label inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import atomic_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    accelerator = Accelerator(
        mixed_precision=str(config.get("precision", "bf16")),
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    config.output_dir = str(Path(str(config.run_root_dir)) / str(config.run_id))
    config.datasets.vla_data.shuffle = False
    loader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    model = build_framework(config)
    model = accelerator.prepare_model(model, evaluation_mode=True)
    loader = accelerator.prepare_data_loader(loader)
    model.eval()
    codec = TrajectoryCodec()
    metric_config = config.evaluation.get("metric_supervisor", {})
    supervisor = None
    if bool(metric_config.get("enabled", False)):
        from starVLA.training.navsim_metric_supervisor import DynamicMetricSupervisor

        supervisor = DynamicMetricSupervisor(
            metric_config,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    totals = torch.zeros(10, device=accelerator.device, dtype=torch.float64)
    start = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(
        int(config.get("seed", 42))
    )
    try:
        with torch.inference_mode():
            for examples in loader:
                output = model(examples)
                selected = output["trajectory_navsim_8"]
                actions = torch.as_tensor(
                    np.asarray([example["action"] for example in examples]),
                    device=selected.device,
                    dtype=torch.float32,
                )
                gt = codec.flow_to_navsim(actions)
                proposals = output.get("all_proposals")
                xy = torch.linalg.vector_norm(
                    selected[..., :2] - gt[..., :2], ord=2, dim=-1
                )
                count = selected.shape[0]
                totals[0] += xy.mean(dim=-1).double().sum()
                totals[1] += xy[:, -1].double().sum()
                totals[2] += count
                if proposals is not None:
                    proposal_xy = torch.linalg.vector_norm(
                        proposals[..., :2] - gt[:, None, ..., :2],
                        ord=2,
                        dim=-1,
                    )
                    totals[3] += proposal_xy[:, 0].mean(dim=-1).double().sum()
                    totals[4] += proposal_xy.mean(dim=-1).min(dim=1).values.double().sum()
                    random_index = torch.randint(
                        proposals.shape[1], (count,), generator=generator
                    ).to(proposals.device)
                    rows = torch.arange(count, device=proposals.device)
                    totals[5] += proposal_xy[rows, random_index].mean(dim=-1).double().sum()
                if supervisor is not None:
                    tokens = [str(example["token"]) for example in examples]
                    final_metrics = supervisor.score(
                        tokens, selected[:, None].float()
                    )
                    totals[6] += final_metrics["aggregate_score"][:, 0].double().sum()
                    if proposals is not None:
                        proposal_metrics = supervisor.score(
                            tokens, proposals.float()
                        )
                        totals[7] += proposal_metrics["aggregate_score"].max(dim=1).values.double().sum()
                        totals[8] += proposal_metrics["aggregate_score"][:, 0].double().sum()
                totals[9] += 1
    finally:
        if supervisor is not None:
            supervisor.close()
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[2].clamp_min(1)
    elapsed = time.perf_counter() - start
    results = {
        "num_scenes": int(totals[2]),
        "selected_min_ade": float(totals[0] / count),
        "selected_min_fde": float(totals[1] / count),
        "proposal_0_min_ade": float(totals[3] / count),
        "oracle_min_ade": float(totals[4] / count),
        "random_proposal_min_ade": float(totals[5] / count),
        "selected_pdms": float(totals[6] / count) if supervisor else None,
        "oracle_at_64_pdms": float(totals[7] / count) if supervisor else None,
        "proposal_0_pdms": float(totals[8] / count) if supervisor else None,
        "wall_time_seconds": elapsed,
        "scenes_per_second": float(totals[2]) / max(elapsed, 1e-9),
    }
    if accelerator.is_main_process:
        output_path = Path(config.output_dir) / "register_evaluation.json"
        atomic_json(output_path, results)
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
