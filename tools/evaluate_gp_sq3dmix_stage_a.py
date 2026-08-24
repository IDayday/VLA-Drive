#!/usr/bin/env python3
"""Evaluate the paired GP-SQ3D-Mix Stage-A-v2 utility contract."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader import build_dataloader
from starVLA.gp_sq3dmix_statistics import (
    absolute_gap_bootstrap_ci,
    absolute_mean_gap,
    evaluate_stage_a_v2_gates,
)
from starVLA.gp_sq3dmix_v2 import sha256_file
from starVLA.model.framework import build_framework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("projected_residual", "gated_residual"), required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--source-datalist", required=True)
    parser.add_argument("--source-cache-root", required=True)
    parser.add_argument("--negative-map", required=True)
    parser.add_argument("--negative-map-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty Stage-A sample CSV")
    fieldnames = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    os.replace(temporary, path)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    # ``build_dataloader`` uses Accelerate's multi-process-aware logger. The
    # evaluator is launched with plain Python, so initialize its singleton
    # before the first dataloader log call.
    PartialState()
    if args.batch_size < 1 or args.num_workers < 0 or args.bootstrap_draws < 1:
        raise ValueError("batch/workers/bootstrap arguments are invalid")
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = run_dir / "config.yaml"
    output_path = Path(args.output).resolve()
    samples_path = (
        Path(args.samples_output).resolve()
        if args.samples_output
        else output_path.with_name(output_path.stem + ".samples.csv")
    )
    cache_manifest = Path(args.cache_root).resolve() / "vggt_dense" / "manifest.json"
    source_cache_manifest = (
        Path(args.source_cache_root).resolve() / "vggt_dense" / "manifest.json"
    )
    stats_manifest = Path(args.stats_root).resolve() / "manifest.json"
    paths = (
        config_path,
        checkpoint,
        Path(args.datalist),
        Path(args.source_datalist),
        cache_manifest,
        source_cache_manifest,
        stats_manifest,
        Path(args.negative_map),
        Path(args.negative_map_manifest),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root or not Path(args.data_root).is_dir():
        raise FileNotFoundError(f"Invalid processed NAVSIM data root: {args.data_root}")
    for path in (output_path, samples_path):
        if path.exists():
            raise FileExistsError(path)

    cfg = OmegaConf.load(config_path)
    action_only_checkpoint = Path(
        str(OmegaConf.select(cfg, "trainer.pretrained_checkpoint", default=""))
    ).resolve()
    if not action_only_checkpoint.is_file():
        raise FileNotFoundError(
            "Stage-A resolved config has no valid action-only checkpoint: "
            f"{action_only_checkpoint}"
        )
    if "BASE_VLM" not in os.environ or not Path(os.environ["BASE_VLM"]).is_dir():
        raise FileNotFoundError("BASE_VLM must name the local converted Qwen model")
    split_tokens = json.loads(Path(args.datalist).read_text(encoding="utf-8"))
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", os.environ["BASE_VLM"], force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.attn_implementation", "sdpa", force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.training.stage", "stage_a", force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.mode", args.variant, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.intervention.mode", "real", force_add=True)
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.evaluation.scene_shuffle_diagnostic",
        args.variant == "gated_residual",
        force_add=True,
    )
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.cache.enabled", True, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.cache.root", args.cache_root, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.cache.allow_datalist_subset", True, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.stats.root", args.stats_root, force_add=True)
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.stats.source_datalist",
        args.source_datalist,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.stats.source_cache_manifest",
        str(source_cache_manifest),
        force_add=True,
    )
    OmegaConf.update(
        cfg, "framework.gp_sq_3d_mix.negative_map.path", args.negative_map, force_add=True
    )
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.negative_map.manifest",
        args.negative_map_manifest,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.negative_map.source_datalist",
        args.source_datalist,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.negative_map.source_cache_root",
        args.source_cache_root,
        force_add=True,
    )
    OmegaConf.update(cfg, "datasets.vla_data.datalist_path", args.datalist, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.data_root", args.data_root, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.split", "train", force_add=True)
    OmegaConf.update(
        cfg, "datasets.vla_data.expected_sample_count", len(split_tokens), force_add=True
    )
    OmegaConf.update(
        cfg, "datasets.vla_data.per_device_batch_size", args.batch_size, force_add=True
    )
    OmegaConf.update(
        cfg, "framework.action_model.repeated_diffusion_steps", 1, force_add=True
    )
    os.environ["NAVSIM_NUM_WORKERS"] = str(args.num_workers)
    os.environ["NAVSIM_PIN_MEMORY"] = "0"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_framework(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    loader = build_dataloader(cfg, dataset_py="navsim_dataset")

    samples: list[dict] = []
    scalar_metrics: dict[str, list[float]] = defaultdict(list)
    gradient_metrics: dict[str, list[float]] = defaultdict(list)
    finite_named_losses = True
    for examples in loader:
        for parameter in model.parameters():
            parameter.grad = None
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            output = model.forward(examples)
            losses = output["losses"]
            expected_names = {
                "action",
                "geometry_rank_hard",
                "geometry_rank_spatial",
                "baseline_fidelity",
            }
            if set(losses) != expected_names:
                raise RuntimeError(
                    f"Stage-A-v2 named losses mismatch: {sorted(losses)}"
                )
            total = (
                losses["action"]
                + 0.05 * losses["geometry_rank_hard"]
                + 0.05 * losses["geometry_rank_spatial"]
                + 0.10 * losses["baseline_fidelity"]
            )
        finite_named_losses &= all(
            loss.ndim == 0 and bool(torch.isfinite(loss).item())
            for loss in losses.values()
        )
        total.backward()
        loss_samples = model._last_gp_gate_samples
        query_samples = model._last_gp_query_samples
        batch_size = len(examples)
        for key in ("base_loss", "real_loss", "hard_loss", "spatial_loss"):
            if loss_samples[key].shape != (batch_size,):
                raise RuntimeError(f"Unexpected per-sample loss shape for {key}")
        residual = query_samples[
            "real_residual_action_ratio_per_horizon"
        ].float().cpu().numpy()
        if residual.shape != (batch_size, 8):
            raise RuntimeError("Unexpected residual ratio shape")
        retention = query_samples.get("real_retention")
        for batch_index, example in enumerate(examples):
            row = {
                "token": str(example["token"]),
                "base_loss": float(loss_samples["base_loss"][batch_index].cpu()),
                "real_loss": float(loss_samples["real_loss"][batch_index].cpu()),
                "hard_loss": float(loss_samples["hard_loss"][batch_index].cpu()),
                "spatial_loss": float(loss_samples["spatial_loss"][batch_index].cpu()),
                "residual_action_ratio": float(residual[batch_index].mean()),
                "alpha": float(model.centered_geometry_reader.alpha.detach().cpu()),
                "retention_mean": "",
                "retention_std": "",
                "retention_min": "",
                "retention_max": "",
                "retention_near_lower_fraction": "",
                "retention_near_upper_fraction": "",
                "scene_shuffled_loss": "",
            }
            for horizon in range(8):
                row[f"residual_action_ratio_h{horizon}"] = float(
                    residual[batch_index, horizon]
                )
            if retention is not None:
                for key in (
                    "mean",
                    "std",
                    "min",
                    "max",
                    "near_lower_fraction",
                    "near_upper_fraction",
                ):
                    row[f"retention_{key}"] = float(
                        retention[key][batch_index].detach().cpu()
                    )
            if "scene_shuffled_loss" in loss_samples:
                row["scene_shuffled_loss"] = float(
                    loss_samples["scene_shuffled_loss"][batch_index].cpu()
                )
            samples.append(row)
        for name, value in output.get("metrics", {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                scalar_metrics[name].append(float(value.detach().cpu()))
        for name, value in model.get_planning_usage_metrics().items():
            gradient_metrics[name].append(float(value.detach().cpu()))

    if [row["token"] for row in samples] != split_tokens:
        raise RuntimeError("Stage-A evaluator changed the immutable split order")
    base = np.asarray([row["base_loss"] for row in samples], dtype=np.float64)
    real = np.asarray([row["real_loss"] for row in samples], dtype=np.float64)
    hard = np.asarray([row["hard_loss"] for row in samples], dtype=np.float64)
    spatial = np.asarray([row["spatial_loss"] for row in samples], dtype=np.float64)
    residual_per_horizon = np.asarray(
        [
            [row[f"residual_action_ratio_h{horizon}"] for horizon in range(8)]
            for row in samples
        ],
        dtype=np.float64,
    )
    metric_means = {
        name: float(np.mean(values)) for name, values in scalar_metrics.items()
    }
    grad_summary = {
        name: {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in gradient_metrics.items()
    }
    retention_lower = (
        float(np.mean([float(row["retention_near_lower_fraction"]) for row in samples]))
        if args.variant == "gated_residual"
        else None
    )
    retention_upper = (
        float(np.mean([float(row["retention_near_upper_fraction"]) for row in samples]))
        if args.variant == "gated_residual"
        else None
    )
    gates = evaluate_stage_a_v2_gates(
        variant=args.variant,
        base_loss=base,
        real_loss=real,
        hard_loss=hard,
        spatial_loss=spatial,
        residual_per_horizon=residual_per_horizon,
        slot_mean_identity_max_abs=max(
            scalar_metrics.get(
                "gp_sq3dmix/slot_mean_identity_max_abs", [float("inf")]
            )
        ),
        adapter_grad_norm=grad_summary.get(
            "gp_sq3dmix/adapter_grad_norm", {"mean": 0.0}
        )["mean"],
        reader_grad_norm=grad_summary.get(
            "gp_sq3dmix/reader_grad_norm", {"mean": 0.0}
        )["mean"],
        gate_grad_norm=(
            grad_summary.get("gp_sq3dmix/gate_grad_norm", {"mean": 0.0})[
                "mean"
            ]
            if args.variant == "gated_residual"
            else None
        ),
        all_named_losses_finite=finite_named_losses,
        alpha=float(model.centered_geometry_reader.alpha.detach().cpu()),
        retention_near_lower_fraction=retention_lower,
        retention_near_upper_fraction=retention_upper,
        seed=args.seed,
        draws=args.bootstrap_draws,
    )
    scene_diagnostic = {
        "scene_summary_batch_variance": metric_means.get(
            "gp_sq3dmix/scene_summary_batch_variance"
        ),
        "cross_scene_pairwise_cosine": metric_means.get(
            "gp_sq3dmix/scene_summary_cross_scene_pairwise_cosine"
        ),
        "scene_shuffled_retention_l2": metric_means.get(
            "gp_sq3dmix/scene_shuffled_retention_l2"
        ),
        "scene_shuffled_residual_l2": metric_means.get(
            "gp_sq3dmix/scene_shuffled_residual_l2"
        ),
    }
    if args.variant == "gated_residual":
        scene_loss = np.asarray(
            [float(row["scene_shuffled_loss"]) for row in samples],
            dtype=np.float64,
        )
        scene_diagnostic.update(
            {
                "scene_shuffled_minus_real_loss": absolute_mean_gap(
                    scene_loss, real
                ),
                "scene_shuffled_minus_real_loss_bootstrap_ci": absolute_gap_bootstrap_ci(
                    scene_loss,
                    real,
                    seed=args.seed + 20,
                    draws=args.bootstrap_draws,
                ),
            }
        )
    report = {
        "schema_version": 2,
        "stage": "stage_a_v2_final_gate",
        "variant": args.variant,
        "code_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "split_datalist": str(Path(args.datalist).resolve()),
        "split_datalist_sha256": sha256_file(args.datalist),
        "source_datalist_sha256": sha256_file(args.source_datalist),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "source_cache_manifest_sha256": sha256_file(source_cache_manifest),
        "stats_manifest_sha256": sha256_file(stats_manifest),
        "negative_map_sha256": sha256_file(args.negative_map),
        "negative_map_manifest_sha256": sha256_file(args.negative_map_manifest),
        "resolved_config_sha256": sha256_file(config_path),
        "action_only_checkpoint": str(action_only_checkpoint),
        "action_only_checkpoint_sha256": sha256_file(action_only_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": len(samples),
        "loss_means": {
            "base": float(base.mean()),
            "real": float(real.mean()),
            "hard": float(hard.mean()),
            "spatial": float(spatial.mean()),
        },
        **gates,
        "metrics": metric_means,
        "gradient_norms": grad_summary,
        "scene_conditioning_diagnostic": scene_diagnostic,
        "sample_csv": str(samples_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(samples_path, samples)
    atomic_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
