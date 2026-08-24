#!/usr/bin/env python3
"""Evaluate the immutable GP-SQ3D-Mix Stage-A utility/gating contract."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000):
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--stats-root", required=True)
    parser.add_argument("--source-datalist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = run_dir / "config.yaml"
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    cache_manifest = Path(args.cache_root) / "vggt_dense" / "manifest.json"
    stats_manifest = Path(args.stats_root) / "manifest.json"
    for path in (
        config_path,
        checkpoint,
        Path(args.datalist),
        Path(args.source_datalist),
        cache_manifest,
        stats_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = OmegaConf.load(config_path)
    action_only_checkpoint = Path(
        str(OmegaConf.select(cfg, "trainer.pretrained_checkpoint", default=""))
    ).resolve()
    if not action_only_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-A resolved config has no valid action-only checkpoint: "
            f"{action_only_checkpoint}"
        )
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", os.environ["BASE_VLM"], force_add=True)
    OmegaConf.update(cfg, "framework.qwenvl.attn_implementation", "sdpa", force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.training.stage", "stage_a", force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.mode", "gated_residual", force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.cache.root", args.cache_root, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.stats.root", args.stats_root, force_add=True)
    OmegaConf.update(cfg, "framework.gp_sq_3d_mix.stats.source_datalist", args.source_datalist, force_add=True)
    OmegaConf.update(
        cfg,
        "framework.gp_sq_3d_mix.stats.source_cache_manifest",
        str(Path(args.cache_root) / "vggt_dense" / "manifest.json"),
        force_add=True,
    )
    OmegaConf.update(cfg, "datasets.vla_data.datalist_path", args.datalist, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.split", "train", force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.per_device_batch_size", args.batch_size, force_add=True)
    os.environ["NAVSIM_NUM_WORKERS"] = str(args.num_workers)
    os.environ["NAVSIM_PIN_MEMORY"] = "0"
    torch.manual_seed(args.seed)
    model = build_framework(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    loader = build_dataloader(cfg, dataset_py="navsim_dataset")

    base_values, real_values, shuffled_values = [], [], []
    scalar_metrics: dict[str, list[float]] = {}
    gradient_metrics: dict[str, list[float]] = {}
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
            total = losses["action"] + 0.10 * losses["geometry_rank"] + 0.10 * losses["baseline_fidelity"]
        finite_named_losses &= all(
            loss.ndim == 0 and bool(torch.isfinite(loss).item())
            for loss in losses.values()
        )
        total.backward()
        samples = model._last_gp_gate_samples
        base_values.extend(samples["base_loss"].float().cpu().tolist())
        real_values.extend(samples["real_loss"].float().cpu().tolist())
        shuffled_values.extend(samples["shuffled_loss"].float().cpu().tolist())
        for name, value in output.get("metrics", {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                scalar_metrics.setdefault(name, []).append(float(value.detach().cpu()))
        for name, value in model.get_planning_usage_metrics().items():
            gradient_metrics.setdefault(name, []).append(float(value.detach().cpu()))

    base = np.asarray(base_values, dtype=np.float64)
    real = np.asarray(real_values, dtype=np.float64)
    shuffled = np.asarray(shuffled_values, dtype=np.float64)
    utility_delta = real - base
    relative_gap = (shuffled - real) / np.maximum(real, 1e-8)
    utility_ci = bootstrap_ci(utility_delta, args.seed)
    gap_ci = bootstrap_ci(relative_gap, args.seed + 1)
    means = {name: float(np.mean(values)) for name, values in scalar_metrics.items()}
    slot_mean_identity_max = max(
        scalar_metrics.get("gp_sq3dmix/slot_mean_identity_max_abs", [float("inf")])
    )
    grad_means = {name: float(np.mean(values)) for name, values in gradient_metrics.items()}
    checks = {
        "slot_mean_identity": slot_mean_identity_max < 1e-6,
        "real_utility": float(utility_delta.mean()) <= 0.0 or utility_ci[0] <= 0.0,
        "causal_ranking": float(relative_gap.mean()) > 0.05 and gap_ci[0] > 0.0,
        "residual_range": 0.01 <= means.get("gp_sq3dmix/residual_action_ratio", -1.0) <= 0.20,
        "geometry_route_active": all(
            grad_means.get(name, 0.0) > 0.0
            for name in (
                "gp_sq3dmix/adapter_grad_norm",
                "gp_sq3dmix/gate_grad_norm",
                "gp_sq3dmix/reader_grad_norm",
            )
        ),
        "gate_not_collapsed": means.get("gp_sq3dmix/retention_near_lower_fraction", 1.0) < 0.80
        and means.get("gp_sq3dmix/retention_near_upper_fraction", 1.0) < 0.80,
        "alpha": 0.05 <= means.get("gp_sq3dmix/alpha", -1.0) <= 0.20,
        "finite_named_losses": finite_named_losses,
    }
    report = {
        "schema_version": 1,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "split_datalist": str(Path(args.datalist).resolve()),
        "split_datalist_sha256": sha256_file(args.datalist),
        "source_datalist_sha256": sha256_file(args.source_datalist),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "stats_manifest_sha256": sha256_file(stats_manifest),
        "resolved_config_sha256": sha256_file(config_path),
        "action_only_checkpoint": str(action_only_checkpoint),
        "action_only_checkpoint_sha256": sha256_file(action_only_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": len(real),
        "mean_real_minus_base": float(utility_delta.mean()),
        "real_minus_base_bootstrap_ci": utility_ci,
        "relative_shuffled_real_flow_gap": float(relative_gap.mean()),
        "relative_gap_bootstrap_ci": gap_ci,
        "metrics": means,
        "gradient_norms": grad_means,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
