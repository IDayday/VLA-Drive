#!/usr/bin/env python3
"""Train and gate a layer-11-global-only 195-token VGGT codec.

This offline teacher-only stage never sees student features or trajectory
labels.  The encoder consumes only VGGT layer-11 global.  The decoder must
reconstruct that source, resume frozen layers 12--23, and preserve the original
camera-head output before its latent may become a V3 target.  Native depth and
point heads are intentionally excluded because they require pre-layer-11 taps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.vggt_query.native_codec import (  # noqa: E402
    VGGTNativeCodecConfig,
    VGGTNativeFeatureCodec,
    save_native_codec_checkpoint,
)
from starVLA.model.modules.vggt_query.native_tail import (  # noqa: E402
    resume_frozen_vggt_tail_from_global,
)
from tools.precompute_vggt_query_cache import (  # noqa: E402
    DEFAULT_VIEWS,
    distributed_context,
    git_revision,
    load_local_vggt,
    load_tokens,
    metadata_path,
    scene_image_paths,
    sha256_file,
)


REPORT_SCHEMA_VERSION = 3


class LatentStatistics:
    def __init__(self, query_count: int, latent_dim: int, device: torch.device) -> None:
        self.count = torch.zeros(query_count, device=device, dtype=torch.float64)
        self.total = torch.zeros(query_count, latent_dim, device=device, dtype=torch.float64)
        self.square_total = torch.zeros_like(self.total)

    @torch.no_grad()
    def update(self, latent: torch.Tensor) -> None:
        value = F.layer_norm(latent.detach().float(), (latent.shape[-1],)).double()
        self.count += value.new_full((value.shape[1],), value.shape[0])
        self.total += value.sum(0)
        self.square_total += value.square().sum(0)

    def distributed_mean_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        if dist.is_initialized():
            for value in (self.count, self.total, self.square_total):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        denominator = self.count.clamp_min(1).unsqueeze(-1)
        mean = self.total / denominator
        variance = (self.square_total / denominator - mean.square()).clamp_min(0)
        return mean.float().cpu(), variance.mean(-1).sqrt().float().cpu()

    def checkpoint_state(self, rank: int) -> dict[str, torch.Tensor] | None:
        values = {
            "count": self.count.clone(),
            "total": self.total.clone(),
            "square_total": self.square_total.clone(),
        }
        if dist.is_initialized():
            for value in values.values():
                dist.reduce(value, dst=0, op=dist.ReduceOp.SUM)
        if rank != 0:
            return None
        return {name: value.cpu() for name, value in values.items()}

    def restore_global_state_on_rank_zero(
        self, state: dict[str, torch.Tensor], rank: int
    ) -> None:
        for name in ("count", "total", "square_total"):
            target = getattr(self, name)
            target.zero_()
            if rank == 0:
                target.copy_(state[name].to(device=target.device, dtype=target.dtype))


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _select_indices(length: int, train_count: int, validation_count: int, seed: int):
    if train_count + validation_count > length:
        raise ValueError("codec train/validation sample counts exceed the datalist")
    validation = np.linspace(0, length - 1, validation_count, dtype=np.int64)
    candidates = np.setdiff1d(np.arange(length, dtype=np.int64), validation)
    generator = np.random.default_rng(seed)
    train = generator.choice(candidates, size=train_count, replace=False)
    return train.tolist(), validation.tolist()


def _load_images(indices, tokens, args, preprocess, device):
    image_batches = []
    for index in indices:
        token = tokens[int(index)]
        with metadata_path(args.data_root, args.split, token).open("rb") as stream:
            raw = pickle.load(stream)
        paths = scene_image_paths(raw, args.views, args.frame_index, args.sensor_root)
        images = preprocess([str(path) for path in paths], mode="pad")
        image_batches.append(images)
    images = torch.stack(image_batches).to(device=device, dtype=torch.float32)
    return images


def _feature_loss(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.float()
    target = target.float()
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-6).mean()
    smooth = F.smooth_l1_loss(prediction, target)
    return 1.0 - cosine + smooth, cosine.detach(), smooth.detach()


def _teacher_forward(model, images, config):
    with torch.inference_mode(), torch.autocast(
        device_type=images.device.type,
        dtype=torch.bfloat16,
        enabled=images.device.type == "cuda",
    ):
        taps, patch_start = model.aggregator(images)
    if patch_start != config.special_per_view:
        raise RuntimeError(
            f"VGGT patch_start_idx mismatch: {patch_start} != {config.special_per_view}"
        )
    with torch.inference_mode():
        float_taps = [value.float() if value is not None else None for value in taps]
        camera = model.camera_head(float_taps)[-1]
    # Tensors created inside inference_mode cannot be saved by autograd even as
    # fixed loss targets. Clone them after leaving the context to obtain normal
    # non-requires-grad tensors for codec backpropagation.
    teacher_taps = {
        layer: taps[layer].detach().clone() for layer in (11, 17, 23)
    }
    return teacher_taps, camera.detach().clone()


def _codec_forward(codec, model, teacher_taps, config):
    # This slice is the only teacher feature that enters the V3 encoder.
    layer11_global = teacher_taps[11][..., config.branch_dim :]
    latent, decoded_layer11_global = codec(layer11_global)
    tail = resume_frozen_vggt_tail_from_global(
        model.aggregator,
        decoded_layer11_global,
        branch_dim=config.branch_dim,
        source_rows=config.source_rows,
        source_cols=config.source_cols,
    )
    # CameraHead natively consumes only the final aggregated tap.  Passing the
    # resumed layer-23 result is therefore the unmodified post-layer-11 path.
    camera = model.camera_head([tail[23]])[-1]
    return latent, decoded_layer11_global, tail, camera


def _losses(
    decoded_layer11_global,
    tail,
    teacher_taps,
    camera,
    teacher_camera,
    args,
):
    metrics = {}
    source_loss, source_cosine, source_smooth = _feature_loss(
        decoded_layer11_global,
        teacher_taps[11][..., teacher_taps[11].shape[-1] // 2 :],
    )
    metrics["layer11_global_cosine"] = source_cosine
    metrics["layer11_global_smooth_l1"] = source_smooth
    # Later native taps are diagnostics only.  They must not become additional
    # teacher feature supervision beyond the layer-11-global contract.
    with torch.no_grad():
        for layer in (17, 23):
            _, cosine, smooth = _feature_loss(tail[layer], teacher_taps[layer])
            metrics[f"layer{layer}_cosine"] = cosine
            metrics[f"layer{layer}_smooth_l1"] = smooth
    camera_loss = F.smooth_l1_loss(camera.float(), teacher_camera.float())
    total = (
        args.source_reconstruction_weight * source_loss
        + args.camera_weight * camera_loss
    )
    metrics.update(
        {
            "camera_smooth_l1": camera_loss.detach(),
        }
    )
    return total, metrics


def _reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    output = {}
    for name, value in metrics.items():
        reduced = value.detach().float().clone()
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= dist.get_world_size()
        output[name] = float(reduced)
    return output


@torch.inference_mode()
def _validate(codec, model, indices, tokens, args, preprocess, device, config):
    module = codec.module if isinstance(codec, DistributedDataParallel) else codec
    module.eval()
    sums: dict[str, float] = {}
    camera_predictions, camera_targets = [], []
    owned = indices[dist.get_rank() :: dist.get_world_size()] if dist.is_initialized() else indices
    iterator = tqdm(owned, desc="codec held-out gate", disable=dist.is_initialized() and dist.get_rank() != 0)
    for start in range(0, len(owned), args.batch_size):
        selected = owned[start : start + args.batch_size]
        images = _load_images(selected, tokens, args, preprocess, device)
        teacher_taps, teacher_camera = _teacher_forward(model, images, config)
        latent, decoded_layer11_global, tail, camera = _codec_forward(
            module, model, teacher_taps, config
        )
        del latent
        _, metrics = _losses(
            decoded_layer11_global,
            tail,
            teacher_taps,
            camera,
            teacher_camera,
            args,
        )
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value) * len(selected)
        camera_predictions.append(camera.float().cpu())
        camera_targets.append(teacher_camera.float().cpu())
        iterator.update(len(selected))
    iterator.close()
    count = torch.tensor(float(len(owned)), device=device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    metric_output = {}
    for name, total in sums.items():
        value = torch.tensor(total, device=device)
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        metric_output[name] = float(value / count.clamp_min(1))

    # Gather compact validation tensors; this is small and gives a scale-free
    # task-retention gate rather than relying only on arbitrary MAE thresholds.
    local_payload = {
        "cp": torch.cat(camera_predictions) if camera_predictions else torch.empty(0),
        "ct": torch.cat(camera_targets) if camera_targets else torch.empty(0),
    }
    gathered = [None] * dist.get_world_size() if dist.is_initialized() else [local_payload]
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_payload)
    camera_prediction = torch.cat([value["cp"] for value in gathered])
    camera_target = torch.cat([value["ct"] for value in gathered])

    def r2(prediction, target):
        prediction, target = prediction.float(), target.float()
        sse = (prediction - target).square().sum()
        centered = (target - target.mean(dim=0, keepdim=True)).square().sum()
        return float(1.0 - sse / centered.clamp_min(1e-12))

    metric_output["camera_r2"] = r2(
        camera_prediction.reshape(-1, 9), camera_target.reshape(-1, 9)
    )
    module.train()
    return metric_output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--encoder-hidden-dim", type=int, default=2048)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--source-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--camera-weight", type=float, default=0.5)
    parser.add_argument("--layer11-global-cosine-threshold", type=float, default=0.90)
    parser.add_argument("--camera-r2-threshold", type=float, default=0.25)
    parser.add_argument("--layer23-cosine-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if len(args.views) != 3:
        raise ValueError("V3 native codec requires exactly three views")
    for path, label in (
        (args.datalist_path, "NAVSIM datalist"),
        (args.data_root, "processed NAVSIM root"),
        (args.sensor_root, "NAVSIM sensor root"),
        (args.vggt_repo, "local VGGT repository"),
        (args.vggt_checkpoint, "local VGGT checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    existing_outputs = [
        path
        for path in (args.output_dir / "native_codec.pt", args.output_dir / "report.json")
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "V3 codec output already exists; choose a new --output-dir or pass "
            f"--overwrite explicitly: {existing_outputs}"
        )
    rank, world_size, device = distributed_context()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tokens = load_tokens(args.datalist_path, None)
    train_indices, validation_indices = _select_indices(
        len(tokens), args.train_samples, args.validation_samples, args.seed
    )
    model, preprocess = load_local_vggt(
        args.vggt_repo,
        args.vggt_checkpoint,
        device,
        enable_camera=True,
        enable_geometry=False,
    )
    config = VGGTNativeCodecConfig(
        latent_dim=args.latent_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_channels=args.decoder_channels,
    )
    trainable = VGGTNativeFeatureCodec(config).to(device)
    codec = (
        DistributedDataParallel(trainable, device_ids=[device.index])
        if world_size > 1 and device.type == "cuda"
        else trainable
    )
    optimizer = torch.optim.AdamW(
        codec.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05
    )
    statistics = LatentStatistics(config.compact_query_count, config.latent_dim, device)
    history = []
    start_step = 1
    training_state_path = args.output_dir / "training_state.pt"
    if training_state_path.is_file() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(
                f"Partial codec state exists: {training_state_path}; pass --resume"
            )
        state = torch.load(
            training_state_path, map_location=device, weights_only=True
        )
        if state.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise RuntimeError(
                "V3 codec resume state uses an obsolete feature-source schema"
            )
        if state.get("config") != asdict(config):
            raise RuntimeError("V3 codec resume configuration changed")
        if int(state.get("world_size", -1)) != world_size:
            raise RuntimeError("V3 codec resume requires the same distributed world size")
        module = codec.module if isinstance(codec, DistributedDataParallel) else codec
        module.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        statistics.restore_global_state_on_rank_zero(state["statistics"], rank)
        history = state.get("history", [])
        start_step = int(state["step"]) + 1
        if rank == 0:
            print(
                f"[V3 native codec] resumed {training_state_path} at step "
                f"{start_step - 1}/{args.steps}"
            )
    started = time.time()
    local_indices = train_indices[rank::world_size]
    if not local_indices:
        raise RuntimeError("codec rank owns no training samples")
    for step in range(start_step, args.steps + 1):
        offset = ((step - 1) * args.batch_size) % len(local_indices)
        selected = [local_indices[(offset + index) % len(local_indices)] for index in range(args.batch_size)]
        images = _load_images(selected, tokens, args, preprocess, device)
        teacher_taps, teacher_camera = _teacher_forward(model, images, config)
        latent, decoded_layer11_global, tail, camera = _codec_forward(
            codec, model, teacher_taps, config
        )
        loss, metrics = _losses(
            decoded_layer11_global,
            tail,
            teacher_taps,
            camera,
            teacher_camera,
            args,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        statistics.update(latent)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            reduced = _reduce_metrics({"loss": loss, **metrics})
            record = {
                "step": step,
                **reduced,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            if rank == 0:
                print("[V3 native codec] " + json.dumps(record, sort_keys=True))
        if (
            args.checkpoint_interval > 0
            and step < args.steps
            and step % args.checkpoint_interval == 0
        ):
            statistics_state = statistics.checkpoint_state(rank)
            if rank == 0:
                module = (
                    codec.module if isinstance(codec, DistributedDataParallel) else codec
                )
                _atomic_torch(
                    {
                        "schema_version": REPORT_SCHEMA_VERSION,
                        "step": step,
                        "world_size": world_size,
                        "config": asdict(config),
                        "model": {
                            name: value.detach().cpu()
                            for name, value in module.state_dict().items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "statistics": statistics_state,
                        "history": history,
                    },
                    training_state_path,
                )
                print(f"[V3 native codec] checkpoint={training_state_path}")
            if dist.is_initialized():
                dist.barrier()
    latent_mean, latent_scale = statistics.distributed_mean_scale()
    validation = _validate(
        codec,
        model,
        validation_indices,
        tokens,
        args,
        preprocess,
        device,
        config,
    )
    gates = {
        "layer11_global_reconstruction": validation["layer11_global_cosine"]
        >= args.layer11_global_cosine_threshold,
        "tail23_continuation": validation["layer23_cosine"]
        >= args.layer23_cosine_threshold,
        "native_camera_retention": validation["camera_r2"]
        >= args.camera_r2_threshold,
    }
    gates["teacher_codec_downstream"] = all(gates.values())
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vggt_checkpoint_sha256": sha256_file(args.vggt_checkpoint),
            "vggt_repo_commit": git_revision(args.vggt_repo),
            "datalist_sha256": sha256_file(args.datalist_path),
            "train_indices_sha256": hashlib.sha256(
                json.dumps(train_indices).encode("utf-8")
            ).hexdigest(),
            "validation_indices": validation_indices,
            "project_commit": git_revision(REPO_ROOT),
        }
        checkpoint_path = args.output_dir / "native_codec.pt"
        module = codec.module if isinstance(codec, DistributedDataParallel) else codec
        thresholds = {
            "layer11_global_cosine": args.layer11_global_cosine_threshold,
            "layer23_cosine": args.layer23_cosine_threshold,
            "camera_r2": args.camera_r2_threshold,
        }
        save_native_codec_checkpoint(
            module,
            checkpoint_path,
            latent_slot_mean=latent_mean,
            latent_slot_scale=latent_scale,
            source=source,
            gates=gates,
            metrics=validation,
            thresholds=thresholds,
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "design": {
                "compact_tokens": config.compact_query_count,
                "latent_dim": config.latent_dim,
                "native_scalar_compression_ratio": module.native_scalar_compression_ratio,
                "trainable_codec_parameters": sum(
                    parameter.numel() for parameter in module.parameters()
                ),
                "encoder_source": {
                    "layer_index": 11,
                    "attention_branch": "global",
                    "shape_per_sample": [
                        config.view_count,
                        config.source_query_count_per_view,
                        config.branch_dim,
                    ],
                },
                "directly_reconstructed_feature": "layer11_global",
                "resumed_frozen_layers": [12, 23],
                "reused_native_heads": ["camera"],
                "excluded_native_heads": {
                    "depth": "requires pre-layer-11 DPT skip taps",
                    "point": "requires pre-layer-11 DPT skip taps",
                },
                "student_or_trajectory_labels_seen": False,
            },
            "source": source,
            "training": {"steps": args.steps, "world_size": world_size, "history": history},
            "thresholds": thresholds,
            "metrics": validation,
            "gates": gates,
            "checkpoint": str(checkpoint_path.resolve()),
        }
        _atomic_json(report, args.output_dir / "report.json")
        print("[V3 native codec gates] " + json.dumps(gates, sort_keys=True))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
