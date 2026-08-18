#!/usr/bin/env python3
"""Compare VGGT spatial pooling layouts without writing feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.vggt_query.resolution_probe import (  # noqa: E402
    NormalizedSlotStatistics,
    crop_and_pool_valid_patches,
    summarize_scene_descriptors,
)
from starVLA.model.modules.vggt_query.targets import (  # noqa: E402
    extract_vggt_query_targets,
)
from tools.precompute_vggt_query_cache import (  # noqa: E402
    git_revision,
    load_local_vggt,
    metadata_path,
    patch_validity_for_image,
    scene_image_paths,
    sha256_file,
)


SCHEMA_VERSION = 1
PATCH_GRID_SIZE = 37
PATCH_START_INDEX = 5
DEFAULT_VIEWS = ("cam_f0", "cam_l0", "cam_r0")


@dataclass(frozen=True)
class Layout:
    name: str
    rows: int
    cols: int
    crop_padding: bool
    native_resolution: bool = False


LAYOUTS = (
    Layout("current_padded_4x4", 4, 4, False),
    Layout("cropped_4x4", 4, 4, True),
    Layout("cropped_6x10", 6, 10, True),
    Layout("cropped_8x12", 8, 12, True),
    Layout("native_valid_37x37", 37, 37, False, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sensor-root", default="")
    parser.add_argument("--vggt-repo", required=True)
    parser.add_argument("--vggt-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_evenly_spaced_indices(sample_count: int, requested: int) -> list[int]:
    if sample_count <= 0:
        raise ValueError("datalist must contain at least one sample")
    if requested < 2:
        raise ValueError("--max-samples must be at least 2")
    selected_count = min(sample_count, requested)
    if selected_count == sample_count:
        return list(range(sample_count))
    return (
        torch.linspace(0, sample_count - 1, steps=selected_count)
        .round()
        .long()
        .tolist()
    )


def batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but the CUDA-compatible runtime is unavailable")
    return device


def layer_normalized_unit(features: torch.Tensor) -> torch.Tensor:
    normalized = F.layer_norm(features.float(), (features.shape[-1],))
    return F.normalize(normalized, dim=-1, eps=1e-6)


class NeighborStatistics:
    def __init__(self) -> None:
        self.horizontal_sum = 0.0
        self.horizontal_count = 0
        self.vertical_sum = 0.0
        self.vertical_count = 0

    def update(self, unit_grid: torch.Tensor, valid_grid: torch.Tensor) -> None:
        assert unit_grid.ndim == 5, "unit_grid must be [B,V,R,C,D]"
        assert valid_grid.shape == unit_grid.shape[:-1]
        horizontal_mask = valid_grid[:, :, :, 1:] & valid_grid[:, :, :, :-1]
        horizontal = (unit_grid[:, :, :, 1:] * unit_grid[:, :, :, :-1]).sum(dim=-1)
        self.horizontal_sum += float(horizontal[horizontal_mask].sum())
        self.horizontal_count += int(horizontal_mask.sum())
        vertical_mask = valid_grid[:, :, 1:] & valid_grid[:, :, :-1]
        vertical = (unit_grid[:, :, 1:] * unit_grid[:, :, :-1]).sum(dim=-1)
        self.vertical_sum += float(vertical[vertical_mask].sum())
        self.vertical_count += int(vertical_mask.sum())

    def summary(self) -> dict[str, float | int]:
        return {
            "horizontal_pair_count": self.horizontal_count,
            "horizontal_neighbor_cosine": self.horizontal_sum / max(1, self.horizontal_count),
            "vertical_pair_count": self.vertical_count,
            "vertical_neighbor_cosine": self.vertical_sum / max(1, self.vertical_count),
        }


class LayoutAccumulator:
    def __init__(self, layout: Layout, views: int, feature_dim: int) -> None:
        self.layout = layout
        spatial_slots = views * layout.rows * layout.cols
        self.spatial = NormalizedSlotStatistics(spatial_slots, feature_dim)
        self.combined = NormalizedSlotStatistics(
            views * PATCH_START_INDEX + spatial_slots,
            feature_dim,
        )
        self.neighbors = NeighborStatistics()
        self.scene_descriptors: list[torch.Tensor] = []
        self.valid_token_total = 0
        self.sample_count = 0

    def update(
        self,
        special: torch.Tensor,
        pooled: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> None:
        batch, views, rows, cols, feature_dim = pooled.shape
        assert (rows, cols) == (self.layout.rows, self.layout.cols)
        spatial = pooled.reshape(batch, views * rows * cols, feature_dim)
        spatial_valid = spatial_mask.reshape(batch, views * rows * cols)
        special_valid = torch.ones(
            special.shape[:2], dtype=torch.bool, device=special.device
        )
        combined = torch.cat((special, spatial), dim=1)
        combined_valid = torch.cat((special_valid, spatial_valid), dim=1)
        spatial_aligned = F.layer_norm(spatial.float(), (feature_dim,))
        combined_aligned = F.layer_norm(combined.float(), (feature_dim,))
        self.spatial.update(spatial_aligned, spatial_valid)
        self.combined.update(combined_aligned, combined_valid)

        spatial_unit = layer_normalized_unit(spatial).reshape(
            batch, views, rows, cols, feature_dim
        )
        self.neighbors.update(spatial_unit, spatial_mask)
        descriptor = (
            spatial_unit.reshape(batch, -1, feature_dim)
            * spatial_valid.unsqueeze(-1)
        ).sum(dim=1) / spatial_valid.sum(dim=1, keepdim=True).clamp_min(1)
        self.scene_descriptors.append(F.normalize(descriptor, dim=-1).cpu())
        self.valid_token_total += int(spatial_valid.sum())
        self.sample_count += batch

    def summary(self, full_dataset_size: int) -> dict[str, object]:
        descriptors = torch.cat(self.scene_descriptors)
        stored_tokens = 3 * PATCH_START_INDEX + 3 * self.layout.rows * self.layout.cols
        estimated_bytes = full_dataset_size * stored_tokens * descriptors.shape[-1] * 2
        return {
            "rows": self.layout.rows,
            "cols": self.layout.cols,
            "crop_padding": self.layout.crop_padding,
            "native_resolution": self.layout.native_resolution,
            "stored_tokens_per_sample": stored_tokens,
            "mean_valid_spatial_tokens_per_sample": self.valid_token_total / self.sample_count,
            "estimated_full_bf16_cache_gib": estimated_bytes / 1024**3,
            "spatial_slot_statistics": self.spatial.summary(),
            "combined_slot_statistics": self.combined.summary(),
            "scene_descriptor_statistics": summarize_scene_descriptors(descriptors),
            "spatial_neighbor_statistics": self.neighbors.summary(),
        }


def current_padded_pool(
    final_tokens: torch.Tensor,
    validity: torch.Tensor,
    *,
    minimum_valid_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined, combined_mask = extract_vggt_query_targets(
        final_tokens,
        patch_start_idx=PATCH_START_INDEX,
        patch_grid_size=PATCH_GRID_SIZE,
        pooled_grid_size=4,
        spatial_validity=validity,
        minimum_valid_ratio=minimum_valid_ratio,
    )
    batch, views = final_tokens.shape[:2]
    special_count = views * PATCH_START_INDEX
    pooled = combined[:, special_count:].reshape(batch, views, 4, 4, -1)
    mask = combined_mask[:, special_count:].reshape(batch, views, 4, 4)
    return pooled, mask


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main(args: argparse.Namespace) -> None:
    datalist_path = Path(args.datalist_path).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    sensor_root = Path(args.sensor_root).expanduser().resolve() if args.sensor_root else None
    repo_path = Path(args.vggt_repo).expanduser().resolve()
    checkpoint_path = Path(args.vggt_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path, label in (
        (datalist_path, "datalist"),
        (checkpoint_path, "VGGT checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing processed NAVSIM root: {data_root}")
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Missing local VGGT repository: {repo_path}")

    values = json.loads(datalist_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("NAVSIM datalist must be a JSON list of token strings")
    indices = select_evenly_spaced_indices(len(values), int(args.max_samples))
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if len(views) != 3:
        raise ValueError("The resolution probe currently requires exactly three views")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, preprocess = load_local_vggt(
        repo_path,
        checkpoint_path,
        device,
        enable_geometry=False,
    )
    aggregator = model.aggregator
    accumulators: dict[str, LayoutAccumulator] | None = None
    started = time.time()

    iterator = tqdm(
        list(batches(indices, int(args.batch_size))),
        desc="VGGT resolution probe",
    )
    for index_batch in iterator:
        image_batches = []
        validity_batches = []
        for sample_index in index_batch:
            token = values[sample_index]
            with metadata_path(data_root, args.split, token).open("rb") as stream:
                raw = pickle.load(stream)
            paths = scene_image_paths(raw, views, int(args.frame_index), sensor_root)
            images = preprocess([str(path) for path in paths], mode="pad")
            assert images.shape == (len(views), 3, 518, 518)
            image_batches.append(images)
            validity_batches.append(
                torch.stack([patch_validity_for_image(path) for path in paths])
            )
        images = torch.stack(image_batches).to(device=device, dtype=dtype, non_blocking=True)
        validity = torch.stack(validity_batches).to(device=device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            aggregated, patch_start_index = aggregator(images)
        if int(patch_start_index) != PATCH_START_INDEX:
            raise RuntimeError(
                f"VGGT patch_start_idx changed: expected {PATCH_START_INDEX}, "
                f"found {patch_start_index}"
            )
        final_tokens = aggregated[-1]
        batch, view_count, token_count, feature_dim = final_tokens.shape
        assert token_count == PATCH_START_INDEX + PATCH_GRID_SIZE**2
        special = final_tokens[:, :, :PATCH_START_INDEX].reshape(
            batch, view_count * PATCH_START_INDEX, feature_dim
        )
        patches = final_tokens[:, :, PATCH_START_INDEX:].reshape(
            batch, view_count, PATCH_GRID_SIZE, PATCH_GRID_SIZE, feature_dim
        )
        if accumulators is None:
            accumulators = {
                layout.name: LayoutAccumulator(layout, view_count, feature_dim)
                for layout in LAYOUTS
            }
        for layout in LAYOUTS:
            if layout.native_resolution:
                assert (layout.rows, layout.cols) == (PATCH_GRID_SIZE, PATCH_GRID_SIZE)
                pooled = patches
                spatial_mask = validity >= float(args.minimum_valid_ratio)
            elif layout.crop_padding:
                pooled, spatial_mask = crop_and_pool_valid_patches(
                    patches,
                    validity,
                    output_size=(layout.rows, layout.cols),
                    minimum_coverage=float(args.minimum_valid_ratio),
                )
            else:
                pooled, spatial_mask = current_padded_pool(
                    final_tokens,
                    validity,
                    minimum_valid_ratio=float(args.minimum_valid_ratio),
                )
            accumulators[layout.name].update(special, pooled, spatial_mask)
        del aggregated, final_tokens, patches, images, validity

    assert accumulators is not None
    elapsed = time.time() - started
    checkpoint_hash = (
        "SKIPPED" if args.skip_checkpoint_hash else sha256_file(checkpoint_path)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_revision(REPO_ROOT),
        "teacher_commit": git_revision(repo_path),
        "checkpoint_sha256": checkpoint_hash,
        "datalist_sha256": file_sha256(datalist_path),
        "full_dataset_size": len(values),
        "sample_count": len(indices),
        "sampling": "evenly_spaced_over_datalist",
        "first_sample_index": indices[0],
        "last_sample_index": indices[-1],
        "views": list(views),
        "frame_index": int(args.frame_index),
        "preprocess": {"mode": "pad", "target_size": 518, "patch_size": 14},
        "feature_contract": {
            "aggregator_layer": "final",
            "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
            "feature_dim": next(iter(accumulators.values())).spatial.feature_dim,
            "normalization": "per-token LayerNorm then L2",
        },
        "elapsed_seconds": elapsed,
        "samples_per_second": len(indices) / elapsed,
        "diagnostic_paths": {
            "data_root": str(data_root),
            "sensor_root": str(sensor_root) if sensor_root else "",
            "vggt_repo": str(repo_path),
            "vggt_checkpoint": str(checkpoint_path),
        },
        "layouts": {
            name: accumulator.summary(len(values))
            for name, accumulator in accumulators.items()
        },
    }
    current = report["layouts"]["current_padded_4x4"]
    for name, summary in report["layouts"].items():
        if name == "current_padded_4x4":
            continue
        summary["relative_to_current"] = {
            "template_cosine_delta": (
                summary["spatial_slot_statistics"]["slot_template_cosine"]
                - current["spatial_slot_statistics"]["slot_template_cosine"]
            ),
            "residual_rms_ratio": (
                summary["spatial_slot_statistics"]["cross_scene_residual_rms"]
                / current["spatial_slot_statistics"]["cross_scene_residual_rms"]
            ),
            "nearest_scene_margin_ratio": (
                summary["scene_descriptor_statistics"]["self_to_nearest_margin_mean"]
                / current["scene_descriptor_statistics"]["self_to_nearest_margin_mean"]
            ),
        }
    write_json_atomic(output_path, report)
    print(f"[vggt-resolution-probe] COMPLETE samples={len(indices)} output={output_path}")


if __name__ == "__main__":
    main(parse_args())
