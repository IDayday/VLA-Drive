#!/usr/bin/env python3
"""Precompute dense VGGT patch hidden features for Spatial-Forcing style alignment.

The cache stores uncompressed VGGT patch tokens per NAVSIM sample:

    features:        bf16 [num_views, num_patches, vggt_dim]
    valid_mask:      bool [num_views, num_patches]
    patch_hw:        long [2]
    image_hw:        long [2]
    patch_start_idx: long [1]
    layer_index:     long [1]

Example:

    source load_env.sh
    export VGGT_MODEL=/path/to/VGGT-1B/model.pt
    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      tools/precompute_vggt_dense_cache.py \
      --cache-root /mnt/workspace/VLA-Drive/navsim_feature_cache/vggt_dense_train_layer-1 \
      --datalist /mnt/workspace/VLA-Drive/train_meta.json \
      --data-root /mnt/workspace/VLA-Drive/navsim_dataset \
      --split train \
      --batch-size 1 \
      --map-size-gb 512 \
      --overwrite
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from starVLA.cache.navsim_feature_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    RankCacheWriter,
    write_manifest,
    write_rank_completion,
)

COMPONENT = "vggt_dense"


def distributed_context() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cpu")
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="gloo")
    return rank, world_size, local_rank, device


def file_sha256(path_value: str) -> str:
    digest = hashlib.sha256()
    with open(path_value, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_tree_signature(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    if not path.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path.parent if path.is_file() else path)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def tensor_cpu(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = value.detach()
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.contiguous().cpu()


def chunks(values: List[int], size: int) -> Iterable[List[int]]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def load_config(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.datalist_path = args.datalist
    cfg.datasets.vla_data.data_root = args.data_root
    cfg.datasets.vla_data.split = args.split
    cfg.datasets.video_data.load_2d_data = 0
    cfg.datasets.gs_data.load_3d_data = 0
    cfg.datasets.reward_data.load_reward_data = 0
    cfg.w_depth = 0
    cfg.enable_image_aug = 0
    return cfg


def make_dataset(cfg, max_samples: int | None):
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    os.environ.pop("NAVSIM_AGENT_DINO_CACHE_ROOT", None)
    from starVLA.dataloader.navsim_dataset import NavSimDataset

    return NavSimDataset(
        datalist_path=cfg.datasets.vla_data.datalist_path,
        split=cfg.datasets.vla_data.split,
        video_data_cfg=cfg.datasets.video_data,
        gs_data_cfg=cfg.datasets.gs_data,
        reward_data_cfg=cfg.datasets.reward_data,
        ver_1225=cfg.ver_1225,
        dataset_cfg=cfg.datasets.vla_data,
        all_cfg=cfg,
        max_samples=max_samples,
        data_root=cfg.datasets.vla_data.data_root,
    )


class VGGTDensePrecomputer:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        spatial_root = Path(args.spatial_forcing_root).expanduser().resolve()
        if not (spatial_root / "vggt" / "models" / "vggt.py").is_file():
            raise FileNotFoundError(f"Spatial-Forcing OpenVLA root not found: {spatial_root}")
        if str(spatial_root) not in os.sys.path:
            os.sys.path.insert(0, str(spatial_root))

        from vggt.models.vggt import VGGT

        model_path = Path(args.vggt_model).expanduser().resolve()
        if model_path.is_dir():
            if (model_path / "model.pt").is_file():
                weight_path = model_path / "model.pt"
            elif (model_path / "model.safetensors").is_file():
                weight_path = model_path / "model.safetensors"
            else:
                raise FileNotFoundError(f"Missing model.pt or model.safetensors under VGGT directory: {model_path}")
        elif model_path.is_file():
            weight_path = model_path
        else:
            raise FileNotFoundError(
                f"Missing VGGT weight path: {model_path}. Set VGGT_MODEL or pass --vggt-model."
            )

        self.device = device
        self.model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        if weight_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(weight_path), device="cpu")
        else:
            state = torch.load(str(weight_path), map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.layer_index = int(args.layer_index)
        self.transform = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
        self.image_size = int(args.image_size)
        self.model_path = str(weight_path)
        self.spatial_root = str(spatial_root)

    @torch.inference_mode()
    def __call__(self, samples: List[dict]) -> List[Mapping[str, torch.Tensor]]:
        batch = torch.stack([
            torch.stack([self.transform(image.convert("RGB")) for image in sample["image"]], dim=0)
            for sample in samples
        ], dim=0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.model(batch)
        features_list = output["features"]
        layer_index = self.layer_index
        features = features_list[layer_index]
        patch_start_idx = int(output["patch_start_idx"])
        patch_features = features[:, :, patch_start_idx:, :]
        images = output.get("images", batch)
        height, width = int(images.shape[-2]), int(images.shape[-1])
        patch_h = height // int(self.model.patch_size)
        patch_w = width // int(self.model.patch_size)
        expected = patch_h * patch_w
        if patch_features.shape[2] != expected:
            raise RuntimeError(
                f"Unexpected VGGT patch count: got {patch_features.shape[2]}, expected {expected} "
                f"from image_hw={(height, width)} patch_size={self.model.patch_size}"
            )

        payloads = []
        valid = torch.ones(patch_features.shape[1:3], dtype=torch.bool)
        for index in range(patch_features.shape[0]):
            payloads.append(
                {
                    "features": tensor_cpu(patch_features[index], torch.bfloat16),
                    "valid_mask": valid.clone(),
                    "patch_hw": torch.tensor([patch_h, patch_w], dtype=torch.long),
                    "image_hw": torch.tensor([height, width], dtype=torch.long),
                    "patch_start_idx": torch.tensor([patch_start_idx], dtype=torch.long),
                    "layer_index": torch.tensor([layer_index], dtype=torch.long),
                }
            )
        return payloads


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = distributed_context()
    cfg = load_config(args)
    dataset = make_dataset(cfg, args.max_samples)
    sample_count = len(dataset)
    owned_indices = list(range(rank, sample_count, world_size))
    precomputer = VGGTDensePrecomputer(args, device)
    started = time.time()

    with RankCacheWriter(
        cache_root=args.cache_root,
        component=COMPONENT,
        rank=rank,
        map_size_bytes=int(args.map_size_gb * 1024**3),
        commit_interval=args.commit_interval,
    ) as writer:
        processed = 0
        for batch_indices in chunks(owned_indices, args.batch_size):
            pending = []
            for sample_index in batch_indices:
                token = dataset.raw_list[sample_index]
                if not args.overwrite and writer.contains(token):
                    writer.skipped += 1
                else:
                    pending.append((sample_index, token))
            if not pending:
                continue
            samples = [dataset[sample_index] for sample_index, _ in pending]
            payloads = precomputer(samples)
            for (_, token), payload in zip(pending, payloads):
                writer.put(token, payload, overwrite=args.overwrite)
                processed += 1
            if rank == 0 and processed % args.log_interval < len(pending):
                elapsed = time.time() - started
                print(
                    f"[vggt-dense-cache] rank=0 processed={processed}/{len(owned_indices)} "
                    f"rate={processed / max(elapsed, 1e-6):.2f} samples/s",
                    flush=True,
                )
        completion = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "component": COMPONENT,
            "rank": rank,
            "world_size": world_size,
            "sample_count": sample_count,
            "owned_samples": len(owned_indices),
            "written": writer.written,
            "skipped": writer.skipped,
            "elapsed_seconds": time.time() - started,
        }

    write_rank_completion(args.cache_root, COMPONENT, rank, completion)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        component_path = Path(args.cache_root) / COMPONENT
        completions = []
        for owner_rank in range(world_size):
            path = component_path / f"rank_{owner_rank:05d}.complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing rank completion marker: {path}")
            completions.append(json.loads(path.read_text(encoding="utf-8")))
        manifest = {
            "component": COMPONENT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "sample_count": sample_count,
            "split": args.split,
            "datalist": str(Path(args.datalist).resolve()),
            "datalist_sha256": file_sha256(args.datalist),
            "model_paths": [precomputer.model_path],
            "model_signatures": {
                precomputer.model_path: file_tree_signature(precomputer.model_path),
            },
            "spatial_forcing_root": precomputer.spatial_root,
            "cache_contract": {
                "feature_type": "dense_vggt_patch_hidden",
                "layer_index": int(args.layer_index),
                "image_size": int(args.image_size),
                "patch_size": int(precomputer.model.patch_size),
                "num_views": 3,
                "feature_dim": int(precomputer.model.embed_dim * 2),
                "dtype": "bfloat16",
                "view_order": ["cam_f0", "cam_l0", "cam_r0"],
            },
            "batch_size_per_rank": args.batch_size,
            "rank_completions": completions,
        }
        manifest_path = write_manifest(args.cache_root, COMPONENT, manifest)
        total_elapsed = max(value["elapsed_seconds"] for value in completions)
        print(
            f"[vggt-dense-cache] COMPLETE samples={sample_count} world_size={world_size} "
            f"elapsed={total_elapsed:.1f}s manifest={manifest_path}",
            flush=True,
        )

    del precomputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--vggt-model", default=os.environ.get("VGGT_MODEL", ""))
    parser.add_argument("--spatial-forcing-root", default=os.environ.get("SPATIAL_FORCING_OPENVLA_ROOT", "/mnt/workspace/Spatial-Forcing/openvla-SF"))
    parser.add_argument("--config-yaml", default="starVLA/config/training/cfg_yaw_1225.yaml")
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--map-size-gb", type=int, default=512)
    parser.add_argument("--commit-interval", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if not parsed.vggt_model:
        raise SystemExit("VGGT model path is required. Set VGGT_MODEL or pass --vggt-model.")
    main(parsed)
