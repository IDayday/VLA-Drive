#!/usr/bin/env python3
"""Precompute critical-agent DINO teacher features for NAVSIM train scenes.

This script follows the existing WAN/PPD feature-cache style: distributed ranks
own disjoint scene indices, each rank writes one LMDB, and rank 0 writes a
manifest after all rank completion files exist.  The cache is independent from
NAVSIM_FEATURE_CACHE_ROOT components and is intended as the teacher cache for
agent-query feature alignment.

Example commands:

    # 1. Mine train critical agents first.  This can be run on a subset for a
    #    smoke test by adding --max-samples.
    python tools/mine_critical_agents_navsim.py \
      --split train \
      --data-root /mnt/data/navsim \
      --log-dir /mnt/data/navsim/train_navsim_logs/train \
      --sensor-dir /mnt/data/navsim/train_sensor_blobs/train \
      --output-dir /mnt/workspace/VLA-Drive/navsim_dataset/critical_agents \
      --overwrite

    # 2. Precompute DINO teacher cache from the train sidecars.
    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      tools/precompute_agent_dino_cache.py \
      --critical-agents-dir /mnt/workspace/VLA-Drive/navsim_dataset/critical_agents/train \
      --cache-root /mnt/workspace/VLA-Drive/navsim_feature_cache/agent_dino_vits14_train \
      --dino-backbone dinov2_vits14 \
      --dino-cache-dir /mnt/workspace/VLA-Drive/weights/derived \
      --sensor-dir /mnt/data/navsim/trainval_sensor_blobs/trainval \
      --batch-size 8 \
      --map-size-gb 64 \
      --overwrite

Important arguments:
    --critical-agents-dir: Directory containing per-token JSON sidecars from
        mine_critical_agents_navsim.py. For teacher training this should be the
        train split sidecar directory, not eval/zero-score sidecars.
    --cache-root: Output cache root. The script writes agent_dino/rank_*.lmdb,
        rank completion JSON files, and agent_dino/manifest.json.
    --dino-backbone: DINOv2 backbone name, e.g. dinov2_vits14.
    --dino-cache-dir: Local torch hub cache root containing hub/checkpoints and
        hub/facebookresearch_dinov2_main. Defaults to weights/derived. The script
        fails if these local files are missing and never downloads weights.
    --sensor-dir: Optional real NAVSIM sensor root used to remap image paths saved
        in sidecars. This is needed when sidecars contain temporary /tmp symlink
        paths from the mining stage.
    --feature-mode: roi_pool (default) pools full-image patch tokens inside the
        projected bbox; crop_patch_mean runs DINO on bbox crops and averages all
        crop patch tokens; crop_cls stores DINO crop CLS tokens.
    --top-k: Optional cap on agents per scene, ordered by sidecar rank.
    --tokens-file: Optional newline-separated allowlist of scene tokens.
    --max-samples: Optional cap for debugging after sorting/filtering sidecars.
    --batch-size: Number of scene sidecars processed per rank step. DINO images
        within the step are internally batched by unique image path.
    --image-batch-size: Maximum number of unique images/crops forwarded through
        DINO at once.
    --map-size-gb: LMDB map size per rank.
    --overwrite: Regenerate existing token records.

Payload per token:
    agent_features: bf16 tensor [num_agents, dino_dim].
    agent_ranks, agent_scores, bbox_xyxy, box_ego, view_ids, patch_counts.
    metadata_json_uint8: JSON bytes with class names, track tokens, image paths,
        feature mode, and original sidecar score terms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

CACHE_SCHEMA_VERSION = 1
COMPONENT = "agent_dino"
VIEW_TO_ID = {"cam_f0": 0, "cam_l0": 1, "cam_r0": 2, "cam_l1": 3, "cam_l2": 4, "cam_r1": 5, "cam_r2": 6, "cam_b0": 7}


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


def serialize(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def component_dir(cache_root: os.PathLike[str] | str) -> Path:
    return Path(cache_root) / COMPONENT


def rank_db_path(cache_root: os.PathLike[str] | str, rank: int) -> Path:
    return component_dir(cache_root) / f"rank_{rank:05d}.lmdb"


class RankCacheWriter:
    def __init__(self, cache_root: str, rank: int, map_size_bytes: int, commit_interval: int = 16) -> None:
        self.path = rank_db_path(cache_root, rank)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            subdir=True,
            map_size=map_size_bytes,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=64,
        )
        self.commit_interval = max(1, int(commit_interval))
        self.transaction = self.env.begin(write=True)
        self.pending = 0
        self.written = 0
        self.skipped = 0

    def contains(self, token: str) -> bool:
        return self.transaction.get(token.encode("utf-8")) is not None

    def put(self, token: str, payload: Mapping[str, Any], overwrite: bool = False) -> bool:
        key = token.encode("utf-8")
        if not overwrite and self.transaction.get(key) is not None:
            self.skipped += 1
            return False
        self.transaction.put(key, serialize(payload), overwrite=True)
        self.pending += 1
        self.written += 1
        if self.pending >= self.commit_interval:
            self.commit()
        return True

    def commit(self) -> None:
        if self.transaction is not None:
            self.transaction.commit()
            self.transaction = self.env.begin(write=True)
            self.pending = 0

    def close(self) -> None:
        if self.transaction is not None:
            self.transaction.commit()
            self.transaction = None
        self.env.sync()
        self.env.close()

    def __enter__(self) -> "RankCacheWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            if self.transaction is not None:
                self.transaction.abort()
                self.transaction = None
            self.env.close()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_sidecar_paths(args: argparse.Namespace) -> List[Path]:
    root = Path(args.critical_agents_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing critical-agents dir: {root}")
    paths = sorted(root.glob("*.json"))
    if not paths:
        paths = sorted(root.rglob("*.json"))
    if args.tokens_file:
        tokens = {line.strip() for line in Path(args.tokens_file).read_text().splitlines() if line.strip()}
        paths = [path for path in paths if path.stem in tokens]
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    return paths


def valid_agents(payload: Dict[str, Any], top_k: Optional[int]) -> List[Dict[str, Any]]:
    agents = [agent for agent in payload.get("critical_agents", []) if agent.get("valid", False)]
    agents = [agent for agent in agents if agent.get("image_path") and agent.get("bbox_xyxy") and agent.get("view")]
    agents.sort(key=lambda item: int(item.get("rank", 10**9)))
    if top_k is not None:
        agents = agents[:top_k]
    return agents


def metadata_to_tensor(metadata: Mapping[str, Any]) -> torch.Tensor:
    data = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def tensor_cpu(value: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    value = value.detach()
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.contiguous().cpu()


def split_sensor_suffix(path: str) -> Optional[str]:
    normalized = os.fspath(path).replace("\\", "/")
    markers = (
        "/sensor_blobs/trainval/",
        "/sensor_blobs/train/",
        "/sensor_blobs/mini/",
        "/sensor_blobs/test/",
    )
    for marker in markers:
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    return None


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def crop_agent(image: Image.Image, bbox_xyxy: Sequence[float], min_size: int) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        cx, cy = width / 2.0, height / 2.0
        half = max(1.0, float(min_size) / 2.0)
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    if x2 - x1 < min_size:
        pad = (min_size - (x2 - x1)) / 2.0
        x1, x2 = x1 - pad, x2 + pad
    if y2 - y1 < min_size:
        pad = (min_size - (y2 - y1)) / 2.0
        y1, y2 = y1 - pad, y2 + pad
    x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2, y2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    return image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))


def dino_feature_dim(backbone_name: str) -> int:
    dims = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1408,
    }
    if backbone_name not in dims:
        raise NotImplementedError(f"DINOv2 backbone {backbone_name} not implemented")
    return dims[backbone_name]


def load_local_dino(backbone_name: str, dino_cache_dir: str) -> torch.nn.Module:
    cache_root = Path(dino_cache_dir).expanduser().resolve()
    hub_dir = cache_root / "hub"
    code_path = hub_dir / "facebookresearch_dinov2_main"
    weights_path = hub_dir / "checkpoints" / f"{backbone_name}_pretrain.pth"
    if not code_path.is_dir():
        raise FileNotFoundError(f"Missing local DINO hub code: {code_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing local DINO weights: {weights_path}")
    os.environ["TORCH_HOME"] = str(cache_root)
    body = torch.hub.load(str(code_path), backbone_name, source="local", pretrained=False)
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    body.load_state_dict(state_dict)
    return body


class AgentDinoPrecomputer:
    def __init__(
        self,
        backbone_name: str,
        device: torch.device,
        image_batch_size: int,
        feature_mode: str,
        min_crop_size: int,
        dino_cache_dir: str,
        sensor_dir: Optional[str] = None,
    ):
        from torchvision import transforms

        self.device = device
        self.model = load_local_dino(backbone_name, dino_cache_dir).to(device).eval()
        self.model.requires_grad_(False)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_batch_size = int(image_batch_size)
        self.feature_mode = feature_mode
        self.min_crop_size = int(min_crop_size)
        self.backbone_name = backbone_name
        self.sensor_dir = Path(sensor_dir).resolve() if sensor_dir else None
        self.feature_dim = dino_feature_dim(backbone_name)
        self.patch_size = 14
        self.input_size = 224

    def resolve_image_path(self, image_path: str) -> str:
        path = Path(image_path)
        if path.is_file():
            return str(path)
        suffix = split_sensor_suffix(image_path)
        if suffix and self.sensor_dir is not None:
            remapped = self.sensor_dir / suffix
            if remapped.is_file():
                return str(remapped)
        searched = [str(path)]
        if suffix and self.sensor_dir is not None:
            searched.append(str(self.sensor_dir / suffix))
        raise FileNotFoundError(f"Missing agent image. searched={searched}")

    def load_rgb(self, image_path: str) -> Image.Image:
        return load_rgb(self.resolve_image_path(image_path))

    @torch.inference_mode()
    def forward_images(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, List[torch.Tensor]] = {"patch": [], "cls": []}
        for start in range(0, len(images), self.image_batch_size):
            batch_images = images[start:start + self.image_batch_size]
            tensor = torch.stack([self.transform(image) for image in batch_images]).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                features = self.model.forward_features(tensor)
            outputs["patch"].append(features["x_norm_patchtokens"].detach())
            outputs["cls"].append(features["x_norm_clstoken"].detach())
        return {key: torch.cat(values, dim=0) for key, values in outputs.items()}

    def roi_pool_feature(self, patch_tokens: torch.Tensor, image_size: tuple[int, int], bbox_xyxy: Sequence[float]) -> tuple[torch.Tensor, int]:
        token_count = int(patch_tokens.shape[0])
        grid = int(round(math.sqrt(token_count)))
        if grid * grid != token_count:
            raise RuntimeError(f"Unexpected DINO patch token count: {token_count}")
        width, height = image_size
        scale_x = self.input_size / max(float(width), 1.0)
        scale_y = self.input_size / max(float(height), 1.0)
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        x1, x2 = sorted((x1 * scale_x, x2 * scale_x))
        y1, y2 = sorted((y1 * scale_y, y2 * scale_y))
        centers = torch.arange(grid, device=patch_tokens.device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(centers * self.patch_size, centers * self.patch_size, indexing="ij")
        mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
        flat_mask = mask.reshape(-1)
        if not bool(flat_mask.any()):
            cx = min(max((x1 + x2) * 0.5, 0.0), float(self.input_size - 1))
            cy = min(max((y1 + y2) * 0.5, 0.0), float(self.input_size - 1))
            col = min(grid - 1, max(0, int(cx // self.patch_size)))
            row = min(grid - 1, max(0, int(cy // self.patch_size)))
            flat_mask[row * grid + col] = True
        selected = patch_tokens[flat_mask]
        return selected.mean(dim=0), int(flat_mask.sum().item())

    @torch.inference_mode()
    def __call__(self, sidecars: List[Dict[str, Any]], top_k: Optional[int]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for sidecar in sidecars:
            token = str(sidecar["token"])
            agents = valid_agents(sidecar, top_k=top_k)
            if not agents:
                outputs.append(self.empty_payload(token, sidecar, reason="no_valid_visible_agents"))
                continue
            if self.feature_mode == "roi_pool":
                outputs.append(self.precompute_roi_pool(token, sidecar, agents))
            else:
                outputs.append(self.precompute_crop(token, sidecar, agents))
        return outputs

    def empty_payload(self, token: str, sidecar: Dict[str, Any], reason: str) -> Dict[str, Any]:
        metadata = {"token": token, "reason": reason, "agents": [], "feature_mode": self.feature_mode}
        return {
            "agent_features": torch.empty((0, self.feature_dim), dtype=torch.bfloat16),
            "agent_ranks": torch.empty((0,), dtype=torch.long),
            "agent_scores": torch.empty((0,), dtype=torch.float32),
            "bbox_xyxy": torch.empty((0, 4), dtype=torch.float32),
            "box_ego": torch.empty((0, 7), dtype=torch.float32),
            "view_ids": torch.empty((0,), dtype=torch.long),
            "patch_counts": torch.empty((0,), dtype=torch.long),
            "metadata_json_uint8": metadata_to_tensor(metadata),
        }

    def precompute_roi_pool(self, token: str, sidecar: Dict[str, Any], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_paths = []
        images = []
        image_sizes = []
        for image_path in dict.fromkeys(str(agent["image_path"]) for agent in agents):
            image = self.load_rgb(image_path)
            image_paths.append(image_path)
            images.append(image)
            image_sizes.append(image.size)
        image_to_index = {path: idx for idx, path in enumerate(image_paths)}
        dino = self.forward_images(images)
        features = []
        patch_counts = []
        for agent in agents:
            image_index = image_to_index[str(agent["image_path"])]
            feature, patch_count = self.roi_pool_feature(
                dino["patch"][image_index],
                image_sizes[image_index],
                agent["bbox_xyxy"],
            )
            features.append(feature)
            patch_counts.append(patch_count)
        return self.build_payload(token, sidecar, agents, features, patch_counts)

    def precompute_crop(self, token: str, sidecar: Dict[str, Any], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        crops = []
        for agent in agents:
            image = self.load_rgb(str(agent["image_path"]))
            crops.append(crop_agent(image, agent["bbox_xyxy"], self.min_crop_size))
        dino = self.forward_images(crops)
        if self.feature_mode == "crop_cls":
            features = [value for value in dino["cls"]]
            patch_counts = [1 for _ in agents]
        elif self.feature_mode == "crop_patch_mean":
            features = [value.mean(dim=0) for value in dino["patch"]]
            patch_counts = [int(dino["patch"].shape[1]) for _ in agents]
        else:
            raise ValueError(self.feature_mode)
        return self.build_payload(token, sidecar, agents, features, patch_counts)

    def build_payload(
        self,
        token: str,
        sidecar: Dict[str, Any],
        agents: List[Dict[str, Any]],
        features: Sequence[torch.Tensor],
        patch_counts: Sequence[int],
    ) -> Dict[str, Any]:
        metadata_agents = []
        for agent in agents:
            metadata_agents.append(
                {
                    "rank": int(agent.get("rank", -1)),
                    "class_name": str(agent.get("class_name", "")),
                    "track_token": str(agent.get("track_token", "")),
                    "instance_token": str(agent.get("instance_token", "")),
                    "view": str(agent.get("view", "")),
                    "image_path": str(agent.get("image_path", "")),
                    "score_terms": agent.get("score_terms", {}),
                    "visible_ratio": float(agent.get("visible_ratio", 0.0)),
                    "occlusion_ratio": float(agent.get("occlusion_ratio", 0.0)),
                    "visible_box_ratio": float(agent.get("visible_box_ratio", 0.0)),
                }
            )
        metadata = {
            "token": token,
            "frame_idx": sidecar.get("frame_idx"),
            "feature_mode": self.feature_mode,
            "dino_backbone": self.backbone_name,
            "agents": metadata_agents,
        }
        return {
            "agent_features": tensor_cpu(torch.stack(list(features), dim=0), torch.bfloat16),
            "agent_ranks": torch.tensor([int(agent.get("rank", -1)) for agent in agents], dtype=torch.long),
            "agent_scores": torch.tensor([float(agent.get("score", 0.0)) for agent in agents], dtype=torch.float32),
            "bbox_xyxy": torch.tensor([agent["bbox_xyxy"] for agent in agents], dtype=torch.float32),
            "box_ego": torch.tensor([agent["box_ego"] for agent in agents], dtype=torch.float32),
            "view_ids": torch.tensor([VIEW_TO_ID.get(str(agent.get("view", "")), -1) for agent in agents], dtype=torch.long),
            "patch_counts": torch.tensor(list(patch_counts), dtype=torch.long),
            "metadata_json_uint8": metadata_to_tensor(metadata),
        }


def chunks(values: List[int], size: int) -> Iterable[List[int]]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = distributed_context()
    paths = load_sidecar_paths(args)
    sample_count = len(paths)
    owned_indices = list(range(rank, sample_count, world_size))
    map_size_bytes = int(args.map_size_gb * 1024**3)
    precomputer = AgentDinoPrecomputer(
        backbone_name=args.dino_backbone,
        device=device,
        image_batch_size=args.image_batch_size,
        feature_mode=args.feature_mode,
        min_crop_size=args.min_crop_size,
        dino_cache_dir=args.dino_cache_dir,
        sensor_dir=args.sensor_dir,
    )
    started = time.time()
    component_path = component_dir(args.cache_root)
    component_path.mkdir(parents=True, exist_ok=True)

    with RankCacheWriter(args.cache_root, rank, map_size_bytes, commit_interval=args.commit_interval) as writer:
        processed = 0
        iterator = chunks(owned_indices, args.batch_size)
        if rank == 0:
            iterator = tqdm(list(iterator), desc="agent-dino cache")
        for batch_indices in iterator:
            pending = []
            sidecars = []
            for sample_index in batch_indices:
                path = paths[sample_index]
                token = path.stem
                if not args.overwrite and writer.contains(token):
                    writer.skipped += 1
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                pending.append((sample_index, token, path))
                sidecars.append(payload)
            if not pending:
                continue
            payloads = precomputer(sidecars, top_k=args.top_k)
            for (_, token, _), payload in zip(pending, payloads):
                writer.put(token, payload, overwrite=args.overwrite)
                processed += 1
            if rank == 0 and processed % args.log_interval < len(pending):
                elapsed = time.time() - started
                print(
                    f"[agent-dino-cache] rank=0 processed={processed}/{len(owned_indices)} "
                    f"rate={processed / max(elapsed, 1e-6):.2f} scenes/s",
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

    write_json_atomic(component_path / f"rank_{rank:05d}.complete.json", completion)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        completions = []
        for owner_rank in range(world_size):
            path = component_path / f"rank_{owner_rank:05d}.complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing rank completion marker: {path}")
            completions.append(json.loads(path.read_text(encoding="utf-8")))
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "complete": True,
            "component": COMPONENT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "sample_count": sample_count,
            "critical_agents_dir": str(Path(args.critical_agents_dir).resolve()),
            "tokens_file": str(Path(args.tokens_file).resolve()) if args.tokens_file else None,
            "dino_backbone": args.dino_backbone,
            "dino_cache_dir": str(Path(args.dino_cache_dir).resolve()),
            "sensor_dir": str(Path(args.sensor_dir).resolve()) if args.sensor_dir else None,
            "dino_model_signature": file_tree_signature(args.dino_cache_dir),
            "feature_mode": args.feature_mode,
            "top_k": args.top_k,
            "payload_contract": {
                "agent_features": "bfloat16[num_agents,dino_dim]",
                "agent_ranks": "int64[num_agents]",
                "agent_scores": "float32[num_agents]",
                "bbox_xyxy": "float32[num_agents,4] original image pixels",
                "box_ego": "float32[num_agents,7] current ego frame",
                "view_ids": VIEW_TO_ID,
                "patch_counts": "int64[num_agents] pooled DINO patch count",
                "metadata_json_uint8": "UTF-8 JSON bytes",
            },
            "batch_size_per_rank": args.batch_size,
            "image_batch_size": args.image_batch_size,
            "rank_completions": completions,
        }
        write_json_atomic(component_path / "manifest.json", manifest)
        total_elapsed = max(value["elapsed_seconds"] for value in completions)
        print(
            f"[agent-dino-cache] COMPLETE samples={sample_count} world_size={world_size} "
            f"elapsed={total_elapsed:.1f}s manifest={component_path / 'manifest.json'}",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critical-agents-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dino-backbone", default="dinov2_vits14")
    parser.add_argument("--dino-cache-dir", default=str(REPO_ROOT / "weights" / "derived"))
    parser.add_argument("--sensor-dir", default=None, help="Real NAVSIM sensor root used to remap stale /tmp sidecar image paths")
    parser.add_argument("--feature-mode", choices=("roi_pool", "crop_patch_mean", "crop_cls"), default="roi_pool")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--tokens-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--commit-interval", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
