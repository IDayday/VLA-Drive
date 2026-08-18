#!/usr/bin/env python3
"""Evaluate V3 through reconstructed layer-11 global and frozen VGGT continuation.

No probe is fitted. Teacher and student 195-token latents use exactly the same
decoder, VGGT layers 12--23, and original camera head.  Depth/point are not
used because their native DPT heads require pre-layer-11 taps.  This makes the
report a strict post-layer-11 knowledge gate rather than another fitted probe.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import VLAAgent  # noqa: E402
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn  # noqa: E402
from starVLA.model.modules.vggt_query.native_codec import (  # noqa: E402
    load_native_codec_checkpoint,
)
from starVLA.model.modules.vggt_query.native_tail import (  # noqa: E402
    resume_frozen_vggt_tail_from_global,
)
from tools.precompute_vggt_query_cache import (  # noqa: E402
    load_local_vggt,
    sha256_file,
)
from tools.validate_vggt_v3_gate import _extract_batch  # noqa: E402


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _dataset(agent: VLAAgent, args) -> NavSimDataset:
    config = deepcopy(agent.model_config)
    config.datasets.video_data.load_2d_data = 0
    config.datasets.gs_data.load_3d_data = 0
    config.datasets.reward_data.load_reward_data = 0
    config.datasets.vla_data.w_neg_traj = None
    config.w_depth = 0
    config.enable_image_aug = 0
    config.framework.vggt.cache.enabled = True
    config.framework.vggt.cache.root = str(args.cache_root)
    config.framework.vggt.cache.strict = True
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    os.environ.pop("NAVSIM_CACHE_COMPONENTS", None)
    os.environ.pop("NAVSIM_AGENT_DINO_CACHE_ROOT", None)
    return NavSimDataset(
        datalist_path=str(args.datalist_path),
        split=args.split,
        video_data_cfg=config.datasets.video_data,
        gs_data_cfg=config.datasets.gs_data,
        reward_data_cfg=config.datasets.reward_data,
        ver_1225=config.ver_1225,
        dataset_cfg=config.datasets.vla_data,
        all_cfg=config,
        data_root=str(args.data_root),
    )


def _decode(codec, vggt, latent, config, device):
    decoded_layer11_global = codec.decode(
        latent.to(device=device, dtype=torch.float32)
    )
    tail = resume_frozen_vggt_tail_from_global(
        vggt.aggregator,
        decoded_layer11_global,
        branch_dim=config.branch_dim,
        source_rows=config.source_rows,
        source_cols=config.source_cols,
    )
    camera = vggt.camera_head([tail[23]])[-1]
    return decoded_layer11_global.float(), camera.float(), tail[23].float()


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    target = target.float()
    prediction = prediction.float()
    return float(
        1.0
        - (prediction - target).square().sum()
        / (target - target.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
    )


def _retention(student: float, teacher: float) -> float | None:
    return student / teacher if teacher > 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--native-codec", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-retention-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.run_dir,
        args.base_vlm,
        args.native_codec,
        args.vggt_repo,
        args.vggt_checkpoint,
        args.cache_root,
        args.datalist_path,
        args.data_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["VGGT_BASE_VLM"] = str(args.base_vlm.resolve())
    os.environ.setdefault("VLM_ATTN_IMPLEMENTATION", "sdpa")
    agent = VLAAgent(
        str(args.run_dir),
        model_iter=args.checkpoint_step,
        device=args.device,
        qwen_forward_mode="auto",
    )
    if int(getattr(agent.model, "vggt_version", -1)) != 3:
        raise RuntimeError("Selected checkpoint is not a formal VGGT V3 model")
    dataset = _dataset(agent, args)
    if not 0 < args.samples <= len(dataset):
        raise ValueError(f"--samples must be in [1,{len(dataset)}]")
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=np.int64).tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_fn,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    codec, codec_metadata = load_native_codec_checkpoint(args.native_codec)
    if not codec_metadata.get("gates", {}).get("teacher_codec_downstream", False):
        raise RuntimeError("Native codec did not pass the teacher downstream gate")
    manifest_path = args.cache_root / "vggt_query" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("teacher_layer_index") != 11 or manifest.get(
        "teacher_attention_branch"
    ) != "global":
        raise RuntimeError("V3 cache source is not VGGT layer-11 global")
    if manifest.get("codec_source_feature") != "layer11_global":
        raise RuntimeError("V3 cache codec source contract changed")
    selected_codec_sha256 = sha256_file(args.native_codec)
    if manifest.get("native_codec_sha256") != selected_codec_sha256:
        raise RuntimeError("Selected native codec does not match the V3 cache identity")
    codec = codec.to(device).freeze_pretrained()
    vggt, _ = load_local_vggt(
        args.vggt_repo,
        args.vggt_checkpoint,
        device,
        enable_camera=True,
        enable_geometry=False,
    )
    config = codec.config
    accumulators = {
        "teacher_layer11_global": [],
        "student_layer11_global": [],
        "teacher_camera": [],
        "student_camera": [],
        "camera_target": [],
        "latent_cosine": [],
        "tail23_cosine": [],
    }
    for examples in tqdm(loader, desc="V3 native downstream"):
        features = _extract_batch(agent.model, examples)
        teacher = features["teacher_memory"].to(device).float()
        student = features["student_memory"].to(device).float()
        camera_targets = []
        for example in examples:
            payload = example["vggt_query_feature_cache"]
            camera_targets.append(payload["camera_target"].float())
        teacher_layer11, teacher_camera, teacher_tail = _decode(
            codec, vggt, teacher, config, device
        )
        student_layer11, student_camera, student_tail = _decode(
            codec, vggt, student, config, device
        )
        accumulators["teacher_layer11_global"].append(teacher_layer11.cpu())
        accumulators["student_layer11_global"].append(student_layer11.cpu())
        accumulators["teacher_camera"].append(teacher_camera.cpu())
        accumulators["student_camera"].append(student_camera.cpu())
        accumulators["camera_target"].append(torch.stack(camera_targets))
        accumulators["latent_cosine"].append(
            F.cosine_similarity(student, teacher, dim=-1).mean(1).cpu()
        )
        accumulators["tail23_cosine"].append(
            F.cosine_similarity(student_tail, teacher_tail, dim=-1).mean((1, 2)).cpu()
        )
    values = {name: torch.cat(parts) for name, parts in accumulators.items()}
    teacher_camera_r2 = _r2(
        values["teacher_camera"].reshape(-1, 9),
        values["camera_target"].reshape(-1, 9),
    )
    student_camera_r2 = _r2(
        values["student_camera"].reshape(-1, 9),
        values["camera_target"].reshape(-1, 9),
    )
    camera_retention = _retention(student_camera_r2, teacher_camera_r2)
    retention_pass = (
        camera_retention is not None
        and camera_retention >= args.student_retention_threshold
    )
    report = {
        "schema_version": 1,
        "design": {
            "student_decoder_fitted": False,
            "codec_output": "layer11_global_only",
            "shared_frozen_path": "codec decoder -> VGGT layers 12-23 -> native camera head",
            "excluded_heads": {
                "depth": "requires pre-layer-11 DPT skip taps",
                "point": "requires pre-layer-11 DPT skip taps",
            },
            "memory_tokens": 195,
        },
        "source": {
            "run_dir": str(args.run_dir.resolve()),
            "checkpoint_step": args.checkpoint_step,
            "native_codec_sha256": selected_codec_sha256,
            "vggt_checkpoint_sha256": sha256_file(args.vggt_checkpoint),
        },
        "samples": args.samples,
        "metrics": {
            "latent_cosine": float(values["latent_cosine"].mean()),
            "decoded_layer11_global_cosine": float(
                F.cosine_similarity(
                    values["student_layer11_global"],
                    values["teacher_layer11_global"],
                    dim=-1,
                ).mean()
            ),
            "decoded_layer23_cosine": float(values["tail23_cosine"].mean()),
            "teacher_camera_r2": teacher_camera_r2,
            "student_camera_r2": student_camera_r2,
            "camera_retention": camera_retention,
        },
        "thresholds": {
            "student_retention": args.student_retention_threshold,
        },
        "gates": {"student_native_downstream_retention": retention_pass},
    }
    _atomic_json(report, args.output)
    print("[V3 student downstream] " + json.dumps(report["gates"], sort_keys=True))
    print(f"[V3 student downstream] report={args.output}")
    if not retention_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
