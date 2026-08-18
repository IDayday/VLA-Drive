#!/usr/bin/env python3
"""Verify layer-11 global alone exactly resumes native VGGT layers 17/23."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.vggt_query.native_tail import (  # noqa: E402
    resume_frozen_vggt_tail_from_global,
)
from tools.precompute_vggt_query_cache import (  # noqa: E402
    DEFAULT_VIEWS,
    git_revision,
    load_local_vggt,
    load_tokens,
    metadata_path,
    scene_image_paths,
    sha256_file,
)


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--vggt-repo", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-absolute-error", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tokens = load_tokens(args.datalist_path, None)
    if not 0 <= args.sample_index < len(tokens):
        raise IndexError("sample index is outside the selected datalist")
    token = tokens[args.sample_index]
    with metadata_path(args.data_root, args.split, token).open("rb") as stream:
        raw = pickle.load(stream)
    paths = scene_image_paths(
        raw, DEFAULT_VIEWS, args.frame_index, args.sensor_root
    )
    model, preprocess = load_local_vggt(
        args.vggt_repo, args.vggt_checkpoint, device, enable_geometry=False
    )
    images = preprocess([str(path) for path in paths], mode="pad").unsqueeze(0)
    images = images.to(device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        taps, patch_start = model.aggregator(images)
        layer11_global = taps[11][..., 1024:]
        resumed = resume_frozen_vggt_tail_from_global(
            model.aggregator,
            layer11_global,
            branch_dim=1024,
            source_rows=37,
            source_cols=37,
        )
    if patch_start != 5:
        raise RuntimeError(f"VGGT patch-start contract changed: {patch_start}")
    metrics = {}
    passed = True
    for layer in (17, 23):
        prediction, target = resumed[layer].float(), taps[layer].float()
        absolute = (prediction - target).abs()
        maximum = float(absolute.max())
        metrics[f"layer{layer}"] = {
            "shape": list(prediction.shape),
            "max_absolute_error": maximum,
            "mean_absolute_error": float(absolute.mean()),
            "cosine": float(F.cosine_similarity(prediction, target, dim=-1).mean()),
        }
        passed = passed and maximum <= args.max_absolute_error
    report = {
        "schema_version": 1,
        "contract": "native layer11 global only -> frozen VGGT layers 12-23",
        "sample_index": args.sample_index,
        "source": {
            "vggt_checkpoint_sha256": sha256_file(args.vggt_checkpoint),
            "vggt_repo_commit": git_revision(args.vggt_repo),
            "datalist_sha256": sha256_file(args.datalist_path),
        },
        "thresholds": {"max_absolute_error": args.max_absolute_error},
        "metrics": metrics,
        "gates": {"native_tail_exact": passed},
    }
    _atomic_json(report, args.output)
    print("[V3 native tail] " + json.dumps(report["gates"], sort_keys=True))
    print(f"[V3 native tail] report={args.output}")
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
