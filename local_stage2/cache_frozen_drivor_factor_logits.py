#!/usr/bin/env python3
"""Cache released DrivOR factor logits on immutable proposal replay.

The cache contains current-observation predictions only.  Offline PDM targets
remain in the separate label replay and are never passed to model forward.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from local_stage2.train_independent_scorer import (
    ReplaySource,
    _ReplayIndexDataset,
    _atomic_torch_save,
    _sha256,
    load_replay_sources,
)
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--drivor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.private_observation_root, args.drivor_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    data, source_lineage = load_replay_sources(
        sources,
        private_observation_root=args.private_observation_root,
    )
    if tuple(data.observation_tokens.shape[1:]) != (64, 256):
        raise RuntimeError("DrivOR register cache must have shape [scene,64,256]")
    if not bool(data.observation_valid_masks.all()):
        raise RuntimeError("DrivOR register cache unexpectedly contains padding")

    config = DrivORRankerConfig()
    model = DrivORInitializedProposalRanker(config)
    initialization_audit = model.load_drivor_checkpoint(
        args.drivor_checkpoint
    )
    device = torch.device(args.device)
    model.to(device).eval()
    logits = torch.empty((len(data), 64, 6), dtype=torch.float16)
    loader = DataLoader(
        _ReplayIndexDataset(data, range(len(data))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        for (
            proposals,
            observation,
            observation_valid_mask,
            status,
            _base_scores,
            _target_factors,
            source_indices,
            *_training_only_targets,
        ) in loader:
            output = model(
                observation.to(device, non_blocking=True).float(),
                status.to(device, non_blocking=True).float(),
                proposals.to(device, non_blocking=True),
                scene_valid_mask=observation_valid_mask.to(
                    device, non_blocking=True
                ),
            )
            logits[source_indices] = output["factor_logits"].half().cpu()

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "ReleasedDrivORFactorLogitReplay",
        "tokens": data.tokens,
        "log_names": data.log_names,
        "physical_logs": data.physical_logs,
        "source_names": data.source_names,
        "factor_logits": logits,
        "model_config": asdict(config),
        "checkpoint": str(args.drivor_checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.drivor_checkpoint),
        "initialization_audit": initialization_audit,
        "source_lineage": source_lineage,
        "private_observation_root": str(
            args.private_observation_root.resolve()
        ),
        "current_observation_only": True,
        "future_or_evaluator_input": False,
    }
    _atomic_torch_save(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
                "scene_count": len(data),
                "shape": list(logits.shape),
                "checkpoint_sha256": payload["checkpoint_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
