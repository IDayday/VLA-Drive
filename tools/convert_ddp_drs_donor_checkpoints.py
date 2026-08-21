#!/usr/bin/env python3
"""Create dimension-tagged DDP-DRS warm starts from local donor files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from starVLA.model.modules.action_model.multi_trajectory.config import (
    DrivoRConfig,
    PlanningConfig,
    SuprimConfig,
)
from starVLA.model.modules.action_model.multi_trajectory.donor_checkpoints import (
    DonorConversionReport,
    convert_drivor_donor_state,
    convert_suprim_donor_state,
    unwrap_tensor_state,
)
from starVLA.model.modules.action_model.multi_trajectory.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.action_model.multi_trajectory.suprim_joint_selector import (
    DriveSuprimJointSelector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return unwrap_tensor_state(checkpoint)


def _write(
    path: Path,
    state: Mapping[str, torch.Tensor],
    report: DonorConversionReport,
    donor_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    metadata = report.as_metadata()
    metadata["donor_sha256"] = donor_sha256
    torch.save({"state_dict": dict(state), "ddp_drs_checkpoint": metadata}, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert local official donor weights to explicit training "
            "initializers. No network access is performed."
        )
    )
    parser.add_argument("--drivor-donor", type=Path, required=True)
    parser.add_argument("--suprim-donor", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ego-status-dim", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ego_status_dim <= 0:
        raise ValueError("--ego-status-dim must be positive")
    planning = PlanningConfig()
    scorer = DrivoRDynamicScorer(
        DrivoRConfig(),
        ego_status_dim=args.ego_status_dim,
        planning_config=planning,
        scene_dim=2048,
    )
    scorer_state, scorer_report = convert_drivor_donor_state(
        _load(args.drivor_donor), scorer.state_dict()
    )
    scorer_path = args.output_dir / "drivor_scorer_2048_memory_warmstart.pth"
    _write(
        scorer_path,
        scorer_state,
        scorer_report,
        _sha256(args.drivor_donor),
    )

    selector = DriveSuprimJointSelector(
        SuprimConfig(vocab_path=str(args.vocab)),
        planning_config=planning,
        scene_dim=2048,
        ego_status_dim=args.ego_status_dim,
    )
    selector_state, selector_report = convert_suprim_donor_state(
        _load(args.suprim_donor), selector.state_dict()
    )
    selector_path = args.output_dir / "suprim_2048_memory_warmstart.pth"
    _write(
        selector_path,
        selector_state,
        selector_report,
        _sha256(args.suprim_donor),
    )
    print(
        json.dumps(
            {
                "drivor": {
                    "path": str(scorer_path.resolve()),
                    "sha256": _sha256(scorer_path),
                    "transferred_key_count": len(scorer_report.transferred_keys),
                    "requires_training": list(scorer_report.requires_training),
                },
                "suprim": {
                    "path": str(selector_path.resolve()),
                    "sha256": _sha256(selector_path),
                    "transferred_key_count": len(selector_report.transferred_keys),
                    "requires_training": list(selector_report.requires_training),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
