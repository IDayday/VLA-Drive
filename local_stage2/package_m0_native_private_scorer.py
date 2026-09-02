#!/usr/bin/env python3
"""Package one selected M0-native private scorer for online deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from local_stage2.m0_native_private_scorer_agent import (
    build_m0_native_private_scorer_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-artifact", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--private-observation-root",
        type=Path,
        default=None,
        help=(
            "Required for raw/spatial private visual-token rankers. Omit for "
            "rankers trained directly on the Base checkpoint's 16 scene tokens."
        ),
    )
    parser.add_argument("--shortlist-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(
        args.ranker_artifact,
        map_location="cpu",
        weights_only=False,
    )
    payload = build_m0_native_private_scorer_artifact(
        source,
        source_path=args.ranker_artifact,
        base_checkpoint=args.base_checkpoint,
        private_observation_root=args.private_observation_root,
        shortlist_size=args.shortlist_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
