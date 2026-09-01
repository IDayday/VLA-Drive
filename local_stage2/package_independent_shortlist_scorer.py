#!/usr/bin/env python3
"""Create an online-deployable independent shortlist scorer artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from local_stage2.independent_scorer_agent import (
    build_independent_shortlist_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker-artifact", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, required=True)
    parser.add_argument(
        "--score-mode", choices=("coarse", "factor", "direct"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ranker = torch.load(
        args.ranker_artifact, map_location="cpu", weights_only=False
    )
    payload = build_independent_shortlist_artifact(
        ranker,
        ranker_artifact_path=args.ranker_artifact,
        base_checkpoint_path=args.base_checkpoint,
        shortlist_size=args.shortlist_size,
        score_mode=args.score_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
