#!/usr/bin/env python3
"""Split a scored [stage0,stage1,stage2,stage3] matrix into stage artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .schema import FACTOR_NAMES


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.combined_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"]
        logs = archive["log_names"]
        factors = archive["candidate_factors"]
        names = archive["candidate_factor_names"].astype(str).tolist()
    if names != list(FACTOR_NAMES) or factors.shape[1:] != (256, 7):
        raise RuntimeError(f"Unexpected combined stage matrix: {factors.shape}, {names}")
    for stage in range(4):
        values = factors[:, stage * 64 : (stage + 1) * 64]
        scores = values[..., -1]
        _atomic_npz(
            args.output_dir / f"stage{stage}_64" / "candidate_scores.npz",
            tokens=tokens,
            log_names=logs,
            candidate_scores=scores,
            oracle_indices=scores.argmax(axis=1).astype(np.int16),
            candidate_factors=values,
            candidate_factor_names=np.asarray(FACTOR_NAMES),
        )
        print(f"stage={stage} scenes={len(tokens)} candidates=64 mean_oracle={scores.max(axis=1).mean():.12f}")


if __name__ == "__main__":
    main()
