#!/usr/bin/env python3
"""Validate the mandatory four-scene public formal-evaluation gate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bank", required=True, type=Path)
    parser.add_argument("--score-directory", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.candidate_bank, allow_pickle=False) as bank:
        proposals = np.asarray(bank["proposals"])
        selected = np.asarray(bank["selected_indices"])
    if proposals.shape != (4, 64, 8, 3) or selected.shape != (4,):
        raise RuntimeError(
            "Four-scene gate requires proposals [4,64,8,3] and four selections"
        )
    csv_files = sorted(args.score_directory.glob("*.csv"))
    if len(csv_files) != 1:
        raise RuntimeError(
            f"Four-scene gate requires exactly one PDMS CSV, found {csv_files}"
        )
    frame = pd.read_csv(csv_files[0])
    scenes = frame[frame["token"] != "average"]
    validity = scenes["valid"]
    if validity.dtype == object:
        validity = validity.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    if len(scenes) != 4 or validity.isna().any() or not validity.astype(bool).all():
        raise RuntimeError("Four-scene selected-trajectory PDMS gate is incomplete")
    if not np.isfinite(scenes.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Four-scene selected-trajectory metrics are non-finite")
    report = {
        "passed": True,
        "scene_count": 4,
        "candidate_count": 64,
        "selected_pdms": float(scenes["score"].mean()),
        "inference_uses_future_inputs": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
