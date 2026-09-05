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
    score_source = parser.add_mutually_exclusive_group(required=True)
    score_source.add_argument("--score-directory", type=Path)
    score_source.add_argument("--scored-candidate-bank", type=Path)
    args = parser.parse_args()
    with np.load(args.candidate_bank, allow_pickle=False) as bank:
        proposals = np.asarray(bank["proposals"])
        selected = np.asarray(bank["selected_indices"])
        tokens = np.asarray(bank["tokens"])
    if proposals.shape != (4, 64, 8, 3) or selected.shape != (4,):
        raise RuntimeError(
            "Four-scene gate requires proposals [4,64,8,3] and four selections"
        )
    if args.scored_candidate_bank is not None:
        with np.load(args.scored_candidate_bank, allow_pickle=False) as scored:
            scored_tokens = np.asarray(scored["tokens"])
            scored_proposals = np.asarray(scored["proposals"])
            names = np.asarray(scored["official_component_names"]).astype(str)
            pdm_columns = np.flatnonzero(names == "pdm_score")
            if len(pdm_columns) != 1:
                raise RuntimeError(
                    "Scored candidate bank must contain exactly one official "
                    f"pdm_score column, got {names.tolist()}"
                )
            scores = np.asarray(scored["pdm_scores"])[
                :, :, int(pdm_columns[0])
            ]
        if not np.array_equal(tokens, scored_tokens):
            raise RuntimeError("Raw/scored candidate-bank token order differs")
        if not np.array_equal(proposals, scored_proposals):
            raise RuntimeError("Raw/scored candidate-bank proposals differ")
        if scores.shape != (4, 64) or not np.isfinite(scores).all():
            raise RuntimeError(
                "Four-scene offline candidate scores must be finite [4,64]"
            )
        selected_scores = scores[np.arange(4), selected]
        score_source_name = "official_offline_candidate_bank"
    else:
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
        selected_scores = scenes["score"].to_numpy(dtype=np.float64)
        score_source_name = "online_selected_trajectory_csv"
    report = {
        "passed": True,
        "scene_count": 4,
        "candidate_count": 64,
        "selected_pdms": float(selected_scores.mean()),
        "score_source": score_source_name,
        "inference_uses_future_inputs": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
