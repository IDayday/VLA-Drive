#!/usr/bin/env python3
"""Measure consequence diversity in the frozen EpisodeDrive proposal bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    write_json,
    write_markdown,
)


def analyze(args: argparse.Namespace) -> dict[str, float | int]:
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    metrics = pd.read_parquet(
        report_dir / "candidates/episode_drive_candidate_metrics.parquet"
    )
    score_range = metrics.groupby("scene_token").aggregate_score.agg(
        lambda values: values.max() - values.min()
    )
    collision_unique = metrics.groupby("scene_token").no_at_fault_collision.nunique()
    ttc_unique = metrics.groupby("scene_token").ttc.nunique()
    variances = []
    pair_rates = []
    target_paths = sorted((cache_dir / "model_candidates/targets_v3").glob("*/*.npz"))
    forbidden = {"aggregate_score", "official_score", "pdm_score", "candidate_type", "candidate_family"}
    leakage_count = 0
    for path in target_paths:
        with np.load(path, allow_pickle=False) as target:
            leakage_count += int(bool(forbidden & set(target.files)))
            consequence = np.concatenate(
                [
                    target["D_state_summary"].reshape(args.num_candidates, -1),
                    target["D_risk"].reshape(args.num_candidates, -1),
                    target["D_signal"].reshape(args.num_candidates, -1),
                ],
                axis=-1,
            )
        variances.append(float(np.var(consequence, axis=0).mean()))
        distances = np.linalg.norm(consequence[:, None] - consequence[None, :], axis=-1)
        pair_rates.append(
            float((distances[np.triu_indices(args.num_candidates, 1)] > 1e-6).mean())
        )
    result = {
        "scene_count": int(metrics.scene_token.nunique()),
        "candidate_count": int(len(metrics)),
        "candidate_score_nonconstant_scene_rate": float((score_range > 1e-9).mean()),
        "mean_within_scene_score_range": float(score_range.mean()),
        "scenes_with_collision_factor_difference": float((collision_unique > 1).mean()),
        "scenes_with_ttc_factor_difference": float((ttc_unique > 1).mean()),
        "candidate_collision_bad_rate": float((metrics.no_at_fault_collision < 0.5).mean()),
        "candidate_ttc_bad_rate": float((metrics.ttc < 0.5).mean()),
        "dynamic_summary_variance_mean": float(np.mean(variances)),
        "nonzero_dynamic_pair_rate_mean": float(np.mean(pair_rates)),
        "target_scene_count": len(target_paths),
        "target_leakage_count": leakage_count,
    }
    write_json(report_dir / "candidates/model_candidate_diversity.json", result)
    write_markdown(
        report_dir / "MODEL_CANDIDATE_DIVERSITY.md",
        f"""# EpisodeDrive Proposal Consequence Diversity

- Scenes / candidates: {result['scene_count']:,} / {result['candidate_count']:,}
- Non-constant official outcome by scene: {result['candidate_score_nonconstant_scene_rate']:.3%}
- Mean within-scene official score range: {result['mean_within_scene_score_range']:.4f}
- Scenes with collision / TTC-factor differences: {result['scenes_with_collision_factor_difference']:.3%} / {result['scenes_with_ttc_factor_difference']:.3%}
- Candidate collision / TTC failure rate: {result['candidate_collision_bad_rate']:.3%} / {result['candidate_ttc_bad_rate']:.3%}
- Non-zero dynamic-consequence pair rate: {result['nonzero_dynamic_pair_rate_mean']:.3%}
- Target leakage violations: {leakage_count}

These are candidate-conditioned relabels of one shared logged future under the
non-reactive assumption. Official scores are used only for this offline audit.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    args = parser.parse_args()
    result = analyze(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.analyze_model_candidate_diversity "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()
