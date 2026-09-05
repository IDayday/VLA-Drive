#!/usr/bin/env python3
"""Strict paired Navtest comparison of two scored PlanReg candidate banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navsim.common.dataloader import MetricCacheLoader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    required = {
        "tokens",
        "proposals",
        "predicted_pdms",
        "selected_indices",
        "candidate_scores",
        "official_component_names",
    }
    missing = required - set(result)
    if missing:
        raise KeyError(f"{path} is missing {sorted(missing)}")
    if len(result["tokens"]) != 12146:
        raise RuntimeError(f"Expected 12,146 Navtest scenes in {path}")
    if result["proposals"].shape != (12146, 64, 8, 3):
        raise RuntimeError(
            f"Expected proposals [12146,64,8,3], got {result['proposals'].shape}"
        )
    return result


def _align(reference: dict, candidate: dict) -> dict:
    reference_tokens = [str(value) for value in reference["tokens"]]
    candidate_tokens = [str(value) for value in candidate["tokens"]]
    if len(set(reference_tokens)) != len(reference_tokens):
        raise RuntimeError("Reference contains duplicate tokens")
    if len(set(candidate_tokens)) != len(candidate_tokens):
        raise RuntimeError("Candidate contains duplicate tokens")
    if set(reference_tokens) != set(candidate_tokens):
        missing = sorted(set(reference_tokens) - set(candidate_tokens))
        extra = sorted(set(candidate_tokens) - set(reference_tokens))
        raise RuntimeError(
            f"Token sets differ: missing={missing[:8]}, extra={extra[:8]}"
        )
    if reference_tokens == candidate_tokens:
        return candidate
    index = {token: row for row, token in enumerate(candidate_tokens)}
    order = np.asarray([index[token] for token in reference_tokens])
    return {
        key: value[order] if value.ndim and value.shape[0] == len(order) else value
        for key, value in candidate.items()
    }


def _state(bank: dict) -> dict:
    official = bank["candidate_scores"].astype(np.float64)
    pdms = official[..., -1]
    selected_indices = bank["selected_indices"].astype(np.int64)
    rows = np.arange(len(pdms))
    selected = pdms[rows, selected_indices]
    oracle = pdms.max(axis=1)
    return {
        "official": official,
        "pdms": pdms,
        "selected_indices": selected_indices,
        "selected": selected,
        "oracle": oracle,
        "regret": oracle - selected,
        "candidate_mean": pdms.mean(axis=1),
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _paired_log_bootstrap(values: dict, groups: np.ndarray, samples: int, seed: int) -> dict:
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    group_sums = {
        name: np.bincount(inverse, weights=np.asarray(value, dtype=np.float64))
        for name, value in values.items()
    }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(samples, dtype=np.float64) for name in values}
    for draw in range(samples):
        chosen = rng.integers(0, len(unique), len(unique))
        denominator = counts[chosen].sum()
        for name in values:
            draws[name][draw] = group_sums[name][chosen].sum() / denominator
    return {
        name: {
            "point_estimate": float(np.asarray(values[name]).mean()),
            "bootstrap_mean": float(value.mean()),
            "ci95": [
                float(np.percentile(value, 2.5)),
                float(np.percentile(value, 97.5)),
            ],
            "probability_gt_zero": float((value > 0).mean()),
        }
        for name, value in draws.items()
    }


def compare(args) -> dict:
    baseline = _load(args.baseline)
    candidate = _align(baseline, _load(args.candidate))
    if tuple(baseline["official_component_names"].tolist()) != tuple(
        candidate["official_component_names"].tolist()
    ):
        raise RuntimeError("Official component ordering differs")
    old = _state(baseline)
    new = _state(candidate)
    deltas = {
        "selected_pdms": new["selected"] - old["selected"],
        "offline_oracle_at_64": new["oracle"] - old["oracle"],
        "scorer_regret_reduction": old["regret"] - new["regret"],
        "candidate_scene_mean_pdms": new["candidate_mean"] - old["candidate_mean"],
    }
    loader = MetricCacheLoader(args.metric_cache)
    logs = np.asarray(
        [
            Path(loader.metric_cache_paths[str(token)])
            .relative_to(args.metric_cache)
            .parts[0]
            for token in baseline["tokens"]
        ]
    )
    component_names = [str(value) for value in baseline["official_component_names"]]
    rows = np.arange(len(logs))
    old_components = old["official"][rows, old["selected_indices"]]
    new_components = new["official"][rows, new["selected_indices"]]
    component_delta = {
        name: _summary(new_components[:, index] - old_components[:, index])
        for index, name in enumerate(component_names)
    }
    proposal_delta = (
        candidate["proposals"].astype(np.float64)
        - baseline["proposals"].astype(np.float64)
    )
    predicted_delta = (
        candidate["predicted_pdms"].astype(np.float64)
        - baseline["predicted_pdms"].astype(np.float64)
    )
    report = {
        "schema_version": 1,
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": _sha256(args.baseline),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": _sha256(args.candidate),
        },
        "scene_count": len(logs),
        "log_count": int(len(np.unique(logs))),
        "candidate_count": 64,
        "inference_uses_future_inputs": False,
        "offline_oracle_definition": (
            "Offline best of each model's same 64 frozen proposals; not a deployable policy."
        ),
        "baseline_metrics": {
            "selected_pdms": float(old["selected"].mean()),
            "offline_oracle_at_64": float(old["oracle"].mean()),
            "scorer_regret": float(old["regret"].mean()),
            "candidate_mean_pdms": float(old["pdms"].mean()),
        },
        "candidate_metrics": {
            "selected_pdms": float(new["selected"].mean()),
            "offline_oracle_at_64": float(new["oracle"].mean()),
            "scorer_regret": float(new["regret"].mean()),
            "candidate_mean_pdms": float(new["pdms"].mean()),
        },
        "paired_deltas_candidate_minus_baseline": {
            name: _summary(value) for name, value in deltas.items()
        },
        "paired_log_cluster_bootstrap": _paired_log_bootstrap(
            deltas, logs, args.bootstrap_samples, args.seed
        ),
        "selected_component_deltas_candidate_minus_baseline": component_delta,
        "selected_index_change_fraction": float(
            (old["selected_indices"] != new["selected_indices"]).mean()
        ),
        "proposal_query_aligned_rms_change": float(
            np.sqrt(np.mean(np.square(proposal_delta)))
        ),
        "proposal_endpoint_rms_change": float(
            np.sqrt(np.mean(np.square(proposal_delta[..., -1, :2])))
        ),
        "predicted_score_rms_change": float(
            np.sqrt(np.mean(np.square(predicted_delta)))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].hist(deltas["selected_pdms"], bins=60)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(title="Selected PDMS paired change", xlabel="candidate - baseline")
    axes[1].hist(deltas["offline_oracle_at_64"], bins=60)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Offline Oracle@64 paired change", xlabel="candidate - baseline")
    axes[2].scatter(old["regret"], new["regret"], s=2, alpha=0.25)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[2].set(title="Scorer regret", xlabel="baseline", ylabel="candidate")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
