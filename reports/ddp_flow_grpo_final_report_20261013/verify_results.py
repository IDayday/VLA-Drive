#!/usr/bin/env python3
"""Read-only reproduction of the archived final DDP Flow-GRPO statistics.

Requires NumPy and pandas; no model, GPU, NAVSIM assets, or original run paths.
All score comparisons use raw 0..1 units with an absolute tolerance of 1e-12.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 47))
SCORE_COLUMNS = [f"{protocol}_{label}" for protocol in ("v1", "v2") for label in ("sft", "rl")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label, scale=1):
    require(
        np.allclose(np.asarray(actual) / scale, np.asarray(expected) / scale, rtol=0, atol=1e-12),
        f"{label}: {actual!r} != {expected!r}",
    )


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def paired_summary(frame, protocol):
    sft = frame[f"{protocol}_sft"].to_numpy()
    rl = frame[f"{protocol}_rl"].to_numpy()
    delta = rl - sft
    grouped = (
        pd.DataFrame({"log_name": frame.log_name.to_numpy(), "delta": delta})
        .groupby("log_name", sort=True).delta.agg(["sum", "count"])
    )
    draws = np.random.default_rng(42).integers(0, len(grouped), size=(2000, len(grouped)))
    estimates = grouped["sum"].to_numpy()[draws].sum(1) / grouped["count"].to_numpy()[draws].sum(1)
    return {
        "scenes": len(frame),
        "logs": len(grouped),
        "mean_paired_delta": float(delta.mean()),
        "log_bootstrap_95ci": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "zero_to_nonzero": int(((sft == 0) & (rl > 0)).sum()),
        "nonzero_to_zero": int(((sft > 0) & (rl == 0)).sum()),
        "high_score_degraded": int(((sft >= 0.9) & (delta < 0)).sum()),
    }


def check_summary(actual, expected, label):
    for key, value in actual.items():
        if key in ("mean_paired_delta", "log_bootstrap_95ci"):
            near(value, expected[key], f"{label}/{key}")
        else:
            require(value == expected[key], f"{label}/{key}: {value} != {expected[key]}")
    require(expected["bootstrap_unit"] == "log", f"{label}: wrong bootstrap unit")
    require(expected["bootstrap_seed"] == 42, f"{label}: wrong bootstrap seed")
    require(expected["high_score_threshold"] == 0.9, f"{label}: wrong high-score threshold")


def check_points(row, sft_mean, rl_mean, summary, label):
    expected = {
        "sft_points": sft_mean * 100,
        "rl_points": rl_mean * 100,
        "delta_points": summary["mean_paired_delta"] * 100,
        "ci95_low_points": summary["log_bootstrap_95ci"][0] * 100,
        "ci95_high_points": summary["log_bootstrap_95ci"][1] * 100,
    }
    for key, value in expected.items():
        near(row[key], value, f"{label}/{key}", scale=100)


def main():
    manifest = read_json("evidence/manifest.json")
    for relative, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        require(actual == expected, f"SHA256 mismatch: {relative}")

    result = read_json("evidence/final_result.json")
    paired = read_json("evidence/final_paired_performance.json")
    require(result["status"] == paired["status"] == "COMPLETE", "Final evaluation incomplete")
    require(result["training"]["status"] == "PASS", "Training not complete")
    require(result["training"]["update"] == 12912, "Wrong final checkpoint")

    frame = pd.read_csv(ROOT / "evidence/final_scene_scores.csv.gz", dtype={"token": str, "log_name": str})
    require(len(frame) == manifest["paired_scene_rows"] == 60730, "Wrong total row count")
    require(set(frame.seed) == set(SEEDS), "Wrong inference seeds")
    require(not frame.duplicated(["seed", "token"]).any(), "Duplicate scene within seed")
    require(not frame.isna().any().any(), "Missing table value")
    numeric = frame.drop(columns=["seed", "token", "log_name"]).to_numpy()
    require(np.isfinite(numeric).all(), "Nonfinite scores/components")
    require(((numeric >= 0) & (numeric <= 1)).all(), "Score/component outside 0..1")

    frames = {seed: frame[frame.seed == seed].sort_values("token").reset_index(drop=True) for seed in SEEDS}
    reference = frames[SEEDS[0]][["token", "log_name"]]
    for seed, rows in frames.items():
        require(len(rows) == 12146 and rows.log_name.nunique() == 136, f"Wrong coverage: seed{seed}")
        require(rows[["token", "log_name"]].equals(reference), f"Unpaired tokens/logs: seed{seed}")

    # Average the five evaluations per scene before resampling whole logs.
    averaged = frame.groupby(["token", "log_name"], as_index=False)[SCORE_COLUMNS].mean()
    aggregate_csv = pd.read_csv(ROOT / "final_metrics.csv").set_index("metric", verify_integrity=True)
    per_seed_csv = pd.read_csv(ROOT / "per_seed_metrics.csv").set_index(["metric", "seed"], verify_integrity=True)
    require(set(aggregate_csv.index) == {"PDMS", "EPDMS"}, "Wrong aggregate CSV metrics")
    require(set(per_seed_csv.index) == {(metric, seed) for metric in ("PDMS", "EPDMS") for seed in SEEDS}, "Wrong per-seed CSV rows")

    for protocol, metric, key in [("v1", "PDMS", "pdms"), ("v2", "EPDMS", "epdms")]:
        archived = paired["results"][protocol]
        sft_means, rl_means = [], []
        for seed, rows in frames.items():
            summary = paired_summary(rows, protocol)
            check_summary(summary, archived["per_seed"][str(seed)], f"{metric}/seed{seed}")
            sft_mean, rl_mean = (float(rows[f"{protocol}_{label}"].mean()) for label in ("sft", "rl"))
            sft_means.append(sft_mean)
            rl_means.append(rl_mean)
            for label, mean in [("sft", sft_mean), ("last", rl_mean)]:
                near(mean, result["evaluations"][label]["seeds"][str(seed)][key], f"result/{metric}/{label}/{seed}")
            csv_row = per_seed_csv.loc[(metric, seed)]
            check_points(csv_row, sft_mean, rl_mean, summary, f"per_seed_csv/{metric}/{seed}")
            for count in ("zero_to_nonzero", "nonzero_to_zero"):
                require(csv_row[count] == summary[count], f"per_seed_csv/{metric}/{seed}/{count}")

        summary = paired_summary(averaged, protocol)
        check_summary(summary, archived["five_seed_mean_paired"], f"{metric}/five_seed_mean")
        for label, means in [("sft", sft_means), ("rl", rl_means)]:
            near(np.mean(means), archived[f"{label}_mean"], f"{metric}/{label}_mean")
            near(np.std(means, ddof=0), archived[f"{label}_seed_std"], f"{metric}/{label}_seed_std")
        csv_row = aggregate_csv.loc[metric]
        check_points(csv_row, np.mean(sft_means), np.mean(rl_means), summary, f"aggregate_csv/{metric}")
        require(csv_row.scenes == 12146 and csv_row.logs == 136 and csv_row.training_seeds == 1, f"{metric}/scope")
        require(csv_row.inference_seeds == "42,43,44,45,46", f"{metric}/inference_seeds")
        for count in ("zero_to_nonzero", "nonzero_to_zero", "high_score_degraded"):
            require(csv_row[f"{count}_on_scene_mean"] == summary[count], f"aggregate_csv/{metric}/{count}")
        lo, hi = np.asarray(summary["log_bootstrap_95ci"]) * 100
        print(f"{metric}: {np.mean(sft_means)*100:.4f} -> {np.mean(rl_means)*100:.4f}; delta {summary['mean_paired_delta']*100:+.4f}; 95% log-bootstrap CI [{lo:+.4f}, {hi:+.4f}] points")

    components = pd.read_csv(ROOT / "final_component_metrics.csv")
    require(len(components) == 14 and not components.duplicated(["protocol", "component"]).any(), "Wrong component rows")
    expected_columns = {"seed", "token", "log_name", *SCORE_COLUMNS}
    for row in components.itertuples(index=False):
        means = {}
        for label in ("sft", "rl"):
            column = f"{row.protocol}_{label}__{row.component}"
            expected_columns.add(column)
            means[label] = float(frame[column].mean()) * 100
            near(getattr(row, f"{label}_points"), means[label], f"components/{column}", scale=100)
        near(row.delta_points, means["rl"] - means["sft"], f"components/{row.protocol}/{row.component}/delta", scale=100)
    require(set(frame.columns) == expected_columns, "Unexpected scene-table columns")
    print(f"PASS: {len(manifest['artifacts'])} archive SHA256 checks; 60,730 paired rows; 12 paired summaries; 14 component means. No inference or rescoring.")
    print(f"Environment: NumPy {np.__version__}, pandas {pd.__version__}; raw-score absolute tolerance 1e-12.")


if __name__ == "__main__":
    main()
