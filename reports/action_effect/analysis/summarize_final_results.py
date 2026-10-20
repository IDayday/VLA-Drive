"""Read existing final comparison/RESULT bytes, never models or native feedback.

Execute only after the parent's independently validated native comparison exits0.
This script deliberately does not call compare_methods.load(), score, simulate,
build datasets, initialize any model, or select an arm/seed/checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.action_effect.common import verify_identity
from tools.action_effect.compare_methods import paired_summary

MULTI_ARMS = ("A0", "A1", "B0", "B1-trajectory", "B1-effect")
SINGLE_ARMS = ("ScoreRefiner", "RandomLocal", "FixedLateBrake")
EXPECTED = {f"{a}_seed{s}" for a in MULTI_ARMS for s in (42, 43, 44)} | {
    f"{a}_seed42" for a in SINGLE_ARMS
}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(data), {
        "path": str(path), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def pinned(pin):
    obj, actual = read(pin["path"])
    if actual != pin:
        raise ValueError("Artifact changed: " + str(pin["path"]))
    return obj


def mean_sd_range(values):
    return {"training_seeds": len(values), "mean": statistics.mean(values),
            "sample_sd_ddof1": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values), "maximum": max(values)}


def existing_rows(pin, protocol, population=None):
    result = pinned(pin)
    parent = Path(pin["path"]).parent
    receipt, rp = read(parent / "COMPLETE.json")
    reg, ep = read(parent / "EVALUATION_REGISTRATION.json")
    verify_identity(reg)
    if (receipt["result"] != pin or result["status"] != "COMPLETE" or
            receipt["registration_identity"] != result["registration_identity"] or
            receipt["registration_identity"] != reg["identity"]):
        raise ValueError("Result completion/registration mismatch")
    if protocol is not None and reg["spec"]["score_protocol_hash"] != protocol:
        raise ValueError("Score protocol mismatch")
    rows = {r["scene_id"]: r for r in result["entries"]}
    if len(rows) != len(result["entries"]) or len(rows) != 1696:
        raise ValueError("Full1696 unique scene population required")
    scene_logs = {s: r["source_log"] for s, r in rows.items()}
    if len(set(scene_logs.values())) != 16 or (population is not None and scene_logs != population):
        raise ValueError("Fixed1696/16-log matched population required")
    if any(r["status"] != "VALID_SCORED" or not math.isfinite(r["metrics"]["score"])
           for r in rows.values()):
        raise ValueError("Invalid official result cannot disappear into a subset")
    summary = result["summary"]
    if (summary["requested_scenes"] != 1696 or summary["valid_refined_scenes"] != 1696 or
            summary["invalid_predictions_or_contexts"] or summary["unprocessed_scenes"]):
        raise ValueError("Full-population scoring incomplete")
    recomputed = statistics.mean(r["metrics"]["score"] for r in rows.values())
    if not math.isclose(recomputed, summary["PDMS"], rel_tol=0, abs_tol=2e-14):
        raise ValueError("Stored mean mismatch")
    return rows, scene_logs, reg["spec"]["score_protocol_hash"], {"result": pin, "completion": rp, "registration": ep}


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("--comparison-exit", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    started = time.monotonic()
    completed, exit_pin = read(a.comparison_exit)
    if completed["returncode"] != 0 or completed["all_requested_methods"] != 18:
        raise ValueError("All18 strict native comparisons must have exited0")
    comparison, comparison_pin = read(a.comparison)
    verify_identity(comparison)
    if (comparison["status"] != "COMPLETE" or set(comparison["methods"]) != EXPECTED or
            set(comparison["results"]) != EXPECTED or comparison["selection_performed"] or
            comparison["navtest_accessed"]):
        raise ValueError("Exact registered18-method DEV comparison required")
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fresh report output directory required")
    out.mkdir(parents=True, exist_ok=True)
    base_rows, population, protocol, base_pins = existing_rows(comparison["baseline"], None)
    methods = {}; sources = {}; rows = []
    for name, pin in comparison["methods"].items():
        data, _, _, proof = existing_rows(pin, protocol, population)
        summary = comparison["results"][name]
        if (summary["requested_scenes"] != 1696 or summary["common_valid_scenes"] != 1696 or
                summary["source_logs"] != 16 or summary["invalid_or_unresolved"] or
                not summary["full_population_score_claim"] or summary["bootstrap_repetitions"] != 2000 or
                summary["bootstrap_seed"] != 42):
            raise ValueError("Incomplete matched summary: " + name)
        mean_score = statistics.mean(r["metrics"]["score"] for r in data.values())
        base_score = statistics.mean(r["metrics"]["score"] for r in base_rows.values())
        if (not math.isclose(mean_score, summary["method_mean"], rel_tol=0, abs_tol=2e-14) or
                not math.isclose(base_score, summary["base_mean"], rel_tol=0, abs_tol=2e-14)):
            raise ValueError("Compared source mean mismatch")
        methods[name] = data; sources[name] = proof
        arm, seed = name.rsplit("_seed", 1)
        rows.append({"method": arm, "training_seed": int(seed), "scenes": 1696, "source_logs": 16,
                     "PDMS_100": 100 * summary["method_mean"], "gain_vs_S0_points": 100 * summary["mean_delta"],
                     "gain_CI95_low_points": 100 * summary["delta_ci95"][0],
                     "gain_CI95_high_points": 100 * summary["delta_ci95"][1],
                     **{k: summary[k] for k in ("method_zero", "zero_repaired", "nonzero_to_zero", "high_to_zero",
                                                  "new_NC_failures", "new_DAC_failures", "new_TTC_failures")},
                     "invalid_or_unresolved": 0})
    aggregates = {}
    for arm in MULTI_ARMS + SINGLE_ARMS:
        selected = [r for r in rows if r["method"] == arm]
        aggregates[arm] = {"PDMS_points": mean_sd_range([r["PDMS_100"] for r in selected]),
                           "gain_vs_fixed_S0_points": mean_sd_range([r["gain_vs_S0_points"] for r in selected]),
                           "seeds": sorted(r["training_seed"] for r in selected),
                           "scene_population_per_seed": 1696, "independent_source_logs": 16,
                           "same_scenes_across_seeds": True,
                           "interpretation": "Descriptive training-seed spread; no5088-independent-scene or hierarchical CI claim."}
    for r in rows:
        ag = aggregates[r["method"]]["PDMS_points"]
        r.update(method_seeds=len(aggregates[r["method"]]["seeds"]), method_seed_mean_PDMS_points=ag["mean"],
                 method_seed_sample_SD_points=ag["sample_sd_ddof1"], method_seed_min_PDMS_points=ag["minimum"],
                 method_seed_max_PDMS_points=ag["maximum"])
    contrasts = {}; contrasts_csv = []
    for first, second in (("A1", "A0"), ("B1-effect", "B0"), ("B1-effect", "B1-trajectory")):
        key = first + "_minus_" + second; contrasts[key] = {}
        for seed in (42, 43, 44):
            contrast = paired_summary(methods[f"{second}_seed{seed}"], methods[f"{first}_seed{seed}"],
                                      repetitions=2000, seed=42)
            # Existing row distance is relative to original S0, not to this method comparator.
            contrast["method_mean_change_m_vs_original_S0"] = contrast.pop("mean_trajectory_change_m")
            contrast["between_compared_methods_trajectory_distance"] = None
            contrast["sign"] = f"{first}-{second} (positive favors {first})"
            contrasts[key][str(seed)] = contrast
            contrasts_csv.append({"contrast": key, "training_seed": seed, "scenes": 1696, "source_logs": 16,
                                  "mean_delta_points": 100 * contrast["mean_delta"],
                                  "CI95_low_points": 100 * contrast["delta_ci95"][0],
                                  "CI95_high_points": 100 * contrast["delta_ci95"][1],
                                  **{k: contrast[k] for k in ("nonzero_to_zero", "high_to_zero", "new_NC_failures",
                                                               "new_DAC_failures", "new_TTC_failures")}})
    original_pins = [comparison_pin, exit_pin, comparison["baseline"], *comparison["methods"].values()]
    for pin in original_pins:
        pinned(pin)
    script_bytes = Path(__file__).read_bytes()
    statistics_source = (ROOT / "tools/action_effect/compare_methods.py").read_bytes()
    result = {"schema": "action_effect_read_only_final_policy_statistics_v1", "status": "COMPLETE",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "analysis_script": {"repo_path": str(Path(__file__).relative_to(ROOT)),
                                  "sha256": hashlib.sha256(script_bytes).hexdigest(), "bytes": len(script_bytes)},
              "paired_summary_source": {"repo_path": "tools/action_effect/compare_methods.py",
                                        "sha256": hashlib.sha256(statistics_source).hexdigest(),
                                        "bytes": len(statistics_source)},
              "strict_native_comparison": comparison_pin, "strict_comparison_exit": exit_pin,
              "strict_comparison_identity": comparison["identity"], "source_baseline": base_pins,
              "source_methods": sources, "score_protocol_hash": protocol,
              "population": {"scenes_per_seed": 1696, "source_logs": 16, "role": "dev_report",
                             "same_scene_log_pairs_for_all_methods": True, "independent_scenes_not_5088": True},
              "seed_aggregates": aggregates, "per_seed_vs_S0": comparison["results"],
              "same_seed_method_contrasts": contrasts,
              "uncertainty": "Each CI resamples16 full logs2000 times with seed42. No training-seed pooling into independent scenes.",
              "method_or_seed_selection_performed": False, "source_artifacts_unchanged": True,
              "new_model_forwards": 0, "new_simulator_calls": 0, "new_optimizer_updates": 0,
              "native_revalidation_in_this_derivation": False, "CPU_wall_seconds": time.monotonic() - started}
    (out / "FINAL_POLICY_STATISTICS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_write(out / "FINAL_SYSTEM_RESULTS.csv", [{"method": "S0", "training_seed": "fixed_Base",
                "scenes": 1696, "source_logs": 16, "PDMS_100": 100 * base_score,
                "gain_vs_S0_points": 0, "method_zero": sum(r["metrics"]["score"] <= 1e-8 for r in base_rows.values()),
                "invalid_or_unresolved": 0}, *rows])
    csv_write(out / "SAME_SEED_METHOD_CONTRASTS.csv", contrasts_csv)
    print(json.dumps({"status": "COMPLETE", "method_count": len(rows), "contrast_count": len(contrasts_csv),
                      "CPU_wall_seconds": result["CPU_wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
