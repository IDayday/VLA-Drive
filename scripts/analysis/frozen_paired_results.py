"""Paired, whole-log bootstrap for the fixed F-only short-run comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from starVLA.rl.flow_grpo.evaluation import paired_scores
from starVLA.rl.flow_grpo.loading import file_sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arm in ("sft", "group", "batch"):
        for protocol in ("v1", "v2"):
            parser.add_argument(f"--{arm}-{protocol}", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources, results = {}, {}
    for protocol in ("v1", "v2"):
        frames = {}
        for arm in ("sft", "group", "batch"):
            path = Path(getattr(args, arm + "_" + protocol))
            if not (path.parent / "COMPLETE").is_file():
                raise ValueError("requires complete source evaluation: " + str(path))
            frames[arm] = pd.read_csv(path, dtype={"token": str, "log_name": str}, float_precision="round_trip")
            sources[arm + "/" + protocol] = {"path": str(path), "sha256": file_sha(path)}
        baseline = frames["sft"]
        result = {}
        for arm, comparison in (("group", baseline), ("batch", baseline), ("batch_minus_group", frames["group"])):
            current = frames["batch" if arm == "batch_minus_group" else arm]
            merged, row = paired_scores(comparison, current)
            merged.to_csv(out / (protocol + "_" + arm + "_paired.csv"), index=False)
            # Override the generic F/U wording: every model here starts from F.
            row["causal_scope"] = "fixed F-SFT, seed42, eight updates; not a long-training or Navtest claim"
            row.update(baseline=float(comparison.score.mean()), trained=float(current.score.mean()))
            components = [key for key in ("no_at_fault_collisions", "drivable_area_compliance",
                "ego_progress", "time_to_collision_within_bound", "comfort", "lane_keeping",
                "history_comfort", "two_frame_extended_comfort") if key in current]
            row["components"] = {key: {"baseline": float(comparison[key].mean()),
                "trained": float(current[key].mean()), "available": int(current[key].notna().sum())}
                for key in components}
            result[arm] = row
        results[protocol] = result
    report = {"status": "COMPLETE", "scope": "1696 fixed dev scenes, 16 logs; single inference seed42",
              "sources": sources, "results": results}
    (out / "paired_results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, protocol, name in zip(axes, ("v1", "v2"), ("v1 PDMS", "v2 EPDMS")):
        data = [results[protocol][arm] for arm in ("group", "batch")]
        means = np.array([r["mean_paired_delta"] for r in data]) * 100
        intervals = np.array([r["log_bootstrap_95ci"] for r in data]) * 100
        ax.bar(range(2), means, color=["#3274a1", "#e1812c"], alpha=.85)
        ax.errorbar(range(2), means, yerr=np.stack([means - intervals[:, 0], intervals[:, 1] - means]),
                    fmt="none", color="black", capsize=5)
        ax.set_xticks(range(2), ["G16 / group", "G16 / global batch"])
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_ylabel("Paired change from own F-SFT (points)")
        ax.set_title(name)
    fig.suptitle("Eight updates; 1696 dev scenes; seed42; 95% whole-log bootstrap")
    fig.tight_layout()
    fig.savefig(out / "paired_deltas.png", dpi=170)
    plt.close(fig)
    print(json.dumps({k: {a: v["mean_paired_delta"] for a, v in p.items()} for k, p in results.items()}))


if __name__ == "__main__":
    main()
