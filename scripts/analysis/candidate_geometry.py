"""Measure coherent trajectory variation versus waypoint jitter in saved banks."""
import argparse
import json
from pathlib import Path
import numpy as np
from starVLA.rl.flow_grpo.loading import file_sha


def coherence(physical):
    xy = np.asarray(physical, dtype=np.float64)[..., :2]
    centered = xy-xy.mean(0)
    left, right = centered[:, :-1], centered[:, 1:]
    norm = float(np.sqrt((left**2).sum()*(right**2).sum()))
    return float((left*right).sum()/norm) if norm > 1e-15 else None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, source, banks = [], {}, {}
    components = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
                  "traffic_light_compliance", "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
    for shard in a.shards:
        root = Path(shard)
        complete = json.loads((root/"COMPLETE").read_text())
        source[str(root)] = file_sha(root/"results.json")
        if source[str(root)] != complete["results_sha256"]:
            raise ValueError("result changed")
        for scene in json.loads((root/"results.json").read_text())["scenes"]:
            token = scene["token"]
            if token in banks:
                raise ValueError("duplicate scene")
            banks[token] = root/(token+".npz")
            with np.load(banks[token], allow_pickle=False) as data:
                for setting, entry in scene["settings"].items():
                    reward = np.asarray([s["score"] for s in entry["scores"]])
                    varied = [k for k in components if np.ptp([s["metrics"][k] for s in entry["scores"]]) > 0]
                    rows.append({"token": token, "setting": setting,
                        "reward_span": float(np.ptp(reward)), "varying_nonprogress_components": varied,
                        "progress_only_differences": bool(np.ptp(reward) > 0 and not varied),
                        "adjacent_waypoint_noise_correlation": coherence(data[setting])})
    if set(banks) != set(manifest["tokens"]):
        raise ValueError("incomplete predeclared scene set")
    summary = {}
    for setting in sorted({r["setting"] for r in rows}):
        selected = [r for r in rows if r["setting"] == setting]
        valid = [r["adjacent_waypoint_noise_correlation"] for r in selected if r["adjacent_waypoint_noise_correlation"] is not None]
        summary[setting] = {"scenes": len(selected),
            "progress_only_nonconstant_groups": sum(r["progress_only_differences"] for r in selected),
            "groups_with_varying_nonprogress_components": sum(bool(r["varying_nonprogress_components"]) for r in selected),
            "median_adjacent_waypoint_noise_correlation": float(np.median(valid)), "coherence_valid": len(valid)}
    (output/"geometry.json").write_text(json.dumps({"status": "COMPLETE", "source": source,
        "definition": "cosine between candidate deviations from the per-waypoint mean at consecutive physical waypoints; not a semantic mode count",
        "summary": summary, "scenes": rows}, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # First four manifest scenes, chosen before scoring. No favorable example selection.
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    for row, token in enumerate(manifest["tokens"][:4]):
        with np.load(banks[token], allow_pickle=False) as data:
            for col, setting in enumerate(["ode", "sde_0.1", "sde_0.3"]):
                xy = data[setting][..., :2].astype(np.float64)
                dx = xy[..., 0]-xy[..., 0].mean(0)
                for candidate in dx:
                    axes[row, col].plot(np.arange(1, 9)*.5, candidate, alpha=.5, linewidth=.8)
                axes[row, col].set_title(f"{token} / {setting}", fontsize=9)
                axes[row, col].set_ylabel("X deviation from group mean (m)")
                axes[row, col].set_ylim(-1., 1.)
                axes[row, col].grid(alpha=.2)
                if row == 3:
                    axes[row, col].set_xlabel("Future time (s)")
    fig.suptitle("All 16 candidates; first four predeclared scenes; common axis scale")
    fig.tight_layout()
    fig.savefig(output/"candidate_deviations.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
