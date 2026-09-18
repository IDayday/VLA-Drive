"""Validate and summarize the complete, predeclared exploration diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from starVLA.rl.flow_grpo.loading import file_sha


def summarize(manifest_path, directories, output):
    manifest = json.loads(Path(manifest_path).read_text())
    scenes, inputs = {}, {}
    for directory in directories:
        path = Path(directory)
        done = json.loads((path/"COMPLETE").read_text())
        actual = file_sha(path/"results.json")
        if actual != done["results_sha256"]:
            raise ValueError("diversity result changed after completion")
        data = json.loads((path/"results.json").read_text())
        if len(data["scenes"]) != done["scenes"] or len(data["scenes"]) != len(data["identity"]["tokens"]):
            raise ValueError("incomplete diagnostic shard")
        inputs[str(path)] = actual
        for row in data["scenes"]:
            if row["token"] in scenes:
                raise ValueError("duplicate diagnostic scene")
            if row["token"] not in data["identity"]["tokens"]:
                raise ValueError("scene outside shard identity")
            scenes[row["token"]] = row
    if set(scenes) != set(manifest["tokens"]):
        raise ValueError("missing/extra predeclared scenes")
    rows, summary = [], {}
    settings = ["ode", *[f"sde_{n}" for n in manifest["noise_levels"]]]
    for setting in settings:
        summary[setting] = {}
        for group in ("g8_prefix", "g16"):
            metrics = [scenes[t]["settings"][setting][group] for t in manifest["tokens"]]
            score_keys = sorted(scenes[manifest["tokens"][0]]["settings"][setting]["scores"][0]["metrics"])
            reward_components = {k: [] for k in score_keys}
            for token, metric in zip(manifest["tokens"], metrics):
                scores = scenes[token]["settings"][setting]["scores"][:metric["group_size"]]
                for k in score_keys:
                    reward_components[k].extend(s["metrics"][k] for s in scores if s["metrics"][k] is not None)
                rows.append({"token": token, "log": manifest["logs"][token], "setting": setting,
                    "group": group, **{k: metric[k] for k in ["reward_mean", "reward_std", "reward_span",
                        "all_equal_rewards", "reward_zero_fraction", "xy_covariance_effective_rank"]},
                    "median_pair_ade_m": metric["pair_ade_m"]["0.5"],
                    "median_pair_fde_m": metric["pair_fde_m"]["0.5"]})
            standard = np.asarray([m["reward_std"] for m in metrics])
            item = {"scenes": len(metrics), "candidates": sum(m["group_size"] for m in metrics),
                "nonconstant_groups": sum(not m["all_equal_rewards"] for m in metrics),
                "reward_std_gt": {str(t): int((standard > t).sum()) for t in [0, .001, .005, .01, .05]},
                "median_pair_ade_m": float(np.median([m["pair_ade_m"]["0.5"] for m in metrics])),
                "median_pair_fde_m": float(np.median([m["pair_fde_m"]["0.5"] for m in metrics])),
                "mean_training_reward": float(np.mean([m["reward_mean"] for m in metrics])),
                "candidate_max_reward_mean_diagnostic_only": float(np.mean([m["diagnostic_candidate_max_reward"] for m in metrics])),
                "mean_reward_std": float(standard.mean()), "median_reward_std": float(np.median(standard)),
                "zero_fraction": float(np.mean([m["reward_zero_fraction"] for m in metrics])),
                "reward_component_means": {k: float(np.mean(v)) for k, v in reward_components.items() if v}}
            if setting != "ode":
                curves = np.asarray([m["normalized_chain_rms_std"] for m in metrics])
                item["median_normalized_chain_rms_std"] = np.median(curves, axis=0).tolist()
            summary[setting][group] = item
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/"per_scene.csv", index=False)
    result = {"status": "COMPLETE", "scope": "training exploration diagnostic, not dev/navtest improvement",
        "manifest_sha256": file_sha(manifest_path), "inputs": inputs,
        "logs": len(set(manifest["logs"].values())), "settings": summary,
        "group_comparison": "first eight vs all sixteen of the SAME bank; not a separate G=8 RNG run"}
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for group, label in [("g8_prefix", "first 8"), ("g16", "all 16")]:
        axes[0].plot(settings, [summary[s][group]["median_pair_ade_m"] for s in settings], "o-", label=label)
        axes[1].plot(settings, [summary[s][group]["nonconstant_groups"] for s in settings], "o-", label=label)
    axes[0].set(ylabel="Median within-scene pair ADE (m)", title="Physical spread (same starting noises)")
    axes[1].set(ylabel="Scenes with nonconstant reward", ylim=(0, len(scenes)), title="Reward distinguishability")
    for s in settings[1:]:
        axes[2].semilogy(range(11), summary[s]["g16"]["median_normalized_chain_rms_std"], "o-", label=s)
    axes[2].set(xlabel="Flow transition", ylabel="Normalized candidate RMS std", title="Diversity along the saved chain")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
        if ax is not axes[2]:
            ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output/"diversity.png", dpi=180)
    plt.close(fig)
    print(json.dumps({s: summary[s]["g16"] for s in settings}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.manifest, args.shards, args.output)
