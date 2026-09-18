"""Summarize actual bounded F runs, without issuing a production acceptance."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.loading import file_sha


def summarize(root):
    rows = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    if [row["update"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("expected a complete fresh diagnostic update sequence")
    cfg = json.loads((root / "rl_config.json").read_text())
    inner = cfg["algorithm"]["inner_epochs"]
    for offset in range(0, len(rows), inner):
        first = rows[offset]
        for row in rows[offset:offset + inner]:
            if row["policy_version"] != first["policy_version"]:
                raise ValueError("behavior version changed inside reuse interval")
            for previous, current in zip(first["ranks"], row["ranks"]):
                for key in ("behavior_sha256", "scene_tokens", "replay_tokens"):
                    if previous[key] != current[key]:
                        raise ValueError("behavior data changed within inner epochs: " + key)
    immutable = []
    for rank in rows[-1]["ranks"]:
        path = root / f"immutable_update{len(rows):06d}_rank{rank['rank']}.json"
        proof = json.loads(path.read_text())
        if proof["status"] != "TESTED" or any(proof["changed"].values()):
            raise ValueError("frozen/reference changed: " + str(path))
        immutable.append({"path": str(path), "sha256": file_sha(path)})
    contract = json.loads((root / "actor_parameter_manifest.json").read_text())
    summaries = []
    for row in rows:
        result = {k: v for k, v in row.items() if k != "ranks"}
        result["gradient_tensors_per_rank"] = [r["gradient_tensors"] for r in row["ranks"]]
        result["max_update_seconds"] = max(r["update_seconds"] for r in row["ranks"])
        result["peak_gpu_gib_by_rank"] = [r["peak_gpu_gib"] for r in row["ranks"]]
        summaries.append(result)
    return {
        "scope": "bounded real-model observations; not production acceptance or RL-only proof",
        "path": str(root), "optimizer_updates": len(rows),
        "fresh_scenes": sum(r["scene_count"] for r in rows if r["inner_epoch"] == 0),
        "fresh_candidates": sum(r["candidate_count"] for r in rows if r["inner_epoch"] == 0),
        "replay_scene_exposures": sum(r["replay_scene_count"] for r in rows),
        "trainable_numel": contract["trainable_numel"],
        "contract_sha256": file_sha(root / "actor_parameter_manifest.json"),
        "execution_context": json.loads((root / "execution_context.json").read_text()),
        "training_sha256": file_sha(root / "training.jsonl"),
        "fixed_behavior_reuse": "PASS", "immutable": immutable,
        "observed_dtype_rank0": rows[-1]["ranks"][0]["dtype"],
        "observed_activation_dtype_rank0": rows[-1]["ranks"][0]["activation_dtypes"],
        "rows": summaries,
    }


def initial_bank_equal(left, right, world):
    checked = 0
    fields = ("group_ids", "chain", "old_elementwise_logprob", "old_logprob",
              "rewards", "times", "reference_mean", "reference_std",
              "dimension_mask", "transition_mask", "noise_seed", "candidate_ids")
    for rank in range(world):
        paths = [p / f"rollout_rank{rank}_v0.pt" for p in (left, right)]
        a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        if len(a) != len(b):
            raise ValueError("first behavior microbatch counts differ")
        for l, r in zip(a, b):
            # Inspect the actual dataclass fields, rather than comparing the
            # overall fingerprint, which intentionally includes advantages.
            for field in fields:
                if not hasattr(l, field) or not hasattr(r, field):
                    raise ValueError("missing expected rollout field: " + field)
                lv, rv = getattr(l, field), getattr(r, field)
                equal = torch.equal(lv, rv) if isinstance(lv, torch.Tensor) else lv == rv
                if not equal:
                    raise ValueError(f"initial bank differs: rank {rank}, {field}")
            checked += 1
    return {"status": "PASS", "scope": "first behavior batch, before policy updates",
            "compared_microbatches": checked, "fields": fields,
            "advantages_excluded": "the intended experimental difference"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, output = Path(a.root), Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    runs = {mode: summarize(root / ("f_g16_" + mode)) for mode in ("group", "global_batch")}
    if runs["group"]["contract_sha256"] != runs["global_batch"]["contract_sha256"]:
        raise ValueError("actor parameter contract changed between arms")
    first = initial_bank_equal(root / "f_g16_group", root / "f_g16_global_batch", 8)
    report = {"scope": "diagnostic only", "initial_bank": first, "runs": runs}
    (output / "training_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, ("reward", "advantage_absolute_mean", "pre_clip_grad_norm",
                                 "post_update_probe_ratio_clip_fraction", "reference", "sft")):
        for mode, run in runs.items():
            ax.plot([r["update"] for r in run["rows"]], [r[key] for r in run["rows"]], "o-", label=mode)
        ax.set_title(key)
        ax.set_xlabel("Actual optimizer update")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle("F-SFT initialization, G16, seed42: bounded 8-update diagnostics")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "initial_bank": first}))


if __name__ == "__main__":
    main()
