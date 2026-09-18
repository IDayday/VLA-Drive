"""Rescore every saved candidate with genuine NAVSIM v1; never resample/select.

Input .pt files must be trusted, locally generated RolloutBatch snapshots. This
CPU diagnostic neither changes training rewards nor publishes an evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from scripts.analysis.paired_dev_pdms_v1 import initialize, publish, score_chunk, sha


def main():
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--raw-logs", required=True)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in [1,16]")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    token_logs = json.loads(Path(args.split_manifest).read_text())["token_logs"]
    tokens, physical, saved = [], [], []
    inputs = {}
    for name in args.rollouts:
        path = Path(name)
        inputs[str(path)] = sha(path)
        for rollout in torch.load(path, map_location="cpu", weights_only=False):
            for i, token in enumerate(rollout.observation.tokens):
                if token in tokens:
                    raise ValueError("duplicate scene: explicitly audit behavior batches separately")
                tokens.append(token)
                physical.append(np.asarray(rollout.physical_trajectories[i]))
                for g, record in enumerate(rollout.score_records[i]):
                    saved.append({"token": token, "label": f"candidate{g}",
                                  "v2_training_reward": record.score,
                                  "saved_advantage": float(rollout.advantages[i, g]),
                                  **{"v2_"+k: v for k, v in record.metrics.items()}})
    physical = np.stack(physical)
    if physical.shape[2:] != (8, 3) or not np.isfinite(physical).all():
        raise ValueError("invalid physical candidate bank")
    directories = {}
    for g in range(physical.shape[1]):
        folder = output / f"candidate{g}"
        folder.mkdir()
        np.savez(folder / "trajectories.npz", tokens=tokens, physical=physical[:, g])
        directories[folder.name] = str(folder)
    spec = {"evaluations": directories, "raw_logs": args.raw_logs,
            "maps": args.maps, "cache_root": args.cache_root}
    publish(output / "identity.json", {"scope": "saved training behavior diagnostic; not dev/navtest",
            "inputs_sha256": inputs, "split_manifest_sha256": sha(args.split_manifest),
            "script_sha256": sha(__file__), "scorer_driver_sha256": sha(score_chunk.__code__.co_filename),
            "tokens": tokens, "group_size": physical.shape[1], "spec": spec})
    jobs = {}
    for token in tokens:
        jobs.setdefault(token_logs[token], []).append(token)
    try:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(spec,)) as pool:
            for part in pool.map(score_chunk, sorted(jobs.items())):
                rows.extend(part)
        scores = pd.DataFrame(rows).merge(pd.DataFrame(saved), on=["token", "label"], validate="one_to_one")
        if len(scores) != len(saved):
            raise ValueError("incomplete scored candidate bank")
        scores.sort_values(["token", "label"]).to_csv(output / "candidates.csv", index=False)
        pairs, disagree = 0, 0
        for _, group in scores.groupby("token"):
            a = group.v2_training_reward.to_numpy()
            b = group.score.to_numpy()
            for i in range(len(a)):
                for j in range(i):
                    if a[i] != a[j] and b[i] != b[j]:
                        pairs += 1
                        disagree += int((a[i]-a[j])*(b[i]-b[j]) < 0)
        summary = {"status": "COMPLETE", "scope": "training-candidate diagnostic, not an RL performance estimate",
            "scenes": len(tokens), "candidates": len(scores), "strict_pairs": pairs,
            "ranking_disagreements": disagree,
            "v1_ttc_failures": int((scores.time_to_collision_within_bound < 1).sum()),
            "positive_v2_advantage_with_v1_ttc_failure": int(((scores.saved_advantage > 0) &
                (scores.time_to_collision_within_bound < 1)).sum()),
            "mean_v1_pdms": float(scores.score.mean()),
            "mean_v2_training_reward": float(scores.v2_training_reward.mean()),
            "csv_sha256": sha(output / "candidates.csv")}
        publish(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        publish(output / "failure.json", {"status": "FAIL", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
