"""Describe paired original-ODE outputs; does not select or tune a policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(sft, rl, output):
    inputs = [p / name for p in (sft, rl)
              for name in ("trajectories.npz", "original_protocol_scores.csv")]
    a, b = np.load(inputs[0]), np.load(inputs[2])
    for data in (a, b):
        if len(set(data["tokens"])) != len(data["tokens"]):
            raise ValueError("duplicate trajectory tokens")
        if not np.isfinite(data["physical"]).all():
            raise ValueError("nonfinite trajectory")
    index = {token: i for i, token in enumerate(b["tokens"])}
    if set(index) != set(a["tokens"]):
        raise ValueError("trajectory token sets differ")
    tokens = a["tokens"]
    x, y = a["physical"], b["physical"][[index[t] for t in tokens]]
    scores = []
    for path in (inputs[1], inputs[3]):
        table = pd.read_csv(path).set_index("token", verify_integrity=True)
        if set(table.index) != set(tokens) or not np.isfinite(table.score).all():
            raise ValueError("score coverage or finiteness failure")
        scores.append(table.loc[tokens])
    sc, rc = scores
    d = pd.DataFrame(dict(token=tokens,
        ade_m=np.linalg.norm(x[..., :2] - y[..., :2], axis=-1).mean(1),
        fde_m=np.linalg.norm(x[:, -1, :2] - y[:, -1, :2], axis=-1),
        endpoint_forward_delta_m=y[:, -1, 0] - x[:, -1, 0],
        endpoint_lateral_delta_m=y[:, -1, 1] - x[:, -1, 1],
        score_sft=sc.score.values, score_rl=rc.score.values))
    groups = dict(all=np.ones(len(d), bool),
        new_zero=(d.score_sft > 0) & (d.score_rl == 0),
        recovered_zero=(d.score_sft == 0) & (d.score_rl > 0),
        collision_component_worse=(rc.no_at_fault_collisions < sc.no_at_fault_collisions).values,
        ttc_component_worse=(rc.time_to_collision_within_bound < sc.time_to_collision_within_bound).values)
    result = {}
    for name, mask in groups.items():
        sub = d.loc[mask]
        result[name] = dict(n=len(sub), ade_mean_m=float(sub.ade_m.mean()),
            ade_median_m=float(sub.ade_m.median()), fde_mean_m=float(sub.fde_m.mean()),
            forward_mean_m=float(sub.endpoint_forward_delta_m.mean()),
            forward_median_m=float(sub.endpoint_forward_delta_m.median()),
            forward_positive_fraction=float((sub.endpoint_forward_delta_m > 0).mean()))
    result["inputs"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result["interpretation"] = ("Descriptive paired displacement, not proof of causation. "
        "Endpoint x is ego-frame forward displacement. No tuning on navtest. "
        "Original evaluation identity checks are performed by intermediate_navtest.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory_changes.json").write_text(json.dumps(result, indent=2) + "\n")
    d.to_csv(output / "trajectory_changes.csv.gz", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for name in ("sft", "rl", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    analyze(args.sft, args.rl, args.output)
