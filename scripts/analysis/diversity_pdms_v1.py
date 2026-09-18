"""Genuine NAVSIM v1 scores for EVERY saved exploration candidate, isolated from v2."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from scripts.analysis.paired_dev_pdms_v1 import initialize, score_chunk, publish, sha


def main():
    import numpy as np
    import pandas as pd
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--raw-logs", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error("workers must be in [1,16]")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(a.manifest).read_text())
    tokens, bank, saved, sources = manifest["tokens"], {}, {}, {}
    for directory in a.shards:
        root = Path(directory)
        done = json.loads((root/"COMPLETE").read_text())
        if done["results_sha256"] != sha(root/"results.json"):
            raise ValueError("changed diagnostic results")
        for row in json.loads((root/"results.json").read_text())["scenes"]:
            token = row["token"]
            if token in saved:
                raise ValueError("duplicate scene")
            saved[token] = row["settings"]
            sources[token] = sha(root/(token+".npz"))
            with np.load(root/(token+".npz"), allow_pickle=False) as arrays:
                bank[token] = {k: arrays[k] for k in row["settings"]}
    if set(bank) != set(tokens):
        raise ValueError("missing predeclared candidates")
    labels, inputs = {}, {}
    for setting in bank[tokens[0]]:
        for g in range(16):
            label = f"{setting}_candidate{g}"
            folder = out/label
            folder.mkdir()
            np.savez(folder/"trajectories.npz", tokens=tokens,
                     physical=np.stack([bank[t][setting][g] for t in tokens]))
            inputs[label] = str(folder)
            labels[label] = (setting, g)
    spec = {"evaluations": inputs, "raw_logs": a.raw_logs, "maps": a.maps, "cache_root": a.cache_root}
    publish(out/"identity.json", {"scope": "fixed training candidate bank; not dev/Navtest", "spec": spec,
            "manifest_sha256": sha(a.manifest), "candidate_archives_sha256": sources,
            "driver_sha256": sha(score_chunk.__code__.co_filename), "script_sha256": sha(__file__)})
    jobs = {}
    for t in tokens:
        jobs.setdefault(manifest["logs"][t], []).append(t)
    try:
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(spec,)) as pool:
            for part in pool.map(score_chunk, sorted(jobs.items())):
                rows.extend(part)
        frame = pd.DataFrame(rows)
        if len(frame) != len(tokens)*len(labels) or frame.duplicated(["token", "label"]).any():
            raise ValueError("incomplete scores")
        frame["setting"] = [labels[l][0] for l in frame.label]
        frame["candidate"] = [labels[l][1] for l in frame.label]
        frame["v2_training_reward"] = [saved[t][s]["scores"][g]["score"] for t, s, g in
                                       zip(frame.token, frame.setting, frame.candidate)]
        frame.sort_values(["setting", "token", "candidate"]).to_csv(out/"candidates.csv", index=False)
        summary = {}
        for setting, group in frame.groupby("setting"):
            pairs, reversals, nonconstant = 0, 0, 0
            for token, part in group.groupby("token"):
                v1, v2 = part.score.to_numpy(), part.v2_training_reward.to_numpy()
                nonconstant += int(np.ptp(v1) > 0)
                i, j = np.triu_indices(len(part), 1)
                dv1, dv2 = v1[i]-v1[j], v2[i]-v2[j]
                comparable = (dv1 != 0) & (dv2 != 0)
                pairs += int(comparable.sum())
                reversals += int(((dv1*dv2 < 0) & comparable).sum())
            summary[setting] = {"scenes": group.token.nunique(), "candidates": len(group),
                "v1_pdms": float(group.score.mean()), "v1_nonconstant_groups": nonconstant,
                "v1_ttc": float(group.time_to_collision_within_bound.mean()),
                "v1_comfort": float(group.comfort.mean()),
                "v1_drivable": float(group.drivable_area_compliance.mean()),
                "v1_zero_fraction": float((group.score == 0).mean()),
                "comparable_pairs": pairs, "v1_v2_ranking_reversals": reversals}
        publish(out/"summary.json", {"status": "COMPLETE", "scope": "training exploration; not evaluation improvement",
                "settings": summary, "csv_sha256": sha(out/"candidates.csv")})
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        publish(out/"failure.json", {"status": "FAIL", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
