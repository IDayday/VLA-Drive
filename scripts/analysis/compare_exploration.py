"""Compare whole saved banks; verify overlapping experiments and trajectory records."""
import argparse
import json
from pathlib import Path
import numpy as np
from starVLA.rl.flow_grpo.diversity import trajectory_diversity
from starVLA.rl.flow_grpo.loading import file_sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    expected = json.loads(Path(a.manifest).read_text())["tokens"]
    banks, inputs, counts = {}, {}, {}
    equal = 0
    for directory in a.shards:
        root = Path(directory)
        done = json.loads((root / "COMPLETE").read_text())
        if done["results_sha256"] != file_sha(root / "results.json"):
            raise ValueError("changed results: " + directory)
        data = json.loads((root / "results.json").read_text())
        seen = set()
        for row in data["scenes"]:
            token = row["token"]
            if token not in expected or token in seen:
                raise ValueError("unexpected or duplicate token")
            seen.add(token)
            path = root / (token + ".npz")
            inputs[str(path)] = file_sha(path)
            with np.load(path, allow_pickle=False) as arrays:
                for setting, record in row["settings"].items():
                    physical = arrays[setting]
                    rewards = np.asarray([s["score"] for s in record["scores"]])
                    actual = trajectory_diversity(physical, rewards,
                        arrays[setting + "_chain"] if setting != "ode" else None)
                    if actual != record["g16"]:
                        raise ValueError("saved trajectories differ from recorded diversity: " + str(path))
                    key = (token, setting)
                    value = (physical, rewards)
                    if key in banks:
                        if not all(np.array_equal(l, r) for l, r in zip(banks[key], value)):
                            raise ValueError("overlapping fixed-noise experiment differs: " + str(key))
                        equal += 1
                    else:
                        banks[key] = value
                        counts[setting] = counts.get(setting, 0) + 1
        if seen != set(data["identity"]["tokens"]) or len(seen) != done["scenes"]:
            raise ValueError("incomplete shard")
    if any(count != len(expected) for count in counts.values()):
        raise ValueError("a setting did not cover every predeclared scene")
    report = {"status": "PASS", "scope": "saved bank integrity and exact overlapping trajectories/rewards; not a performance pass",
              "settings": counts, "exact_duplicate_scene_settings": equal,
              "unique_candidates": sum(len(value[1]) for value in banks.values()), "inputs": inputs}
    path = Path(a.output)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}))


if __name__ == "__main__":
    main()
