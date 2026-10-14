"""Download the explicitly selected sources; retain licenses and immutable hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


PINNED = {
    "rlinf": ("RLinf/RLinf", "f6c0dafbf321cbd876ffe1c90b378bfd33ddd594", [
        "rlinf/models/embodiment/modules/explore_noise_net.py",
        "rlinf/models/embodiment/openpi/sampling/rl_sampler.py",
        "rlinf/models/embodiment/openpi/tasks/rl.py"]),
    "reinflow": ("ReinFlow/ReinFlow", "e722e151bed767f3ffef47527cf697f2358af55d", [
        "model/flow/mlp_flow.py", "model/flow/ft_ppo/ppoflow.py"]),
    "gdpo": ("NVlabs/GDPO", "4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115", [
        "trl-GDPO/trl-0.18.0-gdpo/trl/trainer/grpo_trainer.py"]),
    "verl": ("verl-project/verl", "4f80e465c2ec79ab9c3c30ec74b9745de61d0490", [
        "recipe/dapo/README.md"]),
    "omnisafe": ("PKU-Alignment/omnisafe", None, ["omnisafe/common/lagrange.py"]),
    "dapo_entry": ("BytedTsinghua-SIA/DAPO", None, ["README.md"]),
}


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VLA-Drive-source-lock"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    prior_path = args.lock if args.lock.exists() else args.lock.with_suffix(".json.partial")
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"sources": {}}
    result = {"schema": "constrained_flow_noise_upstreams_v1", "sources": {}}
    for name, (repo, commit, files) in PINNED.items():
        if commit is None:
            commit = prior["sources"].get(name, {}).get("commit")
            if commit is None:
                commit = json.loads(get(f"https://api.github.com/repos/{repo}/commits/HEAD"))["sha"]
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError(f"incomplete upstream tree: {repo}")
        paths = [x["path"] for x in tree["tree"] if x["type"] == "blob"]
        licenses = [p for p in paths if "/" not in p and p.lower().startswith(("license", "copying", "notice"))]
        if not licenses and name != "dapo_entry":
            raise RuntimeError(f"no repository license found: {repo}")
        wanted = list(files) + licenses
        if name == "verl":
            wanted += [p for p in paths if p.startswith("recipe/dapo/src/") and p.endswith(".py")]
            if "docs/algo/dapo.md" in paths:
                wanted += ["docs/algo/dapo.md"]
        record = {"repository": f"https://github.com/{repo}", "commit": commit,
                  "files": {}, "license_files": licenses}
        if not licenses:
            record["reuse_boundary"] = "Reference README only; no code vendoring without an identified license."
        for file in sorted(set(wanted)):
            if file not in paths:
                raise FileNotFoundError(f"{repo}@{commit}: {file}")
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{file}"
            data = get(url)
            target = args.output / name / file
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError(f"refuse to overwrite changed upstream: {target}")
            target.write_bytes(data)
            record["files"][file] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        result["sources"][name] = record
        print(json.dumps({"source": name, "commit": commit, "files": len(record["files"])}), flush=True)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.lock.with_suffix(".json.partial")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.lock)


if __name__ == "__main__":
    main()
