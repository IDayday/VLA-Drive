"""Lock full official token membership and one-data-epoch accounting, without rescanning assets."""
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.full_epoch import epoch_budget
from starVLA.rl.flow_grpo.transactions import atomic_json


def prepare(output, published_root):
    output, published_root = Path(output), Path(published_root)
    official = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
    navtest = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    tokens = list(OmegaConf.load(official).tokens)
    test = list(OmegaConf.load(navtest).tokens)
    source = json.loads((published_root / "split_manifest.json").read_text())
    available = json.loads((published_root / "navtrain_tokens.json").read_text())
    if (len(tokens) != 103288 or len(test) != 12146 or len(set(tokens)) != len(tokens)
            or set(tokens) != set(available) or set(tokens) & set(test)):
        raise ValueError("official full train/test asset token contract differs")
    split = {**source, "train_tokens": sorted(tokens, key=digest), "dev_tokens": [],
             "train_scenes": len(tokens), "dev_scenes": 0, "dev_logs": [], "dev_log_count": 0,
             "train_logs": sorted(set(source["token_logs"].values())),
             "selection": "all official navtrain; seeded permutation per SceneStream epoch",
             "experiment_scope": "full navtrain one data epoch, no held-out navtrain or navtest-driven selection"}
    plan = {"schema_version":1, "budget":epoch_budget(len(tokens)),
            "train_filter_sha256":file_sha(official), "test_filter_sha256":file_sha(navtest),
            "source_asset_publication":str(published_root / "published_v2"),
            "train_logs":len(split["train_logs"]), "navtest_scenes":len(test),
            "evaluation_seeds":[42,43,44,45,46], "primary_checkpoint":12912,
            "selection":"fixed last after one data epoch; navtest never used for stopping or checkpoint selection",
            "baseline":"original F frozen_visual step100000; not the previous 64-update policy",
            "historical_rl_dev":"all 1696 formerly held-out scenes now enter training; not a held-out dev set"}
    output.mkdir(parents=True,exist_ok=True)
    for name, content in (("split_manifest.json",split), ("experiment_manifest.json",plan)):
        path = output/name
        if path.exists():
            if json.loads(path.read_text()) != content:
                raise ValueError("conflicting immutable epoch plan: " + str(path))
        else: atomic_json(path,content)
    return plan


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",default="runs/full_navtrain_epoch1/assets")
    p.add_argument("--published-root",default="/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1")
    a=p.parse_args();print(json.dumps(prepare(a.output,a.published_root),indent=2))
