"""Compare every nonvisual parameter/alias/group and absolute source optimizer LR."""
from pathlib import Path
import argparse
import json
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.contracts import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-manifest",
        default="reports/ddp_flow_grpo/action_run_evidence/action_frozen_short/actor_parameter_manifest.json",
    )
    parser.add_argument(
        "--unfrozen-manifest",
        default="reports/ddp_flow_grpo/action_run_evidence/action_unfrozen_short/actor_parameter_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="reports/ddp_flow_grpo_paired/paired_parameter_optimizer_contract.json",
    )
    args = parser.parse_args()
    manifests = [
        json.loads(Path(p).read_text())
        for p in (args.frozen_manifest, args.unfrozen_manifest)
    ]
    keys = ("name", "shape", "aliases", "optimizer_group")
    entries = [
        {
            p["name"]: {key: p[key] for key in keys}
            for p in m["parameters"]
            if p["requires_grad"]
        }
        for m in manifests
    ]
    differences = {
        n: [entries[0].get(n), entries[1].get(n)]
        for n in entries[0].keys() | entries[1].keys()
        if entries[0].get(n) != entries[1].get(n)
    }
    optimizers = {}
    for variant in ("frozen_visual", "unfrozen_visual"):
        cfg, sft = resolve_config(f"configs/flow_grpo/action_only_{variant}.yaml")
        active_groups = sorted({p["optimizer_group"] for p in entries[0].values()})
        source_lrs = {
            group: float(sft.trainer.learning_rate[group]) for group in active_groups
        }
        optimizers[variant] = {
            "source_absolute_lr": source_lrs,
            "rl_absolute_lr": {
                k: v * cfg["optimizer"]["initial_lr_multiplier"]
                for k, v in source_lrs.items()
            },
            "adam": OmegaConf.to_container(sft.trainer.optimizer),
            "max_grad_norm": cfg["optimizer"]["max_grad_norm"],
        }
    result = {
        "status": "PASS"
        if not differences
        and optimizers["frozen_visual"] == optimizers["unfrozen_visual"]
        else "FAIL",
        "scope": "per-object canonical manifests from fc354f8 real runs; current CPU/GPU policy manifests checked separately",
        "trainable_tensors": [len(e) for e in entries],
        "fields": keys,
        "differences": differences,
        "manifest_entry_hashes": [digest(e) for e in entries],
        "optimizers": optimizers,
        "optimizer_recipe_identical": optimizers["frozen_visual"]
        == optimizers["unfrozen_visual"],
        "source_paths": [args.frozen_manifest, args.unfrozen_manifest],
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "differences"}, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("paired parameter/optimizer contract differs")


if __name__ == "__main__":
    main()
