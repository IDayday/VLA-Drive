"""Lock available scene assets, whole-log holdout and paired experiment budget."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import pickle
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.reproducibility import write_asset_manifest
from starVLA.rl.flow_grpo.reward import build_cache_index
from starVLA.rl.flow_grpo.config import resolve_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/ddp_flow_grpo_paired/data")
    parser.add_argument("--dev-scenes", type=int, default=512)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if 16 % args.world_size or args.dev_scenes < 512:
        raise ValueError("global scene batch 16 and >=512 dev scenes required")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cfg, _ = resolve_config("configs/flow_grpo/action_only_frozen_visual.yaml")
    data = Path(cfg["paths"]["data_root"])
    tokens = json.loads(Path(cfg["paths"]["train_list"]).read_text())
    caches = build_cache_index(cfg["paths"]["metric_cache"])
    groups = defaultdict(list)
    token_logs = {}
    assets = []
    for token in tokens:
        path = data / "meta/train" / f"{token}.pkl"
        sample = pickle.load(path.open("rb"))
        images = [
            Path(sample["glo_images"][view]["image_paths"][3])
            for view in ("cam_f0", "cam_l0", "cam_r0")
        ]
        logs = {p.parent.parent.name for p in images}
        if len(logs) != 1:
            raise ValueError(f"different observation logs: {token}")
        log = logs.pop()
        groups[log].append(token)
        token_logs[token] = log
        assets.extend([path, caches[token], *images])
    logs = sorted(groups, key=lambda log: digest(["rl_log_holdout_v1", log]))
    dev_logs, dev = [], []
    for log in logs:
        dev_logs.append(log)
        dev.extend(sorted(groups[log]))
        if len(dev) >= args.dev_scenes:
            break
    train = sorted(set(tokens) - set(dev), key=lambda token: digest(token))
    manifest = {
        "schema": 1,
        "selection": "whole logs ordered by sha256(rl_log_holdout_v1, log); no rewards used",
        "available_train_scenes": len(tokens),
        "available_train_logs": len(groups),
        "source_u_expected_scenes": 103288,
        "experiment_scope": "prepared metadata subset; raw target inventory recorded separately",
        "train_tokens": train,
        "dev_tokens": dev,
        "dev_logs": dev_logs,
        "train_logs": sorted(set(groups) - set(dev_logs)),
        "token_logs": token_logs,
        "calibration_tokens": train[:16],
        "sft_unseen": False,
        "dev_scenes": len(dev),
        "dev_log_count": len(dev_logs),
        "train_scenes": len(train),
    }
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "dev_tokens.json").write_text(json.dumps(dev))
    (root / "rl_train_tokens.json").write_text(json.dumps(train))
    # Include split files and source token lists in immutable asset identity.
    assets.extend(
        [
            cfg["paths"]["train_list"],
            cfg["paths"]["test_list"],
            root / "split_manifest.json",
        ]
    )
    print(
        f"Locking {len(set(map(str, assets)))} input/cache files for {len(train)} train, {len(dev)} dev in {len(dev_logs)} logs",
        flush=True,
    )
    locked = write_asset_manifest(assets, root / "asset_manifest.json")
    for variant in ("frozen_visual", "unfrozen_visual"):
        raw = OmegaConf.to_container(
            OmegaConf.load(f"configs/flow_grpo/action_only_{variant}.yaml")
        )
        raw["paths"].update(
            split_manifest=str(root / "split_manifest.json"),
            asset_manifest=str(root / "asset_manifest.json"),
            asset_manifest_identity=locked["identity"],
        )
        raw["algorithm"]["inner_epochs"] = 2
        raw["runtime"].update(
            accumulation_steps=16 // args.world_size,
            save_every=100,
            run_mode="formal",
            numerical_profile="bf16_zero2_fp32_accum_v1",
            process_group_timeout=120,
            max_updates=2000,
            reward_workers=1,
            output_dir=None,
            acceptance_record=f"reports/ddp_flow_grpo_paired/release_{variant}.json",
        )
        OmegaConf.save(
            OmegaConf.create(raw), f"configs/flow_grpo/paired_{variant}.yaml"
        )
    budget = {
        "variants": ["F→RL", "U→RL"],
        "optimizer_updates": 2000,
        "inner_epochs": 2,
        "global_scene_batch": 16,
        "world_size": args.world_size,
        "accumulation_steps": 16 // args.world_size,
        "group_size": 8,
        "steps": 10,
        "fresh_scene_rollouts": 16000,
        "fresh_candidates": 128000,
        "sft_replay_draws": 16000,
        "sft_replay_loss_exposures": 32000,
        "train_seed": 42,
        "evaluation_seeds": [42, 43, 44, 45, 46],
        "dev_every_updates": 200,
        "save_every_updates": 100,
        "dev_selection": "highest dev EPDMS at seed42 on every 200th update; earliest update wins ties",
        "final_selection": "last and dev-best; no navtest selection",
        "noise_schedule": "global_scene_v1 for train; sha256(seed, token) for evaluation",
        "baseline_scope": "F and U have different SFT step counts; paired RL gains, not a causal SFT visual ablation",
    }
    (root / "experiment_manifest.json").write_text(json.dumps(budget, indent=2))
    print(
        json.dumps(
            {k: manifest[k] for k in ("train_scenes", "dev_scenes", "dev_log_count")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
