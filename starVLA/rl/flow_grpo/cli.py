import argparse
from pathlib import Path
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Full SFT-parameter DriveDreamer Flow-GRPO"
    )
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "diagnose",
            "train",
            "export",
            "evaluate",
            "calibrate",
            "verify-export",
        ],
    )
    parser.add_argument(
        "--config", default="configs/flow_grpo/action_only_frozen_visual.yaml"
    )
    parser.add_argument("--sft-ckpt")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--export-dir")
    parser.add_argument("--split", choices=["rl_dev", "navtest"])
    parser.add_argument("--tokens")
    parser.add_argument("--data-root")
    parser.add_argument("--metric-cache")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--metric-protocol", choices=["navsim_v2_official_one_stage"])
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--set", nargs="*", action="extend", default=[])
    args = parser.parse_args()
    if args.command == "export":
        from .checkpoint import export_checkpoint

        if not args.checkpoint or not args.output_dir:
            parser.error("export needs --checkpoint and --output-dir")
        print(export_checkpoint(args.checkpoint, args.output_dir))
        return
    from .config import resolve_config, split_tokens

    overrides = list(args.set)
    if os.environ.get("REWARD_WORKERS"):
        overrides.append("runtime.reward_workers=" + os.environ["REWARD_WORKERS"])
    cfg, sft = resolve_config(
        args.config, args.sft_ckpt, args.max_updates, args.output_dir, overrides
    )
    from omegaconf import OmegaConf

    if args.command == "train":
        from .trainer import run

        return run(cfg, sft, args.resume)
    if args.command == "diagnose":
        from .diagnostics import run_diagnostics

        return run_diagnostics(
            cfg,
            sft,
            args.output_dir or "reports/ddp_flow_grpo/diagnostic",
            not args.skip_gradients,
        )
    if args.command == "calibrate":
        from .calibration import calibrate

        return calibrate(
            cfg, sft, args.output_dir or "reports/ddp_flow_grpo/calibration"
        )
    if args.command == "verify-export":
        if not all([args.checkpoint, args.export_dir, args.output_dir]):
            parser.error(
                "verify-export requires --checkpoint, --export-dir, --output-dir"
            )
        from .export_validation import verify_export

        return verify_export(
            cfg, sft, args.checkpoint, args.export_dir, args.output_dir
        )
    if args.command == "evaluate":
        from .evaluation import evaluate

        if any(
            x is None
            for x in (
                args.split,
                args.tokens,
                args.data_root,
                args.metric_cache,
                args.seed,
                args.metric_protocol,
            )
        ):
            parser.error(
                "evaluate requires --split --tokens --data-root --metric-cache --seed --metric-protocol"
            )
        return evaluate(
            cfg,
            sft,
            args.checkpoint,
            args.output_dir,
            split=args.split,
            tokens_file=args.tokens,
            data_root=args.data_root,
            metric_cache=args.metric_cache,
            seed=args.seed,
            metric_protocol=args.metric_protocol,
        )
    from .reward import build_cache_index
    from .loading import file_sha, weight_path

    train, val = split_tokens(cfg)
    index = build_cache_index(cfg["paths"]["metric_cache"])
    missing = set(train + val) - index.keys()
    result = dict(
        code_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        checkpoint=str(weight_path(cfg["sft_checkpoint"])),
        checkpoint_sha256=file_sha(weight_path(cfg["sft_checkpoint"])),
        framework=sft.framework.name,
        head="GR00T_ActionHeader.FlowmatchingActionHead",
        freeze_modules=sft.trainer.freeze_modules,
        train_scenes=len(train),
        validation_scenes=len(val),
        missing_metric_caches=len(missing),
        auxiliary=dict(
            video=int(sft.datasets.video_data.load_2d_data),
            depth=int(sft.w_depth),
            rgb_query=int(sft.rgb_query_loss),
            gs_query=int(sft.gs_query_loss),
        ),
        config=cfg,
        reference_lock=json.loads(Path("reference_lock.json").read_text()),
    )
    if result["checkpoint_sha256"] != cfg["checkpoint_contract"]["sha256"]:
        raise ValueError("SFT checkpoint SHA does not match audited contract")
    output = Path(args.output_dir or "reports/ddp_flow_grpo/preflight")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2))
    OmegaConf.save(sft, output / "resolved_sft.yaml")
    (output / "train_tokens.json").write_text(json.dumps(train))
    (output / "validation_tokens.json").write_text(json.dumps(val))
    print(json.dumps(result, indent=2))
    if missing:
        raise FileNotFoundError(f"{len(missing)} metric caches missing")


if __name__ == "__main__":
    main()
