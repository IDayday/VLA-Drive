"""Thin research-baseline entrypoint using the tested IQE heads and trainer."""
import argparse
import json
import sys
from dataclasses import asdict
import yaml
from .io import read_json, file_hash, atomic_json, atomic_bytes
from .contracts import require
from .internal_baseline.config import load_config


COMMANDS = {"init-baseline": "bind-base", "cache-baseline-context": "cache-features",
    "score-baseline-base": "build-hard-set", "audit-baseline-targets": "build-hard-set",
    "build-expert-data": "build-hard-set", "train-baseline-expert": "train-expert",
    "build-pair-bank": "build-pair-bank", "train-baseline-scorer": "train-scorer",
    "evaluate-two-candidate-baseline": "evaluate", "calibrate-baseline-margin": "evaluate",
    "export-research-bundle": "export-bundle", "run-two-candidate-baseline": "run",
    "audit-online": "audit-online", "benchmark": "benchmark", "diagnostics": "diagnostics"}


def main():
    p = argparse.ArgumentParser(description="70k-or-explicit frozen Base + independent expert + own scratch scorer; research only")
    p.add_argument("command", choices=list(COMMANDS)+["build-baseline-splits", "evaluate-baseline-expert"])
    p.add_argument("--config", required=True)
    p.add_argument("--experiment-id")
    p.add_argument("--output-root")
    p.add_argument("--base-checkpoint")
    p.add_argument("--base-run")
    p.add_argument("--base-policy", choices=["latest_complete"])
    args, rest = p.parse_known_args()
    if args.command == "init-baseline":
        require(not (args.base_checkpoint and args.base_policy), "choose explicit checkpoint OR latest_complete")
        from .internal_baseline.config import Config
        with open(args.config) as f: values = yaml.safe_load(f)
        overrides = {"experiment_id": args.experiment_id, "output_root": args.output_root,
                     "checkpoint": args.base_checkpoint or args.base_policy, "base_run": args.base_run}
        values.update({k:v for k,v in overrides.items() if v is not None})
        c = Config(**values)
        atomic_json(c.root/"resolved_config.json", asdict(c), immutable=True)
        atomic_bytes(c.root/"EXPERIMENT_CONFIG.yaml", yaml.safe_dump(asdict(c), sort_keys=True).encode(), immutable=True)
    else:
        c = load_config(args.config)
    require(c.research_protocol == "two_candidate_v1", "two_candidate_v1 config required")
    # After initialization all commands must read the fixed experiment config.
    for supplied, actual in ((args.experiment_id, c.experiment_id), (args.output_root, c.output_root), (args.base_checkpoint, c.checkpoint),
                              (args.base_run, c.base_run), (args.base_policy, c.checkpoint)):
        require(supplied is None or supplied == actual, "init selection must match the explicit experiment config")
    if args.command == "build-baseline-splits":
        require(not rest, "unexpected split arguments")
        from .internal_baseline.binding import binding
        from .internal_baseline.partition import training_ids
        from .data.manifests import load_scenes
        binding(c)
        scenes = load_scenes(c.root/"scenes.json")
        for role in ("expert_fit", "scorer_fit"): training_ids(c, scenes, role)
        print(json.dumps(read_json(c.root/"DATA_PARTITION_EXPOSURE.json")))
        return
    if args.command == "evaluate-baseline-expert":
        require(not rest, "expert evaluation uses already completed fixed stage_val checkpoints; no new selector inputs")
        result = read_json(c.root/"expert/result.json")
        require(result["status"] == "COMPLETE" and file_hash(c.root/"expert/best.pt") == result["best_checkpoint_hash"], "expert not selected")
        print(json.dumps({"result": result, "validation": [read_json(f) for f in sorted((c.root/"expert").glob("validation_*.json"))]}))
        return
    if args.command == "calibrate-baseline-margin":
        q = argparse.ArgumentParser(); q.add_argument("--device", default="cuda:0")
        options = q.parse_args(rest)
        from .internal_baseline.reporting import calibrate_margin
        print(json.dumps(calibrate_margin(c, options.device)))
        return
    if args.command == "init-baseline":
        require(not rest, "unexpected init arguments")
        atomic_json(c.root/"SELECTION_PROTOCOL.json", {"primary": "raw_predicted_total_argmax_base_tie",
            "margin_grid": [0., .01, .02, .05, .1], "margin_role": "selector_cal",
            "budgets": {"new_NC": 0, "new_DAC": 0, "high_to_zero": 0},
            "fallback": "always_base_if_no_positive_feasible_gain", "high_quality": .8, "zero_epsilon": 1e-8,
            "expert_checkpoint": ["stage_val_oracle_gain", "H_val_direct_gain", "earlier_step"],
            "scorer_checkpoint": ["stage_val_raw_argmax_PDMS", "fewer_new_NC_DAC", "earlier_step"],
            "not_a_deployment_release": True}, immutable=True)
        from .internal_baseline.binding import bind
        print(json.dumps(bind(c)))
        return
    sys.argv = [sys.argv[0], COMMANDS[args.command], "--config", args.config]+rest
    from .internal_baseline.cli import main as run
    run()


if __name__ == "__main__":
    main()
