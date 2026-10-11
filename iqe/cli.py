"""IQE executable pipeline. No large workload is implied by reading a config."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import traceback
from .config import load_config
from .contracts import require
from .io import atomic_json, read_json, BlockedError

COMMANDS = ("capacity-plan", "evaluate-base", "qualify-base", "preflight", "build-splits", "prepare-metric-contexts", "verify-reference", "train-base", "train-policy-copy", "evaluate-policy-copy", "freeze-base", "cache-features", "export-candidates", "score-candidates",
            "build-round", "train-expert", "audit-frozen", "evaluate-oracle", "train-scorer", "train-router", "calibrate",
            "evaluate", "evaluate-baselines", "export-bundle", "run-round", "overfit-32", "benchmark", "load-bundle", "rollback", "final-test")


from .training.resume import resolve_resume


class JSONParser(argparse.ArgumentParser):
    def error(self, message):
        print(__import__("json").dumps({"status":"FAILED", "exit_code":2, "error":message}), file=sys.stderr)
        self.exit(2)


def parser():
    p = JSONParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        c = sub.add_parser(command, help=command.replace("-", " "))
        c.add_argument("--config", required=True)
        c.add_argument("--mode", choices=("smoke", "full", "dry-run", "profile"), default="smoke")
        c.add_argument("--max-samples", type=int, help="explicit per-role sample ceiling; required for full execution")
        c.add_argument("--device", choices=("cpu", "cuda"))
        c.add_argument("--repo", default=".")
        c.add_argument("--round", type=int, default=0)
        c.add_argument("--roles", nargs="+", choices=("incremental_fit", "stage_val", "selector_cal", "dev_report"),
                       default=["incremental_fit", "stage_val", "selector_cal", "dev_report"])
        c.add_argument("--role", choices=("incremental_fit", "stage_val", "selector_cal", "dev_report"), default="dev_report")
        if command in {"evaluate-base", "qualify-base"}:
            c.set_defaults(role="stage_val")
        c.add_argument("--max-steps", type=int)
        c.add_argument("--stop-after", type=int)
        c.add_argument("--resume", nargs="?", const="auto")
        c.add_argument("--non-exact-finetune", action="store_true")
        c.add_argument("--checkpoint")
        c.add_argument("--evaluation")
        if command == "qualify-base":
            c.add_argument("--select", action="store_true", help="explicitly select this qualified version before immutable Base locking")
        c.add_argument("--backend", choices=("reference",), default="reference")
        c.add_argument("--require-gates", action="store_true")
        c.add_argument("--activate", action="store_true")
        c.add_argument("--expert-counts", nargs="+", type=int, default=[1, 2, 3, 5])
        c.add_argument("--warmup", type=int, default=5)
        c.add_argument("--repetitions", type=int, default=20)
        c.add_argument("--bundle")
        c.add_argument("--final-manifest")
        c.add_argument("--selector-only", action="store_true")
        c.add_argument("--shard-index", type=int, default=0)
        c.add_argument("--num-shards", type=int, default=1)
        c.add_argument("--allow-latency-clones", action="store_true")
    return p


def execute(a):
    config = load_config(a.config)
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        require(a.command in {"train-base","train-expert","train-scorer","train-router","train-policy-copy"},
                "DDP wraps individual training jobs; run-round and artifact stages require one coordinator")
    require(not a.selector_only or a.command == "run-round", "--selector-only belongs to run-round")
    require(not a.allow_latency_clones or a.command == "benchmark", "latency clones are diagnostic benchmark only")
    require(0 <= a.shard_index < a.num_shards, "invalid shard index/count")
    require(a.num_shards == 1 or a.command == "cache-features", "scene sharding is supported by cache-features")
    require(not a.non_exact_finetune or (a.resume and a.command in {"train-expert", "train-scorer", "train-router"}), "non-exact finetune requires a supported trainer and explicit resume")
    require(not a.activate or a.command == "export-bundle", "--activate belongs to export-bundle")
    require(a.round >= 0, "negative round")
    require(a.mode != "full" or a.max_samples is not None, "full mode needs explicit --max-samples")
    maximum = a.max_samples if a.max_samples is not None else config["execution"]["max_samples"]
    require(maximum > 0 and (a.max_steps is None or a.max_steps > 0), "positive budgets")
    if a.mode == "dry-run":
        return {"status": "DRY_RUN", "command": a.command, "mode": a.mode, "max_samples": maximum, "max_steps": a.max_steps,
                "scientific_results_generated": False, "required_contract": config["s0"]["contract"]}
    from .resources import qualify_distributed
    resource = qualify_distributed(a.device or config["execution"]["device"])
    if a.command == "preflight":
        from .preflight import audit
        contract = audit(config, a.repo)
        from .config import resolve
        # The historical framework contract intentionally stays untrained. Do
        # not erase an explicit experiment checkpoint from the launch config.
        configured_checkpoint = config['s0']['query_checkpoint'] or contract.get('query_checkpoint')
        resolve(config,contract | {'query_checkpoint':configured_checkpoint},Path(config["output_root"])/"preflight_resolved_config.json")
        return {"status": contract["binding_status"], "framework_contract_hash": contract["framework_contract_hash"],
                "learned_driving_weights_loaded": False, "query_checkpoint": configured_checkpoint,
                "scope":"historical_framework_contract; experiment Base qualification/lock is checked by freeze-base"}
    from .pipeline import Pipeline
    pipeline = Pipeline(config, mode=a.mode, max_samples=maximum, device=a.device)
    if a.command == "capacity-plan":
        from .storage import capacity_plan
        from .training.resume import inspect_checkpoint
        require(a.checkpoint, "capacity-plan requires a real complete --checkpoint")
        info = inspect_checkpoint(a.checkpoint)
        report = capacity_plan(pipeline.root, Path(a.checkpoint).stat().st_size,
            config["execution"]["checkpoint_every_steps"], config["query_base"]["max_optimizer_steps"], info["optimizer_step"])
        atomic_json(pipeline.root / "CAPACITY_PLAN.json", report)
        return report
    if a.command == "build-splits":
        return pipeline.build_splits()
    if a.command == "verify-reference":
        from .scoring.audit import verify_reference
        return verify_reference(pipeline)
    if a.command == "prepare-metric-contexts":
        from .scoring.prepare import prepare_contexts
        return prepare_contexts(pipeline,a.roles)
    if a.command == "train-base":
        from .training.base_trainer import train_base
        resume = a.resume
        path = train_base(pipeline.config, pipeline.contract, pipeline.scenes(["incremental_fit"]), pipeline.root / "query_base", mode=a.mode,
                          steps=a.max_steps, stop_after=a.stop_after, resume=resume, device=pipeline.device)
        return read_json(pipeline.root / "query_base/result.json") | {"resources": resource}
    if a.command in {"evaluate-base", "qualify-base"}:
        from .base_qualification import evaluate_base, qualify_base
        require(a.role == "stage_val", "Base qualification uses stage_val only")
        require(a.checkpoint, "explicit complete --checkpoint required")
        if a.command == "evaluate-base":
            result = evaluate_base(pipeline, a.checkpoint)
            if result['status'] != 'COMPLETE':
                raise BlockedError(result['status'] + ': ' + result['result_path'])
            return result
        require(a.evaluation, "qualify-base requires --evaluation")
        result = qualify_base(pipeline, a.checkpoint, a.evaluation, select=a.select)
        if result["status"] != "QUALIFIED":
            raise BlockedError(result["status"] + ": " + result['qualification_path'])
        return result
    if a.command == "freeze-base":
        checkpoint = a.checkpoint or config["s0"]["query_checkpoint"]
        require(checkpoint is not None, "freeze-base requires the actually trained Query checkpoint")
        return pipeline.freeze_base(checkpoint)
    if a.command in {"rollback", "load-bundle", "final-test"}:
        from .export import load_bundle, rollback
        if a.command == "rollback":
            return rollback(pipeline.root)
        require(a.bundle, "explicit locked --bundle required")
        model, rule, metadata = load_bundle(a.bundle, pipeline.device)
        if a.command == "load-bundle":
            return {"status": "COMPLETE", "bundle_hash": metadata["bundle_hash"], "experts": list(model.experts)}
        require(a.final_manifest and metadata["mode"] == "full" and metadata["gate_results"]["passed"], "final_test needs locked accepted full bundle and independent explicit manifest")
        from .final_test import evaluate_final
        return evaluate_final(model, rule, metadata, a.final_manifest, pipeline.root / "final_test_access", maximum, pipeline.config["metric"])
    pipeline.use_locked_base()
    if a.command in {"train-policy-copy", "evaluate-policy-copy"}:
        from .training.policy_copy import train_copy, evaluate_copy
        if a.command == "evaluate-policy-copy":
            result = evaluate_copy(pipeline, a.round, a.role)
            atomic_json(pipeline.round_root(a.round) / "policy_copy" / ("evaluation_" + a.role + ".json"), result, immutable=True)
            return result
        resume = a.resume
        return train_copy(pipeline, a.round, steps=a.max_steps, resume=resume, stop_after=a.stop_after)
    if a.command == "cache-features":
        return pipeline.cache_features(a.roles, shard_index=a.shard_index, num_shards=a.num_shards)
    if a.command == "export-candidates":
        return pipeline.export_candidates(a.round, a.roles)
    if a.command == "score-candidates":
        return pipeline.score_candidates(a.round)
    if a.command == "build-round":
        return pipeline.build_round(a.round, a.max_steps)
    if a.command in {"train-expert", "overfit-32"}:
        require(a.round > 0, "new expert round required")
        if a.command == "overfit-32":
            from .overfit import overfit_32
            return overfit_32(pipeline, a.round, a.max_steps or 32, resume=a.resume)
        resume = a.resume
        return pipeline.train_expert(a.round, steps=a.max_steps, resume=resume, stop_after=a.stop_after, non_exact_finetune=a.non_exact_finetune)
    if a.command == "audit-frozen":
        return pipeline.frozen_audit(a.round)
    if a.command == "evaluate-oracle":
        return pipeline.evaluate_oracle(a.round, a.role)
    router = config["router_ablation"]["enabled"]
    if a.command in {"train-scorer", "train-router"}:
        router = a.command == "train-router"
        resume = a.resume
        return pipeline.train_selector(a.round, router=router, steps=a.max_steps, resume=resume, stop_after=a.stop_after, non_exact_finetune=a.non_exact_finetune)
    if a.command == "calibrate":
        return pipeline.calibrate(a.round, a.role, router=router)
    if a.command == "evaluate":
        return pipeline.evaluate(a.round, a.role, router=router)
    if a.command == "evaluate-baselines":
        from .evaluation.baselines import evaluate_baselines
        result = evaluate_baselines(pipeline,a.round,a.role)
        atomic_json(pipeline.round_root(a.round) / ("baselines_"+a.role+".json"),result,immutable=True)
        return result
    if a.command == "export-bundle":
        return pipeline.export_bundle(a.round, require_gates=a.require_gates, activate=a.activate, router=router)
    if a.command == "run-round":
        from .rounds import run_round
        return run_round(pipeline, a.round, resume=bool(a.resume), steps=a.max_steps, selector_only=a.selector_only)
    if a.command == "benchmark":
        from .evaluation.latency import benchmark
        from .data.sources import load_observation
        model = pipeline.model(a.round)
        scorer = model.scorer
        rule = router_rule = None
        if a.round > 0:
            if (pipeline.round_root(a.round) / "calibration.json").exists():
                scorer, rule, _ = pipeline.calibrated_selector(a.round)
            if (pipeline.round_root(a.round) / "router_calibration.json").exists():
                model.router, router_rule, _ = pipeline.calibrated_selector(a.round, router=True)
        diagnostic_clones = []
        if a.allow_latency_clones:
            while len(model.experts) < max(a.expert_counts):
                expert_id = "expert_" + str(max(int(e.split("_")[-1]) for e in model.experts) + 1)
                model.append_expert(expert_id)
                diagnostic_clones.append(expert_id)
            model.set_trainable_stage("inference")
        scene = pipeline.scenes(["stage_val"])[0]
        result = benchmark(model, load_observation(scene, pipeline.contract), scorer, model.router,
                           a.expert_counts, a.warmup, a.repetitions, pipeline.device,
                           observation_loader=lambda:load_observation(scene,pipeline.contract), rule=rule, router_rule=router_rule)
        result["untrained_capacity_probe_clones"] = diagnostic_clones
        result["registry_or_candidate_bank_modified"] = False
        atomic_json(pipeline.root / "latency.json", result)
        return result
    raise ValueError("unhandled command")


def main(argv=None):
    args = parser().parse_args(argv)
    config = None
    started = time.perf_counter()
    try:
        import torch
        config = load_config(args.config)
        torch.set_num_threads(config["execution"]["num_threads"])
        result = execute(args)
        status = {"command": args.command, "exit_code": 0, "elapsed_seconds": time.perf_counter() - started, "result": result}
        print(__import__("json").dumps(status, allow_nan=False))
        return 0
    except Exception as e:
        status = {"command": args.command, "exit_code": 2, "status": "BLOCKED" if isinstance(e, BlockedError) else "FAILED",
                  "error": str(e), "elapsed_seconds": time.perf_counter() - started}
        if config is not None:
            atomic_json(Path(config["output_root"]) / (args.command + ".FAILED.json"), status)
        print(__import__("json").dumps(status, allow_nan=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
