"""Auditable stage receipts and idempotent multi-round execution."""
from __future__ import annotations
from pathlib import Path
import time
from datetime import datetime, timezone
from .contracts import require
from .io import atomic_json, digest, file_hash, read_json, BlockedError, complete, reusable

FLOW = ("export-candidates", "score-candidates", "build-round", "train-expert", "audit-frozen",
        "evaluate-oracle", "train-scorer", "calibrate", "evaluate", "export-bundle")


def run_round(pipeline, number, *, resume=False, steps=None, selector_only=False):
    require(number >= 1, "round must be positive")
    require(selector_only or len(pipeline.registry.latest(pipeline.registry.read())) < pipeline.config["model"]["max_experts_including_base"] or resume,
            "configured scientific expert budget reached")
    if number > 1 and pipeline.mode == "full" and not selector_only:
        prior = read_json(pipeline.round_root(number - 1) / "oracle_stage_val.json")
        if not prior["oracle_gate"]:
            raise BlockedError("NO_COVERAGE_GAIN: automatic scientific expansion stopped; existing Scorer/export functionality remains executable")
    roles = ["incremental_fit", "stage_val", "selector_cal", "dev_report"]
    router = pipeline.config["router_ablation"]["enabled"]
    kind = "router" if router else "scorer"
    stage_log = []
    def stage(name, fn, *, version=number):
        folder = pipeline.round_root(version)
        log = folder / (name.replace("-", "_") + ".stage.json")
        inputs = {"round": version, "mode": pipeline.mode, "config": pipeline.config["config_hash"],
                  "contract": digest(pipeline.contract), "max_samples": pipeline.max_samples, "steps": steps, "selector_only":selector_only}
        if log.exists():
            saved = read_json(log)
            require(saved["inputs"] == inputs, "round-stage version mismatch")
            if saved["status"] == "COMPLETE":
                # All completed stage results are bound to actual output checksums.
                for p, h in saved["outputs"].items():
                    require(Path(p).is_file() and file_hash(p) == h, "round stage output changed/incomplete")
                stage_log.append({"stage": name, "status": "REUSED"})
                return
            require(resume, f"incomplete stage {name}; use --resume")
        begin = datetime.now(timezone.utc).isoformat(); start = time.perf_counter()
        atomic_json(log, {"status": "RUNNING", "inputs": inputs, "begin": begin})
        before = {str(p): file_hash(p) for p in folder.rglob("*.json") if p != log and not p.name.endswith(".lock")}
        try:
            result = fn()
            require(result.get("status") != "STOPPED_AT_BOUNDARY", "training stopped at a boundary; resume required")
            outputs = {str(p): file_hash(p) for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".pt"} and p != log and not p.name.endswith(".stage.json") and not p.name.endswith(".lock")
                       and (str(p) not in before or before[str(p)] != file_hash(p))}
            record = {"status": "COMPLETE", "inputs": inputs, "outputs": outputs, "begin": begin,
                      "end": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - start, "result": result}
            atomic_json(log, record)
            stage_log.append({"stage": name, "status": "COMPLETE"})
        except BaseException as e:
            atomic_json(log, {"status": "BLOCKED" if isinstance(e, BlockedError) else "FAILED", "inputs": inputs,
                             "begin": begin, "end": datetime.now(timezone.utc).isoformat(), "error": repr(e)})
            raise
    # Old pool snapshots are generated and scored using the exact previous registered checkpoints.
    if not (pipeline.round_root(number - 1) / "scored.json").exists():
        stage("export-old-pool", lambda: pipeline.export_candidates(number - 1, roles), version=number - 1)
        stage("score-old-pool", lambda: pipeline.score_candidates(number - 1), version=number - 1)
    if not selector_only:
        stage("build-round", lambda: pipeline.build_round(number, steps))
    def resume_path(kind):
        folder = pipeline.round_root(number) / kind
        results = folder / "result.json"
        if resume and results.exists() and read_json(results).get("status") == "STOPPED_AT_BOUNDARY":
            return read_json(results)["checkpoint"]
        receipts = sorted(folder.glob("step_*.pt.COMPLETE.json"))
        return str(receipts[-1]).removesuffix(".COMPLETE.json") if resume and receipts else None
    if selector_only:
        def reuse_pool():
            from .evaluation.retention import frozen_snapshot
            for filename in ("candidates.json", "scored.json"):
                atomic_json(pipeline.round_root(number) / filename, read_json(pipeline.round_root(number-1) / filename), immutable=True)
            atomic_json(pipeline.round_root(number) / "frozen_audit.json", {"passed":True, "selector_only":True,
                        "frozen_hashes":frozen_snapshot(pipeline.model(number-1)), "output_drift":"no generator update; selector trainer also verifies hashes"}, immutable=True)
            atomic_json(pipeline.round_root(number) / "oracle_stage_val.json", {"oracle_gate":False,"selector_only":True,
                        "scientific_gain":"NO_NEW_CANDIDATE", "pool_hash":read_json(pipeline.round_root(number-1) / "scored.json")["pool_hash"]}, immutable=True)
            return {"status":"COMPLETE", "selector_only":True}
        stage("reuse-accepted-pool", reuse_pool)
    else:
        stage("train-expert", lambda: pipeline.train_expert(number, steps=steps, resume=resume_path("expert")))
        stage("audit-frozen", lambda: pipeline.frozen_audit(number))
        stage("export-candidates", lambda: pipeline.export_candidates(number, roles))
        stage("score-candidates", lambda: pipeline.score_candidates(number))
        stage("evaluate-oracle", lambda: pipeline.evaluate_oracle(number, "stage_val"))
    stage("train-"+kind, lambda: pipeline.train_selector(number, router=router, steps=steps, resume=resume_path(kind)))
    stage("calibrate", lambda: pipeline.calibrate(number, "selector_cal",router=router))
    stage("evaluate", lambda: pipeline.evaluate(number, "dev_report",router=router))
    stage("evaluate-fit-selector", lambda: pipeline.evaluate(number, "incremental_fit",router=router))
    # Failed science gates retain the previous bundle; they never truncate implementation stages.
    if pipeline.mode == "smoke":
        stage("export-bundle", lambda: pipeline.export_bundle(number, require_gates=False,router=router))
    else:
        from .export import release_gate
        folder = pipeline.round_root(number)
        prefix = "router_" if router else ""
        gate = release_gate(read_json(folder / (prefix+"evaluation_dev_report.json")), read_json(folder / "oracle_stage_val.json"),
                            read_json(folder / "frozen_audit.json"), read_json(folder / (prefix+"calibration.json"))["report"])
        if gate["passed"]:
            stage("export-bundle", lambda: pipeline.export_bundle(number, require_gates=True,router=router))
        else:
            atomic_json(folder / "release_gate.json", gate, immutable=True)
            stage_log.append({"stage": "export-bundle", "status": "RETAIN_PREVIOUS_BUNDLE", "gate": gate})
    return {"status": "COMPLETE", "round": number, "stages": stage_log, "science": "UNTESTED" if pipeline.mode == "smoke" else "REPORTED"}
