"""Publish a release only from complete semantic evidence accepted by the training gate.

This command runs no model tests and cannot manufacture their results. Evidence
must be schema-versioned test bundles from actual executions; unit-test fixtures
or a directory of logs are not a release. A missing/failed/stale gate fails closed.
"""
import argparse
import json
from pathlib import Path
from starVLA.rl.flow_grpo.acceptance import (
    GATES, acceptance_context, validate_gate_evidence, enforce_training_budget,
)
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, resume_assets
from starVLA.rl.flow_grpo.transactions import atomic_json, publication_lock


def publish(cfg, context, paths, output):
    pointers = {}
    for value in paths:
        path = Path(value).resolve()
        bundle = json.loads(path.read_text())
        for test in bundle.get("tests", []):
            gate = test.get("test_id")
            if gate not in GATES:
                continue
            if gate in pointers:
                raise ValueError(f"duplicate test evidence for {gate}")
            pointer = {"path": str(path), "sha256": file_sha(path), "test_id": gate, "status": test.get("status")}
            validate_gate_evidence(gate, pointer, context, cfg)
            pointers[gate] = pointer
    if set(pointers) != set(GATES):
        raise ValueError("NOT_READY: missing gates " + ", ".join(sorted(set(GATES) - set(pointers))))
    record = {"schema_version": 1, "status": "READY_FOR_THIS_PROFILE", "context": context, "gates": pointers}
    output = Path(output).resolve()
    if output != Path(cfg["runtime"]["acceptance_record"]).resolve():
        raise ValueError("release output differs from the requested training configuration")
    enforce_training_budget(cfg, context, record=record)
    with publication_lock(output.with_suffix(output.suffix + ".lock")):
        if output.exists():
            if json.loads(output.read_text()) != record:
                raise ValueError("refusing to replace an existing conflicting release")
        else:
            atomic_json(output, record)
        # Exercise the exact formal-train entry guard, including recipe/budget.
        enforce_training_budget(cfg, context)
    return record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--evidence", nargs="+", required=True)
    args = p.parse_args()
    configure_numerics()
    cfg, sft = resolve_config(args.config)
    if cfg["runtime"]["run_mode"] != "formal":
        p.error("release must target an explicit formal configuration")
    context = acceptance_context(cfg, resume_assets(cfg, sft))
    record = publish(cfg, context, args.evidence, cfg["runtime"]["acceptance_record"])
    print(json.dumps({"status": record["status"], "gates": len(record["gates"])}))


if __name__ == "__main__":
    main()
