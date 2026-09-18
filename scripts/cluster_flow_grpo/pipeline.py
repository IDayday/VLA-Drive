"""Continuous native training with independent, immutable-checkpoint consumers.

Only COMPLETE publications are read. The native optimizer, saving transaction,
rewards and evaluation protocol are unchanged. A slow evaluator never determines
how many optimizer updates the trainer performs.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from starVLA.rl.flow_grpo.orchestration import checkpoint_inventory
from starVLA.rl.flow_grpo.transactions import atomic_json, preserve_attempt


def continuous_pipeline(run, targets, *, validate, train, export, evaluate,
                        on_result, baseline, cancelled, adopt=None, poll_seconds=2):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    targets = list(targets)
    if not targets or targets != sorted(set(targets)):
        raise ValueError("unique ascending evaluation targets required")
    maximum = targets[-1]
    checked = {}
    validation_lock = threading.Lock()
    producer_errors = []

    def cancelled_error(producer):
        # The producer sets cancellation before its future becomes done. Keep
        # the original failure visible during that interval (including slow I/O
        # while publishing the failure report), not a generic cancellation.
        if producer_errors:
            raise producer_errors[0]
        if producer.done():
            producer.result()
        raise RuntimeError("paired pipeline cancelled")

    def validated(path):
        # Cache a full validation only while every file's stat identity remains
        # unchanged. New/changed publications still use the native validator.
        path = Path(path)
        with validation_lock:
            fingerprint = tuple((str(p.relative_to(path)), p.stat().st_ino,
                p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                for p in sorted(path.rglob("*")) if p.is_file())
            prior = checked.get(path)
            if prior:
                if prior[0] != fingerprint:
                    raise ValueError(f"completed checkpoint changed: {path}")
                return prior[1]
            state = validate(path)
            checked[path] = (fingerprint, state)
            return state

    def produce():
        try:
            if adopt:
                adopt()
            complete, incomplete = checkpoint_inventory(run, validated)
            latest = max(complete, default=0)
            if latest > maximum:
                raise ValueError("checkpoint exceeds the fixed training budget")
            if any(t < latest and t not in complete for t in targets):
                raise ValueError("missing historical evaluation checkpoint; cannot replay updates")
            atomic_json(run/"continuous_training.json", {
                "status": "RUNNING" if latest < maximum else "COMPLETE",
                "resume_update": latest, "target_update": maximum})
            if latest < maximum:
                for path in incomplete:
                    preserve_attempt(path)
                train(complete.get(latest), maximum)
            complete, _ = checkpoint_inventory(run, validated)
            if maximum not in complete:
                raise RuntimeError("training exited without the final complete checkpoint")
            atomic_json(run/"continuous_training.json", {
                "status": "COMPLETE", "target_update": maximum,
                "latest_complete_update": max(complete)})
        except BaseException as exc:
            producer_errors.append(exc)
            cancelled.set()
            atomic_json(run/"continuous_training.json", {"status": "FAIL", "error": str(exc)})
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        producer = pool.submit(produce)
        try:
            # Baseline identity checks/evaluation also run alongside training.
            baseline()
            progress = {"mode": "asynchronous", "exports": {}, "dev_evaluations": {}}
            for target in targets:
                checkpoint = run/"checkpoints"/f"update_{target:06d}"
                while not (checkpoint/"COMPLETE").is_file():
                    if producer.done():
                        producer.result()
                        raise RuntimeError(f"missing required checkpoint {target}")
                    if cancelled.wait(poll_seconds):
                        cancelled_error(producer)
                if cancelled.is_set():
                    cancelled_error(producer)
                validated(checkpoint)
                atomic_json(run/"artifact_activity.json", {
                    "stage": "cpu_export", "update": target, "time": time.time()})
                exported = export(checkpoint, target)
                progress["exports"][str(target)] = str(exported)
                atomic_json(run/"orchestration_progress.json", progress)
                atomic_json(run/"artifact_activity.json", {
                    "stage": "dev_evaluation", "update": target, "time": time.time()})
                result = evaluate(exported, target)
                progress["dev_evaluations"][str(target)] = result
                atomic_json(run/"orchestration_progress.json", progress)
                on_result(checkpoint, exported, target, result)
            producer.result()
            atomic_json(run/"artifact_activity.json", {"stage": "COMPLETE", "time": time.time()})
        except BaseException:
            cancelled.set()
            raise
