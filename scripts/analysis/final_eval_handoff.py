"""Evaluate the fixed final checkpoint on its training host after native exit.

This is a dependency on one explicit owned training job, not a GPU availability
queue. Never stops a training rank. The old evaluation controller is interrupted
only AFTER the native training supervisor reports success and the final complete
checkpoint exists. Original evaluation, export and pairing functions are reused.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

from scripts.cluster_flow_grpo.cluster import exclusive_controller, write_json
from scripts.analysis.accelerated_epoch_run import validate_baseline
from scripts.analysis.full_epoch_run import evaluate_policy, summarize
from starVLA.rl.flow_grpo.loading import file_sha


def training_finished(native_result, checkpoint, target):
    path = Path(native_result)
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get("status") != "PASS" or result.get("exit_codes") != [0]:
        raise RuntimeError("owned native single-host training did not exit successfully")
    checkpoint = Path(checkpoint)
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("native exit without complete final checkpoint")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("update") != target or state.get("world_size") != 8:
        raise ValueError("final checkpoint budget/world mismatch")
    return True


def retire_controller(pid, expected_spec, *, proc_root=Path("/proc"), send=os.kill):
    """Exact owned command match; PID reuse or an unrelated process is rejected."""
    process = proc_root / str(pid)
    if not process.exists():
        return "already_exited"
    command = process.joinpath("cmdline").read_bytes().split(b"\0")
    module = b"scripts.analysis.accelerated_epoch_run"
    if (module not in command or b"--spec" not in command
            or command[command.index(b"--spec") + 1] != os.fsencode(str(expected_spec))):
        raise ValueError("refusing to signal an unrecognized controller")
    send(pid, signal.SIGINT)
    return "SIGINT_owned_evaluation_controller_after_native_training_exit"


def execute(spec):
    root = Path(spec["control_dir"])
    started = time.time()
    write_json(root / "progress.json", {"status": "WAITING_FOR_OWN_TRAINING_COMPLETION",
               "native_result": spec["native_result"], "checkpoint": spec["final_checkpoint"], "started": started})
    deadline = time.monotonic() + spec["dependency_timeout_seconds"]
    while not training_finished(spec["native_result"], spec["final_checkpoint"], spec["target"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("registered native training dependency exceeded its deadline")
        time.sleep(15)
    operation = retire_controller(spec["old_controller_pid"], spec["old_experiment_spec"])
    write_json(root / "handoff.json", {"status": "NATIVE_TRAINING_COMPLETE", "operation": operation,
               "native_result_sha256": file_sha(spec["native_result"]), "time": time.time()})
    # Export uses its existing filesystem transaction/lock if the prior
    # controller was already publishing the exact same completed checkpoint.
    from starVLA.rl.flow_grpo.checkpoint import export_checkpoint
    exported = export_checkpoint(spec["final_checkpoint"], spec["final_export"])
    state = {"status": "RUNNING", "training": {"status": "PASS", "update": spec["target"],
             "checkpoint": spec["final_checkpoint"], "exported": str(exported)}, "evaluations": {}}
    binding = spec["external_baseline"]
    baseline = json.loads((Path(binding["control_dir"]) / "progress.json").read_text())["evaluations"]["sft"]
    state["evaluations"]["sft"] = validate_baseline(baseline, binding, spec)
    write_json(root / "progress.json", state)
    state["evaluations"]["last"] = evaluate_policy(spec, "last", str(exported), threading.Event())
    state.update(status="COMPLETE", finished=time.time())
    summarize(spec, state)
    write_json(root / "result.json", state)
    write_json(root / "progress.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--spec", required=True)
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    if spec["handoff_controller_sha256"] != file_sha(__file__):
        raise ValueError("final evaluation handoff source changed")
    with exclusive_controller(spec["control_dir"]):
        try:
            execute(spec)
        except BaseException as exc:
            write_json(Path(spec["control_dir"]) / "failure.json", {"status": "FAIL", "error": repr(exc)})
            raise
