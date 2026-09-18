"""Attach to this experiment's live, independently supervised native jobs.

The old controller must be stopped before the replacement obtains its lock.
Workers retain their own process groups, timeout and exit receipts. No optimizer
step is restarted and no process is selected by a broad name/pkill expression.
"""
import json
from pathlib import Path
import shlex
import subprocess
import time

from scripts.cluster_flow_grpo.cluster import PYTHON, command_for, write_json
from starVLA.rl.flow_grpo.loading import file_sha


def validate_adoption(receipt, group, run, maximum):
    path = Path(receipt["spec"]).resolve()
    if file_sha(path) != receipt["sha256"]:
        raise ValueError("adopted spec changed")
    spec = json.loads(path.read_text())
    entry = spec["entry"]
    def argument(name):
        return entry[entry.index(name)+1]
    if ("direct_command" in spec or spec["nodes"] != group["nodes"] or
        entry[:3] != ["-m", "starVLA.rl.flow_grpo.cli", "train"] or
        argument("--config") != group["config"] or
        Path(argument("--output-dir")).resolve() != Path(run).resolve() or
        not 0 < int(argument("--max-updates")) <= maximum or
        spec.get("timeout_seconds", 0) <= 0):
        raise ValueError("adopted job does not match the experiment")
    return path, spec


def supervisor_command(spec_path, rank, state, terminate=False):
    spec = json.loads(Path(spec_path).read_text())
    if state.get("job_id") != spec["job_id"] or state.get("command") != command_for(spec, rank):
        raise ValueError("supervisor identity mismatch")
    pid = int(state["supervisor_pid"])
    code = ("import pathlib,os,signal; p="+str(pid)+"; "
            "c=pathlib.Path(f'/proc/{p}/cmdline').read_bytes(); "
            "assert b'cluster.py' in c and "+repr(str(Path(spec_path).resolve()).encode())+" in c; ")
    code += "os.kill(p,signal.SIGTERM)" if terminate else "print('RUNNING')"
    command = [PYTHON, "-c", code]
    host = spec["nodes"][rank]["host"]
    if host != "local":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, shlex.join(command)]
    return command


def wait_existing(receipt, group, run, maximum, cancelled, poll_seconds=2):
    path, spec = validate_adoption(receipt, group, run, maximum)
    control = Path(spec["control_dir"])
    def states():
        result = []
        for rank in range(len(spec["nodes"])):
            state = json.loads((control/f"node{rank}.json").read_text())
            supervisor_command(path, rank, state)  # Match job and exact command even after exit.
            result.append(state)
        return result
    next_probe = 0
    try:
        while True:
            rows = states()
            if any(r["status"] == "FAILED" or r.get("exit_code", 0) != 0 for r in rows):
                raise RuntimeError("adopted training peer failed")
            if all(r["status"] == "COMPLETE" and r.get("exit_code") == 0 for r in rows):
                result = {"status": "PASS", "exit_codes": [0]*len(rows),
                    "adopted": True, "spec_sha256": receipt["sha256"],
                    "seconds": max(r["finished"] for r in rows)-min(r["started"] for r in rows)}
                # The displaced controller never fabricated a cluster receipt;
                # completion derives from every independent supervisor's exit.
                write_json(control/"adopted_result.json", result)
                return result
            if cancelled.is_set():
                raise RuntimeError("adopted training cancelled")
            if time.time()-min(r["started"] for r in rows) > spec["timeout_seconds"]+60:
                raise TimeoutError("adopted training exceeded original deadline")
            if time.monotonic() >= next_probe:
                for rank, state in enumerate(rows):
                    if state["status"] != "RUNNING":
                        continue
                    probe = subprocess.run(supervisor_command(path, rank, state), timeout=15,
                                           capture_output=True, text=True)
                    if probe.returncode:
                        # A normal exit may race the read-only /proc probe.
                        current = states()[rank]
                        if current["status"] != "COMPLETE" or current.get("exit_code") != 0:
                            raise RuntimeError("adopted supervisor disappeared or changed")
                next_probe = time.monotonic()+30
            cancelled.wait(poll_seconds)
    except BaseException as exc:
        cancelled.set()
        for rank, state in enumerate(states()):
            if state["status"] == "RUNNING":
                subprocess.run(supervisor_command(path, rank, state, terminate=True),
                               timeout=15, capture_output=True)
        write_json(control/"adoption_failure.json", {"status": "FAIL", "error": str(exc)})
        raise
