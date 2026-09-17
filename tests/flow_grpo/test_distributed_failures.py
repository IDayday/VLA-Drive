from pathlib import Path
import json
import os
import subprocess
import sys
import time
import pytest


@pytest.mark.parametrize(
    "case",
    [
        "output_conflict",
        "checkpoint_mkdir",
        "rank0_write",
        "reward_failure",
        "rank_exit",
    ],
)
def test_torchrun_failure_is_bounded(case, tmp_path):
    start = time.monotonic()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--max-restarts=0",
            "--monitor-interval=1",
            "tests/flow_grpo/fault_worker.py",
            case,
            str(tmp_path),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=45,
    )
    assert result.returncode != 0, result.stdout
    failures = [json.loads(p.read_text()) for p in tmp_path.glob("rank*.json")]
    assert failures and all(x["status"] == "FAIL" for x in failures), result.stdout
    if case != "rank_exit":
        assert len(failures) == 2 and all(
            "collective phase failed" in x["error"] for x in failures
        ), result.stdout
    evidence = {
        "case": case,
        "returncode": result.returncode,
        "seconds": time.monotonic() - start,
        "rank_failures": failures,
        "status": "PASS",
        "backend": "two-process torchrun/Gloo control-plane; CUDA failures NOT_RUN",
    }
    report = os.getenv("FLOW_FAILURE_REPORT_DIR")
    if report:
        Path(report).mkdir(parents=True, exist_ok=True)
        (Path(report) / f"{case}.json").write_text(json.dumps(evidence, indent=2))
        (Path(report) / f"{case}.log").write_text(result.stdout)
