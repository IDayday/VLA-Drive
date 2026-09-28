"""Append-only per-run resource accounting, including startup and failed runs."""
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import socket
import time


@contextmanager
def metered_run(root, run_id, gpu_count, details):
    out = Path(root) / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    record = {"run_id": run_id, "host": socket.gethostname(), "pid": os.getpid(),
              "start_unix": start, "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "gpu_count": gpu_count, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "status": "RUNNING", "real_optimizer_updates": 0, "synthetic_optimizer_updates": 0,
              "inference_scenes": 0, **details}
    def save():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(record, indent=2, default=str))
        os.replace(tmp, out / "status.json")
    save()
    try:
        yield record, out, save
        record["status"] = "COMPLETE"
    except BaseException as error:
        record["status"] = "FAILED"
        record["error"] = repr(error)
        raise
    finally:
        record["end_unix"] = time.time()
        record["wall_seconds"] = record["end_unix"]-start
        record["gpu_hours"] = record["wall_seconds"]*gpu_count/3600
        save()
