"""Read the current dispatch policy at each job boundary; never evict another job."""
from __future__ import annotations
import os
import socket
import subprocess
import json
from datetime import timedelta
from .io import read_json, BlockedError


def qualify(device, policy_path="/mnt/project/server_dispatch_policy.json"):
    policy = read_json(policy_path)
    hostname = socket.gethostname()
    authorization = policy.get("task_authorizations", {}).get("iqe_v1", {})
    permitted = set(authorization.get("allowed_hosts", []))
    permitted.update(h for h in ("training-vla-zt", "training-vla-zt2")
                     if policy.get("hosts", {}).get(h, {}).get("new_task_dispatch_allowed"))
    canonical = next((h for h in sorted(permitted)
                      if hostname in {h, h + "-worker-0"}
                      or hostname in policy.get("hosts", {}).get(h, {}).get("aliases", [])), None)
    if canonical is None:
        raise BlockedError(f"IQE dispatch outside task authorization: {hostname}")
    result = {"host": hostname, "canonical_host": canonical, "policy": policy_path, "device": device}
    if device.startswith("cuda"):
        lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).splitlines()
        selected = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed_indices = set(selected.split(",")) if selected else {line.split(",")[0].strip() for line in lines}
        devices = [{"index": i.strip(), "used_mib": int(m), "utilization": int(u)} for i, m, u in (line.split(",") for line in lines)
                   if i.strip() in allowed_indices]
        if not devices or any(d["used_mib"] > 1024 or d["utilization"] > 5 for d in devices):
            raise BlockedError(f"BLOCKED_GPU_RESOURCES: selected GPUs occupied; existing jobs preserved: {devices}")
        result["GPUs"] = devices
    return result


def qualify_distributed(device, policy_path="/mnt/project/server_dispatch_policy.json"):
    """All local ranks observe one pre-CUDA resource check before model creation.

    A CPU TCPStore uses torchrun's existing rendezvous. Creating a CUDA process
    group first would make our own contexts look like competing GPU workloads.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return qualify(device, policy_path)
    import torch.distributed as dist
    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                          is_master=False, timeout=timedelta(seconds=120))
    prefix = "iqe/resource/" + os.environ.get("TORCHELASTIC_RUN_ID", "run") + "/" + socket.gethostname() + "/"
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        try:
            result = {"result": qualify(device, policy_path)}
        except Exception as error:
            result = {"error": str(error)}
        store.set(prefix + "result", json.dumps(result))
    result = json.loads(store.get(prefix + "result").decode())
    store.set(prefix + "rank/" + str(local_rank), "ready")
    store.wait([prefix + "rank/" + str(i) for i in range(int(os.environ["LOCAL_WORLD_SIZE"]))])
    if "error" in result:
        raise BlockedError(result["error"])
    return result["result"]
