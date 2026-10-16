"""Read-only host/cgroup/topology inventory; never admits, stops, or starts work.

Run on the actual intended worker. Host-wide RAM/free disk and container quota
are reported separately. Unknown GPU owners stay unknown, including PIDs that
are not visible in this PID namespace. Raw inventories belong in private runtime
artifacts, not the source repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def command(argv, timeout=15):
    try:
        value = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return {"argv": argv, "returncode": value.returncode,
                "stdout": value.stdout, "stderr": value.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": repr(error)}


def parse_mounts(text):
    """Only cgroup mount metadata; escaped mount paths use proc octal escapes."""
    def decode(value):
        for old, new in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
            value = value.replace(old, new)
        return value
    result = []
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        fields, tail = left.split(), right.split()
        if separator and len(fields) > 5 and tail[0] in ("cgroup", "cgroup2"):
            result.append({"root": decode(fields[3]), "mount": decode(fields[4]),
                           "type": tail[0], "controllers": tail[2].split(",")})
    return result


def cgroup_paths(cgroups, mounts):
    result = []
    for line in cgroups.splitlines():
        _, controllers, relative = line.split(":", 2)
        names = controllers.split(",") if controllers else []
        for mount in mounts:
            if mount["type"] == "cgroup2" and names:
                continue
            if mount["type"] == "cgroup" and not set(names).intersection(mount["controllers"]):
                continue
            root = mount["root"]
            if relative == root:
                suffix = ""
            elif root == "/":
                suffix = relative.lstrip("/")
            elif relative.startswith(root.rstrip("/") + "/"):
                suffix = relative[len(root):].lstrip("/")
            else:
                # A namespaced root can expose / while proc/self reports a host
                # path. Never guess that a nonexistent leaf is the quota.
                suffix = relative.lstrip("/")
            leaf = Path(mount["mount"]) / suffix
            boundary = Path(mount["mount"])
            if not leaf.exists() and boundary.exists():
                leaf = boundary
            result.append({**mount, "controllers": names, "leaf": str(leaf)})
    return result


def cgroup_inventory(proc_root=Path("/proc")):
    raw = (proc_root / "self/cgroup").read_text()
    paths = cgroup_paths(raw, parse_mounts((proc_root / "self/mountinfo").read_text()))
    records, cpu_limits, memory_limits = [], [], []
    wanted = ("cpu.max", "cpu.stat", "cpuset.cpus.effective", "cpuset.cpus",
              "memory.max", "memory.current", "memory.peak", "memory.events",
              "cpu.cfs_quota_us", "cpu.cfs_period_us", "cpuacct.usage",
              "memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    seen = set()
    for entry in paths:
        leaf, boundary = Path(entry["leaf"]), Path(entry["mount"])
        for folder in (leaf, *leaf.parents):
            if folder != boundary and boundary not in folder.parents:
                break
            if str(folder) not in seen:
                seen.add(str(folder))
                values = {}
                for name in wanted:
                    try:
                        values[name] = (folder / name).read_text().strip()
                    except (FileNotFoundError, PermissionError, NotADirectoryError):
                        continue
                records.append({"path": str(folder), "values": values})
                if "cpu.max" in values:
                    quota, period = values["cpu.max"].split()
                    if quota != "max":
                        cpu_limits.append(int(quota) / int(period))
                if "cpu.cfs_quota_us" in values and "cpu.cfs_period_us" in values:
                    quota, period = int(values["cpu.cfs_quota_us"]), int(values["cpu.cfs_period_us"])
                    if quota > 0:
                        cpu_limits.append(quota / period)
                for key in ("memory.max", "memory.limit_in_bytes"):
                    value = values.get(key)
                    if value is not None and value != "max" and int(value) < 1 << 60:
                        memory_limits.append(int(value))
            if folder == boundary:
                break
    return {"raw_self_cgroup": raw, "resolved": paths, "visible_ancestors": records,
            "effective_visible_cpu_quota_cores": min(cpu_limits) if cpu_limits else None,
            "effective_visible_memory_limit_bytes": min(memory_limits) if memory_limits else None,
            "hidden_ancestor_limits": "not_observable; no unlimited-host claim"}


def process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        stat = (path / "stat").read_text()
        # comm can contain spaces/parentheses; the final ')' ends it.
        remaining = stat[stat.rfind(")") + 2:].split()
        cmdline = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return {"pid": int(pid), "uid": path.stat().st_uid, "start_ticks": int(remaining[19]),
                "cwd": os.readlink(path / "cwd"), "cmdline": cmdline,
                "cgroup": (path / "cgroup").read_text(), "visible_in_namespace": True}
    except (OSError, ValueError) as error:
        return {"pid": int(pid), "visible_in_namespace": False, "error": type(error).__name__}


def inspect_resources(storage_path):
    storage_path = Path(storage_path).resolve(strict=True)
    gpu = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id",
                   "--format=csv,noheader,nounits"])
    compute = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"])
    pids = []
    for line in compute.get("stdout", "").splitlines():
        try:
            pids.append(int(line.split(",", 1)[0].strip()))
        except ValueError:
            continue
    known_markers = ("/idle_pressure/", "/zt-idle-pressure-v1/agent.py", "gpu_stress.py", "gpu_pressure.py")
    pressure, visible_workloads = [], []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            cmd = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(marker in cmd for marker in known_markers):
            identity = process_identity(int(child.name))
            identity["classification"] = "marker_match_requires_owner_review; not_authorized_for_signal"
            pressure.append(identity)
        elif any(marker in cmd for marker in ("torchrun", "evaluate_policy", "train_world", "train_student",
                                               "train_shared_a_base", "cache_teacher")):
            visible_workloads.append(process_identity(int(child.name)))
    stat = os.statvfs(storage_path)
    return {"schema": "st_readonly_resource_inventory_v1", "hostname": socket.gethostname(),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "unix": time.time(),
            "pid": os.getpid(), "cpu_count_host": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "cgroup": cgroup_inventory(),
            "memory_host": Path("/proc/meminfo").read_text(),
            "cpu_topology": command(["lscpu", "-J"]),
            "numa": command(["numactl", "--hardware"]),
            "gpu": gpu, "gpu_topology": command(["nvidia-smi", "topo", "-m"]),
            "compute": compute, "compute_processes": [process_identity(p) for p in sorted(set(pids))],
            "pressure_marker_processes": pressure,
            "visible_workload_processes": visible_workloads,
            "gpu_pid_namespace_warning": "NVIDIA host PIDs need not equal visible container PIDs; never infer idle from missing /proc/PID",
            "storage": {"path": str(storage_path), "free_bytes": stat.f_bavail * stat.f_frsize,
                        "total_bytes": stat.f_blocks * stat.f_frsize,
                        "mount": command(["findmnt", "-T", str(storage_path), "-J"]),
                        "quota": command(["lfs", "quota", "-u", str(os.getuid()), str(storage_path)]),
                        "statfs_free_is_not_user_quota": True},
            "actions": "READ_ONLY; no process signals, locks, GPU allocation, or admission"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--output", help="Private JSON inventory; stdout if omitted")
    args = parser.parse_args()
    result = inspect_resources(args.storage_path)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A historical observation must not be silently overwritten.
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"status": "RECORDED", "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
