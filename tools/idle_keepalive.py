#!/usr/bin/env python3
"""Adaptive, low-priority CPU keepalive for an interactive dev container.

The daemon consumes a small fraction of one CPU core and runs a low-duty matrix
multiplication worker on each PPU while the container is otherwise idle. It
subtracts its own CPU time and PPU worker PIDs from container activity, so it
can stop the load when real work starts and resume after the container is quiet.

Commands:
    python tools/idle_keepalive.py start
    python tools/idle_keepalive.py status
    python tools/idle_keepalive.py pause
    python tools/idle_keepalive.py resume
    python tools/idle_keepalive.py stop

Optional environment variables:
    IDLE_KEEPALIVE_DUTY=0.12          Fraction of one core used when idle.
    IDLE_KEEPALIVE_BUSY_CORES=0.20    External CPU cores that count as busy.
    IDLE_KEEPALIVE_BUSY_SAMPLES=3     Consecutive busy CPU samples required.
    IDLE_KEEPALIVE_PPU_BUSY=3         PPU utilization percent that counts busy.
    IDLE_KEEPALIVE_PPU_DUTY=0.05      PPU target duty fraction when idle.
    IDLE_KEEPALIVE_PPU_INTERVAL=2     PPU work period in seconds.
    IDLE_KEEPALIVE_PPU_MATRIX=1024    Half-precision matrix dimension.
    IDLE_KEEPALIVE_BUSY_HOLD=20       Quiet seconds before load resumes.
    IDLE_KEEPALIVE_INTERVAL=1.0       Sampling/load period in seconds.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from typing import Any, Optional


RUNTIME_DIR = Path(os.environ.get("IDLE_KEEPALIVE_RUNTIME_DIR", "/tmp/idle-keepalive"))
PID_FILE = RUNTIME_DIR / "daemon.pid"
LOCK_FILE = RUNTIME_DIR / "daemon.lock"
STATE_FILE = RUNTIME_DIR / "state.json"
LOG_FILE = RUNTIME_DIR / "daemon.log"
PAUSE_FILE = RUNTIME_DIR / "manual.pause"

CGROUP_CPU_USAGE_FILES = (
    Path("/sys/fs/cgroup/cpu.stat"),
    Path("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"),
    Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"),
)


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise SystemExit(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)


def read_pid() -> Optional[int]:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def process_is_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"idle_keepalive.py" in cmdline and b"run" in cmdline


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_state() -> dict[str, Any]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def read_cgroup_cpu_seconds() -> Optional[float]:
    for path in CGROUP_CPU_USAGE_FILES:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if path.name == "cpu.stat":
            fields = dict(line.split(maxsplit=1) for line in raw.splitlines())
            if "usage_usec" in fields:
                return int(fields["usage_usec"]) / 1_000_000.0
        else:
            return int(raw) / 1_000_000_000.0
    return None


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def process_cpu_seconds(pids: set[int]) -> float:
    """Return accumulated user+system CPU seconds for live worker processes."""
    clock_ticks = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for pid in pids:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            total += (int(fields[11]) + int(fields[12])) / clock_ticks
        except (FileNotFoundError, OSError, ValueError, IndexError):
            continue
    return total


def process_namespace_pids(pids: set[int]) -> set[int]:
    """Expand container PIDs to every PID namespace identity in /proc status."""
    identities = set(pids)
    for pid in pids:
        try:
            lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            continue
        for line in lines:
            if line.startswith("NSpid:"):
                try:
                    identities.update(int(value) for value in line.split()[1:])
                except ValueError:
                    pass
                break
    return identities


def own_cpu_seconds(worker_pids: Optional[set[int]] = None) -> float:
    """CPU used by the daemon, monitors, and live PPU keepalive workers."""
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed_children = children.ru_utime + children.ru_stime
    return time.process_time() + completed_children + process_cpu_seconds(worker_pids or set())


def initialize_ppu_monitor() -> Optional[tuple[Any, list[Any]]]:
    """Initialize the PPU SDK's in-process NVML-compatible management API."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
            import pynvml

        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:
        return None
    return pynvml, handles


def probe_ppu_monitor(
    monitor: tuple[Any, list[Any]],
) -> Optional[tuple[float, set[int]]]:
    """Return maximum utilization and PIDs holding PPU compute contexts."""
    pynvml, handles = monitor
    utilization: list[float] = []
    pids: set[int] = set()
    try:
        for handle in handles:
            utilization.append(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
            pids.update(
                int(process.pid)
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            )
    except Exception:
        return None
    return max(utilization, default=0.0), pids


def shutdown_ppu_monitor(monitor: Optional[tuple[Any, list[Any]]]) -> None:
    if monitor is None:
        return
    pynvml, _handles = monitor
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def ppu_device_process_pids(
    monitor: Optional[tuple[Any, list[Any]]], device: int
) -> set[int]:
    if monitor is None:
        return set()
    pynvml, handles = monitor
    try:
        return {
            int(process.pid)
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handles[device])
        }
    except Exception:
        return set()


def ppu_worker_identity_file(local_pid: int) -> Path:
    return RUNTIME_DIR / f"ppu-worker-{local_pid}.json"


def ppu_worker_host_pids(local_pids: set[int]) -> set[int]:
    host_pids: set[int] = set()
    for local_pid in local_pids:
        try:
            value = json.loads(
                ppu_worker_identity_file(local_pid).read_text(encoding="utf-8")
            )
            host_pids.update(int(pid) for pid in value.get("host_pids", []))
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return host_pids


def run_ppu_worker(device: int) -> int:
    """Run low-duty FP16 matrix multiplication on one PPU."""
    duty = env_float("IDLE_KEEPALIVE_PPU_DUTY", 0.05, 0.005, 0.25)
    interval = env_float("IDLE_KEEPALIVE_PPU_INTERVAL", 2.0, 0.5, 30.0)
    matrix_size = int(env_float("IDLE_KEEPALIVE_PPU_MATRIX", 1024, 256, 4096))
    parent_pid = os.getppid()

    import warnings

    warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
    monitor = initialize_ppu_monitor()
    pids_before_context = ppu_device_process_pids(monitor, device)
    import torch

    identity_file = ppu_worker_identity_file(os.getpid())
    try:
        torch.cuda.set_device(device)
        tensor_a = torch.randn(
            (matrix_size, matrix_size), device=f"cuda:{device}", dtype=torch.float16
        )
        tensor_b = torch.randn(
            (matrix_size, matrix_size), device=f"cuda:{device}", dtype=torch.float16
        )
        output = torch.empty_like(tensor_a)
        torch.mm(tensor_a, tensor_b, out=output)
        torch.cuda.synchronize(device)

        host_pids = ppu_device_process_pids(monitor, device) - pids_before_context
        write_json_atomic(
            identity_file,
            {"local_pid": os.getpid(), "device": device, "host_pids": sorted(host_pids)},
        )

        while os.getppid() == parent_pid:
            cycle_started = time.monotonic()
            work_deadline = cycle_started + duty * interval
            # Always submit at least one operation per period. Synchronization keeps
            # the configured duty based on actual PPU execution rather than enqueue time.
            first_operation = True
            while first_operation or time.monotonic() < work_deadline:
                torch.mm(tensor_a, tensor_b, out=output)
                torch.cuda.synchronize(device)
                first_operation = False
            remaining = interval - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)

        del output, tensor_a, tensor_b
        return 0
    finally:
        try:
            identity_file.unlink()
        except FileNotFoundError:
            pass
        shutdown_ppu_monitor(monitor)


def start_ppu_workers(device_count: int) -> dict[int, subprocess.Popen[str]]:
    workers: dict[int, subprocess.Popen[str]] = {}
    for device in range(device_count):
        workers[device] = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "ppu-worker",
                "--device",
                str(device),
            ],
            stdin=subprocess.DEVNULL,
            text=True,
            close_fds=True,
        )
    return workers


def stop_ppu_workers(workers: dict[int, subprocess.Popen[str]]) -> None:
    for process in workers.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    while any(process.poll() is None for process in workers.values()):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for process in workers.values():
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        try:
            ppu_worker_identity_file(process.pid).unlink()
        except FileNotFoundError:
            pass
    workers.clear()


def burn_cpu(cpu_seconds: float) -> None:
    """Consume approximately cpu_seconds on this process's current thread."""
    deadline = time.thread_time() + cpu_seconds
    accumulator = 0.123456789
    while time.thread_time() < deadline:
        for index in range(512):
            accumulator = math.sin(accumulator + index) ** 2 + 0.000001
    # Retain an observable use so an alternative interpreter cannot remove the loop.
    if accumulator < 0:
        raise AssertionError("unreachable")


def daemon_state(
    mode: str,
    external_cores: float,
    duty: float,
    busy_threshold: float,
    ppu_available: bool,
    ppu_utilization: Optional[float],
    ppu_device_total: int,
    ppu_worker_pids: set[int],
    external_ppu_pids: set[int],
    ppu_duty: float,
) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "mode": mode,
        "external_cpu_cores": round(external_cores, 4),
        "idle_load_single_core_percent": round(duty * 100, 2),
        "busy_threshold_cores": busy_threshold,
        "ppu_monitoring": "enabled" if ppu_available else "unavailable",
        "ppu_utilization_percent": ppu_utilization,
        "ppu_device_count": ppu_device_total,
        "ppu_idle_load_target_percent": round(ppu_duty * 100, 2),
        "ppu_keepalive_pids": sorted(ppu_worker_pids),
        "external_ppu_pids": sorted(external_ppu_pids),
        "cpu_count": available_cpu_count(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_daemon() -> int:
    ensure_runtime_dir()
    lock_handle = LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("idle keepalive is already running", file=sys.stderr)
        return 1

    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        os.nice(19)
    except OSError:
        pass

    duty = env_float("IDLE_KEEPALIVE_DUTY", 0.12, 0.01, 0.50)
    busy_threshold = env_float("IDLE_KEEPALIVE_BUSY_CORES", 0.20, 0.02, 8.0)
    ppu_busy_threshold = env_float("IDLE_KEEPALIVE_PPU_BUSY", 3.0, 0.0, 100.0)
    ppu_duty = env_float("IDLE_KEEPALIVE_PPU_DUTY", 0.05, 0.005, 0.25)
    busy_hold = env_float("IDLE_KEEPALIVE_BUSY_HOLD", 20.0, 1.0, 600.0)
    interval = env_float("IDLE_KEEPALIVE_INTERVAL", 1.0, 0.25, 10.0)
    busy_samples_required = int(
        env_float("IDLE_KEEPALIVE_BUSY_SAMPLES", 3, 1, 20)
    )

    ppu_monitor = initialize_ppu_monitor()
    initial_ppu_status = probe_ppu_monitor(ppu_monitor) if ppu_monitor else None
    ppu_available = initial_ppu_status is not None
    ppu_utilization, observed_ppu_pids = initial_ppu_status or (None, set())
    ppu_device_total = len(ppu_monitor[1]) if ppu_monitor else 0
    external_ppu_pids = observed_ppu_pids
    ppu_workers: dict[int, subprocess.Popen[str]] = {}
    ppu_worker_retry_after = 0.0
    ppu_worker_started_at = 0.0
    ppu_worker_baseline_external_pids: set[int] = set()
    next_ppu_probe = time.monotonic() + (1.0 if ppu_available else 60.0)
    hold_until = time.monotonic() + min(3.0, busy_hold)
    previous_wall = time.monotonic()
    previous_own_cpu = own_cpu_seconds()
    previous_cgroup_cpu = read_cgroup_cpu_seconds()
    cpu_busy_streak = 0

    try:
        while not stopping:
            cycle_started = time.monotonic()
            worker_pids = {
                process.pid for process in ppu_workers.values() if process.poll() is None
            }
            if ppu_workers and len(worker_pids) != len(ppu_workers):
                stop_ppu_workers(ppu_workers)
                worker_pids.clear()
                ppu_worker_retry_after = cycle_started + 60.0
                ppu_worker_started_at = 0.0

            current_own_cpu = own_cpu_seconds(worker_pids)
            current_cgroup_cpu = read_cgroup_cpu_seconds()
            elapsed = max(cycle_started - previous_wall, 0.001)

            if current_cgroup_cpu is None or previous_cgroup_cpu is None:
                external_cores = 0.0
            else:
                total_cpu = max(0.0, current_cgroup_cpu - previous_cgroup_cpu)
                own_cpu = max(0.0, current_own_cpu - previous_own_cpu)
                external_cores = max(0.0, total_cpu - own_cpu) / elapsed

            previous_wall = cycle_started
            previous_own_cpu = current_own_cpu
            previous_cgroup_cpu = current_cgroup_cpu

            if external_cores >= busy_threshold:
                cpu_busy_streak += 1
            else:
                cpu_busy_streak = 0
            cpu_is_busy = cpu_busy_streak >= busy_samples_required

            if cycle_started >= next_ppu_probe:
                if ppu_monitor is None:
                    ppu_monitor = initialize_ppu_monitor()
                    if ppu_monitor is not None:
                        ppu_device_total = len(ppu_monitor[1])
                probed = probe_ppu_monitor(ppu_monitor) if ppu_monitor else None
                if probed is None:
                    ppu_available = False
                    ppu_utilization = None
                    external_ppu_pids = set()
                    next_ppu_probe = cycle_started + 60.0
                else:
                    ppu_available = True
                    ppu_utilization, observed = probed
                    worker_host_pids = ppu_worker_host_pids(worker_pids)
                    worker_identities = process_namespace_pids(worker_pids) | worker_host_pids
                    unrecognized_pids = observed - worker_identities
                    identities_ready = len(worker_host_pids) >= len(worker_pids)
                    in_startup_grace = (
                        bool(worker_pids)
                        and not identities_ready
                        and cycle_started < ppu_worker_started_at + 8.0
                    )
                    if in_startup_grace:
                        external_ppu_pids = (
                            unrecognized_pids & ppu_worker_baseline_external_pids
                        )
                    else:
                        external_ppu_pids = unrecognized_pids
                    next_ppu_probe = cycle_started + 1.0

            ppu_is_busy = bool(external_ppu_pids) or (
                not worker_pids
                and ppu_utilization is not None
                and ppu_utilization >= ppu_busy_threshold
            )
            if (
                cpu_is_busy
                or ppu_is_busy
            ):
                hold_until = cycle_started + busy_hold

            manual_pause = PAUSE_FILE.exists()
            if manual_pause:
                mode = "paused-manually"
            elif cycle_started < hold_until:
                mode = "paused-busy"
            else:
                mode = "keeping-alive"

            if mode != "keeping-alive":
                if ppu_workers:
                    stop_ppu_workers(ppu_workers)
                worker_pids = set()
                ppu_worker_started_at = 0.0
            elif (
                ppu_available
                and ppu_device_total > 0
                and not ppu_workers
                and cycle_started >= ppu_worker_retry_after
            ):
                ppu_worker_baseline_external_pids = set(external_ppu_pids)
                ppu_workers = start_ppu_workers(ppu_device_total)
                worker_pids = {process.pid for process in ppu_workers.values()}
                ppu_worker_started_at = cycle_started

            write_json_atomic(
                STATE_FILE,
                daemon_state(
                    mode,
                    external_cores,
                    duty,
                    busy_threshold,
                    ppu_available,
                    ppu_utilization,
                    ppu_device_total,
                    worker_pids,
                    external_ppu_pids,
                    ppu_duty,
                ),
            )

            if mode == "keeping-alive":
                burn_cpu(duty * interval)

            remaining = interval - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        stop_ppu_workers(ppu_workers)
        shutdown_ppu_monitor(ppu_monitor)
        write_json_atomic(
            STATE_FILE,
            daemon_state(
                "stopped",
                0.0,
                duty,
                busy_threshold,
                ppu_available,
                ppu_utilization,
                ppu_device_total,
                set(),
                external_ppu_pids,
                ppu_duty,
            ),
        )
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    return 0


def start_daemon() -> int:
    ensure_runtime_dir()
    pid = read_pid()
    if process_is_alive(pid):
        print(f"idle keepalive is already running (pid {pid})")
        return 0

    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass

    log_handle = LOG_FILE.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run"],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_handle.close()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        pid = read_pid()
        if process_is_alive(pid):
            print(f"idle keepalive started (pid {pid})")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    print(f"failed to start idle keepalive; inspect {LOG_FILE}", file=sys.stderr)
    return 1


def stop_daemon() -> int:
    pid = read_pid()
    if not process_is_alive(pid):
        print("idle keepalive is not running")
        return 0
    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            print("idle keepalive stopped")
            return 0
        time.sleep(0.1)
    print(f"idle keepalive did not stop within 5 seconds (pid {pid})", file=sys.stderr)
    return 1


def set_manual_pause(paused: bool) -> int:
    ensure_runtime_dir()
    if paused:
        PAUSE_FILE.touch()
        print("idle keepalive load paused")
    else:
        try:
            PAUSE_FILE.unlink()
        except FileNotFoundError:
            pass
        print("idle keepalive load resumed (automatic busy detection remains enabled)")
    return 0


def show_status() -> int:
    pid = read_pid()
    alive = process_is_alive(pid)
    state = read_state()
    state["running"] = alive
    state["pid"] = pid if alive else None
    state["log_file"] = str(LOG_FILE)
    state["manual_pause"] = PAUSE_FILE.exists()
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if alive else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("start", "stop", "status", "pause", "resume", "run", "ppu-worker"),
    )
    parser.add_argument("--device", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    command = arguments.command
    if command == "start":
        return start_daemon()
    if command == "stop":
        return stop_daemon()
    if command == "status":
        return show_status()
    if command == "pause":
        return set_manual_pause(True)
    if command == "resume":
        return set_manual_pause(False)
    if command == "ppu-worker":
        if arguments.device is None:
            parser.error("ppu-worker requires --device")
        return run_ppu_worker(arguments.device)
    return run_daemon()


if __name__ == "__main__":
    raise SystemExit(main())
