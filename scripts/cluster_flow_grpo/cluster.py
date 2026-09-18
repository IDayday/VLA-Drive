"""Explicit, bounded SSH process orchestration around the unchanged training CLI.

This layer never grants training acceptance or alters the optimizer recipe.
Each remote supervisor owns exactly one process group and records its exit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/root/miniconda3/envs/ddp/bin/python"
# SSH executes this file directly in a login directory with no PYTHONPATH.
# Bootstrap the supervisor before base_env configures the actual child process.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def exclusive_controller(root):
    """Reject a second controller; never wait behind a live experiment."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/"controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"experiment controller already active: {root}") from exc
        stream.seek(0);stream.truncate()
        json.dump({"pid":os.getpid(),"host":socket.gethostname(),"started":time.time()},stream)
        stream.flush()
        yield


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def base_env():
    from scripts.cluster_flow_grpo.identity import configure_release
    configure_release()
    return {
        **os.environ,
        "PYTHONPATH": f"{ROOT}/navsim:{ROOT}",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "NO_ALBUMENTATIONS_UPDATE": "1", "FLASH_ATTENTION_DETERMINISTIC": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
    }


def command_for(spec, node_rank):
    node = spec["nodes"][node_rank]
    if "direct_command" in spec:
        if len(spec["nodes"]) != 1:
            raise ValueError("direct commands require one supervisor")
        return list(spec["direct_command"])
    entry = list(spec["entry"])
    if spec.get("asset_verification_receipt"):
        if entry[:2] != ["-m", "starVLA.rl.flow_grpo.cli"] or "train" not in entry:
            raise ValueError("verified asset reuse only wraps the native train CLI")
        receipt = spec["asset_verification_receipt"]
        entry = ["-m", "scripts.cluster_flow_grpo.assets", "--receipt", receipt["path"],
                 "--sha256", receipt["sha256"], "--module", entry[1], *entry[2:]]
    return [
        PYTHON, "-m", "torch.distributed.run",
        "--nnodes", str(len(spec["nodes"])),
        "--nproc-per-node", str(len(node["devices"])),
        "--node-rank", str(node_rank), "--master-addr", spec["master_addr"],
        "--master-port", str(spec["master_port"]),
        "--max-restarts=0", "--monitor-interval=1", *entry,
    ]


def supervise(spec_path, node_rank):
    spec = json.loads(Path(spec_path).read_text())
    node = spec["nodes"][node_rank]
    output = Path(spec["control_dir"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / f"node{node_rank}.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    if node.get("cpu_affinity"):
        os.sched_setaffinity(0, node["cpu_affinity"])
    warm = spec.get("warm_files", [])
    entry = spec.get("entry", [])
    if not warm and "train" in entry and "--config" in entry:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(ROOT/entry[entry.index("--config")+1])
        warm = [str(p) for p in Path(cfg.paths.base_vlm).glob("*.safetensors")]
        source = Path(cfg.sft_checkpoint)
        warm.append(str(source/"pytorch_model.pt" if source.is_dir() else source))
    if warm:
        def warm_file(value):
            path=Path(value);count=0
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b""):
                    count+=len(chunk)
            return {"path":str(path),"bytes":count}
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=min(4,len(warm))) as pool:
            warmed=list(pool.map(warm_file,warm))
        write_json(output/f"warmup_node{node_rank}.json",{
            "files":warmed,"seconds":time.monotonic()-started,
            "scope":"page-cache warmup only; native trainer still verifies original input hashes"})
    if spec.get("require_idle_gpus", False):
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
        used = {int(a): (int(b),int(c)) for a,b,c in (line.split(",") for line in rows.splitlines())}
        # A live rank doing CPU asset checks already owns a~420MiB CUDA context.
        # It must not be mistaken for an idle GPU merely because utilization is0.
        if any(used.get(int(gpu),(10**9,100))[0] > 64 or
               used.get(int(gpu),(10**9,100))[1] > 10 for gpu in node["devices"]):
            raise RuntimeError("allocated GPU is occupied; refusing to launch")
    env = base_env()
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(spec.get("environment", {}))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, node["devices"]))
    env["TRITON_CACHE_DIR"] = str(output / f"triton_node{node_rank}")
    command = command_for(spec, node_rank)
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    state = {"status": "RUNNING", "host": socket.gethostname(),
             "supervisor_pid": os.getpid(), "pid": process.pid,
             "started": time.time(), "command": command,
             "cpu_affinity": sorted(os.sched_getaffinity(0)),
             "environment": {k: env[k] for k in base_env() if k in {
                 "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NCCL_SOCKET_IFNAME", "FLASH_ATTENTION_DETERMINISTIC"}},
             "job_id": spec["job_id"]}

    def terminate(signum, frame):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, terminate)
    try:
        write_json(status_path, state)
        code = process.wait(timeout=spec.get("timeout_seconds"))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    finally:
        # Even an initial status-file write failure must not orphan torchrun.
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        state.update(status="COMPLETE" if process.returncode == 0 else "FAILED",
                     exit_code=process.returncode, finished=time.time())
        write_json(status_path, state)
    return code


def run(spec_path, cancel_event=None):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    sizes = {len(n["devices"]) for n in spec["nodes"]}
    cpu_only = spec.get("cpu_only", False)
    if cpu_only and (len(spec["nodes"]) != 1 or sizes != {0} or not spec.get("direct_command")):
        raise ValueError("CPU-only supervision requires one direct-command node with no GPU slots")
    if not sizes or (0 in sizes and not cpu_only):
        raise ValueError("nonempty rank counts required on each node")
    occupied = set()
    for node in spec["nodes"]:
        for device in node["devices"]:
            key = (node["host"], device)
            if key in occupied:
                raise ValueError("duplicate GPU slot in cluster job")
            occupied.add(key)
    output = Path(spec["control_dir"])
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    streams = []
    try:
        for rank, node in enumerate(spec["nodes"]):
            command = [PYTHON, str(Path(__file__).resolve()), "supervise",
                       str(spec_path), "--node-rank", str(rank)]
            if node["host"] != "local":
                command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           node["host"], shlex.join(command)]
            stream = (output / f"node{rank}.log").open("x")
            streams.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        started = time.monotonic()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("cluster experiment cancelled after peer failure")
            codes = [p.poll() for p in processes]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f"cluster peer failed: {codes}")
            if all(c is not None for c in codes):
                break
            if time.monotonic() - started > spec.get("timeout_seconds", 86400) + 60:
                raise TimeoutError("cluster operation exceeded its deadline")
            time.sleep(1)
        result = {"status": "PASS", "exit_codes": codes, "seconds": time.monotonic()-started}
        write_json(output / "result.json", result)
        return result
    except BaseException as exc:
        # Signal the job-specific remote supervisors, not arbitrary SSH sessions.
        for rank, node in enumerate(spec["nodes"]):
            status = output / f"node{rank}.json"
            if status.exists():
                state = json.loads(status.read_text())
                if state["job_id"] != spec["job_id"] or state["status"] != "RUNNING":
                    continue
                code = ("import os,signal,pathlib; p=" + str(state["supervisor_pid"]) + "; "
                        "c=pathlib.Path(f'/proc/{p}/cmdline').read_bytes(); "
                        "assert b'cluster.py' in c and " + repr(str(spec_path).encode()) +
                        " in c; os.kill(p,signal.SIGTERM)")
                cmd = [PYTHON, "-c", code]
                if node["host"] != "local":
                    cmd = ["ssh", "-o", "ConnectTimeout=5", node["host"], shlex.join(cmd)]
                subprocess.run(cmd, timeout=15, check=False, stdout=subprocess.DEVNULL)
        write_json(output / "result.json", {"status": "FAIL", "error": str(exc)})
        raise
    finally:
        for p in processes:
            try:
                p.wait(timeout=25)
            except subprocess.TimeoutExpired:
                p.terminate()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["run", "supervise"])
    p.add_argument("spec")
    p.add_argument("--node-rank", type=int, default=0)
    a = p.parse_args()
    if a.mode == "supervise":
        sys.exit(supervise(a.spec, a.node_rank))
    print(json.dumps(run(a.spec)))
