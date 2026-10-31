"""Exact-host, physical-GPU lease dispatch for the independent ABCD RL task."""
import json
import os
from pathlib import Path
import time
from starVLA.rl.qwen_world_compare.contracts import ROOT, TASK, file_sha
from starVLA.rl.qwen_world_compare.resources import remote, MODEL_PYTHON
from starVLA.rl.flow_grpo.transactions import atomic_json


def launch(host, source, command, name, *, gpus='', expected_uuids=None,
           minimum=None, share_minimum=None, lease_slot=None):
    if TASK != 'qwen_world_abcd_rl_best_20261031':
        raise ValueError('Explicit WORLD_COMPARE_TASK/ROOT required before importing dispatch')
    directory = ROOT/'jobs'/name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'DISPATCH.json').exists():
        raise ValueError('Invocation already dispatched')
    executor = str(Path(source)/'tools/qwen_world_abcd_research/job_worker.py')
    args = [MODEL_PYTHON, executor, '--receipt-root', str(directory), '--gpus', gpus,
            '--command-json', json.dumps(command), '--cwd', str(source)]
    if expected_uuids is not None:
        args += ['--expected-gpu-uuids-json', json.dumps(expected_uuids)]
    if minimum and share_minimum:
        raise ValueError('Choose exclusive or measured sharing admission')
    if minimum:
        args += ['--minimum-free-mib', str(minimum)]
    if share_minimum:
        args += ['--share-min-free-mib', str(share_minimum)]
    args += ['--lease-slot', lease_slot or ('shared' if share_minimum else 'primary')]
    env = dict(WORLD_COMPARE_ROOT=str(ROOT), WORLD_COMPARE_TASK=TASK, PYTHONPATH=str(source),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUTF8='1', LC_ALL='C.UTF-8')
    code = ('import subprocess,json,os;from pathlib import Path;'
            'p=subprocess.Popen('+repr(args)+',cwd='+repr(str(source))+
            ',stdout=open('+repr(str(directory/'worker.log'))+',"ab"),stderr=subprocess.STDOUT,'
            'start_new_session=True,env=dict(os.environ,**'+repr(env)+'));'
            'print(json.dumps(dict(worker_pid=p.pid,worker_start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]))))')
    started = json.loads(remote(host, [MODEL_PYTHON, '-c', code]))
    receipt = dict(host=host, source=str(source), command=command, invocation=name,
                   dispatched_unix=time.time(), executor=executor,
                   executor_sha256=file_sha(executor), **started)
    atomic_json(directory/'DISPATCH.json', receipt)
    return directory


def job_state(directory):
    for name in ('RESULT.json', 'RUNNING.json', 'DISPATCH.json'):
        path = Path(directory)/name
        if path.exists():
            value = json.loads(path.read_text())
            value.setdefault('status', 'DISPATCHED')
            return value
    raise FileNotFoundError(directory)
