"""Bounded local CPU tests only; never invokes calibration measure or dispatch."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = '/root/miniconda3/envs/ddp/bin/python'


def io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def child():
    import pytest
    sys.path.insert(0, str(REPO))
    code = pytest.main(['-vv', '-p', 'no:cacheprovider',
        'tests/st_world/test_temporal_scale_calibration.py',
        'tests/st_world/test_calibration_runtime_preflight.py',
        'reports/st_world_policy/20261017/temporal_scale_calibration_v1/test_controller.py',
        str(HERE / 'test_fix_independent.py')])
    import torch
    assert not torch.cuda.is_initialized()
    assert sorted(os.sched_getaffinity(0)) == [6, 7]
    print(json.dumps({'child_exit': code, 'CUDA_initialized': torch.cuda.is_initialized(),
                      'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}))
    return code


def parent():
    os.sched_setaffinity(0, {6, 7})
    before, started = io(), time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(REPO), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    log = HERE / 'FIX_TESTS.log'
    peak = 0
    reason = None
    with log.open('x') as stream:
        proc = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--child'],
                                cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                status = Path(f'/proc/{proc.pid}/status').read_text()
                rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
                peak = max(peak, rss)
                child_io = Path(f'/proc/{proc.pid}/io').read_text()
                rchar = next(int(line.split(':')[1]) for line in child_io.splitlines() if line.startswith('rchar:'))
            except (FileNotFoundError, StopIteration, ProcessLookupError):
                continue
            if time.monotonic() - started > 115 or peak > 1 << 30 or rchar > (1 << 30) - (16 << 20) or log.stat().st_size > 15 << 20:
                reason = 'OWN_CPU_TEST_BUDGET_EXCEEDED'
                proc.kill()  # Only the Popen child created above; no campaign or other process.
                break
            time.sleep(.05)
        code = proc.wait()
    after = io()
    value = {'status': 'PASS_CPU_TESTS_ONLY' if code == 0 and reason is None else 'FAIL',
        'exit_code': code, 'reason': reason, 'elapsed_seconds': time.monotonic() - started,
        'sampled_child_peak_RSS_bytes': peak,
        'wait_accounted_rchar_delta': after['rchar'] - before['rchar'],
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'GPU_dispatches': 0,
        'model_image_teacher_payload_reads': 0,
        'limits': {'wall_seconds': 120, 'RSS_bytes': 1 << 30, 'logical_read_bytes': 1 << 30, 'output_bytes': 16 << 20}}
    assert not (HERE / 'FIX_CPU_COST.json').exists()
    (HERE / 'FIX_CPU_COST.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))
    return code or (2 if reason else 0)


if __name__ == '__main__':
    raise SystemExit(child() if '--child' in sys.argv else parent())
