"""Bounded reviewer CPU fixtures; never launches a real trainer or GPU work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TMP = Path('/tmp/st-t1-native4-adapter-correctness-final-20261017')
if TMP.exists():
    raise SystemExit('Prior test evidence exists; refuse overwrite')
TMP.mkdir()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(ROOT), TMPDIR=str(TMP))
env.pop('ST_TEST_NATIVE_CPU', None)
cmd = ['taskset', '-c', '0,1', '/root/miniconda3/envs/ddp/bin/python', '-m', 'pytest', '-vv',
    '-p', 'no:cacheprovider', '--basetemp', str(TMP / 'pytest'),
    'reports/st_world_policy/20261017/official_T1_native4_execution_v1/test_run_once.py',
    'reports/st_world_policy/20261017/official_T1_native4_execution_v1/test_independent_deadline.py',
    str(OUT / 'test_adapter_independent.py')]
started = time.monotonic()
peak_rss = peak_output = 0
reason = None
with (OUT / 'PYTEST_FULL.log').open('x') as log:
    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        table = {}
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                table[int(path.parent.name)] = (int(fields[1]), int(fields[2]),
                    int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
            except (OSError, ValueError, IndexError):
                continue
        owned = {process.pid} | {pid for pid, (_, group, _) in table.items() if group == process.pid}
        while True:
            expanded = owned | {pid for pid, (parent, _, _) in table.items() if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        rss = sum(row[2] for pid, row in table.items() if pid in owned)
        peak_rss = max(peak_rss, rss)
        size = 0
        for directory in (TMP, OUT):
            for base, _, files in os.walk(directory):
                for name in files:
                    try:
                        size += (Path(base) / name).stat().st_blocks * 512
                    except FileNotFoundError:
                        continue
        peak_output = max(peak_output, size)
        if time.monotonic() - started > 180:
            reason = 'WALL_180S'
        if peak_rss > 4 * 1024**3:
            reason = 'RSS_4GiB'
        if peak_output > 32 * 1024**2:
            reason = 'OUTPUT_32MiB'
        if reason:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(.1)
    code = process.wait()
receipt = dict(command=cmd, exit_code=code, wall_seconds=time.monotonic()-started,
    observed_peak_owned_descendant_rss_bytes=peak_rss, observed_peak_output_allocated_bytes=peak_output,
    sampling_interval_seconds=.1, cpu_affinity=[0, 1], library_threads=2, cuda_visible_devices='',
    violation=reason, actual_model_image_target_payload_reads=0, actual_native_checkpoint_test=False,
    scope='CPU synthetic adapter/comparator/control tests; no admission or resource permission')
(OUT / 'TEST_EXECUTION.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(code or bool(reason))
