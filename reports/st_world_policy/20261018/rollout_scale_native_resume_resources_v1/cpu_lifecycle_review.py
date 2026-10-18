"""Independent bounded controller-only fixtures; never import torch or dispatch GPU.

All route/receipt/model-completion helpers are explicitly synthetic. Only child
creation, catchable signals, ownership registration and cleanup are exercised.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


SOURCE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
SPEC = importlib.util.spec_from_file_location('review_native_resume', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HELPER = MODULE.helper()
REAL_POPEN = subprocess.Popen
REAL_STOP = HELPER.stop_owned


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('fixture immutable conflict')
    path.write_text(json.dumps(value))


def run_case(case):
    spawned = []
    with tempfile.TemporaryDirectory(prefix='native_resume_review_') as folder:
        root = Path(folder)/'owned'
        binding = {'scope': 'SYNTHETIC_CPU_LIFECYCLE_ONLY'}
        write(root/'BINDING.json', binding)
        route = {'gpus': list(range(8)), 'gpu_uuids': ['fixture'+str(i) for i in range(8)]}
        if case == 'cleanup_unconfirmed':
            write(root/'PAUSE_REQUEST.json', {'run': str(root/'interrupted/run'), 'completed': 2})

        def popen(*args, **kwargs):
            # The production argv is deliberately never executed in this review.
            delay = '0.02' if case == 'cleanup_unconfirmed' else '60'
            child = REAL_POPEN([sys.executable, '-c', 'import time;time.sleep('+delay+')'], **kwargs)
            spawned.append(child)
            if case in ('SIGTERM_before_Popen_return', 'SIGINT_before_Popen_return'):
                os.kill(os.getpid(), signal.SIGTERM if case.startswith('SIGTERM') else signal.SIGINT)
            return child

        def fixture_write(path, value):
            if case == 'LAUNCH_write_failure' and Path(path).name == 'LAUNCH.json':
                raise OSError('independent CPU fixture receipt failure')
            write(path, value)

        def stop_owned(jobs):
            REAL_STOP(jobs)
            if case == 'cleanup_unconfirmed':
                # Real children have exited; only the acknowledgement is negative.
                for job in jobs:
                    job['cleanup_complete'] = False

        try:
            with contextlib.ExitStack() as stack:
                for owner, name, value in (
                    (MODULE, 'ROOT', root), (MODULE, 'helper', lambda: HELPER),
                    (MODULE, 'route_checks', lambda *a, **k: (route, {'fresh_gpu_v3': lambda *a: None})),
                    (MODULE, 'write', fixture_write), (MODULE, 'run_path', lambda arm: root/arm/'run'),
                    (MODULE, 'group_ready', lambda *a: True),
                    (MODULE, 'validate_result', lambda *a: {'fixture': True}),
                    (HELPER, 'gpu_monitor', lambda *a: {}), (HELPER, 'stop_owned', stop_owned),
                    (HELPER, 'immutable', write),
                    (subprocess, 'Popen', popen),
                ):
                    stack.enter_context(patch.object(owner, name, value))
                result = MODULE.execute('initial', {'fixture': 'CPU_ONLY'}, binding)
            admissions = list((root/'control').glob('invocations/*/*/ADMISSION.json'))
            exits = [json.loads(path.with_name('EXIT.json').read_text()) for path in admissions]
            alive = [child.pid for child in spawned if child.poll() is None]
            assert result['status'] == 'PARTIAL_OR_FAILED', result
            assert not alive, alive
            expected = 2 if case == 'cleanup_unconfirmed' else 1
            assert len(spawned) == len(admissions) == expected
            if case != 'cleanup_unconfirmed':
                assert all(row['owned_cleanup_complete'] for row in exits)
            else:
                assert all(not row['owned_cleanup_complete'] for row in exits)
            return {'case': case, 'status': 'PASS', 'real_short_lived_children': len(spawned),
                    'charged_attempts': len(admissions), 'residual_children': alive,
                    'production_dispatch': False, 'result': result}
        finally:
            for child in spawned:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)


started = time.monotonic()
results = []
for case in ('SIGTERM_before_Popen_return', 'SIGINT_before_Popen_return',
             'LAUNCH_write_failure', 'cleanup_unconfirmed'):
    results.append(run_case(case))
usage = resource.getrusage(resource.RUSAGE_SELF)
children = resource.getrusage(resource.RUSAGE_CHILDREN)
report = {'reviewer': '/root/road_failure_resources_review_v1', 'source': str(SOURCE),
          'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'cases': results,
          'wall_seconds': time.monotonic()-started,
          'CPU_seconds_self_and_waited_children': usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime,
          'self_peak_RSS_bytes': usage.ru_maxrss*1024,
          'post_cases_current_RSS_bytes': int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
          'children_max_peak_RSS_bytes': children.ru_maxrss*1024,
          'large_payload_reads': 0, 'model_calls': 0, 'official_calls': 0, 'GPU_calls': 0,
          'torch_imported': 'torch' in sys.modules,
          'scope': 'Controller lifecycle only. Synthetic route/completion; real CPU Popen/signals/cleanup.'}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
assert report['CPU_seconds_self_and_waited_children'] < 60
assert report['wall_seconds'] < 600
assert not report['torch_imported']
print(json.dumps({'status': 'PASS', 'cases': len(results), 'wall_seconds': report['wall_seconds']}))
