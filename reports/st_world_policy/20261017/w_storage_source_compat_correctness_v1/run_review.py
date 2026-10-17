"""Bounded independent source/fixture run; never dispatches an experiment."""
import os
import time
import resource
import signal
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
os.sched_setaffinity(0, {6, 7})
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Independent review deadline')))
signal.alarm(110)
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', CUDA_VISIBLE_DEVICES='',
                  PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

def io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}

protected = [
    '/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/dev_plan/SIGNATURE.json',
    '/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/train_plan/SIGNATURE.json',
    '/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP1_5000_v1/stages/dev_plan/SIGNATURE.json',
]
def pins():
    result = {}
    for name in protected:
        p = Path(name); st = p.stat()
        assert st.st_size <= 1 << 20
        result[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'file_version': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]}
    return result

started = time.monotonic()
initial_io = io()
before = pins()
import pytest
os.chdir(ROOT)
targets = ([str(HERE / 'test_independent.py') + '::test_observer_exact_delta_and_install_before_first_runtime_import']
    if '--single-fixture' in sys.argv else [str(HERE.parent / 'w_storage_source_compat_v1/test_compat.py'),
    str(HERE / 'test_independent.py')])
code = pytest.main(['-v', '-p', 'no:cacheprovider', '--tb=short', *targets,
    '--basetemp', '/tmp/st-w-source-compat-independent-fixtures-v1' + ('-single' if '--single-fixture' in sys.argv else '')])
after = pins()
end_io = io()
assert before == after, 'Protected completed signature bytes/file versions changed'
assert 'torch' not in sys.modules, 'Unexpected model import'
usage = resource.getrusage(resource.RUSAGE_SELF)
result = {
    'test_exit_code': int(code), 'wall_seconds': time.monotonic() - started,
    'cpu_seconds': usage.ru_utime + usage.ru_stime,
    'peak_rss_bytes': usage.ru_maxrss * 1024, 'affinity': sorted(os.sched_getaffinity(0)),
    'process_logical_read_including_startup_bytes': end_io['rchar'],
    'proc_io_delta': {k: end_io[k] - initial_io[k] for k in end_io},
    'protected_signatures_unchanged': before == after, 'protected_signatures': before,
    'torch_imported': 'torch' in sys.modules,
    'gpu_or_experiment_worker_launched': False,
    'reads_note': 'Includes Python/pytest startup and git stdout; source metadata only. Git read-only subprocesses inspect fixed trees. No checkpoint/tensor/database/image/ledger content read.',
}
(HERE / ('SINGLE_FIXTURE_COST.json' if '--single-fixture' in sys.argv else 'COST_AND_PROTECTED_PINS.json')).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
assert result['process_logical_read_including_startup_bytes'] <= 64 << 20
signal.alarm(0)
raise SystemExit(code)
