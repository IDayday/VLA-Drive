"""Independent regression of the two previously demonstrated source-boundary defects."""
import time
START = time.monotonic()
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path('/mnt/project/VLA-Drive-st-indexprep-source-boundary-20261017/reports/st_world_policy/20261017/future_index_sequential_control_v1/prepare_once.py')
SOURCE_SHA = '49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
spec = importlib.util.spec_from_file_location('independent_fixed_index_preparer', SOURCE)
module = importlib.util.module_from_spec(spec)
exec(compile(raw, str(SOURCE), 'exec'), module.__dict__)
EVENTS = []


class BoundaryRegression(unittest.TestCase):
    def test_fifo_rejected_before_open_wait(self):
        with tempfile.TemporaryDirectory(prefix='independent-index-source-') as directory:
            source = Path(directory) / 'source.py'
            os.mkfifo(source)
            previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
                TimeoutError('Independent 200ms deadline: FIFO blocked')))
            signal.setitimer(signal.ITIMER_REAL, .2)
            before = time.monotonic()
            try:
                with self.assertRaisesRegex(ValueError, 'regular'):
                    module.read_source(source, '0' * 64)
            finally:
                elapsed = time.monotonic() - before
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            self.assertLess(elapsed, .15)
            EVENTS.append({'case': 'fifo', 'elapsed_seconds': elapsed,
                           'fixture_content_bytes_read': 0, 'result': 'PASS'})

    def test_actual_main_bounds_first_load_by_original_absolute_deadline(self):
        previous_handler = signal.getsignal(signal.SIGALRM)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
        limits = dict(module.LIMITS, wall_seconds=.10)
        # Half the absolute budget is already consumed. The first loader sleeps
        # beyond both deadlines; main must interrupt before its normal return.
        with patch.object(module, 'START', time.monotonic() - .05), \
             patch.object(module, 'LIMITS', limits), \
             patch.object(module, 'options', return_value=object()), \
             patch.object(module.os, 'sched_setaffinity'), \
             patch.object(module, 'load', side_effect=lambda *args: time.sleep(.5)) as loader, \
             patch.object(module, 'run') as run:
            before = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'wall deadline'):
                module.main()
            elapsed = time.monotonic() - before
            loader.assert_called_once()
            run.assert_not_called()
        self.assertLess(elapsed, .09)
        self.assertGreater(elapsed, .02)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertEqual(os.sched_getaffinity(0), {8, 9})
        EVENTS.append({'case': 'first_load_absolute_deadline', 'elapsed_seconds': elapsed,
                       'actual_external_source_loads': 0, 'result': 'PASS'})


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryRegression)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    io = {key: int(value) for key, value in
          (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    cost = {'schema': 'independent_source_boundary_review_cost_v2',
            'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
            'source_bytes_read': len(raw), 'tests': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'wall_seconds': time.monotonic() - START,
            'process_rchar_bytes_since_start': io['rchar'],
            'process_vm_hwm_bytes': int(status['VmHWM'].split()[0]) * 1024,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'fixture_payload_read_bytes': 0, 'fixture_payload_write_bytes': 0,
            'events': EVENTS, 'no_actual_payload_ledger_gpu_or_dispatch': True}
    assert cost['wall_seconds'] < 120
    assert cost['process_rchar_bytes_since_start'] <= 256 << 20
    assert cost['process_vm_hwm_bytes'] <= 1 << 30
    Path(__file__).with_name('COST.json').write_text(json.dumps(cost, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
