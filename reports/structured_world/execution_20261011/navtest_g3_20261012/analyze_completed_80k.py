"""One-shot CPU analysis after the existing controller validates full G3 80k."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DISPATCH = ROOT.parent / 'navsim_navtest_dispatch_20261011T0225'
KEY = 'G3_EVENT_LOCAL_0080000'
ANALYZER = Path('/mnt/project/VLA-Drive-structured-world-navtest-cohost-frozen-20261012/tools/structured_world/analyze_navsim_navtest.py')
EXPECTED_SOURCE = 'ba288ea8fa8f897eab96ffb2f44174363cc31d02b07d12a8b249d22edf5c409e'
OUTPUT = ROOT / 'G3_80K_ANALYSIS.json'


def state(status, **fields):
    payload = {'status': status, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'pid': os.getpid(), 'model_or_evaluator_changed': False,
               'GPU_jobs_launched': False, **fields}
    path = ROOT / 'ANALYSIS80_STATE.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main():
    with (ROOT / 'analysis80.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert hashlib.sha256(ANALYZER.read_bytes()).hexdigest() == EXPECTED_SOURCE
        command = [sys.executable, '-B', str(ANALYZER), '--scores',
                   str(DISPATCH / 'jobs' / KEY / 'scores')]
        for updates in (50000, 70000, 75000):
            reference = DISPATCH / 'jobs' / ('G3_EVENT_LOCAL_' + str(updates).zfill(7)) / 'scores/q_final/scenes.csv'
            command += ['--reference', 'G3_' + str(updates) + '=' + str(reference)]
        command += ['--output', str(OUTPUT), '--bootstrap-replicates', '20000']
        assert not OUTPUT.exists(), 'Preserve existing aggregate result'
        state('WAITING_FOR_VALID_FULL_COMPLETION', command=command)
        started = time.monotonic()
        while time.monotonic() - started < 9000:
            if (ROOT / 'STOP_ANALYSIS80').exists():
                state('STOPPED_BEFORE_ANALYSIS')
                return
            completion = DISPATCH / 'completed' / (KEY + '.json')
            if completion.exists():
                receipt = json.loads(completion.read_text())
                assert receipt['request']['key'] == KEY and receipt['valid'] is True
                assert receipt['request']['scenes'] == 12146
                state('ANALYZING_COMPLETE_POPULATION', command=command)
                environment = os.environ.copy()
                environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                subprocess.run(command, env=environment, check=True)
                result = json.loads(OUTPUT.read_text())
                assert result['scenes'] == 12146 and result['logs'] == 136
                state('COMPLETE', output=str(OUTPUT), sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      q0_PDMS=100 * result['planning']['q0']['metrics']['score'],
                      q_final_PDMS=100 * result['planning']['q_final']['metrics']['score'])
                return
            time.sleep(15)
        state('WAIT_TIMEOUT', inference_or_training_signaled=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('FAILED', error=repr(error))
        raise
