"""Independent bounded CPU controls; never opens a training payload."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'tools/st_world_policy/analyze_full_feedback_support.py'
PLAN = ROOT / 'reports/st_world_policy/20261018/full_train_feedback_support_v1/PLAN_CANDIDATE.json'
spec = importlib.util.spec_from_file_location('system_review_target', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []

def ok(name):
    checks.append(name)

def rejected(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('Unexpected acceptance: ' + text)

def main():
    started = time.monotonic()
    plan_raw = PLAN.read_bytes()
    plan = m.validate_plan(plan_raw)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca'
    assert hashlib.sha256(plan_raw).hexdigest() == '2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06'
    assert m.torch is None and m.np is None
    assert [len(s['items']) for s in plan['shards']] == [512] * 8
    assert 8 * m.SHARD_READ + m.CONTROL_READ == 32 << 30
    assert 8 * m.SHARD_OUTPUT + (16 << 20) + (16 << 20) == 128 << 20
    assert 8 * m.READ_RATE == 32 << 20
    assert 8 * m.READ_BURST == 64 << 20
    ok('fixed_source_plan_partition_and_global_budget_arithmetic')
    with tempfile.TemporaryDirectory(prefix='full-feedback-system-review-') as td:
        root = Path(td)
        clock = [1000.0]
        sleeps = []
        def sleep(dt):
            sleeps.append(dt)
            clock[0] += dt
        with mock.patch.object(m.time, 'time', lambda: clock[0]), mock.patch.object(m.time, 'sleep', sleep):
            b = m.Budget(root / 'pace', 'plan', 'test', 2000., 1 << 30, 1 << 20, False)
            b.pace(m.READ_BURST)
            assert not sleeps
            b.pace(m.READ_RATE)
            assert sleeps == [1.0]
            clock[0] += 50
            b.pace(m.READ_BURST)
            b.pace(m.READ_RATE)
            assert sleeps == [1.0, 1.0]
            b.state['io_tokens'] = 0
            b.state['io_updated_unix'] = clock[0] + 1
            b.save()
            resumed = m.Budget(root / 'pace', 'plan', 'test', 2000., 1 << 30, 1 << 20, True)
            resumed.pace(m.READ_RATE)
            assert sleeps[-1] == 2.0
            rejected(lambda: m.Budget(root / 'pace', 'plan', 'test', 2001., 1 << 30, 1 << 20, True), 'reset deadline')
            resumed.deadline = clock[0] + .5
            rejected(lambda: resumed.pace(m.READ_RATE), 'pacing would exceed')
        ok('rate_burst_idle_credit_interrupted_reservation_and_resume_debt')

        p = root / 'payload_fixture'; p.write_bytes(b'0123456789')
        b = m.Budget(root / 'read', 'plan', 'read', time.time() + 60, 19, 1 << 20, False)
        with mock.patch.object(Path, 'open', side_effect=AssertionError('Input must not open')):
            rejected(lambda: b.read(p, payload=True), 'before read/load')
        assert b.state['actual_read_bytes'] == 0
        ok('logical_read_and_nominal_load_charged_before_open')

        b = m.Budget(root / 'out', 'plan', 'out', time.time()+60, 1 << 20, 80 << 10, False)
        rejected(lambda: b.publish(root / 'rejected', b'x' * (17 << 10)), 'Output budget exhausted')
        assert not (root / 'rejected').exists()
        m.atomic(root / 'immutable', b'first', immutable=True)
        m.atomic(root / 'immutable', b'first', immutable=True)
        rejected(lambda: m.atomic(root / 'immutable', b'second', immutable=True), 'Immutable/stale')
        ok('output_precharge_and_atomic_immutable_idempotence')

        path = root / 'LOCK'
        child = "import fcntl,sys; f=open(sys.argv[1],'a+b');\ntry: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(7)\nsys.exit(0)"
        with m.lock(path):
            r = subprocess.run([sys.executable, '-c', child, str(path)], timeout=5)
            assert r.returncode == 7
        r = subprocess.run([sys.executable, '-c', child, str(path)], timeout=5)
        assert r.returncode == 0
        ok('actual_owned_child_duplicate_lock_refused_then_release_acquired')

        try:
            with m.wall_guard(time.time() + .03):
                time.sleep(.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError('Common deadline not enforced')
        ok('real_signal_deadline_interrupts_cpu_worker_wait')

        source = root / 'changed_source'; source.write_text('changed')
        with mock.patch.object(m, '__file__', str(source)):
            rejected(lambda: m.validate_plan(plan_raw), 'Analyzer source changed')
        altered = copy.deepcopy(plan); altered['limits']['wall_seconds'] += 1
        altered.pop('plan_id'); altered['plan_id'] = m.identity(altered)
        rejected(lambda: m.validate_plan(m.encoded(altered)), 'Budget or topology changed')
        ok('source_hash_and_resource_plan_mutation_rejected')

        affinity = os.sched_getaffinity(0)
        try:
            m.configure_cpu(min(affinity), numeric=False)
            assert len(os.sched_getaffinity(0)) == 1
            assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
            assert all(os.environ[x] == '1' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
            assert m.torch is None
        finally:
            os.sched_setaffinity(0, affinity)
        ok('CPU_affinity_threads_and_no_numeric_or_CUDA_import')

    result = {'status': 'PASS', 'checks': checks, 'count': len(checks),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'plan_sha256': hashlib.sha256(plan_raw).hexdigest(),
              'wall_seconds': time.monotonic()-started, 'actual_training_payload_reads': 0,
              'gpu_initializations': 0, 'remote_dispatches': 0, 'new_scores': 0}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
