"""Independent synthetic control checks. No actual compact ledgers are opened."""
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
SOURCE = ROOT / 'tools/st_world_policy/analyze_feedback_scale.py'
spec = importlib.util.spec_from_file_location('derived_test_fixtures', ROOT / 'tests/st_world/test_feedback_scale.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
m = fixture.m

def rejected(fn, fragment):
    try:
        fn()
    except ValueError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('Unexpected acceptance: ' + fragment)

def main():
    started = time.monotonic()
    checks = []
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '026f49fb34f22307439fc6c2672e0323758b8d65710c836ffbbff0b3017ff72e'
    assert m._HELPER_SHA == '30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca'
    assert m.PRIMARY_PLAN_SHA == '2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06'
    assert m.PRIMARY_RESULT_SHA == '0a6a84d3652122aeb69760b5237a56c71fec269c16b4066e78a0ddb05876d8cc'
    assert m.base.torch is None and m.base.np is None
    checks.append('all_executable_and_primary_content_pins_no_torch_or_numpy_import')
    affinity = os.sched_getaffinity(0); cpu = min(affinity)
    try:
        with tempfile.TemporaryDirectory(prefix='derived-scale-systems-') as td:
            root = Path(td)
            f = fixture.PrimaryFixture(root)
            snapshot = {p.relative_to(f.root): p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
            for unsafe in (f.root/'never-create', root/'original-run'/'never-create'):
                rejected(lambda: m.execute(f.plan_path, unsafe, cpu, production=False), 'overlaps primary or original run')
                assert not unsafe.exists()
            assert not (root/'original-run').exists()
            assert snapshot == {p.relative_to(f.root): p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
            checks.append('output_isolation_precedes_all_mkdir_or_primary_mutation')

            copied = root/'copied'; copied.mkdir()
            (copied/SOURCE.name).write_bytes(SOURCE.read_bytes())
            (copied/'analyze_full_feedback_support.py').write_text("raise AssertionError('Unverified helper was executed')\n")
            child = subprocess.run([sys.executable, str(copied/SOURCE.name), '--help'], capture_output=True, text=True, timeout=5)
            assert child.returncode != 0
            assert 'Frozen primary helper source hash mismatch before import' in child.stderr
            assert 'Unverified helper was executed' not in child.stderr
            checks.append('helper_hash_rejection_precedes_unverified_helper_execution')

            folder = root/'budget'; folder.mkdir(); data = root/'data.json'; data.write_bytes(b'1')
            b = m.ReadBudget(folder, 'synthetic', time.monotonic())
            initial = json.loads((folder/'READ_BUDGET.json').read_bytes())
            assert initial['reserved_once_only_allowance_bytes'] == 64 << 20
            assert initial['read_operations'] == 0
            with mock.patch.object(b, 'save', side_effect=RuntimeError('interrupt-before-atomic-replace')):
                for _ in range(63): b.read(data, 1)
                try:
                    b.read(data, 1)
                except RuntimeError as exc:
                    assert str(exc) == 'interrupt-before-atomic-replace'
                else:
                    raise AssertionError('64-read flush boundary not reached')
            persisted = json.loads((folder/'READ_BUDGET.json').read_bytes())
            assert b.state['read_operations'] == 64 and persisted['read_operations'] == 0
            assert persisted['abrupt_kill_counter_may_lag_reads'] == 64
            assert persisted['reserved_once_only_allowance_bytes'] == 64 << 20
            assert b.state['precharged_read_bytes'] == initial['precharged_read_bytes'] + 128
            b.state['precharged_read_bytes'] = m.READ_LIMIT - 1
            with mock.patch.object(Path, 'open', side_effect=AssertionError('must not open')):
                rejected(lambda: b.read(data, 1), 'before read')
            checks.append('64_read_flush_interruption_keeps_full_reservation_and_memory_hard_cap')

            real_open = Path.open
            def guarded(path, *args, **kwargs):
                assert path.suffix != '.pt'
                assert 'original-run' not in path.parts
                return real_open(path, *args, **kwargs)
            with mock.patch.object(Path, 'open', guarded):
                result = m.execute(f.plan_path, root/'derived', cpu, production=False)
            assert result['evidence']['primary_actual_logical_read_bytes'] == 1000
            assert result['evidence']['primary_plus_full_derived_allowance_bytes'] == 1000 + (64 << 20)
            assert result['status'] == 'COMPLETE_DESCRIPTIVE_SCALE_EVIDENCE_NOT_APPROVAL'
            assert result['new_scores'] == result['model_loads'] == result['optimizer_updates'] == 0
            assert snapshot == {p.relative_to(f.root): p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
            assert json.loads((f.root/'EXECUTION.json').read_bytes())['deadline_unix'] == 1.0
            rejected(lambda: m.execute(f.plan_path, root/'derived', cpu, production=False), 'already exists')
            assert sum(p.stat().st_size for p in (root/'derived').iterdir()) <= 2 << 20
            checks.append('compact_only_hash_join_primary_deadline_immutable_once_only_and_output_cap')
    finally:
        os.sched_setaffinity(0, affinity)
    print(json.dumps({'status':'PASS','count':len(checks),'checks':checks,'wall_seconds':time.monotonic()-started,
                      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      'real_compact_inputs_read':0,'original_payload_reads':0,'remote_dispatches':0,'GPU_initializations':0},indent=2))

if __name__ == '__main__':
    main()
