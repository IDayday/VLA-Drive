"""Small synthetic binding tests only; no builder, ledger, images or tensors."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('history32_binding', HERE / 'run_history32_once.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
raw = m.pinned({'path': str(m.ORIGINAL), 'sha256': m.PINS[str(m.ORIGINAL.relative_to(m.RUNTIME))]}, decode=False)


def admission():
    now = time.time()
    a = dict(schema='st_history32_owned_admission_v1', status='APPROVED_HISTORY32_PILOT_ONLY',
        reviewer='SYNTHETIC_INDEPENDENT_REVIEWER_NOT_REAL_APPROVAL', hostname=socket.gethostname(),
        limits=m.limits([2, 3]), runtime=str(m.RUNTIME), output_root=str(m.ROOT),
        control_root=str(m.CONTROL), control_reserved_bytes=1 << 20,
        issued_at_unix=now - 1, latest_start_unix=now + 60, valid_until_unix=now + 1000,
        checkpoint_receipt={'path': '/SYNTHETIC_NOT_REAL.json', 'sha256': '0' * 64})
    cp = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT', remaining_bytes=123456789,
              free_floor_bytes=1 << 40, checked_at_unix=now - 1, valid_until_unix=now + 1100)
    return a, cp, now


class BindingTests(unittest.TestCase):
    def test_exact32_new_cp_and_old_controller_are_separate(self):
        a, cp, now = admission()
        self.assertEqual(m.validate_binding(a, cp, now=now), [2, 3])
        c = m.make_controller(raw, a, cp, [2, 3], m.ENTRY_IO)
        self.assertEqual(c.ROOT, m.ROOT)
        self.assertEqual(c.CP, a['checkpoint_receipt'])
        self.assertEqual(c.BOUND_CP_REMAINING, 123456789)
        self.assertEqual(c.LIMITS, m.limits([2, 3]))
        self.assertIn("expected={'max_scenes':32", m.bind_source(raw))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), m.PINS[str(m.ORIGINAL.relative_to(m.RUNTIME))])

    def test_old64_or_unsigned_or_author_admission_rejected(self):
        a, cp, now = admission()
        for key, value in [('schema', 'st_history64_owned_admission_v1'),
                           ('status', 'DRAFT'), ('reviewer', '/root/st_indexprep_boundary_fix'),
                           ('output_root', str(m.ROOT) + '_other')]:
            b = copy.deepcopy(a); b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_binding(b, cp, now=now)

    def test_limits_cannot_expand(self):
        a, cp, now = admission()
        for key, value in [('read_bytes', (16 << 30) + 1), ('rss_bytes', (4 << 30) + 1),
                           ('max_disk_bytes', 33 << 20), ('wall_seconds', 901),
                           ('max_invocations', 2), ('cpu_affinity', [1, 2, 3])]:
            b = copy.deepcopy(a); b['limits'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_binding(b, cp, now=now)

    def test_current_cp_time_capacity_and_floor_required(self):
        a, cp, now = admission()
        for key, value in [('status', 'EXPIRED'), ('remaining_bytes', -1),
                           ('free_floor_bytes', 0), ('valid_until_unix', now + 10)]:
            b = dict(cp); b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_binding(a, b, now=now)

    def test_mechanical_binding_rejects_changed_source(self):
        with self.assertRaises(ValueError):
            m.bind_source(raw.replace(b"expected={'max_scenes':64", b"expected={'max_scenes':63"))

    def test_nonblocking_pin_rejects_fifo_and_wrong_hash(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fifo'; os.mkfifo(p)
            with self.assertRaises(ValueError):
                m.pinned({'path': str(p), 'sha256': '0' * 64})
            p = Path(d) / 'small.json'; p.write_text('{}')
            with self.assertRaises(ValueError):
                m.pinned({'path': str(p), 'sha256': '0' * 64})

    def test_original_parent_routes_child_and_monitor_without_retry(self):
        from contextlib import nullcontext
        a, cp, now = admission()
        for failure in (None, 'SYNTHETIC_OWNED_DESCENDANT_FAILURE'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                c = m.make_controller(raw, a, cp, [2, 3], m.ENTRY_IO)
                c.ROOT = Path(d) / 'artifact'; c.ROOT.mkdir()
                c.CONTROL_ROOT = Path(d) / 'control'; c.CONTROL_ROOT.mkdir()
                calls = []
                def monitor(command, **kw):
                    calls.append((command, kw))
                    return dict(returncode=0, failure=failure, read_accounting={}, cleanup={'live': 0},
                                child_accounting={'cli_result': {'artifact': {'status': 'COMPLETE'}}})
                core = types.SimpleNamespace(monitor=monitor, publish_guarded=lambda *args: None,
                                             charge_after_reap=lambda *args: 10)
                args = types.SimpleNamespace(admission='/fixture.json', admission_sha256='0' * 64)
                with patch.object(c, 'nonblocking_original_ledger', lambda _: nullcontext()):
                    code = c.run_parent(args, a, {}, {}, None, types.SimpleNamespace(proc_io=lambda: {}),
                                        None, None, core, lambda: 1 << 50, work_deadline=time.monotonic() + 900)
                self.assertEqual(code, 0 if failure is None else 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0][1], str(HERE / 'run_history32_once.py'))
                self.assertEqual(calls[0][1]['cpu_affinity'], (2, 3))
                self.assertEqual(calls[0][1]['budget']['read_bytes'], 16 << 30)
                with self.assertRaises(FileExistsError):
                    c.control_once(c.CONTROL_ROOT, 'INVOCATION.json', {})

    def test_registration_exact_command_and_combined_disk(self):
        a, _, _ = admission()
        reg = json.loads((HERE / 'STORAGE_REGISTRATION_DRAFT.json').read_text())
        reg['cpu_affinity'] = [2, 3]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'budget.json'
            b = dict(phase='data_preparation', split='train', max_scenes=32,
                     max_seconds=900, workers=1, target_times_s=[0.0])
            p.write_text(json.dumps(b))
            a['builder_budget'] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            reg['command'][reg['command'].index('--registration') + 1] = str(p)
            m.validate_registration(reg, a, [2, 3])
            for what in ('population', 'targets', 'disk'):
                bad = copy.deepcopy(reg)
                if what == 'population': bad['command'][bad['command'].index('--max-scenes') + 1] = '64'
                if what == 'targets': bad['command'][bad['command'].index('--target-times') + 1] = '4'
                if what == 'disk': bad['limits']['max_disk_bytes'] = 32 << 20
                with self.subTest(what=what), self.assertRaises(ValueError):
                    m.validate_registration(bad, a, [2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
