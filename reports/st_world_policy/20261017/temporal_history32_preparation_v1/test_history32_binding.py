"""Small synthetic binding tests only; no builder, ledger, images or tensors."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
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
        w_source_compatibility=copy.deepcopy(m.W_COMPAT),
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

    def test_compatibility_pins_required_in_admission(self):
        a, cp, now = admission()
        for dependency in m.W_COMPAT:
            bad = copy.deepcopy(a)
            bad['w_source_compatibility'][dependency]['sha256'] = '0' * 64
            with self.subTest(dependency=dependency), self.assertRaises(ValueError):
                m.validate_binding(bad, cp, now=now)
        del a['w_source_compatibility']
        with self.assertRaises(ValueError): m.validate_binding(a, cp, now=now)

    def test_parent_and_child_install_before_original_main(self):
        # Exercise actual wrapper main; only control dispatch is mocked, never
        # invoke the real builder or storage guard in these fixtures.
        a, cp, _ = admission()
        reg_path = HERE / 'actual_preparation/HISTORY32_STORAGE_REGISTRATION.json'
        reg = json.loads(reg_path.read_text())
        a['registration'] = {'path': str(reg_path), 'sha256': hashlib.sha256(reg_path.read_bytes()).hexdigest()}
        a['builder_budget'] = json.loads((HERE / 'actual_preparation/ADMISSION_PROPOSAL.json').read_text())['builder_budget']
        a['controller_sources'] = [{'path': str(HERE / 'run_history32_once.py'),
            'sha256': hashlib.sha256((HERE / 'run_history32_once.py').read_bytes()).hexdigest()}]
        original_read = m.pinned
        def read(pin, **kw):
            if pin['path'] == '/synthetic-admission': return a
            if pin == a['checkpoint_receipt']: return cp
            return original_read(pin, **kw)
        for child in (False, True):
            events = []
            controller = types.SimpleNamespace(main=lambda: events.append('original_main') or 17)
            with self.subTest(child=child), patch.object(m, 'pinned', read), \
                    patch.object(m.os, 'sched_setaffinity'), patch.object(m.signal, 'getitimer', return_value=(0., 0.)), \
                    patch.object(m.signal, 'signal'), patch.object(m.signal, 'setitimer'), \
                    patch.object(m, 'make_controller', return_value=controller), \
                    patch.object(m, 'install_current_w_compatibility', side_effect=lambda pins: events.append('install') or {'fixture': True}), \
                    patch.dict(m.os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                    patch.object(sys, 'argv', ['wrapper', '--admission', '/synthetic-admission', '--admission-sha256', '0' * 64] + (['--child'] if child else [])):
                self.assertEqual(m.main(), 17)
            self.assertEqual(events, ['install', 'original_main'])


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_path = list(sys.path)
        cls.old_modules = set(sys.modules)
        cls.receipt = m.install_current_w_compatibility(m.W_COMPAT)
        cls.w = sys.modules['tools.st_world_policy.spatial_w_storage']
        cls.metadata = m.importlib.import_module('tools.st_world_policy.metadata_storage')

    @classmethod
    def tearDownClass(cls):
        for name in set(sys.modules) - cls.old_modules:
            if name == 'tools' or name.startswith('tools.'): del sys.modules[name]
        sys.path[:] = cls.old_path

    def test_real_verified_bridge_retains_current_runtime_without_torch(self):
        self.assertEqual(Path(self.w.__file__), Path(m.W_COMPAT['current_validator']['path']))
        self.assertEqual(Path(self.metadata.__file__), m.RUNTIME / 'tools/st_world_policy/metadata_storage.py')
        self.assertEqual(Path(self.w.b.__file__), m.RUNTIME / 'tools/st_world_policy/cache_storage_budget.py')
        self.assertNotIn('torch', sys.modules)
        old, new = self.receipt['actual_source_shas_preserved']
        self.assertEqual(self.w._verified_common_compute_source(old), self.w._verified_common_compute_source(new))
        with self.assertRaises(ValueError): self.w._verified_common_compute_source('f' * 40)
        self.assertFalse(self.receipt['policy_or_ledger_written'])
        with self.assertRaises(ValueError): m.install_current_w_compatibility(m.W_COMPAT)

    def test_exactly_common_expression_changed_endpoint_and_original_untouched(self):
        raw_source = m.pinned(m.W_COMPAT['current_validator'], decode=False).decode()
        adapter = types.ModuleType('fixture_adapter'); adapter.__file__ = m.W_COMPAT['adapter']['path']
        exec(compile(m.pinned(m.W_COMPAT['adapter'], decode=False), adapter.__file__, 'exec'), adapter.__dict__)
        changed = adapter.patched_source(raw_source)
        self.assertEqual(changed.replace("_verified_common_compute_source(binding['source_sha'])", "binding['source_sha']"), raw_source)
        self.assertIn("endpoint = (binding['checkpoint_identity'], binding['training_registration'], binding['source_sha']", changed)
        with self.assertRaises(ValueError): adapter.patched_source(changed)
        with self.assertRaises(ValueError): adapter.verify({**m.W_COMPAT['proof'], 'sha256': '0' * 64})

    def test_real_common_validator_accepts_only_two_sources_keeps_other_guards(self):
        # Synthetic small signatures isolate the original package comparison;
        # no claim of re-running deep signatures, payloads or live policy.
        old, new = self.receipt['actual_source_shas_preserved']
        signatures, rows = {}, []
        for arm, role, source in [('S-P0', 'train', old), ('S-P0', 'dev', old), ('S-P1', 'dev', new)]:
            path = '/SYNTHETIC/' + arm + role
            binding = dict(arm=arm, source_sha=source, environment={'fixture': True},
                observations=dict(split=role, identity=role, manifest=role),
                checkpoint_identity=arm, training_registration=arm, hidden_dim=2048, memory_tokens=144)
            signature = dict(binding=binding, checkpoint={'base_binding': 'base'}, identity='a' * 64)
            signatures[path] = signature
            rows.append(dict(claim_id=arm + role, root=path, cache_identity=signature['identity'],
                identity_kind=self.w.SIGNATURE_SCHEMA, signature_path=path, signature_sha256='b' * 64,
                tensor_schema=self.w.tensor_schema(binding), **self.w._requirements(1, 144, 2048)))
        declarations = {row['root']: row for row in rows}
        core = dict(metadata_package={}, legacy_reservations=[], frozen_w_package=dict(
            schema=self.w.PACKAGE_SCHEMA, maximum_bytes=1 << 30, baseline_policy_sha256='c' * 64,
            baseline_ledger_sha256='d' * 64, baseline_claims={}, baseline_legacy_reservations=[],
            metadata_package_sha256=self.w.b._digest({}), members=declarations))
        before = copy.deepcopy(signatures)
        with patch.object(self.w, '_pin', side_effect=lambda pin: signatures[pin['path']]), \
                patch.object(self.w, 'validate_signature', side_effect=lambda sig: (sig['binding'], self.w._requirements(1, 144, 2048))), \
                patch.object(self.metadata, 'original_package', side_effect=lambda value: value):
            self.assertGreater(self.w.validate_package(core, declarations), 0)
            self.assertEqual(signatures, before)
            for fault in ('third_source', 'base', 'environment', 'same_arm_endpoint'):
                signatures.clear(); signatures.update(copy.deepcopy(before))
                if fault == 'third_source': signatures['/SYNTHETIC/S-P1dev']['binding']['source_sha'] = 'e' * 40
                if fault == 'base': signatures['/SYNTHETIC/S-P1dev']['checkpoint']['base_binding'] = 'different'
                if fault == 'environment': signatures['/SYNTHETIC/S-P1dev']['binding']['environment'] = {'different': True}
                if fault == 'same_arm_endpoint': signatures['/SYNTHETIC/S-P0train']['binding']['source_sha'] = new
                with self.subTest(fault=fault), self.assertRaises(Exception):
                    self.w.validate_package(core, declarations)


if __name__ == '__main__':
    unittest.main(verbosity=2)
