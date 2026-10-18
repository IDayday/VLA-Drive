"""CPU-only checks of exact external control source; no launcher main/GPU imports."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest


RESOURCES = Path('/mnt/project/st-world-policy-20261016/resources')
LAUNCHER = RESOURCES / 'launch_reviewed_rollout_scale_formal_v1.py'
MONITOR = RESOURCES / 'monitor_rollout_scale_formal_v1.py'
RUN_ID = 'C1_rollout_std_fulltrain_seed42_official_4096_v1'
REG_SHA = '928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f'
GIB = 2**30


def functions():
    """Use unchanged function ASTs; deliberately do not execute launcher imports/main."""
    tree = ast.parse(LAUNCHER.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'native_admission', 'validate_storage', 'fresh_resources'}]
    namespace = dict(RUN_ID=RUN_ID, REG_SHA=REG_SHA, GIB=GIB, ROOT=Path('/mnt/project'))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(LAUNCHER), 'exec'), namespace)
    return namespace


def reservations():
    return dict(run_id=RUN_ID, registration_sha256=REG_SHA,
                own_run_reserve_bytes=209*GIB, shared_free_floor_bytes=1024*GIB,
                unresolved_old_remaining_bytes=4*GIB,
                other_active_remaining_writes=[dict(bytes=292*GIB)])


class Controls(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def test_storage_1529_exact_boundary(self):
        self.assertEqual(self.ns['validate_storage'](reservations(), 1529*GIB), 1529*GIB)
        with self.assertRaises(ValueError):
            self.ns['validate_storage'](reservations(), 1529*GIB-1)

    def test_changed_scientific_identity_rejected(self):
        value = reservations(); value['registration_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.ns['validate_storage'](value, 2000*GIB)

    def test_negative_external_commitment_rejected(self):
        value = reservations(); value['other_active_remaining_writes'][0]['bytes'] = -1
        with self.assertRaises(ValueError):
            self.ns['validate_storage'](value, 2000*GIB)

    def test_first_invocation_native_budget(self):
        result = self.ns['native_admission']([], 0, False)
        self.assertEqual(result['maximum_directories'], 11)
        self.assertEqual(result['initial_writes'], 1)
        self.assertEqual(result['remaining_periodic_steps'][-1], 4096)

    def test_resume_counts_existing_partial_directories(self):
        result = self.ns['native_admission'](['initial', 'partial', 'step512'], 512, True)
        self.assertEqual(result['maximum_directories'], 12)
        self.assertEqual(result['initial_writes'], 0)
        with self.assertRaises(ValueError):
            self.ns['native_admission'](list(range(8)), 512, True)

    def fake_resources(self, apps=''):
        def query(argv, **kwargs):
            return apps if '--query-compute-apps=gpu_uuid,pid' in argv else '0, GPU-A, 0, 0\n'
        self.ns.update(os=types.SimpleNamespace(sched_getaffinity=lambda _: set(range(128))),
                       subprocess=types.SimpleNamespace(check_output=query),
                       Path=lambda _: types.SimpleNamespace(
                           read_text=lambda: 'MemAvailable: 900000000 kB\n', is_dir=lambda: True),
                       shutil=types.SimpleNamespace(disk_usage=lambda _: types.SimpleNamespace(free=2000*GIB)),
                       time=types.SimpleNamespace(time=lambda: 1.0))

    def test_zero_memory_gpu_app_still_rejected(self):
        self.fake_resources('GPU-A, 123\n')
        with self.assertRaisesRegex(RuntimeError, 'compute process'):
            self.ns['fresh_resources']('0', ['GPU-A'], reservations())

    def test_idle_selected_gpu_and_unrelated_app(self):
        self.fake_resources('GPU-B, 456\n')
        result = self.ns['fresh_resources']('0', ['GPU-A'], reservations())
        self.assertEqual(result['required_free_bytes'], 1529*GIB)

    def test_uuid_mismatch_rejected(self):
        self.fake_resources()
        with self.assertRaisesRegex(RuntimeError, 'UUID'):
            self.ns['fresh_resources']('0', ['GPU-C'], reservations())

    def test_monitor_preserves_existing_operator_stop(self):
        spec = importlib.util.spec_from_file_location('reviewed_monitor', MONITOR)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); own = root / 'own'; own.mkdir(); other = root / 'other'; other.mkdir()
            (own / 'STOP_REQUESTED').write_text('operator reason\n')
            self.assertTrue(mod.request_stop(own, {'reason': 'resource'}))
            self.assertEqual((own / 'STOP_REQUESTED').read_text(), 'operator reason\n')
            self.assertFalse((other / 'STOP_REQUESTED').exists())
            self.assertFalse(mod.request_stop(root / 'absent', {'reason': 'resource'}))

    def test_monitor_new_stop_is_owned_only(self):
        spec = importlib.util.spec_from_file_location('reviewed_monitor2', MONITOR)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(mod.request_stop(root, {'reason': 'resource'}))
            self.assertEqual(json.loads((root / 'STOP_REQUESTED').read_text()), {'reason': 'resource'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
