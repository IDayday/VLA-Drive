"""Synthetic review of immutable PROFILE controls; no real GPU/process/data access."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path('/mnt/project/st-world-policy-20261016/resources')
LAUNCHER = ROOT / 'launch_rollout_scale_diagnostic_v2.py'
MONITOR = ROOT / 'monitor_rollout_scale_owned_run_v2.py'
EXPECTED_LAUNCHER = '1f2828ffed12d1f27bd78b143fb97ee65000c79f90a7742a9e593fb5564c9e30'
EXPECTED_MONITOR = 'b24a26a27e79cd71919ae2e769061e5b6b033d5993c9fde8260f0065098b2ce2'

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

legacy = module(Path(__file__).parents[1] / 'v2/test_launcher_resources.py', 'prior_synthetic_resource_review')
legacy.LAUNCHER = LAUNCHER
FreshHostControls = legacy.FreshHostControls

class MonitorDeltaControls(unittest.TestCase):
    def run_once(self, smoke_bytes=1, profile_bytes=1, native_counts=(0, 0),
                 gpu_rows='GPU-0, 102, 8', monitoring_error=False, old_stop=None):
        m = module(MONITOR, 'synthetic_monitor')
        with tempfile.TemporaryDirectory(prefix='rollout-resource-review-') as temporary:
            root = Path(temporary)
            smoke = root / 'smoke/scene_center_rollout_std_v1_4update'
            profile = root / 'profiles/scene_center_rollout_std_v1_fulltrain240'
            run = profile / 'runs/owned'
            run.mkdir(parents=True)
            smoke.mkdir(parents=True)
            (profile / 'LAUNCH.json').write_text(json.dumps(dict(host='synthetic-host', pid=101,
                output=str(run), source_sha='synthetic-sha')))
            if old_stop is not None:
                (run / 'STOP_REQUESTED').write_text(old_stop)
            for parent, count in zip([smoke, profile], native_counts):
                for i in range(count):
                    (parent / ('runs/owned/checkpoints/step_%d' % i)).mkdir(parents=True)
            state = dict(done=False)
            child = types.SimpleNamespace(pid=101, poll=lambda: 0 if state['done'] else None,
                                          wait=lambda: 0)
            reads = []
            def size(path):
                reads.append(path)
                return {smoke: smoke_bytes, profile: profile_bytes}[path]
            def command(*args, **kwargs):
                if monitoring_error:
                    raise RuntimeError('synthetic nvidia query failure')
                return gpu_rows
            with mock.patch.object(m.socket, 'gethostname', return_value='synthetic-host'), \
                 mock.patch.object(m, 'stat', return_value=dict(start_ticks=123)), \
                 mock.patch.object(m, 'subtree', return_value=({101, 102}, 8 * 2**30)), \
                 mock.patch.object(m.subprocess, 'check_output', side_effect=command), \
                 mock.patch.object(m.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2048 * 2**30)), \
                 mock.patch.object(m, 'directory_bytes', side_effect=size), \
                 mock.patch.object(m.time, 'sleep', side_effect=lambda _: state.update(done=True)):
                result = m.wait_owned(child, profile, run, ['GPU-0', 'GPU-1', 'GPU-2', 'GPU-3'])
            marker = (run / 'STOP_REQUESTED').read_text() if (run / 'STOP_REQUESTED').exists() else None
            rows = [json.loads(line) for line in (profile / 'RESOURCE_MONITOR.jsonl').read_text().splitlines()]
            self.assertFalse((smoke / 'STOP_REQUESTED').exists())
            return result, marker, rows, len(reads)

    def test_exact_aggregate_limit_and_six_native_are_allowed(self):
        result, marker, rows, reads = self.run_once(48*2**30, 48*2**30, (3, 3))
        self.assertIsNone(result['stop_request'])
        self.assertIsNone(marker)
        self.assertEqual(rows[0]['study_output_bytes'], 96*2**30)
        self.assertEqual(rows[0]['study_native_directory_count'], 6)
        self.assertEqual(reads, 2)

    def test_combined_output_over_limit_stops_owned_run(self):
        result, marker, _, _ = self.run_once(50*2**30, 47*2**30)
        self.assertIn('STUDY_OUTPUT_OVER_96_GIB', result['stop_request']['reasons'])
        self.assertIsNotNone(marker)

    def test_seven_native_directories_across_stages_stop(self):
        result, _, rows, _ = self.run_once(native_counts=(3, 4))
        self.assertIn('STUDY_NATIVE_DIRECTORIES_OVER_6', result['stop_request']['reasons'])
        self.assertEqual(rows[0]['study_native_directory_count'], 7)

    def test_selected_unmapped_gpu_pid_recorded_and_stops_only_owned_run(self):
        result, marker, rows, _ = self.run_once(gpu_rows='GPU-0, 999, 0')
        self.assertIn('SELECTED_GPU_PROCESS_NOT_MAPPED_TO_OWNED_TREE', result['stop_request']['reasons'])
        self.assertEqual(rows[0]['unmapped_selected_gpu_processes'][0]['pid'], 999)
        self.assertIsNotNone(marker)

    def test_other_gpu_does_not_trigger(self):
        result, marker, _, _ = self.run_once(gpu_rows='GPU-OTHER, 999, 80000')
        self.assertIsNone(result['stop_request'])
        self.assertIsNone(marker)

    def test_monitor_error_requests_safe_stop(self):
        result, marker, _, _ = self.run_once(monitoring_error=True)
        self.assertIn('RESOURCE_MONITOR_ERROR', result['stop_request']['reasons'])
        self.assertIsNotNone(marker)

    def test_operator_marker_not_overwritten(self):
        _, marker, _, _ = self.run_once(gpu_rows='GPU-0, 999, 0', old_stop='operator-owned-evidence')
        self.assertEqual(marker, 'operator-owned-evidence')

    def test_source_pins_and_profile_only(self):
        self.assertEqual(hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(), EXPECTED_LAUNCHER)
        self.assertEqual(hashlib.sha256(MONITOR.read_bytes()).hexdigest(), EXPECTED_MONITOR)
        tree = ast.parse(LAUNCHER.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument' and
                 n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--stage']
        self.assertEqual(len(calls), 1)
        choices = next(k.value for k in calls[0].keywords if k.arg == 'choices')
        self.assertEqual(ast.literal_eval(choices), ['PROFILE'])
        monitor_calls = [n.func.attr for n in ast.walk(ast.parse(MONITOR.read_text()))
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('kill', monitor_calls)
        self.assertNotIn('killpg', monitor_calls)
        self.assertNotIn('Popen', monitor_calls)

if __name__ == '__main__':
    print(json.dumps(dict(scope='synthetic only; no real subprocess/GPU/model/manifest reads',
        launcher_sha256=hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(),
        monitor_sha256=hashlib.sha256(MONITOR.read_bytes()).hexdigest())), flush=True)
    unittest.main(verbosity=2)
