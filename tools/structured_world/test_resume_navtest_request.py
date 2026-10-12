"""CPU-only lifecycle controls; no dispatcher jobs, signals or subprocesses."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('resume_wrapper', Path(__file__).with_name('resume_navtest_request.py'))
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = wrapper.parse_args(['--dispatcher', str(root / 'run_navsim_navtest.py'),
            '--request-key', 'G3_EVENT_LOCAL_0080000', '--output', str(root / 'queue'),
            '--inputs', str(root / 'inputs'), '--cadence-policy', str(root / 'policy.json')])
        self.req = dict(key=self.args.request_key, group='G3_EVENT_LOCAL', run=str(root / 'origin'),
            tag='latest_0080000', updates=80000, training_identity='train', population='navtest',
            scenes=12146, logs=136, seed=42, precision='FP32 / TF32 disabled',
            variants=['q0', 'q_final'], checkpoint_files_sha256={'model.pt': 'sha'}, priority=100)
        self.job = self.args.output / 'jobs' / self.req['key']
        self.policy = dict(dispatch_output=str(self.args.output), base_policy_path=str(root / 'base.json'),
            runs=[dict(group=self.req['group'], training_identity='train')])
        save(self.args.output / 'requests' / (self.req['key'] + '.json'), self.req)
        save(self.args.cadence_policy, self.policy); save(root / 'base.json', {})
        save(self.args.inputs / 'identity.json', dict(identity='inputs'))
        save(self.job / 'REGISTRATION.json', dict(request=self.req, frozen_evaluation_source=wrapper.MODEL_SOURCE,
                                                 inputs=str(self.args.inputs)))
        save(self.job / 'identity.json', dict(training_identity='train', checkpoint=self.req['tag'],
            updates=80000, checkpoint_files_sha256=self.req['checkpoint_files_sha256'],
            evaluation_source_sha=wrapper.MODEL_SOURCE, input_identity='inputs'))
        self.failure = self.job / 'recoveries' / '20261012_recovered' / 'INFERENCE_FAILED.json'
        save(self.failure, dict(classification='infrastructure'))
        self.recovery = self.failure.with_name('RECOVERY.json')
        digest = wrapper.sha(self.failure)
        save(self.recovery, dict(evidence=dict(classification='infrastructure', reason='verified SSH255',
            request_key=self.req['key'], failure_sha256=digest), original_failure_sha256=digest,
            reconciled=dict(verified_prior_scenes=8000), worker_exit_proofs=[dict(host='remote-fixture',
                pid=31415, expected_start_ticks='123', observed=dict(exists=False))]))
        # Deliberately nonexistent fixture PIDs; existing owned process checks use /proc only.
        save(self.args.output / 'CONTROLLER.json', dict(pid=2147483640))
        save(self.job / 'SCORER_LAUNCH.json', dict(pid=2147483641))
        save(self.job / 'attempts/old/launches/shard_000.json', dict(pid=2147483642, host='remote-fixture'))
        save(self.job / 'attempts/old/shards/shard_000.json',
             dict(host='remote-fixture', pid=31415, proc_start_ticks='123'))
        self.events = []
        self.execution_error = None
        self.observer_error = None

        def execute(args, req):
            self.assert_lock_held()
            self.events.append('execute')
            if self.execution_error:
                raise self.execution_error
            save(args.output / 'completed' / (req['key'] + '.json'), dict(valid=True, request=req))

        def observer(*unused):
            self.assert_lock_held()
            self.events.append('observe')
            return SimpleNamespace(error=self.observer_error, close=lambda: self.events.append('close'))

        self.dispatcher = SimpleNamespace(authorized=lambda: None, now=lambda: '2026-10-12T04:10:00+00:00',
            atomic=save, execute=execute, request=object(), valid_score_completion=object())
        self.cadence = SimpleNamespace(load_policy=lambda path: wrapper.read(path),
            eligible=lambda policy, req: req['updates'] == 80000, Observer=observer)

    def assert_lock_held(self):
        with (self.args.output / 'controller.lock').open('a') as stream:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def run_resume(self):
        wrapper.resume(self.args, self.dispatcher, self.cadence)

    def test_lock_held_refuses_before_validation(self):
        with (self.args.output / 'controller.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            with patch.object(self.dispatcher, 'authorized') as authorized:
                with self.assertRaises(BlockingIOError): self.run_resume()
                authorized.assert_not_called()

    def test_unresolved_failure_refuses_without_execute_or_clearing(self):
        p = self.job / 'INFERENCE_FAILED.json'; save(p, dict(classification='infrastructure'))
        with patch.object(wrapper.os, 'execv') as execv:
            with self.assertRaisesRegex(ValueError, 'Unresolved failure'): self.run_resume()
            execv.assert_not_called()
        self.assertTrue(p.exists()); self.assertNotIn('execute', self.events)

    def test_missing_recovery_refuses(self):
        self.recovery.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing original'): self.run_resume()

    def test_scientific_invalid_refuses(self):
        save(self.job / 'scores/INVALID.json', dict(reason='nonfinite'))
        with self.assertRaisesRegex(ValueError, 'Scientific score failure'): self.run_resume()

    def test_wrong_request_contract_refuses(self):
        changed = dict(self.req, seed=43)
        save(self.args.output / 'requests' / (self.req['key'] + '.json'), changed)
        with self.assertRaisesRegex(ValueError, 'scientific contract'): self.run_resume()

    def test_archive_hash_change_refuses(self):
        save(self.failure, dict(classification='scientific'))
        with self.assertRaisesRegex(ValueError, 'original failure'): self.run_resume()

    def test_remote_worker_still_alive_refuses(self):
        d = wrapper.read(self.recovery)
        d['worker_exit_proofs'] = [dict(expected_start_ticks='123', observed=dict(exists=True, start_ticks='123'))]
        save(self.recovery, d)
        with self.assertRaisesRegex(ValueError, 'remote worker'): self.run_resume()

    def test_live_local_launcher_refuses(self):
        save(self.job / 'attempts/old/launches/shard_000.json', dict(pid=os.getppid(), host='remote-fixture'))
        with self.assertRaisesRegex(ValueError, 'still alive'): self.run_resume()

    def test_stale_recovery_does_not_cover_new_worker(self):
        save(self.job / 'attempts/new/shards/shard_000.json',
             dict(host='remote-fixture', pid=31416, proc_start_ticks='456'))
        with self.assertRaisesRegex(ValueError, 'not covered'): self.run_resume()

    def test_execute_failure_preserved_no_serve_and_lock_released(self):
        self.execution_error = RuntimeError('fixture execute failure')
        with patch.object(wrapper.os, 'execv') as execv:
            with self.assertRaisesRegex(RuntimeError, 'fixture execute failure'): self.run_resume()
            execv.assert_not_called()
        self.assertEqual(self.events, ['observe', 'execute', 'close'])
        self.assertEqual(len(list((self.args.output / 'resume_requests').glob('*/*/FAILED.json'))), 1)
        with (self.args.output / 'controller.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_observer_error_refuses_no_execute(self):
        self.observer_error = RuntimeError('fixture observer failure')
        with patch.object(wrapper.os, 'execv') as execv:
            with self.assertRaisesRegex(ValueError, 'observer failed'): self.run_resume()
            execv.assert_not_called()
        self.assertEqual(self.events, ['observe', 'close'])

    def test_success_execs_serve_only_after_observer_closed_and_lock_released(self):
        def execv(executable, argv):
            self.assertEqual(executable, wrapper.sys.executable)
            self.assertEqual(argv, wrapper.serve_argv(self.args))
            self.assertEqual(self.events[:2], ['observe', 'execute'])
            self.assertIn('close', self.events)
            with (self.args.output / 'controller.lock').open('a') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.events.append('exec')
        with patch.object(wrapper.os, 'execv', side_effect=execv): self.run_resume()
        self.assertEqual(self.events[-1], 'exec')
        self.assertTrue(self.failure.exists())
        self.assertEqual(len(list((self.args.output / 'resume_requests').glob('*/*/COMPLETE.json'))), 1)

    def test_actual_frozen_import_and_namespace_defaults_cpu_only(self):
        # The wrapper pins its exact companion dispatcher hash, including after
        # an explicitly versioned infrastructure-only recovery patch.
        path = Path(wrapper.__file__).with_name('run_navsim_navtest.py')
        dispatcher, cadence = wrapper.load_modules(path)
        self.assertEqual(dispatcher.SHA, wrapper.MODEL_SOURCE)
        self.assertEqual(cadence.SOURCE, wrapper.MODEL_SOURCE)
        for name in ['exit_when_idle', 'cohost_duty_cycle', 'fp32_export', 'attempt', 'canonical_python', 'routes']:
            self.assertTrue(hasattr(self.args, name))
        self.assertFalse(self.args.exit_when_idle)
        with patch.object(wrapper, 'sha', return_value='tampered'):
            with self.assertRaisesRegex(ValueError, 'source hash differs'): wrapper.load_modules(path)


if __name__ == '__main__':
    unittest.main()
