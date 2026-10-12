"""CPU checks for bounded resource permission and owned-command timeout."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dispatch', Path(__file__).with_name('run_navsim_navtest.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class TemporaryHostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'authorization.json'
        self.grant = dict(schema='structured_world_temporary_host_authorization_v1',
            task='structured_world_fgtr_round1', lease_id='test', hosts=['temporary'],
            starts_unix=100, expires_unix=1000)
        self.policy = dict(temporary_task_authorizations={})
        self.bind()

    def bind(self):
        self.path.write_text(json.dumps(self.grant))
        self.policy['temporary_task_authorizations']['test'] = dict(
            **{k: self.grant[k] for k in ['task', 'hosts', 'starts_unix', 'expires_unix']},
            authorization_path=str(self.path), authorization_sha256=d.digest(self.path))

    def test_exact_active_host_only(self):
        self.assertIsNotNone(d.temporary_authorization('temporary', self.policy, 200))
        self.assertIsNone(d.temporary_authorization('another', self.policy, 200))

    def test_expired_or_not_started_is_rejected(self):
        for instant in [99, 940, 1000, 1001]:
            with self.assertRaisesRegex(ValueError, 'not active'):
                d.temporary_authorization('temporary', self.policy, instant)

    def test_other_task_does_not_authorize_this_task(self):
        self.policy['temporary_task_authorizations']['test']['task'] = 'iqe_v1'
        self.assertIsNone(d.temporary_authorization('temporary', self.policy, 200))

    def test_hash_and_registry_bindings(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            d.temporary_authorization('temporary', self.policy, 200)
        self.bind()
        self.policy['temporary_task_authorizations']['test']['expires_unix'] += 1
        with self.assertRaisesRegex(ValueError, 'identity/window'):
            d.temporary_authorization('temporary', self.policy, 200)

    def test_window_cannot_be_extended_over_nine_hours(self):
        self.grant['expires_unix'] = self.grant['starts_unix'] + 9 * 3600 + 1
        self.bind()
        with self.assertRaisesRegex(ValueError, 'identity/window'):
            d.temporary_authorization('temporary', self.policy, 200)

    def test_dynamic_hosts_deduplicated_and_expiry_fail_closed(self):
        with patch.object(d.time, 'time', return_value=200):
            self.assertEqual(d.evaluation_hosts(self.policy, ['base']), ['base', 'temporary'])
            self.assertEqual(d.evaluation_hosts(self.policy, ['temporary']), ['temporary'])
        with patch.object(d.time, 'time', return_value=1000):
            self.assertEqual(d.evaluation_hosts(self.policy, ['base']), ['base'])
            # An old baseline can list it, but authorization must still reject it.
            with self.assertRaises(ValueError):
                d.temporary_authorization('temporary', self.policy)

    def test_plain_command_unchanged_and_grace_reserved(self):
        command = ['python', 'worker.py']
        self.assertIs(d.deadline_command(command, None), command)
        argv = d.deadline_command(command, self.grant, 200)
        self.assertEqual(argv[-2:], command)
        self.assertEqual(argv[3], '1000')
        self.assertIn("'--kill-after=10s'", argv[2])
        with self.assertRaisesRegex(ValueError, 'deadline reached'):
            d.deadline_command(command, self.grant, 940)

    def test_real_timeout_leaves_unrelated_process_alive(self):
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            command = [sys.executable, '-c', 'import time; time.sleep(30)']
            argv = d.deadline_command(command, dict(expires_unix=time.time() + 60.3))
            result = subprocess.run(argv, timeout=3, start_new_session=True)
            self.assertEqual(result.returncode, 124)
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=3)

    def test_delayed_remote_start_does_not_extend_deadline(self):
        marker = Path(self.tmp.name) / 'should_not_run'
        command = [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']
        argv = d.deadline_command(command, dict(expires_unix=time.time() + 60.1))
        time.sleep(.2)
        result = subprocess.run(argv, timeout=3, start_new_session=True)
        self.assertEqual(result.returncode, 124)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
