#!/usr/bin/python3
"""Owned CPU-only pressure-lease validation. Never queries NVIDIA or real fillers.

Run /usr/bin/python3 -B tools/structured_world/validate_navtest_pressure_lease.py.
Every signalled process is created by this validator under its temporary trusted
fixture path. Inventories/compute apps are fake; no torch/GPU/SSH/install is used.
"""

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import navtest_pressure_lease as lease


FIXTURE = '''import os, signal, subprocess, sys, time
def option(name):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else None
if '--ignore-term' in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if '--spawn-child' in sys.argv:
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
ready = option('--ready-file')
if ready:
    with open(ready, 'w') as stream:
        stream.write(str(os.getpid()))
marker = option('--exec-marker')
while True:
    if marker and os.path.exists(marker):
        os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(120)'])
    time.sleep(.01)
'''


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('owned fixture did not reach expected state')


class FakeHost(lease.LocalHost):
    def __init__(self, script):
        super().__init__(script)
        self.cards = [{'index': 0, 'uuid': 'GPU-a'}, {'index': 1, 'uuid': 'GPU-b'},
                      {'index': 2, 'uuid': 'GPU-c'}]
        self.extra_apps = []
        self.children = []
        self.spawn_count = 0
        self.after_capture = None
        # Deliberately independent of nvidia-smi index order.
        self.cuda_uuid_order = ['GPU-c', 'GPU-a', 'GPU-b']
        self.suppress_pressure_apps = False

    def inventory(self):
        return [dict(row) for row in self.cards]

    def processes(self):
        # Limit the fake world to processes owned by this test plus their tree.
        all_processes = super().processes()
        roots = {p.pid for p in self.children}
        selected = self.guard._descendants(all_processes, roots)
        return {pid: row for pid, row in all_processes.items() if pid in selected}

    def apps(self):
        result = list(self.extra_apps)
        if self.suppress_pressure_apps:
            return result
        processes = self.processes()
        for pid, process in processes.items():
            if self.is_pressure(pid, process):
                app_pid = pid
                if '--spawn-child' in process['argv']:
                    descendants = self.guard._descendants(processes, {pid}) - {pid}
                    if not descendants:
                        continue
                    app_pid = min(descendants)
                # The fake driver is the ground truth, not binding() under test.
                env = lease.read_environment(pid)
                raw = env.get('CUDA_VISIBLE_DEVICES')
                visible = list(self.cuda_uuid_order) if raw is None else [
                    self.cuda_uuid_order[int(token)] if token.strip().isdecimal() else token.strip()
                    for token in raw.split(',')]
                indices = self.guard._pressure_root_gpu_indices(
                    dict(process, command=lease.shlex.join(process['argv'])), len(visible))
                result.extend({'pid': app_pid, 'gpu_uuid': visible[i]} for i in indices)
        return result

    def capture(self, process):
        captured = super().capture(process)
        if self.after_capture:
            self.after_capture()
            self.after_capture = None
        return captured

    def spawn(self, captured, log_fd):
        child = super().spawn(captured, log_fd)
        self.children.append(child)
        self.spawn_count += 1
        return child


class PressureLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='owned-navtest-pressure-')
        self.directory = Path(self.temp.name)
        self.script = self.directory / 'trusted_cpu_fixture.py'
        self.script.write_text(FIXTURE)
        self.script.chmod(0o600)
        self.host = FakeHost(self.script)
        self.helper = lease.PressureLease(self.directory / 'private-art', self.host, timeout=.4)
        self.known_owned = {}
        self.secret = 'FAKE_PRIVATE_SENTINEL_NOT_A_CREDENTIAL'

    def remember(self):
        for process in self.host.processes().values():
            self.known_owned[(process['pid'], process['start_ticks'])] = process

    def tearDown(self):
        self.remember()
        # Cleanup ONLY the saved PID/start identities of our owned CPU fixtures.
        for record in self.known_owned.values():
            fd = lease.open_pidfd(record)
            if fd is not None:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
        for child in self.host.children:
            child.wait(timeout=2)
        self.temp.cleanup()

    def launch(self, extra=(), visible='GPU-c,GPU-a', gpus='0', genuine=False, interpreter_flags=()):
        ready = self.directory / ('ready-' + str(len(self.host.children)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible,
                   OWNED_FIXTURE_PRIVATE_KEY=self.secret)
        if visible is None:
            env.pop('CUDA_VISIBLE_DEVICES')
        def limits():
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(128, soft), hard))
        if genuine:
            argv = ['/usr/bin/python3', *interpreter_flags, '-c', 'import time; time.sleep(120)', str(self.script)]
        else:
            argv = ['/usr/bin/python3', *interpreter_flags, str(self.script), '--gpus', gpus,
                    '--memory-gb', '7.25', '--ready-file', str(ready), *extra]
        child = subprocess.Popen(argv, env=env, cwd=self.directory,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True, preexec_fn=limits)
        self.host.children.append(child)
        if not genuine:
            wait_for(lambda: ready.exists())
        wait_for(lambda: lease.proc_identity(child.pid))
        self.remember()
        return child

    def journal(self, name='job'):
        return json.loads((self.helper.root / f'lease-{name}.json').read_text())

    def acquire(self, selected=('GPU-c',), name='job'):
        return self.helper.acquire(name, selected)

    def test_capture_release_restore_limits_descendants_and_redaction(self):
        child = self.launch(extra=('--spawn-child',))
        wait_for(lambda: len(self.host.snapshot()['roots'][child.pid]['tree']) == 2)
        self.remember()
        before = self.host.snapshot()
        self.assertTrue(all(a['pid'] != child.pid for a in before['apps']))
        root = self.host.capture(before['processes'][child.pid])
        tree = before['roots'][child.pid]['tree']
        acquired = self.acquire()
        self.assertTrue(acquired['changed'])
        self.assertFalse(lease.same_process(root))
        for pid in tree:
            current = lease.proc_identity(pid)
            self.assertTrue(current is None or current['state'] == 'Z')
        journal = self.journal()
        private = self.helper.private(journal)
        captured = private['roots'][str(child.pid)]
        self.assertEqual(captured['root']['argv'], root['argv'])
        self.assertEqual(captured['root']['executable'], root['executable'])
        self.assertEqual(captured['root']['cwd'], str(self.directory))
        self.assertEqual(captured['root']['environment']['OWNED_FIXTURE_PRIVATE_KEY'], self.secret)
        self.assertEqual(captured['root']['rlimits']['RLIMIT_NOFILE'][0], 128)
        self.assertEqual(len(captured['tree']), 2)
        self.assertEqual(self.helper.root.stat().st_mode & 0o777, 0o700)
        for path in self.helper.root.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        public_text = json.dumps(acquired) + (self.helper.root / 'lease-job.json').read_text()
        self.assertNotIn(self.secret, public_text)
        self.assertNotIn('OWNED_FIXTURE_PRIVATE_KEY', public_text)
        self.assertNotIn('argv', public_text)
        restored = self.helper.restore('job', no_workers_started=True)
        self.assertFalse(restored.get('deferred', False))
        self.assertEqual(self.host.spawn_count, 1)
        new_pid = restored['journal']['roots'][0]['replacement']['pid']
        actual = self.host.capture(self.host.snapshot()['processes'][new_pid])
        self.assertEqual(actual['rlimits'], root['rlimits'])
        self.assertEqual(actual['argv'], root['argv'])
        self.assertEqual(actual['environment']['OWNED_FIXTURE_PRIVATE_KEY'], self.secret)
        self.remember()
        self.assertEqual(self.helper.restore('job')['journal']['state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 1)

    def test_no_original_never_creates_a_filler(self):
        result = self.acquire()
        self.assertEqual(result['journal']['state'], 'ACQUIRED_NO_PRESSURE')
        self.assertFalse(self.acquire()['changed'])
        self.assertFalse(self.helper.restore('job')['changed'])
        self.assertEqual(self.host.spawn_count, 0)

    def test_exact_entrypoint_and_logical_CVM_binding(self):
        pressure = self.launch()
        genuine = self.launch(genuine=True)
        self.host.extra_apps = [{'pid': genuine.pid, 'gpu_uuid': 'GPU-a'}]
        inspected = self.helper.inspect(['GPU-c', 'GPU-a'])
        self.assertEqual([r['pid'] for r in inspected['pressure_roots']], [pressure.pid])
        self.assertEqual(inspected['pressure_roots'][0]['gpu_uuids'], ['GPU-c'])
        self.assertEqual(inspected['genuine_or_unreadable_apps'][0]['pid'], genuine.pid)
        self.acquire()
        self.assertIsNone(genuine.poll())

    def test_different_script_same_basename_is_not_pressure(self):
        other = self.directory / 'other' / self.script.name
        other.parent.mkdir()
        other.write_text(FIXTURE)
        child = subprocess.Popen(['/usr/bin/python3', str(other)], cwd=self.directory,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.host.children.append(child)
        wait_for(lambda: lease.proc_identity(child.pid))
        self.assertEqual(self.helper.inspect(['GPU-c'])['pressure_roots'], [])
        self.acquire()
        self.assertIsNone(child.poll())

    def test_complete_multi_GPU_parent_required_and_restored_once(self):
        child = self.launch(gpus='all')
        with self.assertRaisesRegex(lease.LeaseError, 'complete_pressure_parent'):
            self.acquire()
        self.assertIsNone(child.poll())
        result = self.acquire(['GPU-c', 'GPU-a'])
        self.assertEqual(result['journal']['roots'][0]['gpu_uuids'], ['GPU-a', 'GPU-c'])
        self.assertFalse(self.acquire(['GPU-a', 'GPU-c'])['changed'])
        self.assertFalse(self.helper.restore('job', no_workers_started=True).get('deferred', False))
        self.assertEqual(self.host.spawn_count, 1)

    def test_lease_selection_and_overlapping_lease_are_rejected(self):
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseError, 'selection_cannot_change'):
            self.acquire(['GPU-a'])
        with self.assertRaisesRegex(lease.LeaseError, 'outstanding_local_lease'):
            self.acquire(name='another')

    def test_worker_identity_proof_and_CPU_preloading_worker_defer(self):
        self.launch()
        self.acquire()
        self.assertEqual(self.helper.restore('job')['reason'], 'WORKER_EXIT_IDENTITIES_REQUIRED')
        worker = self.launch(genuine=True)
        identity = lease.proc_identity(worker.pid)
        raw = f'{worker.pid}:{identity["start_ticks"]}'
        result = self.helper.restore('job', workers=[raw])
        self.assertEqual(result['reason'], 'EVALUATION_WORKERS_STILL_RUNNING')
        self.assertIsNone(worker.poll())
        self.assertEqual(self.host.spawn_count, 0)
        fd = lease.open_pidfd(identity)
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
        finally:
            os.close(fd)
        worker.wait(timeout=2)
        self.assertFalse(self.helper.restore('job', workers=[raw]).get('deferred', False))

    def test_new_genuine_work_defers_without_signalling_it(self):
        self.launch()
        self.acquire()
        genuine = self.launch(genuine=True)
        self.host.extra_apps = [{'pid': genuine.pid, 'gpu_uuid': 'GPU-c'}]
        result = self.helper.restore('job', no_workers_started=True)
        self.assertEqual(result['reason'], 'GPU_ACQUIRED_BY_ANOTHER_JOB')
        self.assertIsNone(genuine.poll())
        self.assertEqual(self.host.spawn_count, 0)
        self.host.extra_apps = []
        self.assertFalse(self.helper.restore('job', no_workers_started=True).get('deferred', False))

    def test_numeric_CVM_index_mapping_change_defers(self):
        self.launch(visible='0,1')
        self.acquire()
        self.host.cards[0]['uuid'], self.host.cards[2]['uuid'] = 'GPU-c', 'GPU-a'
        result = self.helper.restore('job', no_workers_started=True)
        self.assertEqual(result['reason'], 'GPU_INDEX_UUID_MAPPING_CHANGED')
        self.assertEqual(self.host.spawn_count, 0)

    def test_integer_mask_uses_actual_apps_not_NVML_index(self):
        child = self.launch(visible='0,1')
        inspected = self.helper.inspect(['GPU-c'])
        self.assertEqual(inspected['pressure_roots'][0]['gpu_uuids'], ['GPU-c'])
        self.assertEqual(inspected['pressure_roots'][0]['binding_proof'], 'actual_compute_apps')
        # NVML index zero is GPU-a, but CUDA ordinal zero is GPU-c here.
        self.assertEqual(self.acquire(['GPU-a'])['journal']['state'], 'ACQUIRED_NO_PRESSURE')
        self.assertIsNone(child.poll())
        self.helper.restore('job')
        self.acquire(['GPU-c'], name='correct')
        self.assertFalse(self.helper.restore('correct', no_workers_started=True).get('deferred', False))

    def test_partial_default_order_uses_actual_apps_and_replays_original(self):
        self.launch(visible=None)
        inspected = self.helper.inspect(['GPU-c'])
        self.assertEqual(inspected['pressure_roots'][0]['binding_proof'], 'actual_compute_apps')
        self.acquire()
        self.assertFalse(self.helper.restore('job', no_workers_started=True).get('deferred', False))
        self.assertEqual(self.host.spawn_count, 1)

    def test_numeric_or_partial_default_order_without_apps_is_ambiguous(self):
        for visible in ('0,1', None):
            with self.subTest(visible=visible):
                child = self.launch(visible=visible)
                self.host.suppress_pressure_apps = True
                inspected = self.helper.inspect(['GPU-c'])
                self.assertTrue(inspected['unresolved_pressure_roots'])
                self.assertEqual(inspected['pressure_roots'], [])
                with self.assertRaisesRegex(lease.LeaseError, 'ambiguous_pressure_binding'):
                    self.acquire()
                self.assertIsNone(child.poll())

    def test_unmasked_all_is_safe_without_resolving_ordinal_order(self):
        self.launch(visible=None, gpus='all')
        self.host.suppress_pressure_apps = True
        inspected = self.helper.inspect(['GPU-a', 'GPU-b', 'GPU-c'])
        self.assertEqual(inspected['pressure_roots'][0]['binding_proof'], 'all_inventory')
        with self.assertRaisesRegex(lease.LeaseError, 'complete_pressure_parent'):
            self.acquire()
        self.acquire(['GPU-a', 'GPU-b', 'GPU-c'])
        self.host.suppress_pressure_apps = False
        self.assertFalse(self.helper.restore('job', no_workers_started=True).get('deferred', False))

    def test_changed_actual_CUDA_order_stops_only_our_invalid_replacement(self):
        self.launch(visible='0,1')
        self.acquire()
        self.host.cuda_uuid_order = ['GPU-a', 'GPU-c', 'GPU-b']
        result = self.helper.restore('job', no_workers_started=True)
        self.assertEqual(result['reason'], 'RESTORED_GPU_BINDING_CHANGED_REPLACEMENT_STOPPED')
        replacement = result['journal']['roots'][0]['replacement']
        self.assertFalse(lease.same_process(replacement))
        self.assertEqual(self.host.spawn_count, 1)

    def test_missing_GPU_UUID_defers_and_optional_selection_is_checked(self):
        self.launch()
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseError, 'does_not_match_journal'):
            self.helper.restore('job', no_workers_started=True, gpu_uuids=['GPU-a'])
        self.host.cards = self.host.cards[:2]
        result = self.helper.restore('job', no_workers_started=True)
        self.assertEqual(result['reason'], 'SELECTED_GPU_UUID_UNAVAILABLE')
        self.assertEqual(self.host.spawn_count, 0)

    def test_common_interpreter_flags_retain_exact_entrypoint_semantics(self):
        for flags in (('-B',), ('-W', 'ignore'), ('-X', 'dev')):
            with self.subTest(flags=flags):
                child = self.launch(interpreter_flags=flags)
                process = self.host.processes()[child.pid]
                self.assertTrue(self.host.is_pressure(child.pid, process))
                genuine = self.launch(genuine=True)
                self.assertFalse(self.host.is_pressure(genuine.pid, self.host.processes()[genuine.pid]))

    def test_failed_restore_launch_keeps_obligation_and_can_retry(self):
        self.launch()
        self.acquire()
        with patch.object(self.host, 'spawn', side_effect=OSError('private text must not be published')):
            result = self.helper.restore('job', no_workers_started=True)
        self.assertEqual(result['reason'], 'ORIGINAL_LIMITS_OR_LAUNCH_FAILED')
        self.assertNotIn('private text', json.dumps(result))
        self.assertEqual(self.host.spawn_count, 0)
        self.assertFalse(self.helper.restore('job', no_workers_started=True).get('deferred', False))
        self.assertEqual(self.host.spawn_count, 1)

    def test_pressure_path_as_interpreter_option_operand_is_genuine(self):
        for option in ('-W', '-X'):
            with self.subTest(option=option):
                genuine = self.launch(genuine=True, interpreter_flags=(option, str(self.script)))
                process = self.host.processes()[genuine.pid]
                self.assertFalse(self.host.is_pressure(genuine.pid, process))
                self.host.extra_apps = [{'pid': genuine.pid, 'gpu_uuid': 'GPU-c'}]
                result = self.helper.inspect(['GPU-c'])
                self.assertEqual(result['pressure_roots'], [])
                self.assertEqual(result['genuine_or_unreadable_apps'][0]['pid'], genuine.pid)
                self.assertIsNone(genuine.poll())

    def test_replacement_that_exited_is_not_respawned(self):
        self.launch()
        self.acquire()
        result = self.helper.restore('job', no_workers_started=True)
        replacement = result['journal']['roots'][0]['replacement']
        fd = lease.open_pidfd(replacement)
        try:
            signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
        finally:
            os.close(fd)
        wait_for(lambda: not lease.same_process(replacement))
        self.assertFalse(self.helper.restore('job')['changed'])
        self.assertFalse(self.helper.status('job')['journal']['roots'][0]['replacement_alive'])
        self.assertEqual(self.host.spawn_count, 1)

    def test_PID_reuse_before_and_during_pidfd_open_is_never_signalled(self):
        worker = self.launch(genuine=True)
        identity = lease.proc_identity(worker.pid)
        reused = dict(identity, start_ticks=identity['start_ticks'] + 1)
        with patch.object(lease, 'proc_identity', return_value=reused), patch.object(lease.os, 'pidfd_open') as opened:
            self.assertIsNone(lease.open_pidfd(identity))
            opened.assert_not_called()
        with patch.object(lease, 'proc_identity', side_effect=[identity, reused]), patch.object(lease.os, 'pidfd_open', return_value=987), patch.object(lease.os, 'close') as closed:
            self.assertIsNone(lease.open_pidfd(identity))
            closed.assert_called_once_with(987)
        self.assertIsNone(worker.poll())

    def test_same_PID_exec_to_genuine_command_before_stop_is_rejected(self):
        marker = self.directory / 'exec-marker'
        child = self.launch(extra=('--exec-marker', str(marker)))
        original = lease.proc_identity(child.pid)
        def exec_now():
            marker.touch()
            wait_for(lambda: '-c' in self.host.processes()[child.pid]['argv'])
        self.host.after_capture = exec_now
        with self.assertRaisesRegex(lease.LeaseError, 'execution_changed_before_stop'):
            self.acquire()
        self.assertIsNone(child.poll())
        self.assertEqual(lease.proc_identity(child.pid)['start_ticks'], original['start_ticks'])
        self.assertEqual(self.helper.restore('job')['journal']['state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 0)

    def test_signal_intent_and_spawn_intent_are_not_guessed(self):
        self.launch()
        self.acquire()
        journal = self.journal()
        journal['roots'][0]['state'] = 'SIGNAL_INTENT'
        self.helper.write_lease(journal)
        self.assertEqual(self.helper.restore('job', no_workers_started=True)['reason'], 'AMBIGUOUS_SIGNAL_INTENT')
        journal['roots'][0]['state'] = 'SPAWN_INTENT'
        self.helper.write_lease(journal)
        self.assertEqual(self.helper.restore('job', no_workers_started=True)['reason'], 'AMBIGUOUS_SPAWN_INTENT')
        self.assertEqual(self.host.spawn_count, 0)

    def test_unlogged_spawn_is_adopted_without_a_second_filler(self):
        self.launch()
        self.acquire()
        self.helper.restore('job', no_workers_started=True)
        journal = self.journal()
        journal['state'] = 'DEFERRED'
        journal['roots'][0]['state'] = 'SPAWN_INTENT'
        journal['roots'][0]['replacement'] = None
        self.helper.write_lease(journal)
        result = self.helper.restore('job', no_workers_started=True)
        self.assertFalse(result.get('deferred', False))
        self.assertEqual(self.host.spawn_count, 1)

    def test_uncooperative_owned_fixture_times_out_without_KILL_escalation(self):
        child = self.launch(extra=('--ignore-term',))
        result = self.acquire()
        self.assertEqual(result['reason'], 'ORIGINAL_PROCESSES_STILL_RUNNING')
        self.assertIsNone(child.poll())
        self.assertEqual(self.helper.restore('job', no_workers_started=True)['reason'], 'ORIGINAL_PROCESSES_STILL_RUNNING')
        self.assertEqual(self.host.spawn_count, 0)

    def test_private_capture_tamper_permissions_and_host_context_fail_closed(self):
        self.launch()
        self.acquire()
        journal = self.journal()
        private = self.helper.root / journal['private_capture']
        private.chmod(0o644)
        with self.assertRaisesRegex(lease.LeaseError, '0600'):
            self.helper.restore('job', no_workers_started=True)
        private.chmod(0o600)
        private.write_text(private.read_text() + ' ')
        with self.assertRaisesRegex(lease.LeaseError, 'hash_mismatch'):
            self.helper.restore('job', no_workers_started=True)
        journal['context']['boot_id'] = 'not-this-boot'
        self.helper.write_lease(journal)
        with self.assertRaisesRegex(lease.LeaseError, 'namespace_mismatch'):
            self.helper.status('job')
        self.assertEqual(self.host.spawn_count, 0)

    def test_state_root_rejects_git_and_world_readable_directory(self):
        self.helper.root.mkdir(mode=0o755)
        with self.assertRaisesRegex(lease.LeaseError, '0700'):
            self.acquire()
        self.helper.root.chmod(0o700)
        (self.directory / '.git').mkdir()
        with self.assertRaisesRegex(lease.LeaseError, 'outside_git'):
            self.acquire()

    def test_cli_help_and_safe_error_output(self):
        result = subprocess.run(['/usr/bin/python3', '-B', str(Path(lease.__file__)), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--gpu-uuids', result.stdout)
        self.assertIn('--worker-pid', result.stdout)
        result = subprocess.run(['/usr/bin/python3', '-B', str(Path(lease.__file__)), 'acquire'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)['error'], 'gpu_uuids_required')
        self.assertNotIn(self.secret, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
