"""CPU-only sidecar integration: real leases, fake GPU apps, owned fixtures.

Never invokes sidecar.main(), SSH, NVIDIA, Torch, or either production pressure
script. All signalled processes belong to validate_navtest_pressure_lease's
temporary CPU fixture and are cleaned up with saved PID/start-time identities.
"""
import os
import signal
import socket
import subprocess
import unittest
from unittest.mock import patch

import navtest_cadence as cadence
import navtest_idle_pressure as idle
import navtest_pressure_lease as lease
import validate_navtest_pressure_lease as fixtures


class IdlePressureTests(unittest.TestCase):
    # Reuse the mature fixture lifecycle without rediscovering its full test suite.
    launch = fixtures.PressureLeaseTests.launch
    remember = fixtures.PressureLeaseTests.remember
    tearDown = fixtures.PressureLeaseTests.tearDown

    def setUp(self):
        fixtures.PressureLeaseTests.setUp(self)
        self.output = self.directory / 'dispatch'
        self.elapsed = 0.0
        self.activity = {'workers': [], 'unaccounted_busy_gpu_uuids': [],
                         'GPUs': [], 'cpu_load_1min': 0}
        no_real_query = patch.object(self.host, 'query',
                                    side_effect=AssertionError('Real GPU query forbidden in CPU fixtures'))
        no_real_query.start()
        self.addCleanup(no_real_query.stop)
        self.base = self.directory / 'base-policy.json'
        cadence.atomic(self.base, {'original_training_recipe_unchanged': True})
        self.cadence = {
            'schema': cadence.SCHEMA, 'population': 'navtest',
            'expected_scenes': 12146, 'sampling_seed': 42,
            'training_recipe_change': False,
            'observation_updates': [25000, 50000, 60000, 70000, 80000, 90000, 100000],
            'base_policy_path': str(self.base),
            'base_policy_sha256': cadence.sha(self.base),
            'runs': [{'group': f'G{n}', 'training_identity': f'train{n}'} for n in range(4)],
        }
        policy_path = self.directory / 'cadence.json'
        cadence.atomic(policy_path, self.cadence)
        self.worker_script = self.directory / 'owned_navtest_cpu_worker.py'
        self.worker_script.write_text(fixtures.FIXTURE)
        self.policy = {
            'identity': 'owned-cpu-idle-policy-v1', 'host': socket.gethostname(),
            'dispatch_output': str(self.output), 'cadence_policy': str(policy_path),
            'initial_lease_id': 'owned-initial',
            'gpu_uuids': ['GPU-a', 'GPU-c'],
            'worker_scripts': [str(self.worker_script)],
            'idle_grace_seconds': 600, 'poll_seconds': 5,
            'status_output': str(self.directory / 'public' / 'STATUS.json'),
        }

    def stop_owned(self, child):
        self.remember()
        record = lease.proc_identity(child.pid)
        if record and record['state'] not in ('Z', 'X'):
            fd = lease.open_pidfd(record)
            if fd is not None:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
                finally:
                    os.close(fd)
        child.wait(timeout=2)

    def start(self, *, pressure=True, extra=(), prior_worker=True):
        original = self.launch(gpus='all', extra=extra) if pressure else None
        workers = []
        if prior_worker:
            old = self.launch(genuine=True)
            record = lease.proc_identity(old.pid)
            workers.append(f'{old.pid}:{record["start_ticks"]}')
            self.stop_owned(old)
        self.helper.acquire('owned-initial', self.policy['gpu_uuids'], workers)
        controller = idle.IdlePressure(self.policy, self.helper, lambda: self.elapsed,
                                       activity=lambda *_: dict(self.activity))
        return controller, original

    def request(self, update=70000, identity='train3'):
        value = {
            'key': f'G3_{update:07d}', 'group': 'G3', 'updates': update,
            'tag': f'latest_{update:07d}', 'training_identity': identity,
            'run': str(self.directory / 'pinned-native'),
            'population': 'navtest', 'scenes': 12146, 'seed': 42,
            'variants': ['q0', 'q_final'], 'priority': 100,
            'checkpoint_files_sha256': {'synthetic_only.pt': 'fixture'},
        }
        cadence.atomic(self.output / 'requests' / (value['key'] + '.json'), value)
        return value

    def complete(self, request):
        cadence.atomic(self.output / 'completed' / (request['key'] + '.json'),
                       {'valid': True, 'request': request})

    def saved_worker(self, key, *, stop=True, qualification=False, host=None):
        child = self.launch(genuine=True)
        record = lease.proc_identity(child.pid)
        row = {'host': host or self.policy['host'], 'pid': child.pid,
               'proc_start_ticks': record['start_ticks'], 'status': 'COMPLETE'}
        if qualification:
            root = self.output / 'qualification' / key / self.policy['host'].removesuffix('-worker-0') / 'shards'
        else:
            root = self.output / 'jobs' / key / 'attempts' / 'owned-attempt' / 'shards'
        cadence.atomic(root / 'shard_000.json', row)
        if stop:
            self.stop_owned(child)
        return child, f'{child.pid}:{record["start_ticks"]}'

    def preload_worker(self, key):
        ready = self.directory / ('preload-' + str(len(self.host.children)))
        child = subprocess.Popen(
            ['/usr/bin/python3', '-B', str(self.worker_script), '--worker',
             '--output', str(self.output / 'jobs' / key), '--ready-file', str(ready)],
            cwd=self.directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        self.host.children.append(child)
        fixtures.wait_for(ready.exists)
        self.remember()
        return child

    def restore_after_grace(self, controller):
        first = controller.step()
        self.assertEqual(first['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 0)
        self.elapsed += 599
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 0)
        self.elapsed += 1
        return controller.step()

    def test_real_restore_then_pending_reacquire_and_restore_new_cycle(self):
        controller, original = self.start()
        self.assertIsNotNone(original.poll())
        first = self.restore_after_grace(controller)
        self.assertEqual(first['lease_state'], 'RESTORED')
        self.assertTrue(first['pressure_alive'])
        self.assertEqual(self.host.spawn_count, 1)
        request = self.request()
        released = controller.step()
        self.assertEqual(released['status'], 'AVAILABLE_FOR_NAVTEST')
        self.assertNotEqual(released['lease_id'], 'owned-initial')
        self.assertEqual(released['lease_state'], 'ACQUIRED')
        self.assertFalse(released['pressure_alive'])
        self.assertEqual(controller.step()['lease_id'], released['lease_id'])
        self.assertEqual(self.host.spawn_count, 1)
        self.saved_worker(request['key'])
        self.complete(request)
        controller.step()
        self.elapsed += 599
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 1)
        self.elapsed += 1
        second = controller.step()
        self.assertEqual(second['lease_state'], 'RESTORED')
        self.assertTrue(second['pressure_alive'])
        self.assertEqual(self.host.spawn_count, 2)
        restored = self.helper.read_lease(second['lease_id'])
        capture = self.helper.private(restored)
        old_capture = self.helper.private(self.helper.read_lease('owned-initial'))
        self.assertEqual(capture['script_sha256'], old_capture['script_sha256'])
        old = next(iter(old_capture['roots'].values()))['root']
        new = next(iter(capture['roots'].values()))['root']
        for key in ('argv', 'environment', 'rlimits', 'affinity', 'uid', 'gid', 'cgroup'):
            # A failed equality check must not dump inherited credentials.
            self.assertTrue(new[key] == old[key], key + ' equality')

    def test_pending_never_restores_pressure(self):
        controller, _ = self.start()
        request = self.request()
        for elapsed in (0, 599, 600, 1200):
            self.elapsed = elapsed
            value = controller.step()
            self.assertEqual(value['pending_requests'], [request['key']])
            self.assertEqual(value['status'], 'AVAILABLE_FOR_NAVTEST')
            self.assertFalse(value['pressure_alive'])
        self.assertEqual(self.host.spawn_count, 0)

    def test_genuine_fake_gpu_app_survives_pressure_release(self):
        controller, _ = self.start()
        self.restore_after_grace(controller)
        genuine = self.launch(genuine=True)
        self.host.extra_apps = [{'pid': genuine.pid, 'gpu_uuid': 'GPU-c'}]
        value = controller.step()
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['workers_alive'], 1)
        self.assertFalse(value['pressure_alive'])
        self.assertIsNone(genuine.poll())
        self.elapsed += 1200
        controller.step()
        self.assertEqual(self.host.spawn_count, 1)
        self.assertIsNone(genuine.poll())

    def test_unreadable_fake_app_blocks_idle_restore(self):
        controller, _ = self.start()
        self.host.extra_apps = [{'pid': 99999999, 'gpu_uuid': 'GPU-c'}]
        for elapsed in (0, 600, 1200):
            self.elapsed = elapsed
            self.assertEqual(controller.step()['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(self.host.spawn_count, 0)

    def test_loading_worker_without_cuda_app_is_protected(self):
        controller, _ = self.start()
        worker = self.preload_worker('G3_0070000')
        value = controller.step()
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['workers_alive'], 1)
        self.assertIsNone(worker.poll())
        self.elapsed = 1200
        controller.step()
        self.assertEqual(self.host.spawn_count, 0)
        self.assertIsNone(worker.poll())

    def test_completed_receipt_does_not_override_live_saved_worker(self):
        controller, _ = self.start()
        request = self.request()
        controller.step()
        worker, _ = self.saved_worker(request['key'], stop=False, qualification=True)
        self.complete(request)
        self.assertEqual(controller.step()['status'], 'YIELDING_TO_REAL_WORK')
        self.elapsed += 600
        value = controller.step()
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['idle_seconds'], 0)
        self.assertEqual(self.host.spawn_count, 0)
        self.assertIsNone(worker.poll())
        self.stop_owned(worker)
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.elapsed += 599
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 0)
        self.elapsed += 1
        self.assertEqual(controller.step()['lease_state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 1)

    def test_request_arriving_at_idle_confirmation_prevents_restore(self):
        controller, _ = self.start()
        controller.step()
        self.elapsed += 600
        original = idle.queue_demand
        reads = []

        def enqueue_on_confirmation(output, policy):
            reads.append(None)
            if len(reads) == 2:
                self.request()
            return original(output, policy)

        with patch.object(idle, 'queue_demand', enqueue_on_confirmation):
            value = controller.step()
        self.assertEqual(len(reads), 2)
        self.assertEqual(value['status'], 'AVAILABLE_FOR_NAVTEST')
        self.assertEqual(self.host.spawn_count, 0)

    def test_report_from_another_host_is_not_local_exit_evidence(self):
        request = self.request()
        self.saved_worker(request['key'], host='another-worker-0')
        self.assertEqual(idle.saved_workers(self.output, [request['key']],
                                           self.policy['host']), [])

    def test_registered_75k_filtered_and_foreign_75k_kept(self):
        request = self.request(75000)
        self.assertEqual(idle.queue_demand(self.output, self.cadence), [])
        # Off-cadence history is retained rather than revalidated/rewritten here.
        cadence.atomic(self.output / 'completed' / (request['key'] + '.json'), {'historical': True})
        self.assertEqual(idle.queue_demand(self.output, self.cadence), [])
        self.assertTrue((self.output / 'completed' / (request['key'] + '.json')).exists())
        (self.output / 'completed' / (request['key'] + '.json')).unlink()
        foreign = self.request(75000, identity='other-formal-identity')
        self.assertEqual(idle.queue_demand(self.output, self.cadence), [foreign['key']])

    def test_invalid_or_foreign_completion_receipt_stops_before_restore(self):
        controller, _ = self.start()
        request = self.request()
        for marker in ({'valid': False, 'request': request},
                       {'valid': True, 'request': request | {'training_identity': 'foreign'}}):
            with self.subTest(marker=marker):
                cadence.atomic(self.output / 'completed' / (request['key'] + '.json'), marker)
                with self.assertRaisesRegex(ValueError, 'Completion receipt'):
                    controller.step()
        self.assertEqual(self.host.spawn_count, 0)

    def test_request_filename_mismatch_rejected(self):
        request = self.request()
        path = self.output / 'requests' / (request['key'] + '.json')
        path.rename(path.with_name('foreign.json'))
        with self.assertRaisesRegex(ValueError, 'key/path'):
            idle.queue_demand(self.output, self.cadence)

    def test_changed_base_policy_rejected_without_spawning(self):
        controller, _ = self.start()
        cadence.atomic(self.base, {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Base evaluation policy changed'):
            controller.step()
        self.assertEqual(self.host.spawn_count, 0)

    def test_private_policy_identity_mismatch_rejected(self):
        self.start()
        with self.assertRaisesRegex(ValueError, 'another policy'):
            idle.IdlePressure(self.policy | {'identity': 'different'}, self.helper)

    def test_host_boot_namespace_mismatch_rejected(self):
        controller, _ = self.start()
        context = self.host.context() | {'boot_id': 'changed-boot'}
        with patch.object(self.host, 'context', return_value=context):
            with self.assertRaisesRegex(lease.LeaseError, 'host_boot_or_PID_namespace'):
                controller.step()
        self.assertEqual(self.host.spawn_count, 0)

    def test_changed_capture_script_defers_original_restore(self):
        controller, _ = self.start()
        self.script.write_text(fixtures.FIXTURE + '\n# owned fixture change\n')
        value = self.restore_after_grace(controller)
        self.assertEqual(value['status'], 'RESTORE_DEFERRED')
        self.assertEqual(value['deferred_reason'], 'TRUSTED_SCRIPT_CHANGED')
        self.assertEqual(self.host.spawn_count, 0)

    def test_restart_reuses_acquired_cycle_without_duplicate_restore(self):
        controller, _ = self.start()
        self.restore_after_grace(controller)
        self.request()
        value = controller.step()
        restarted = idle.IdlePressure(self.policy, self.helper, lambda: self.elapsed,
                                     activity=lambda *_: dict(self.activity))
        self.assertEqual(restarted.step()['lease_id'], value['lease_id'])
        self.assertEqual(self.host.spawn_count, 1)

    def test_no_original_never_creates_a_new_filler(self):
        controller, _ = self.start(pressure=False)
        value = self.restore_after_grace(controller)
        self.assertEqual(value['lease_state'], 'RESTORED')
        self.assertFalse(value['pressure_alive'])
        self.assertEqual(self.host.spawn_count, 0)

    def test_resource_activity_resets_full_600_second_interval(self):
        controller, _ = self.start()
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.elapsed = 499
        self.activity['unaccounted_busy_gpu_uuids'] = ['GPU-c']
        value = controller.step()
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['idle_seconds'], 0)
        self.elapsed = 500
        self.activity['unaccounted_busy_gpu_uuids'] = []
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.elapsed = 1099
        value = controller.step()
        self.assertEqual(value['idle_seconds'], 599)
        self.assertEqual(value['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 0)
        self.elapsed = 1100
        value = controller.step()
        self.assertEqual(value['idle_required_seconds'], 600)
        self.assertEqual(value['lease_state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 1)

    def test_pending_activity_also_resets_entire_interval(self):
        controller, _ = self.start()
        controller.step()
        self.elapsed = 599
        request = self.request()
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.complete(request)
        self.elapsed = 600
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.elapsed = 1199
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 0)
        self.elapsed = 1200
        self.assertEqual(controller.step()['lease_state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 1)

    def test_new_resource_activity_at_restore_boundary_prevents_spawn(self):
        controller, _ = self.start()
        controller.step()
        self.elapsed = 600
        fake_idle = dict(self.activity)
        fake_busy = fake_idle | {'unaccounted_busy_gpu_uuids': ['GPU-c']}
        with patch.object(controller, 'activity', side_effect=[fake_idle, fake_busy]) as scan:
            value = controller.step()
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['idle_seconds'], 0)
        self.assertEqual(self.host.spawn_count, 0)

    def test_resource_worker_releases_pressure_without_signalling_real_work(self):
        controller, _ = self.start()
        self.restore_after_grace(controller)
        real_work = self.launch(genuine=True)
        record = lease.proc_identity(real_work.pid)
        self.activity['workers'] = [f'{real_work.pid}:{record["start_ticks"]}']
        value = controller.step()
        self.assertEqual(value['status'], 'YIELDING_TO_REAL_WORK')
        self.assertEqual(value['workers_alive'], 1)
        self.assertFalse(value['pressure_alive'])
        self.assertIsNone(real_work.poll())
        self.stop_owned(real_work)
        self.activity['workers'] = []
        controller.step()
        self.elapsed += 599
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 1)
        self.elapsed += 1
        self.assertEqual(controller.step()['lease_state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 2)

    def test_real_resource_scanner_flags_memory_and_util_without_gpu_apps(self):
        controller, _ = self.start()
        inspection = self.helper.inspect(self.policy['gpu_uuids'])
        rows = [['GPU-a', '65', '0'], ['GPU-c', '0', '1']]
        with patch.object(self.host, 'query', return_value=rows):
            value = idle.resource_activity(self.helper, inspection, self.policy['gpu_uuids'])
        self.assertEqual(set(value['unaccounted_busy_gpu_uuids']), {'GPU-a', 'GPU-c'})
        self.assertEqual(self.host.spawn_count, 0)
        with patch.object(self.host, 'query', return_value=[['GPU-a', '64', '0'], ['GPU-c', '0', '0']]):
            value = idle.resource_activity(self.helper, inspection, self.policy['gpu_uuids'])
        self.assertEqual(value['unaccounted_busy_gpu_uuids'], [])

    def test_real_resource_scanner_protects_pre_cuda_script(self):
        controller, _ = self.start()
        real_work = self.preload_worker('pre-cuda-training-stand-in')
        inspection = self.helper.inspect(self.policy['gpu_uuids'])
        rows = [['GPU-a', '0', '0'], ['GPU-c', '0', '0']]
        with patch.object(self.host, 'query', return_value=rows):
            value = idle.resource_activity(self.helper, inspection, self.policy['gpu_uuids'])
        self.assertTrue(any(row.startswith(f'{real_work.pid}:') for row in value['workers']))
        self.assertIsNone(real_work.poll())
        self.assertEqual(self.host.spawn_count, 0)

    def test_registered_coordinator_does_not_mask_workers_or_foreign_scripts(self):
        self.start()
        inspection = self.helper.inspect(self.policy['gpu_uuids'])
        script = str(self.directory / 'run_navsim_navtest.py')
        def process(argv):
            return {'argv': argv, 'cwd': str(self.directory), 'start_ticks': 17}
        processes = {
            910001: process(['python3', '-B', '-u', script, '--serve']),
            910002: process(['python3', script, '--worker']),
            910003: process(['python3', script, '--serve', '--worker']),
            910004: process(['python3', '/unregistered/run_navsim_navtest.py', '--serve']),
            910005: process(['python3', '/owned/canonical_stream.py']),
        }
        with patch.object(self.host, 'processes', return_value=processes), \
                patch.object(self.host, 'query', return_value=[['GPU-a', '0', '0'], ['GPU-c', '0', '0']]):
            value = idle.resource_activity(self.helper, inspection, self.policy['gpu_uuids'], [script])
        self.assertEqual(value['workers'], [f'{pid}:17' for pid in range(910002, 910006)])

    def test_project_pressure_requires_bound_authorization_and_host(self):
        authorization = {
            'schema': 'navtest_captured_pressure_authorization_v1',
            'hosts': [socket.gethostname().removesuffix('-worker-0')],
            'pressure_restore_after_use': True,
            'pressure_script': str(lease.TRUSTED_SCRIPT),
            'pressure_script_sha256': 'owned-fixture-no-production-file-read',
            'private_state_root': str(self.directory / 'private'),
            'idle_grace_seconds': 600,
        }
        path = self.directory / 'authorization.json'
        cadence.atomic(path, authorization)
        policy = self.policy | {'pressure_kind': 'captured_project_pressure',
            'authorization_path': str(path), 'authorization_sha256': cadence.sha(path),
            'private_state_root': authorization['private_state_root']}
        self.assertEqual(idle.pressure_settings(policy),
                         (lease.TRUSTED_SCRIPT, authorization['pressure_script_sha256']))
        with self.assertRaisesRegex(ValueError, 'actual local host'):
            idle.pressure_settings(policy | {'host': 'not-authorized-worker-0'})
        authorization['hosts'] = []
        cadence.atomic(path, authorization)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            idle.pressure_settings(policy)
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            idle.pressure_settings(policy | {'authorization_sha256': cadence.sha(path)})

    def test_legacy_pressure_host_binding_is_preserved(self):
        with patch.object(idle.socket, 'gethostname', return_value='training-vlawm-zt4-worker-0'):
            policy = self.policy | {'host': 'training-vlawm-zt4-worker-0',
                'private_state_root': '/var/tmp/structured-world-fgtr-execution-private-20261011'}
            self.assertEqual(idle.pressure_settings(policy), (idle.pressure.SCRIPT, idle.pressure.SHA256))
        with self.assertRaises(ValueError):
            idle.pressure_settings(self.policy | {'private_state_root': '/var/tmp/unrelated'})

    def test_real_resource_scanner_excludes_only_identified_pressure(self):
        controller, _ = self.start()
        self.restore_after_grace(controller)
        inspection = self.helper.inspect(self.policy['gpu_uuids'])
        rows = [['GPU-a', '4096', '100'], ['GPU-c', '4096', '100']]
        with patch.object(self.host, 'query', return_value=rows):
            value = idle.resource_activity(self.helper, inspection, self.policy['gpu_uuids'])
        self.assertEqual(value['workers'], [])
        self.assertEqual(value['unaccounted_busy_gpu_uuids'], [])

    def test_no_local_workers_for_new_cycle_defers_instead_of_inventing_exit_proof(self):
        controller, _ = self.start()
        self.restore_after_grace(controller)
        request = self.request()
        controller.step()
        self.complete(request)
        controller.step()
        self.elapsed += 600
        value = controller.step()
        self.assertEqual(value['status'], 'DEFERRED_WORKER_EXIT_EVIDENCE_REQUIRED')
        self.assertEqual(self.host.spawn_count, 1)

    def test_pending_during_verifying_stops_only_owned_replacement_then_waits_600(self):
        controller, _ = self.start()
        self.host.suppress_pressure_apps = True
        value = self.restore_after_grace(controller)
        self.assertEqual(value['status'], 'RESTORE_DEFERRED')
        self.assertEqual(value['deferred_reason'], 'RESTORED_GPU_BINDING_NOT_YET_VERIFIED')
        journal = self.helper.read_lease('owned-initial')
        self.assertEqual(journal['roots'][0]['state'], 'VERIFYING')
        replacement = journal['roots'][0]['replacement']
        capture_sha = journal['private_capture_sha256']
        request = self.request()
        real_work = self.preload_worker(request['key'])
        value = controller.step()
        self.assertEqual(value['status'], 'AVAILABLE_FOR_NAVTEST')
        self.assertEqual(value['lease_state'], 'ACQUIRED')
        self.assertEqual(value['lease_id'], 'owned-initial')
        self.assertFalse(value['pressure_alive'])
        self.assertIsNone(real_work.poll())
        self.assertFalse(lease.same_process(replacement))
        self.assertEqual(self.host.spawn_count, 1)
        journal = self.helper.read_lease('owned-initial')
        self.assertEqual(journal['private_capture_sha256'], capture_sha)
        root = journal['roots'][0]
        self.assertEqual(root['state'], 'RELEASED')
        self.assertIsNone(root['replacement'])
        self.assertTrue(root['displaced'])
        cancellation = root['interrupted_restorations'][0]
        self.assertEqual(cancellation['replacement'], replacement)
        self.assertTrue(cancellation['all_captured_processes_exited'])
        evidence = self.helper.root / cancellation['tree_capture']
        self.assertEqual(lease.digest(evidence), cancellation['tree_capture_sha256'])
        self.assertEqual(evidence.stat().st_mode & 0o777, 0o600)
        self.complete(request)
        self.assertEqual(controller.step()['status'], 'YIELDING_TO_REAL_WORK')
        self.stop_owned(real_work)
        self.host.suppress_pressure_apps = False
        self.assertEqual(controller.step()['idle_seconds'], 0)
        self.elapsed += 599
        self.assertEqual(controller.step()['status'], 'WAITING_FOR_CONFIRMED_IDLE')
        self.assertEqual(self.host.spawn_count, 1)
        self.elapsed += 1
        self.assertEqual(controller.step()['lease_state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 2)

    def test_uncontrolled_replacement_exit_keeps_obligation_and_never_respawns(self):
        controller, _ = self.start()
        self.host.suppress_pressure_apps = True
        self.assertEqual(self.restore_after_grace(controller)['status'], 'RESTORE_DEFERRED')
        replacement = self.helper.read_lease('owned-initial')['roots'][0]['replacement']
        child = next(child for child in self.host.children if child.pid == replacement['pid'])
        self.stop_owned(child)  # External fixture exit, without helper's controlled-stop capture.
        self.request()
        for elapsed in (600, 1200):
            self.elapsed = elapsed
            value = controller.step()
            self.assertEqual(value['status'], 'PRESSURE_RELEASE_DEFERRED')
            self.assertEqual(value['deferred_reason'], 'REPLACEMENT_EXIT_WITHOUT_CONTROLLED_STOP_EVIDENCE')
            self.assertEqual(self.host.spawn_count, 1)
            self.assertEqual(self.helper.read_lease('owned-initial')['roots'][0]['replacement'], replacement)

    def test_replacement_execution_identity_change_is_not_signalled(self):
        marker = self.directory / 'owned-exec-marker'
        controller, _ = self.start(extra=('--exec-marker', str(marker)))
        self.host.suppress_pressure_apps = True
        self.assertEqual(self.restore_after_grace(controller)['status'], 'RESTORE_DEFERRED')
        replacement = self.helper.read_lease('owned-initial')['roots'][0]['replacement']
        marker.write_text('owned fixture exec now')
        fixtures.wait_for(lambda: replacement['pid'] not in self.host.snapshot()['roots'])
        self.request()
        value = controller.step()
        self.assertEqual(value['status'], 'PRESSURE_RELEASE_DEFERRED')
        self.assertEqual(value['deferred_reason'], 'OWNED_REPLACEMENT_IDENTITY_NOT_VERIFIED')
        self.assertTrue(lease.same_process(replacement))
        self.assertEqual(self.host.spawn_count, 1)

    def test_reverify_existing_replacement_then_fresh_acquire_without_respawn(self):
        controller, _ = self.start()
        self.host.suppress_pressure_apps = True
        self.assertEqual(self.restore_after_grace(controller)['status'], 'RESTORE_DEFERRED')
        self.request()
        self.host.suppress_pressure_apps = False
        # No actual worker/genuine task exists: this call verifies the already
        # recorded replacement, then ordinary pending handling recaptures it.
        result = self.helper.restore('owned-initial')
        self.assertEqual(result['journal']['state'], 'RESTORED')
        self.assertEqual(self.host.spawn_count, 1)
        value = controller.step()
        self.assertEqual(value['status'], 'AVAILABLE_FOR_NAVTEST')
        self.assertEqual(value['lease_state'], 'ACQUIRED')
        self.assertNotEqual(value['lease_id'], 'owned-initial')
        self.assertFalse(value['pressure_alive'])
        self.assertEqual(self.host.spawn_count, 1)

    def test_deferred_acquire_must_not_claim_gpu_available(self):
        controller, original = self.start(extra=('--ignore-term',))
        self.request()
        value = controller.step()
        self.assertEqual(value['lease_state'], 'DEFERRED')
        self.assertIsNone(original.poll())
        self.assertNotEqual(value['status'], 'AVAILABLE_FOR_NAVTEST')
        self.assertTrue(value['pressure_alive'])


if __name__ == '__main__':
    unittest.main()
