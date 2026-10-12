"""CPU fixtures only; no real processes, GPU queries, training or enqueue."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continue_migrated_navsim_group as op


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.output = self.root/'control'; self.output.mkdir(mode=0o700)
        self.identity = {'identity': 'fixture-id', 'training_source_sha': op.guard.SHA,
            'training_source_tree_sha256': op.guard.TREE, 'schedule_horizon': 100000,
            'effective_global_batch': 32, 'FM_repeat': 8, 'microbatch': 4, 'world_size': 8,
            'training_seed': 42, 'scenes_per_epoch': 101592}
        op.write(self.run/'identity.json', self.identity)
        self.argv = ['fixture-python', '-m', 'torch.distributed.run', '--',
            str(op.guard.ROOT/'tools/structured_world/train_vla.py'), '--output', str(self.run),
            '--stop-after', '25000', '--resume']
        self.reference = {'pid': 42, 'start_ticks': 123, 'argv': self.argv, 'parent_completed': 23269}
        self.manifest = {'run': str(self.run), 'identity': 'fixture-id', 'references': {},
            'current': self.reference, 'locks': [str(self.root/f'{i}.lock') for i in range(3)],
            'navtest_targets': [25000, 50000, 100000], 'cache': {}, 'uuids': [f'GPU-{i}' for i in range(8)]}
        self.save(25000)
        self.set_boundary(23269)
        self.manifest['initial_boundary_sha256'] = op.digest(self.run/'RESUME_BOUNDARY_VERIFIED.json')
        op.write(self.output/'ENV_PRIVATE.json', {'OMP_NUM_THREADS': '2'})
        self.manifest['environment_sha256'] = op.digest(self.output/'ENV_PRIVATE.json')
        op.write(self.output/'REGISTRATION.json', self.manifest)
        op.write(self.output/'STATE.json', {'status': 'PREPARED'})

    def save(self, completed, status=None):
        tag = f'milestone_{completed:07d}'
        folder = self.run/'checkpoints'/tag; folder.mkdir(parents=True, exist_ok=True)
        inventory = {f'payload_{i}': 'fixture-hash' for i in range(17)}
        for name in inventory: (folder/name).touch()
        op.write(folder/'COMPLETE.json', {'identity': 'fixture-id', 'completed': completed,
            'ranks': 8, 'file_sha256': inventory, 'status': status or ('COMPLETE' if completed == 100000 else 'PAUSED')})
        (self.run/'checkpoints/latest').write_text(tag)

    def set_boundary(self, completed):
        op.write(self.run/'RESUME_BOUNDARY_VERIFIED.json', {'passed': True, 'identity': 'fixture-id', 'completed': completed})

    def mock_execute(self, returncode=0, same=None, workers=None, enqueue=None, checks=None, sleep=None):
        commands = []; queued = []; fixture = self
        def launch(command, **kwargs):
            target = int(command[command.index('--stop-after')+1]) if '--stop-after' in command else 100000
            commands.append((target, command, kwargs))
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
            self.assertEqual(kwargs['env']['HOME'], os.environ['HOME'])
            self.assertEqual(kwargs['env']['SECRET_FIXTURE'], 'private-not-public')
            self.assertEqual(len(kwargs['pass_fds']), 3)
            for name in self.manifest['locks']:
                with Path(name).open('a') as f:
                    with self.assertRaises(BlockingIOError): fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            class Child:
                pid = 99
                def wait(self):
                    if returncode == 0:
                        previous = op.guard.native(fixture.run)['completed']
                        fixture.set_boundary(previous); fixture.save(target)
                        if target == 100000: op.write(fixture.run/'TRAINING_COMPLETE.json', {'identity': 'fixture-id', 'updates': 100000})
                    return returncode
            return Child()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'SECRET_FIXTURE': 'private-not-public'}))
            for name, value in [('checks', checks or (lambda m: None)), ('same', same or (lambda r: None)),
                    ('workers', workers or (lambda r: set())), ('gpu_ready', lambda m: True)]:
                stack.enter_context(patch.object(op, name, side_effect=value))
            stack.enter_context(patch.object(op.subprocess, 'Popen', side_effect=launch))
            stack.enter_context(patch.object(op.guard, 'process', return_value={'start_ticks': 999}))
            stack.enter_context(patch.object(op.guard, 'queue_navtest', side_effect=enqueue or (lambda p: queued.append(p['target']) or {'fixture': True})))
            stack.enter_context(patch.object(op.time, 'sleep', side_effect=sleep))
            stack.enter_context(patch.object(op.os, 'kill', side_effect=AssertionError('No signals')))
            stack.enter_context(patch.object(op.guard.signal, 'pidfd_send_signal', side_effect=AssertionError('No signals'), create=True))
            op.execute(self.output)
        return commands, queued

    def test_stages_queue_locks_and_private_environment(self):
        commands, queued = self.mock_execute()
        self.assertEqual([c[0] for c in commands], [50000, 75000, 100000])
        self.assertEqual(queued, [25000, 50000, 100000])
        for target, command, _ in commands: self.assertEqual(command, op.stage_command(self.argv, target))
        self.assertFalse(op.read(self.output/'STATE.json')['evaluation_complete'])
        public = ''.join(p.read_text() for p in self.output.glob('*.json*'))
        self.assertNotIn('SECRET_FIXTURE', public); self.assertNotIn('private-not-public', public)
        self.assertEqual((self.output/'ENV_PRIVATE.json').stat().st_mode & 0o777, 0o600)

    def test_waits_current_launcher_without_launching_first_stage(self):
        count = [0]
        def current(r):
            count[0] += 1
            return {**r, 'state': 'S', 'children': [43]} if count[0] == 1 else None
        commands, _ = self.mock_execute(same=current, workers=lambda r: {42, 43} if count[0] == 1 else set())
        self.assertGreaterEqual(count[0], 2); self.assertEqual(commands[0][0], 50000)

    def test_non_paused_checkpoint_fails_without_launch(self):
        self.save(25000, 'RUNNING')
        with self.assertRaisesRegex(RuntimeError, 'PAUSED'): self.mock_execute()
        self.assertEqual(op.read(self.output/'STATE.json')['status'], 'FAILED')

    def test_failed_process_retained_and_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'no retry'): self.mock_execute(returncode=1)
        journal = (self.output/'executed_commands.jsonl').read_text()
        self.assertIn('"returncode": 1', journal)
        with self.assertRaisesRegex(RuntimeError, 'Already attempted'): self.mock_execute()
        self.assertEqual((self.output/'executed_commands.jsonl').read_text(), journal)

    def test_duplicate_execute_does_not_overwrite_state(self):
        (self.output/'EXECUTION_STARTED.json').write_text('{}')
        with self.assertRaises(FileExistsError): self.mock_execute()
        self.assertEqual(op.read(self.output/'STATE.json')['status'], 'PREPARED')

    def test_queue_lock_conflict_does_not_launch(self):
        with Path(self.manifest['locks'][0]).open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.mock_execute()
        self.assertFalse((self.output/'executed_commands.jsonl').exists())

    def test_waits_inherited_lineage_lock(self):
        with Path(self.manifest['locks'][2]).open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            waited = []
            def release(seconds):
                waited.append(seconds)
                self.assertFalse((self.output/'train_to_50000.log').exists())
                fcntl.flock(f, fcntl.LOCK_UN)
            self.mock_execute(sleep=release)
            self.assertEqual(waited, [5])

    def test_other_same_run_worker_blocks(self):
        with self.assertRaisesRegex(RuntimeError, 'same-run'): self.mock_execute(workers=lambda r: {666})

    def test_missing_native_payload_and_wrong_identity(self):
        (self.run/'checkpoints/milestone_0025000/payload_0').unlink()
        with self.assertRaisesRegex(RuntimeError, 'payload missing'): op.bank(self.manifest, 25000)
        self.save(25000); self.manifest['identity'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'identity'): op.bank(self.manifest, 25000)

    def test_pid_reuse_rejected(self):
        with patch.object(op.guard, 'process', return_value={'start_ticks': 456}):
            with self.assertRaisesRegex(RuntimeError, 'PID reused'): op.same(self.reference)

    def test_memory_floor_and_genuine_exclusion(self):
        self.manifest['cache'] = {'80': {'pid': 80, 'start_ticks': 4, 'argv': ['iqe']}}
        def query(free, owner='80'):
            return patch.object(op.subprocess, 'check_output', side_effect=[f'{owner}, GPU-0\n',
                ''.join(f'GPU-{i}, {free}\n' for i in range(8))])
        with patch.object(op, 'same', return_value={'state': 'S'}):
            with query(51200): self.assertTrue(op.gpu_ready(self.manifest))
            with query(51199): self.assertFalse(op.gpu_ready(self.manifest))
            with query(70000, owner='666'):
                with self.assertRaisesRegex(RuntimeError, 'Unregistered'): op.gpu_ready(self.manifest)
        with patch.object(op, 'same', side_effect=RuntimeError('PID reused')), query(70000):
            with self.assertRaisesRegex(RuntimeError, 'PID reused'): op.gpu_ready(self.manifest)

    def test_boundary_change_and_private_mode_rejected(self):
        self.set_boundary(23268)
        with self.assertRaisesRegex(RuntimeError, 'Initial resume report'): self.mock_execute()
        (self.output/'ENV_PRIVATE.json').chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'unsafe'): self.mock_execute()

    def test_saved_credentials_key_is_rejected(self):
        op.write(self.output/'ENV_PRIVATE.json', {'HF_TOKEN': 'fixture-secret'})
        self.manifest['environment_sha256'] = op.digest(self.output/'ENV_PRIVATE.json')
        op.write(self.output/'REGISTRATION.json', self.manifest)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected saved environment key'): self.mock_execute()

    def test_prepare_captures_allowlist_only_no_tokens_home_or_gpu(self):
        migration = self.root/'migration'; migration.mkdir()
        proc = self.root/'proc'/'42'; proc.mkdir(parents=True)
        (proc/'environ').write_bytes(b'PATH=/fixture/bin\0OMP_NUM_THREADS=2\0HF_TOKEN=fixture-secret\0HOME=/source/home\0')
        launch = {**self.reference, 'host': op.HOST, 'cwd': str(op.guard.ROOT),
            'identity': 'fixture-id', 'parent_tag': 'latest_0023269', 'parent_receipt_sha256': 'fixture-sha',
            'cache_workers_preserved': list(range(80, 88)), 'lineage_lock': str(self.run/'HOST_MIGRATION_RUNTIME.lock')}
        queue = op.guard.ART/'physical_gpu_locks'/('queue_'+op.hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()+'.lock')
        original = self.root/'original.json'
        op.write(original, {'expected_native_identity': 'fixture-id', 'groups': [op.GROUP],
            'queue_lock': str(queue), 'preserved_files_sha256': {}, 'training_plan': [
                {'target': target, 'command': self.argv[:-1] if target == 25000 else op.stage_command(self.argv, target)}
                for target in op.guard.TARGETS]})
        records = {'TARGET_LAUNCH.json': launch,
            'SOURCE_STOP_REQUEST.json': {'source_argv': self.argv[:-1], 'source_cwd': str(op.guard.ROOT)},
            'SOURCE_EXIT_CONFIRMED.json': {'status': 'SOURCE_EXIT_CONFIRMED', 'latest_tag': 'latest_0023269'},
            'PARENT_CHECKPOINT_VERIFIED.json': {'status': 'PASS', 'tag': 'latest_0023269', 'receipt_sha256': 'fixture-sha'},
            'SHARED_RESOURCE_AUTHORIZATION.json': {'host': op.HOST, 'shared_GPU_deployment_authorized': True},
            'SHARED_GPU_ADMISSION.json': {'gpus': [{'uuid': uuid} for uuid in self.manifest['uuids']]}}
        for name, value in records.items(): op.write(migration/name, value)
        cadence = self.root/'cadence.json'
        op.write(cadence, {'observation_updates': [25000, 50000, 60000, 70000, 80000, 90000, 100000],
            'runs': [{'run': str(self.run), 'training_identity': 'fixture-id'}]})
        real_path = Path
        def path(*parts):
            if parts and parts[0] == '/proc': return real_path(self.root/'proc', *parts[1:])
            return real_path(*parts)
        prepared = self.root/'prepared'
        with patch.object(op, 'Path', side_effect=path), patch.object(op, 'CADENCE', cadence), \
             patch.object(op, 'same', return_value={**launch, 'state': 'S', 'children': list(range(43, 51))}), \
             patch.object(op, 'workers', return_value={42, *range(43, 51)}), patch.object(op, 'checks'), \
             patch.object(op.guard, 'process', side_effect=lambda pid: {'pid': pid, 'argv': ['iqe.vision_cache.cli', 'cache-build'], 'start_ticks': 4}), \
             patch.object(op.subprocess, 'Popen', side_effect=AssertionError('No process launch')), \
             patch.object(op.subprocess, 'check_output', side_effect=AssertionError('No GPU queries')):
            op.prepare(prepared, migration, original)
        self.assertEqual(op.read(prepared/'ENV_PRIVATE.json'), {'PATH': '/fixture/bin', 'OMP_NUM_THREADS': '2'})
        for p in prepared.glob('*.json'): self.assertNotIn('fixture-secret', p.read_text())
        self.assertEqual(op.read(prepared/'STATE.json')['GPU_jobs_started'], 0)

    def test_stop_marker_rejected_by_runtime_checks(self):
        (self.run/'STOP_AFTER_CHECKPOINT').touch()
        with patch.object(op.socket, 'gethostname', return_value='training-vlawm-zt4-worker-0'), \
             patch.object(op.guard, 'authorize'), patch.object(op.guard, 'clean_source'), patch.object(op.guard, 'evaluation_policy'):
            with self.assertRaisesRegex(RuntimeError, 'stop marker'): op.checks(self.manifest)


if __name__ == '__main__': unittest.main()
