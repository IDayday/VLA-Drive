"""Filesystem-only tests for rolling checkpoint pinning and cadence admission."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import navtest_cadence as cadence


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='navtest-cadence-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'dispatch'
        self.policy = {'schema': cadence.SCHEMA, 'population': 'navtest',
            'expected_scenes': 12146, 'sampling_seed': 42, 'training_recipe_change': False,
            'observation_updates': [25000, 50000, 60000, 70000, 80000, 90000, 100000],
            'pins_root': str(self.root / 'pins'), 'runs': []}
        base = self.root / 'base.json'
        cadence.atomic(base, {'historical_policy': 'unchanged'})
        self.policy.update(base_policy_path=str(base), base_policy_sha256=cadence.sha(base))
        for n in range(4):
            run = self.root / f'G{n}'
            identity = {'identity': f'train{n}', 'training_source_sha': cadence.SOURCE,
                'scope': 'formal', 'dataset': 'navsim', 'config': {'structured_world': {'group': f'G{n}'}}}
            cadence.atomic(run / 'identity.json', identity)
            self.policy['runs'].append({'group': f'G{n}', 'training_identity': f'train{n}',
                'run': str(run), 'identity_sha256': cadence.sha(run / 'identity.json')})
        self.registration = self.policy['runs'][3]
        self.run = Path(self.registration['run'])
        self.folder = self.bank(70000)
        self.bank(75000, 'milestone')
        (self.run / 'checkpoints/latest').write_text('milestone_0075000\n')
        self.calls = []

    def bank(self, update, prefix='latest'):
        tag = f'{prefix}_{update:07d}'
        folder = self.run / 'checkpoints' / tag
        folder.mkdir(parents=True)
        hashes = {}
        for n in range(17):
            path = folder / f'payload{n}.pt'
            path.write_bytes(f'synthetic native {update} {n}'.encode())
            hashes[path.name] = cadence.sha(path)
        cadence.atomic(folder / 'COMPLETE.json', {'completed': update, 'tag': tag, 'ranks': 8,
            'identity': 'train3', 'file_sha256': hashes})
        return folder

    def enqueue(self, args):
        self.calls.append(args)
        identity = cadence.read(args.run / 'identity.json')
        complete = cadence.read(args.run / 'checkpoints' / args.tag / 'COMPLETE.json')
        key = 'G3_' + str(complete['completed']).zfill(7)
        cadence.atomic(args.output / 'requests' / (key + '.json'), {'key': key,
            'run': str(args.run), 'tag': args.tag, 'training_identity': identity['identity'],
            'updates': complete['completed'], 'checkpoint_files_sha256': complete['file_sha256']})

    def poll(self, valid=True):
        cadence.poll(self.policy, self.output, self.enqueue, lambda *_: valid)

    def exit_proof(self, key, request):
        cadence.atomic(self.output / 'jobs' / key / 'OWNED_WORKERS_EXITED.json', {
            'key': key, 'training_identity': request['training_identity'],
            'checkpoint_files_sha256': request['checkpoint_files_sha256'],
            'all_gpu_launchers_exited': True, 'scorer_exited': True})

    def test_only_extra_cadence_enqueued_once_and_history_preserved(self):
        self.poll()
        self.poll()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0].tag, 'latest_0070000')
        states = {x['key']: x['status'] for x in cadence.read(self.output / 'CADENCE_STATE.json')['observations']}
        self.assertEqual(states['G3_0060000'], 'PAST_CHECKPOINT_UNAVAILABLE')
        self.assertEqual(states['G3_0080000'], 'WAITING_FOR_NATIVE_CHECKPOINT')
        self.assertEqual(states['G3_0100000'], 'ORIGINAL_SUPERVISOR_OWNS_REQUEST')
        self.assertFalse(cadence.eligible(self.policy, {'training_identity': 'train3', 'updates': 75000}))
        self.assertTrue(cadence.eligible(self.policy, {'training_identity': 'foreign', 'updates': 75000}))

    def explicit_observation(self, **changes):
        return dict(group='G3', training_identity='train3', updates=45000,
                    user_instruction='Evaluate the latest saved checkpoint',
                    requested_utc='2026-10-13T02:00:00Z') | changes

    def test_explicit_latest_checkpoint_keeps_other_groups_and_cadence_unchanged(self):
        self.bank(45000)
        self.policy['additional_observations'] = [self.explicit_observation()]
        path = self.root / 'cadence.json'
        cadence.atomic(path, self.policy)
        self.policy = cadence.load_policy(path)
        self.poll()
        self.poll()
        self.assertEqual([a.tag for a in self.calls], ['latest_0045000', 'latest_0070000'])
        self.assertTrue(cadence.eligible(self.policy, {'training_identity': 'train3', 'updates': 45000}))
        self.assertFalse(cadence.eligible(self.policy, {'training_identity': 'train2', 'updates': 45000}))
        self.assertFalse(cadence.eligible(self.policy, {'training_identity': 'train3', 'updates': 75000}))
        self.assertEqual(cadence.observation_updates(self.policy, 'train2'), self.policy['observation_updates'])
        key = 'G3_0045000'
        request = cadence.read(self.output / 'requests' / (key + '.json'))
        cadence.atomic(self.output / 'completed' / (key + '.json'), {'valid': True})
        self.exit_proof(key, request)
        self.poll()
        self.assertTrue((self.root / 'pins' / key / 'RELEASED.json').exists())
        self.assertTrue((self.run / 'checkpoints/latest_0045000/payload0.pt').exists())

    def test_invalid_explicit_observations_rejected(self):
        bad = [dict(training_identity='foreign'), dict(group='G2'), dict(updates=True),
               dict(updates=0), dict(updates=100001), dict(updates=50000),
               dict(user_instruction=''), dict(requested_utc='')]
        path = self.root / 'cadence.json'
        for changes in bad:
            with self.subTest(changes=changes):
                self.policy['additional_observations'] = [self.explicit_observation(**changes)]
                cadence.atomic(path, self.policy)
                with self.assertRaises(ValueError):
                    cadence.load_policy(path)
        self.policy['additional_observations'] = [self.explicit_observation()] * 2
        cadence.atomic(path, self.policy)
        with self.assertRaisesRegex(ValueError, 'Duplicate explicit'):
            cadence.load_policy(path)

    def test_pin_survives_rolling_deletion_and_interrupted_enqueue(self):
        complete = cadence.read(self.folder / 'COMPLETE.json')
        pinned, tag = cadence.pin(self.policy, self.registration, 70000, self.folder, complete)
        before = (self.folder / 'payload0.pt').stat()
        after = (pinned / 'checkpoints' / tag / 'payload0.pt').stat()
        self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
        self.assertEqual((pinned / 'identity.json').read_bytes(), (self.run / 'identity.json').read_bytes())
        shutil.rmtree(self.folder)
        self.poll()
        self.assertEqual(len(self.calls), 1)
        for name, digest in complete['file_sha256'].items():
            self.assertEqual(cadence.sha(pinned / 'checkpoints' / tag / name), digest)

    def test_release_requires_valid_complete_and_preserves_native_source(self):
        self.poll()
        key = 'G3_0070000'
        request = cadence.read(self.output / 'requests' / (key + '.json'))
        cadence.atomic(self.output / 'completed' / (key + '.json'), {'valid': True})
        with self.assertRaisesRegex(ValueError, 'Invalid full score'):
            self.poll(valid=False)
        pinned = Path(request['run']) / 'checkpoints' / request['tag']
        self.assertTrue((pinned / 'payload0.pt').exists())
        self.poll()
        self.assertTrue((pinned / 'payload0.pt').exists())  # Completion alone is insufficient.
        self.exit_proof(key, request)
        export = self.output / 'FP32_exports' / key
        export.mkdir(parents=True)
        (export / 'model_fp32.pt').write_bytes(b'synthetic export')
        cadence.atomic(export / 'FP32_EXPORT_COMPLETE.json', request | {'checkpoint': request['tag']})
        self.poll()
        self.assertFalse((pinned / 'payload0.pt').exists())
        self.assertTrue((self.folder / 'payload0.pt').exists())
        self.assertFalse((export / 'model_fp32.pt').exists())
        self.assertTrue((export / 'FP32_EXPORT_COMPLETE.json').exists())
        self.assertTrue((pinned / 'COMPLETE.json').exists())
        self.poll()
        self.assertEqual(len(self.calls), 1)

    def test_changed_identity_and_foreign_export_rejected(self):
        self.poll()
        key = 'G3_0070000'
        request = cadence.read(self.output / 'requests' / (key + '.json'))
        cadence.atomic(self.output / 'completed' / (key + '.json'), {'valid': True})
        self.exit_proof(key, request)
        export = self.output / 'FP32_exports' / key
        export.mkdir(parents=True)
        (export / 'model_fp32.pt').write_bytes(b'foreign export')
        cadence.atomic(export / 'FP32_EXPORT_COMPLETE.json', request | {'checkpoint': request['tag'], 'training_identity': 'foreign'})
        with self.assertRaisesRegex(ValueError, 'foreign FP32'):
            self.poll()
        self.assertTrue((Path(request['run']) / 'checkpoints' / request['tag'] / 'payload0.pt').exists())
        (self.run / 'identity.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Formal run identity changed'):
            self.poll()

    def test_partial_banks_and_modified_base_policy_rejected(self):
        (self.folder / 'payload0.pt').unlink()
        self.poll()
        self.assertEqual(self.calls, [])
        path = self.root / 'cadence.json'
        cadence.atomic(path, self.policy)
        self.assertEqual(cadence.load_policy(path), self.policy)
        Path(self.policy['base_policy_path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Base evaluation policy changed'):
            cadence.load_policy(path)

    def test_source_pruned_immediately_after_last_link(self):
        real_link = cadence.os.link
        calls = []
        def pruning_link(source, destination):
            real_link(source, destination)
            calls.append(source)
            if len(calls) == 17:
                shutil.rmtree(self.folder)
        with patch.object(cadence.os, 'link', pruning_link):
            self.poll()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.folder.exists())
        bank = self.calls[0].run / 'checkpoints' / self.calls[0].tag
        self.assertEqual(len(list(bank.glob('*.pt'))), 17)

    def test_corruption_is_not_queued(self):
        (self.folder / 'payload0.pt').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Pinned payload hash differs'):
            self.poll()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'pins/G3_0070000').exists())

    def test_partial_prune_does_not_stop_other_observations(self):
        real_link = cadence.os.link
        def pruning_link(source, destination):
            real_link(source, destination)
            shutil.rmtree(self.folder)
        with patch.object(cadence.os, 'link', pruning_link):
            self.poll()
        states = {x['key']: x['status'] for x in cadence.read(self.output / 'CADENCE_STATE.json')['observations']}
        self.assertEqual(states['G3_0070000'], 'CHECKPOINT_PRUNED_DURING_PIN')
        self.assertEqual(states['G3_0090000'], 'WAITING_FOR_NATIVE_CHECKPOINT')
        self.assertEqual(self.calls, [])

    def test_observer_close_waits_for_inflight_capture(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        def capture(*_):
            entered.set()
            release.wait(5)
        with patch.object(cadence, 'poll', capture):
            observer = cadence.Observer(self.policy, self.output, self.enqueue, lambda *_: True)
            self.assertTrue(entered.wait(2))
            closer = threading.Thread(target=lambda: (observer.close(), closed.set()))
            closer.start()
            self.assertFalse(closed.wait(.05))
            release.set()
            closer.join(2)
            self.assertTrue(closed.is_set())
            self.assertFalse(observer.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
