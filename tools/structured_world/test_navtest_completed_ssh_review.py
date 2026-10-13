"""CPU-only checks: completed remote inference is not a live SSH exit status."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_navsim_navtest as dispatch


class CompletedSSHReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.attempt = self.root / 'attempts/a'
        dispatch.atomic(self.root/'identity.json', {'identity': 'prediction'})
        self.report = self.attempt/'shards/shard_000.json'
        self.log = self.attempt/'shard_000.log'
        self.launch = self.attempt/'launches/shard_000.json'
        self.record = {'shard': 0, 'shards': 1, 'attempt': 'a', 'status': 'COMPLETE',
                       'prediction_identity': 'prediction', 'host': 'owned-worker-0'}
        dispatch.atomic(self.report, self.record)
        self.log.write_text('Timeout, server owned not responding.\n')
        dispatch.atomic(self.launch, {'shard': 0, 'shards': 1, 'host': 'owned',
                                      'command': ['ssh', 'owned', 'qualified worker']})
        self.index = self.root/'index.json'
        dispatch.atomic(self.index, [{'token': str(i), 'log': str(i % 136)} for i in range(12146)])
        p = patch.object(dispatch, 'INDEX', self.index)
        p.start(); self.addCleanup(p.stop)
        # The mature union validator is separately exercised by full-bank validation.
        # Verify this review invokes it and propagates every corruption failure.
        p = patch.object(dispatch, 'validate_record_union')
        self.union = p.start(); self.addCleanup(p.stop)
        self.failure = {'classification': 'unclassified', 'exit_codes': [255]}
        self.evidence = {'attempt': 'a'}
        self.bind()

    def bind(self):
        self.evidence['attempt_files_sha256'] = {
            str(p.relative_to(self.root)): dispatch.digest(p)
            for p in (self.report, self.log, self.launch) if p.exists()}

    def review(self):
        dispatch.review_completed_ssh_launcher_exit(self.root, self.failure, self.evidence)

    def test_complete_bank_is_checked_without_mutating_outputs(self):
        before = self.report.read_bytes()
        self.review()
        self.union.assert_called_once_with(self.root, {str(i) for i in range(12146)}, 'prediction')
        self.assertEqual(self.report.read_bytes(), before)
        self.assertFalse((self.root/'COMPLETE.json').exists())

    def test_model_errors_and_non_ssh_exits_are_rejected(self):
        for failure in ({'exit_codes': [1]}, {'exit_codes': [0]},
                        {'exit_codes': [True]}, {'exit_codes': [255], 'error': 'bad model'}):
            self.failure = failure
            with self.assertRaises(ValueError): self.review()

    def test_missing_or_modified_evidence_is_rejected(self):
        self.log.write_text(self.log.read_text() + 'changed\n')
        with self.assertRaisesRegex(ValueError, 'hash differs'): self.review()
        self.report.unlink()
        with self.assertRaisesRegex(ValueError, 'Every original'): self.review()

    def test_failed_worker_foreign_identity_and_wrong_mapping_are_rejected(self):
        for change in ({'status': 'FAILED'}, {'error': 'bad geometry'},
                       {'prediction_identity': 'other'}, {'attempt': 'other'},
                       {'shard': 1}, {'host': 'foreign-worker-0'}):
            dispatch.atomic(self.report, self.record | change)
            self.bind()
            with self.assertRaisesRegex(ValueError, 'Incomplete/foreign/scientific'): self.review()

    def test_timeout_and_real_ssh_launcher_are_required(self):
        self.log.write_text('a different process failure\n'); self.bind()
        with self.assertRaisesRegex(ValueError, 'SSH timeout'): self.review()
        self.log.write_text('Timeout, server owned not responding.\n')
        x = json.loads(self.launch.read_text()); x['command'][0] = 'python'
        dispatch.atomic(self.launch, x); self.bind()
        with self.assertRaisesRegex(ValueError, 'SSH timeout'): self.review()

    def test_partial_population_and_corrupt_payload_propagate(self):
        self.union.side_effect = ValueError('payload mismatch')
        with self.assertRaisesRegex(ValueError, 'payload mismatch'): self.review()
        dispatch.atomic(self.index, [{'token': 'one', 'log': 'log'}])
        with self.assertRaisesRegex(ValueError, 'population'): self.review()

    def test_path_traversal_attempt_is_rejected(self):
        self.evidence['attempt'] = '../other'
        with self.assertRaisesRegex(ValueError, 'Exact completed'): self.review()


if __name__ == '__main__':
    unittest.main()
