"""CPU-only evidence checks for one routing-only scorer startup failure."""
import json
from pathlib import Path
import tempfile
import unittest

import run_navsim_navtest as dispatch


class ScoringStartupReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failure = {'error': "RuntimeError('Canonical Navtest scoring failed; failed scenes retained')"}
        self.log = self.root / 'canonical_score.log'
        self.log.write_text('Traceback (most recent call last):\nValueError: Host outside latest five-host authorization\n')
        dispatch.atomic(self.root / 'SCORER_LAUNCH.json', {'pid': 123})
        self.full = {'scenes': 12146, 'full_population': True, 'final_attempt': 'a',
                     'shards': 1, 'identity': 'fixture'}
        dispatch.atomic(self.root / 'COMPLETE.json', self.full)
        self.report = self.root / 'attempts/a/shards/shard_000.json'
        self.record = {'status': 'COMPLETE', 'prediction_identity': 'fixture', 'attempt': 'a'}
        dispatch.atomic(self.report, self.record)
        self.evidence = {'scorer_log_sha256': dispatch.digest(self.log),
                         'scorer_launch_sha256': dispatch.digest(self.root / 'SCORER_LAUNCH.json')}
        self.bind()

    def bind(self):
        self.evidence['attempt_files_sha256'] = {
            str(self.report.relative_to(self.root)): dispatch.digest(self.report)}

    def test_complete_predictions_and_exact_startup_failure_are_reviewable(self):
        before = self.report.read_bytes()
        dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)
        self.assertEqual(self.report.read_bytes(), before)

    def test_changed_log_or_another_failure_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch.review_scoring_startup_authorization(self.root, {'error': 'bad trajectory'}, self.evidence)
        self.log.write_text('ValueError: Host outside latest five-host authorization\nchanged')
        with self.assertRaises(ValueError):
            dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)

    def test_partial_prediction_bank_is_rejected(self):
        dispatch.atomic(self.root / 'COMPLETE.json', self.full | {'scenes': 12145})
        with self.assertRaisesRegex(ValueError, 'Complete original'):
            dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)

    def test_prediction_error_or_foreign_identity_is_rejected(self):
        for change in [{'status': 'FAILED'}, {'error': 'bad model'}, {'prediction_identity': 'other'}]:
            dispatch.atomic(self.report, self.record | change)
            self.bind()
            with self.assertRaisesRegex(ValueError, 'scientific failures'):
                dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)

    def test_missing_or_changed_shard_is_rejected(self):
        dispatch.atomic(self.report, self.record | {'scenes_done': 4})
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)
        self.report.unlink()
        with self.assertRaisesRegex(ValueError, 'Every completed'):
            dispatch.review_scoring_startup_authorization(self.root, self.failure, self.evidence)


if __name__ == '__main__':
    unittest.main()
