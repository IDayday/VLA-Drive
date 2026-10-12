"""Narrow CPU tests: resource recovery must not clear scientific failures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dispatch', Path(__file__).with_name('run_navsim_navtest.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class LoadOOMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root/'attempts/a/shards/shard_000.json'
        self.log = self.root/'attempts/a/shard_000.log'
        self.record = dict(status='FAILED', scenes_done=0, shards=1,
                           error="OutOfMemoryError('CUDA out of memory')")
        d.atomic(self.report, self.record)
        self.log.write_text('Original CUDA loading traceback')
        self.failure = dict(attempt='a', error=self.record['error'],
            error_traceback="model.to('cuda').eval(); of which 12.31 MiB is free")
        self.evidence = dict(reviewed_failure_kind='CUDA_load_oom_after_qualification')
        self.bind()

    def bind(self):
        self.evidence['attempt_files_sha256'] = {
            str(p.relative_to(self.root)): d.digest(p) for p in (self.report, self.log)}

    def test_bound_loading_failure_is_reviewable(self):
        d.review_load_oom(self.root, self.failure, self.evidence)
        self.assertTrue(self.report.exists())  # Review itself never mutates evidence.

    def test_changed_log_fails(self):
        self.log.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            d.review_load_oom(self.root, self.failure, self.evidence)

    def test_executed_scene_or_scientific_exception_fails(self):
        for change in [dict(scenes_done=1), dict(error="ValueError('bad trajectory')")]:
            d.atomic(self.report, dict(self.record, **change)); self.bind()
            with self.assertRaisesRegex(ValueError, 'executed scene or another error'):
                d.review_load_oom(self.root, self.failure, self.evidence)

    def test_scene_compute_stack_or_non_exhaustion_fails(self):
        for suffix in ['; plan_from_current', '; scene_noise(']:
            with self.assertRaises(ValueError):
                d.review_load_oom(self.root, dict(self.failure,
                    error_traceback=self.failure['error_traceback']+suffix), self.evidence)
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            d.review_load_oom(self.root, dict(self.failure,
                error_traceback="model.to('cuda').eval(); of which 4096 MiB is free"), self.evidence)

    def test_missing_shard_and_existing_output_fail(self):
        d.atomic(self.report, dict(self.record, shards=2)); self.bind()
        with self.assertRaisesRegex(ValueError, 'every original shard'):
            d.review_load_oom(self.root, self.failure, self.evidence)
        d.atomic(self.report, self.record); self.bind()
        d.atomic(self.root/'records/scene.json', {})
        with self.assertRaisesRegex(ValueError, 'scene outputs'):
            d.review_load_oom(self.root, self.failure, self.evidence)


if __name__ == '__main__':
    unittest.main(verbosity=2)
