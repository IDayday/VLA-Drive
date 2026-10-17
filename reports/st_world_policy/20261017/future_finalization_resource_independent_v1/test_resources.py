"""Independent resource regressions; one two-shard synthetic case only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import pytest
from tools.st_world_policy import finalize_future_metadata as m
from tools.st_world_policy import prepare_future_metadata as f
from tests.st_world.test_finalize_future_metadata import make_case, repin

HERE = Path(__file__).resolve().parent
FIXTURE = Path(os.environ['INDEPENDENT_FIXTURE_ROOT'])


class ResourceReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURE.mkdir(exist_ok=False)
        cls.mp = pytest.MonkeyPatch()
        cls.case = make_case(FIXTURE, cls.mp)
        cls.original_reg = json.loads(json.dumps(cls.case.reg))
        cls.old = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in cls.case.source.rglob('*') if p.is_file()}
        cls.details = {'fixture': str(FIXTURE), 'source_shards': 2, 'cache_scenes': 128}
        def deny_real(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0])).absolute()
            if str(path).startswith('/mnt/project/st-world-policy-20261016/'):
                raise AssertionError('Actual campaign payload/control access forbidden')
            if path.suffix in ('.pkl', '.png', '.pt', '.safetensors'):
                raise AssertionError('No raw/image/tensor opens after synthetic setup')
        sys.addaudithook(deny_real)

    @classmethod
    def tearDownClass(cls):
        cls.mp.undo()
        cls.details['fixture_file_bytes'] = sum(p.stat().st_size for p in FIXTURE.rglob('*') if p.is_file())
        cls.details['fixture_file_count'] = sum(p.is_file() for p in FIXTURE.rglob('*'))
        (HERE/'DETAILS.json').write_text(json.dumps(cls.details, indent=2, sort_keys=True)+'\n')

    def test_01_actual_finalize_single_read_and_failure_retained(self):
        counts = {}
        old_reader = f.metadata_reader
        def counted(path):
            if Path(path).parent.name == 'log_shards':
                counts[str(path)] = counts.get(str(path), 0) + 1
            return old_reader(path)
        old_terminal = self.case.terminal.read_bytes()
        with patch.object(f, 'metadata_reader', counted):
            result = m.run(self.case.args)
        self.assertEqual(list(counts.values()), [1, 1])
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertFalse(result['raw_log_deserialized'])
        self.assertTrue(result['source_shards_reused'])
        self.assertEqual(self.case.terminal.read_bytes(), old_terminal)
        self.assertEqual({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in self.case.source.rglob('*') if p.is_file()}, self.old)
        self.assertFalse((self.case.source/'COMPLETE.json').exists())
        self.details['actual_finalizer_shard_open_counts'] = counts
        self.details['synthetic_finalizer_cost'] = result['cost']
        self.details['compressed_cache_bytes'] = result['compressed_cache_bytes']
        self.assertLess(result['cost']['source_bytes_read'], 8*m.MiB)
        m.validate_completion(result, self.case.output/m.COMPLETE_NAME)

    def test_02_three_ram_replays_no_source_open(self):
        rows = [{'token':str(i),'log':'ABC'[i%3]} for i in range(128)]
        scenes = [dict(row, payload='abc'*100, nested=[i, None, True]) for i,row in enumerate(rows)]
        budget = f.PreparationBudget(self.case.output, self.original_reg['limits'])
        cache = m.CompressedScenes(rows, budget, 4*m.MiB, 64*1024)
        for log in 'ABC':
            for scene in scenes:
                if scene['log'] == log: cache.add(scene)
        actual_bytes = sys.getsizeof(cache.values)
        for key,value in cache.values.items():
            actual_bytes += sys.getsizeof(key)+sys.getsizeof(value)+sum(sys.getsizeof(x) for x in value)
        self.assertEqual(cache.bytes, actual_bytes)
        with patch.object(f, 'metadata_reader', side_effect=AssertionError('Replay source read')):
            for _ in range(3): self.assertEqual(list(cache()), scenes)
        self.assertEqual(budget.bytes, 0)
        self.details['cache_128_rows_actual_accounted_bytes'] = cache.bytes

    def test_03_read_preflight_rejects_before_open(self):
        path = next((self.case.source/'log_shards').glob('*.json'))
        budget = f.PreparationBudget(self.case.output, self.original_reg['limits'])
        with patch.object(f, 'metadata_reader', side_effect=AssertionError('Must refuse before read')):
            with self.assertRaisesRegex(f.BudgetExceeded, 'DECODED_SHARD_PREFLIGHT'):
                m._read(path, budget, decoded_limit=1)
        self.assertEqual(budget.bytes, 0)

    def test_04_remaining_read_budget_rejects_before_bytes(self):
        path = next((self.case.source/'log_shards').glob('*.json'))
        limits = dict(self.original_reg['limits'], max_source_read_bytes=1)
        budget = f.PreparationBudget(self.case.output, limits)
        with self.assertRaisesRegex(f.BudgetExceeded, 'READ_BYTES'):
            m._read(path, budget)
        self.assertEqual(budget.bytes, 0)

    def test_05_cache_limit_refuses_before_insert(self):
        budget = f.PreparationBudget(self.case.output, self.original_reg['limits'])
        cache = m.CompressedScenes([{'token':'x'}], budget, 32, 128)
        with self.assertRaisesRegex(f.BudgetExceeded, 'COMPRESSED_RAM'):
            cache.add({'token':'x', 'value':'small'})
        self.assertEqual(cache.values, {})

    def test_06_compression_bomb_bounded_output(self):
        budget = f.PreparationBudget(self.case.output, self.original_reg['limits'])
        cache = m.CompressedScenes([{'token':'x'}], budget, m.MiB, 128)
        cache.add({'token':'x'})
        cache.values['x'] = (zlib.compress(b'x'*1000000), 1000000, b'0'*32)
        with self.assertRaisesRegex(ValueError, 'length/checksum'): next(cache())

    def test_07_truncation_trailing_and_digest_fail(self):
        budget = f.PreparationBudget(self.case.output, self.original_reg['limits'])
        for kind in ('truncated','trailing','digest'):
            cache = m.CompressedScenes([{'token':'x'}], budget, m.MiB, 128)
            cache.add({'token':'x'})
            body,length,digest = cache.values['x']
            cache.values['x'] = (body[:-2] if kind=='truncated' else body+b'junk' if kind=='trailing' else body,
                                 length, b'0'*32 if kind=='digest' else digest)
            with self.assertRaises(ValueError): next(cache())

    def test_08_partial_file_counts_toward_output_cap(self):
        path = self.case.output/'retained.partial'
        path.write_bytes(b'x'*4096)
        before = path.read_bytes()
        with self.assertRaisesRegex(f.BudgetExceeded, 'METADATA_DISK_BUDGET'):
            f.PreparationBudget(self.case.output, dict(self.original_reg['limits'], max_metadata_bytes=4095))
        self.assertEqual(path.read_bytes(), before)

    def test_09_completed_retry_preserves_receipt(self):
        before = (self.case.output/m.COMPLETE_NAME).read_bytes()
        result = m.run(self.case.args)
        self.assertTrue(result['reused'])
        self.assertEqual((self.case.output/m.COMPLETE_NAME).read_bytes(), before)

    def test_10_output_resource_ceilings_fail_closed(self):
        for key,value in [('max_rss_bytes',4*m.GiB+1),('max_metadata_bytes',2*m.GiB+1),('workers',2),
                          ('max_source_read_bytes',16*m.GiB+1)]:
            reg = json.loads(json.dumps(self.original_reg)); reg['limits'][key] = value
            reg.pop('identity'); reg = f.signed(reg)
            with self.assertRaises(ValueError): m._registration(reg, self.case.output)

    def test_11_original_writer_lock_excludes_finalizer(self):
        import fcntl
        with (self.case.source/'BUILD.lock').open('rb') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaises(BlockingIOError):
                    with m._source_lock(self.case.source): self.fail('Acquired occupied source lock')
            finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__': unittest.main(verbosity=2)
