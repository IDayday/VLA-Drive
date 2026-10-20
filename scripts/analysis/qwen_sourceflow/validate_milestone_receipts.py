"""CPU regression for worker completion racing queue liveness checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis.qwen_sourceflow import milestone_queue as queue


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.item=dict(receipt_dir=str(self.root))
        (self.root/'operator.json').write_text(json.dumps(dict(operator_pid=42,start_ticks=123)))

    def tearDown(self):self.temp.cleanup()

    def complete(self):
        (self.root/'result.json').write_text(json.dumps(dict(status='PROCESS_COMPLETE',exit_code=0)))

    def test_already_complete_needs_no_liveness_probe(self):
        self.complete()
        with patch.object(queue,'operator_alive',side_effect=AssertionError('unexpected probe')):
            self.assertTrue(queue.worker_finished(self.item))

    def test_receipt_published_during_dead_owner_probe(self):
        def probe(*_):self.complete();return False
        with patch.object(queue,'operator_alive',side_effect=probe):
            self.assertTrue(queue.worker_finished(self.item))

    def test_receipt_published_during_alive_owner_probe(self):
        def probe(*_):self.complete();return True
        with patch.object(queue,'operator_alive',side_effect=probe):
            self.assertTrue(queue.worker_finished(self.item))

    def test_finite_visibility_grace(self):
        with patch.object(queue,'operator_alive',return_value=False),patch.object(queue.time,'sleep',side_effect=lambda _:self.complete()) as sleep:
            self.assertTrue(queue.worker_finished(self.item));self.assertEqual(sleep.call_count,1)

    def test_live_worker_remains_pending(self):
        with patch.object(queue,'operator_alive',return_value=True):
            self.assertFalse(queue.worker_finished(self.item))

    def test_missing_receipt_still_fails_without_retry(self):
        with patch.object(queue,'operator_alive',return_value=False),patch.object(queue.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'without completion receipt'):queue.worker_finished(self.item)
            self.assertEqual(sleep.call_count,20)

    def test_failed_result_is_not_success(self):
        (self.root/'result.json').write_text(json.dumps(dict(status='FAILED',exit_code=1)))
        with self.assertRaisesRegex(RuntimeError,'Worker failed'):queue.worker_finished(self.item)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptTests))
    if args.output:
        Path(args.output).write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
