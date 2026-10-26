import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.analysis.qwen_sourceflow.summarize_resumed_navtest import invocation_costs


class TransportCostsTest(unittest.TestCase):
    def record(self):
        return dict(status='COMPLETE', end_unix=30., wall_seconds=20., physical_inference=3,
                    reused=4, model_loading_seconds=2., forward_seconds=10., scoring_wait_seconds=3.)

    def test_unlost_costs_preserve_original_arithmetic(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'; p.write_text(json.dumps(self.record()))
            x=invocation_costs([p], {})
            self.assertEqual(x['worker_wall_seconds'],20.)
            self.assertEqual(x['elapsed_span_seconds'],20.)
            self.assertFalse(x['physical_cost_is_lower_bound'])

    def test_stale_is_not_ignored_or_given_fake_exit_time(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'; r=self.record();r['status']='RUNNING';r.pop('end_unix');p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):invocation_costs([p],{})
            evidence={str(p.resolve()):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                rank_lock_unheld=True,worker_lock_unheld=True,last_observed_write_unix=30.)}
            x=invocation_costs([p],evidence)
            self.assertTrue(x['physical_cost_is_lower_bound']);self.assertEqual(x['physical_inference_including_retries'],3)
            self.assertIsNone(x['transport_lost_invocations'][0]['unmeasured_tail_seconds'])
            self.assertNotIn('end_unix',json.loads(p.read_text()))
            evidence[str(p.resolve())]['sha256']='changed'
            with self.assertRaises(ValueError):invocation_costs([p],evidence)


if __name__=='__main__':unittest.main()
