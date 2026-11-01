import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.il_report.navtest_burst import adopt_four_jobs, prechild_refusal


class BurstRecoveryTest(unittest.TestCase):
    def test_resource_wait_only_before_child(self):
        good=dict(status='REFUSED_RESOURCE',child_not_started=True,returncode=75)
        self.assertTrue(prechild_refusal(good))
        for delta in [dict(child_not_started=False),dict(returncode=1),dict(status='FAILED')]:
            self.assertFalse(prechild_refusal(dict(good,**delta)))

    def fixture(self,base):
        root=Path(base);request=SimpleNamespace(root=root,source=Path('/frozen/source'),source_sha='abc',
            export=root/'export',output=root/'evaluation',world=32,attempts={},jobs={})
        for name,num in [('export',1),('parity',1),('rank0',1),('rank7',1),('rank7',4)]:
            p=root/'jobs'/f'{name}_attempt{num:03d}';p.mkdir(parents=True)
            (p/'spec.json').write_text(json.dumps(dict(source=str(request.source),source_sha='abc',receipt_dir=str(p),
                command=['python','-m','evaluator','--output',str(request.export if name=='export' else request.output)])))
        return request

    def test_latest_attempt_and_active_shard_adopted_without_relaunch(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.fixture(d);adopt_four_jobs(r)
            self.assertEqual(r.attempts['rank7'],4)
            self.assertEqual(set(r.jobs),{'export','parity','rank0','rank7'})
            self.assertEqual(r.jobs['rank7']['receipt_dir'],str(r.root/'jobs/rank7_attempt004'))

    def test_changed_source_or_output_is_rejected(self):
        for change in ['source_sha','output']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                r=self.fixture(d);p=r.root/'jobs/rank0_attempt001/spec.json';s=json.loads(p.read_text())
                if change=='source_sha':s['source_sha']='different'
                else:s['command'][-1]='/other/evaluation'
                p.write_text(json.dumps(s))
                with self.assertRaises(ValueError):adopt_four_jobs(r)


if __name__=='__main__':unittest.main()
