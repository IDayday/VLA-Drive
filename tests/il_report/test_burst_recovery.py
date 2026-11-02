import json
from pathlib import Path
import tempfile
import fcntl
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.il_report.navtest_burst import Pool, adopt_four_jobs, prechild_refusal,capacity_probe,vacant_slot,resource_worker_overrides,review_unacknowledged_startup


class BurstRecoveryTest(unittest.TestCase):
    def test_dead_prechild_sharing_rejection_is_reviewed_without_fake_exit_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);spec=dict(receipt_dir=str(root),command=['python','evaluate'],host='authorized')
            (root/'spec.json').write_text(json.dumps(spec))
            (root/'operator.json').write_text(json.dumps(dict(pid=123,start_ticks=456)))
            (root/'STARTUP_PENDING.json').write_text(json.dumps(dict(at=0)))
            (root/'transport.log').write_text('in _gpu_admission\nValueError: Sharing requires an explicitly authorized host and measured valid headroom\n')
            proof=dict(worker_alive=False,matching_processes=[],worker_lock_free=True,receipt_children=[])
            def remote(host,args):self.assertEqual(host,'authorized');return json.dumps(proof)
            self.assertTrue(review_unacknowledged_startup(spec,remote))
            decision=json.loads((root/'OPERATOR_REQUEUE.json').read_text())
            self.assertEqual(decision['spec_sha256'],hashlib.sha256((root/'spec.json').read_bytes()).hexdigest())
            self.assertFalse((root/'RESULT.json').exists())
            for change in [dict(worker_alive=True),dict(worker_lock_free=False),dict(matching_processes=[123]),dict(receipt_children=['RUNNING.json'])]:
                self.assertFalse(review_unacknowledged_startup(spec,lambda h,a:json.dumps(dict(proof,**change))))
            (root/'RUNNING.json').write_text('{}')
            self.assertFalse(review_unacknowledged_startup(spec,lambda *a:self.fail('Never inspect/requeue a started child')))

    def test_utf8_policy_read_survives_native_locale_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy.json';p.write_text(json.dumps({'request':'评测'},ensure_ascii=False),encoding='utf-8')
            program="import locale,json;from pathlib import Path;locale.setlocale(locale.LC_ALL,'C');assert json.loads(Path("+repr(str(p))+").read_text())['request']=='评测'"
            overrides=resource_worker_overrides('/qualified/worker')
            self.assertEqual(set(overrides),{'PYTHONPATH','PYTHONUTF8'})
            subprocess.run([sys.executable,'-c',program],env=dict(os.environ,**overrides),check=True)

    def test_legacy_probe_requires_real_used_total_and_RAM_enrichment(self):
        full=dict(MemAvailable_bytes=200*1024**3,worker_python_exists=True,
            gpus=[dict(total_MiB=80000)],owners=[])
        calls=[]
        delegate=SimpleNamespace(probe=lambda h:dict(available_memory_bytes=200*1024**3,gpus=[{}],owners=[]),
            remote=lambda h,args:calls.append(h)or json.dumps(full))
        self.assertEqual(capacity_probe(delegate,'host'),full)
        self.assertEqual(calls,['host'])

    def test_free_primary_is_used_without_stealing_held_shared_slot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shared=root/'runtime_gpu_shared_workers/GPU-test.lock';shared.parent.mkdir()
            with shared.open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(vacant_slot('GPU-test',family='four',root=root),'primary')
                self.assertIsNone(vacant_slot('GPU-test',family='world',root=root))
                primary=root/'runtime_gpu_leases/GPU-test.lock'
                with primary.open('a')as other:
                    fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    self.assertIsNone(vacant_slot('GPU-test',family='four',root=root))

    def test_inventory_uses_the_qualified_tasks_probe(self):
        calls=[]
        def probe(host):
            calls.append(host)
            return dict(gpus=[dict(index=0,uuid='GPU-test',total_MiB=80000,free_MiB=80000,utilization=0)],
                owners=[],MemAvailable_bytes=500*1024**3,worker_python_exists=True)
        delegate=SimpleNamespace(allowed_hosts=lambda:['authorized-task-host'],probe=probe,
            remote=lambda host,args:json.dumps(dict(accepted=['GPU-test'],rejected={})))
        routes,inventory=Pool(delegate,admission_source='/frozen/source').available()
        self.assertEqual(len(routes),1)
        self.assertEqual(calls,['authorized-task-host']*2)
        self.assertFalse(inventory['errors'])

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

    def test_reviewed_terminal_attempt_retains_number_and_does_not_adopt_older_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.fixture(d);p=r.root/'jobs/rank7_attempt004/spec.json'
            (p.parent/'OPERATOR_REQUEUE.json').write_text(json.dumps(dict(
                status='REVIEWED_TERMINAL_REQUEUE',spec_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
            adopt_four_jobs(r)
            self.assertEqual(r.attempts['rank7'],4)
            self.assertNotIn('rank7',r.jobs)
            self.assertIn('rank0',r.jobs)

    def test_requeue_decision_must_bind_exact_spec(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.fixture(d);p=r.root/'jobs/parity_attempt001/OPERATOR_REQUEUE.json'
            p.write_text(json.dumps(dict(status='REVIEWED_TERMINAL_REQUEUE',spec_sha256='different')))
            with self.assertRaises(ValueError):adopt_four_jobs(r)


if __name__=='__main__':unittest.main()
