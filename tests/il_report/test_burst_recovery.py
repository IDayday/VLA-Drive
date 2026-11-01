import json
from pathlib import Path
import tempfile
import fcntl
from types import SimpleNamespace
import unittest

from tools.il_report.navtest_burst import Pool, adopt_four_jobs, prechild_refusal,capacity_probe,vacant_slot


class BurstRecoveryTest(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
