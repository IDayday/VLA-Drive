"""Bounded source/control fixtures; never opens metadata DBs or dispatches."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:BOOT={k:int(v) for k,v in (line.split(':') for line in f)}
import copy
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('subject',HERE/'prepare_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PLAN=json.loads((HERE/'PLAN.json').read_text())

class Reads:
    def __init__(self):self.paths=[]
    def raw(self,path,sha):
        self.paths.append(str(path))
        if str(path) in (str(HERE/'prepare_once.py'),str(HERE/'launch_once.sh')):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('fixture controller mismatch')
        return b''
    def document(self,pin):
        self.paths.append(pin['path']);return {'status':'COMPLETE','stage':'fit'}

class Preparation(unittest.TestCase):
    def setUp(self):
        self.plan=copy.deepcopy(PLAN);self.reads=Reads();now=time.time()
        self.ad={'schema':'st_full_population_provider_preparation_admission_v1',
            'status':'APPROVED_PROVIDER_METADATA_PREPARATION_ONLY','reviewer':'independent_fixture',
            'hostname':socket.gethostname(),'limits':m.LIMITS,'cpu_affinity':[0,1],
            'output_root':str(m.ROOT),'max_output_bytes':32<<20,
            'control_reservation':copy.deepcopy(PLAN['control_reservation']),
            'gpu':False,'automatic_retry':False,'ledger_written':False,'checkpoint_reservation_created':False,
            'plan':{'path':str(HERE/'PLAN.json'),'sha256':m.PLAN_SHA},
            'issued_at_unix':now-10,'latest_start_unix':now+60,'valid_until_unix':now+2000,
            'controller_sources':[{'path':str(HERE/name),'sha256':hashlib.sha256((HERE/name).read_bytes()).hexdigest()}
                                  for name in ('prepare_once.py','launch_once.sh')]}

    def validate(self):
        with patch.object(m.os,'sched_getaffinity',return_value={0,1}):
            return m.validate(self.ad,self.plan,None,self.reads,time.time())

    def test_exact_plan_and_bounds(self):
        self.assertEqual(hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),m.PLAN_SHA)
        self.assertEqual(m.LIMITS,dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30))
        self.validate()
        self.assertFalse(any(p.endswith('.sqlite') or p.endswith('.safetensors') or p.endswith('SELECTED_SCENES.json') for p in self.reads.paths))

    def test_exact_two_original_cli_paths_and_scientific_budget(self):
        calls=m.commands(self.plan)
        self.assertEqual([c[0] for c in calls],['tools.st_world_policy.qualify_indexed_provider','tools.st_world_policy.teacher_provider'])
        self.assertEqual(calls[0][1][0],'prepare-profile-migration')
        a=calls[1][1]
        self.assertIn('--profile-full-population',a)
        self.assertEqual(a[a.index('--max-cached-bytes')+1],'14159232')
        self.assertEqual(a[a.index('--max-batch-images')+1],'8')
        self.assertEqual(a[a.index('--profile-data-manifest')+1],self.plan['data_manifest']['path'])
        self.assertEqual(self.plan['recipe']['execution']['sample_limit'],101592)
        self.assertEqual(self.plan['recipe']['execution']['max_updates'],20000)
        self.assertEqual(self.plan['recipe']['optimizer']['warmup'],1000)
        self.assertEqual(self.plan['consumption']['cumulative_optimizer_updates'],64)
        self.assertEqual(self.plan['consumption']['cumulative_scene_exposures'],128)

    def test_wrong_runtime_output_budget_or_lease_rejected(self):
        cases=[('output_root','/tmp/other'),('gpu',True),('automatic_retry',True),('ledger_written',True),
               ('checkpoint_reservation_created',True),('valid_until_unix',time.time()+100),
               ('latest_start_unix',0),('max_output_bytes',33<<20)]
        for name,value in cases:
            saved=self.ad[name];self.ad[name]=value
            with self.subTest(field=name),self.assertRaises(ValueError):self.validate()
            self.ad[name]=saved
        self.plan['runtime']['source_sha']='0'*40
        with self.assertRaises(ValueError):self.validate()

    def test_missing_changed_or_null_controller_pin_rejected(self):
        saved=copy.deepcopy(self.ad['controller_sources'])
        for pins in (saved[:1],[saved[0],saved[0]],
                     [{**saved[0],'sha256':None},saved[1]],[{**saved[0],'sha256':'0'*64},saved[1]]):
            self.ad['controller_sources']=pins
            with self.subTest(pins=pins),self.assertRaises(ValueError):self.validate()

    def test_incomplete_fit_prerequisite_rejected(self):
        with patch.object(self.reads,'document',return_value={'status':'STOPPED_NO_AUTO_RETRY','stage':'fit'}):
            with self.assertRaises(ValueError):self.validate()

    def test_original_cli_argv_return_and_failure_semantics(self):
        before=sys.argv
        with patch.object(m.runpy,'run_module',side_effect=lambda module,run_name:self.assertEqual(sys.argv,['actual.module','--flag','value'])):
            m.original_cli('actual.module',['--flag','value'])
        self.assertIs(sys.argv,before)
        with patch.object(m.runpy,'run_module',side_effect=SystemExit(0)):m.original_cli('actual.module',[])
        with patch.object(m.runpy,'run_module',side_effect=SystemExit(2)),self.assertRaises(RuntimeError):m.original_cli('actual.module',[])
        self.assertIs(sys.argv,before)

    def test_original_cli_prints_preserve_single_stdout_envelope(self):
        stdout=io.StringIO();stderr=io.StringIO()
        def real_prints(module,run_name):
            print(json.dumps({'original_cli':module}));print('original stderr',file=sys.stderr)
            raise SystemExit(0)
        with redirect_stdout(stdout),redirect_stderr(stderr),patch.object(m.runpy,'run_module',side_effect=real_prints):
            for module,args in m.commands(self.plan):m.original_cli(module,args)
            print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{},'counters':{}}))
        self.assertEqual(json.loads(stdout.getvalue())['schema'],'st_full_future_child_result_v2')
        self.assertEqual(stderr.getvalue().count('original_cli'),2)
        self.assertEqual(stderr.getvalue().count('original stderr'),2)

    def test_original_monitor_charge_lock_and_timeout_preserved(self):
        source=(HERE/'prepare_once.py').read_text();launch=(HERE/'launch_once.sh').read_text()
        self.assertIn('core.monitor(',source);self.assertIn('core.charge_after_reap(',source)
        self.assertIn('lock=s.acquire(s.LOCK,stack)',source)
        self.assertIn("ST_PROVIDER_PREPARATION_DEADLINE=str(deadline)",source)
        self.assertIn('census(1<<20)',source)
        self.assertNotIn('subprocess.Popen(',source)
        self.assertNotIn('load_state_dict(',source)
        self.assertIn('timeout --signal=TERM --kill-after=40s 1500s',launch)
        self.assertIn('taskset -c 0,1',launch)

    def test_bootstrap_reuses_reviewed_source_and_core_pin(self):
        bootstrap=m.bootstrap()
        self.assertEqual(bootstrap.CORE_SHA,'4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2')
        self.assertTrue(callable(bootstrap.read_source));self.assertTrue(callable(bootstrap.first_source_alarm))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Preparation))
    with open('/proc/self/io') as f:final={k:int(v) for k,v in (line.split(':') for line in f)}
    print('COST_JSON='+json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        wall_seconds=time.monotonic()-START,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        logical_read_once_bytes=final['rchar']-BOOT['rchar'],kernel_read_bytes=final['read_bytes']-BOOT['read_bytes'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),fixture_disk_written_bytes=0,
        database_or_payload_read=False,ledger_access=False,actual_preparation=False),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
