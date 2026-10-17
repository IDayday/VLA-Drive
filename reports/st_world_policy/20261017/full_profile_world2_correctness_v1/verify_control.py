"""Independent bounded control fixtures, never dispatches or reads real payloads."""
import time
START=time.monotonic()
with open('/proc/self/io') as f: BOOT={k:int(v) for k,v in (line.split(':') for line in f)}
import ast
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Independent review deadline')))
signal.alarm(110)
HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'full_population_profile_launch_v1'
spec=importlib.util.spec_from_file_location('author_control_cases',OWNER/'test_controller.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
c=author.c

class Independent(unittest.TestCase):
    def test_exact_source_and_inputs_pins(self):
        pins=json.loads((OWNER/'SOURCE_PINS.json').read_text())
        self.assertEqual(hashlib.sha256((OWNER/'controller.py').read_bytes()).hexdigest(),
            'bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1')
        self.assertEqual(pins['controller_sha256'],hashlib.sha256((OWNER/'controller.py').read_bytes()).hexdigest())
        self.assertEqual(pins['inputs_sha256'],hashlib.sha256((OWNER/'INPUTS.json').read_bytes()).hexdigest())
        self.assertEqual(c.SOURCE,'a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a')
        self.assertEqual(c.RESERVE,80<<30)

    def test_output_relocation_is_one_deepcopied_field(self):
        cfg=author.recipe()
        cfg['optimizer'].update(lr=1e-5,minimum_lr=5e-7,betas=[.9,.95],weight_decay=.001)
        cfg['space_time']={'cell':'T2','teacher':{'channels':384,'views':['F','L','R']},'future_weight':1.0}
        cfg['data']={'teacher_provider':{'descriptor':{'path':'/fixture/provider','sha256':'a'*64}}}
        before=deepcopy(cfg); relocated=c.registered_config(cfg)
        self.assertEqual(cfg,before)
        self.assertEqual(relocated['runtime']['output_root'],str(c.OUT/'runs'))
        relocated['runtime']['output_root']=before['runtime']['output_root']
        self.assertEqual(relocated,before)
        relocated['space_time']['teacher']['views'][0]='changed'
        self.assertEqual(cfg,before)

    def test_all_interior_resume_boundaries_and_no_budget_restart(self):
        for completed in range(1,64):
            c.validate_resume_state({'status':'PAUSED','source_sha':c.SOURCE,'completed':completed,'exposure':2*completed})
        for completed,exposure in [(64,128),(65,130),(32,128),(32,63),(-1,-2),(1.0,2)]:
            with self.subTest(completed=completed,exposure=exposure),self.assertRaises(RuntimeError):
                c.validate_resume_state({'status':'PAUSED','source_sha':c.SOURCE,'completed':completed,'exposure':exposure})
        self.assertEqual(c.command(True)[-2:],['--resume','--acknowledge-stop'])
        self.assertNotIn('--stop-after',c.command(True))
        self.assertNotIn('--max-updates',c.command(True))

    def exercise_execute(self,resume=False,completed=64,exposure=128,reason=None):
        with tempfile.TemporaryDirectory(prefix='st-world2-control-') as directory:
            out=Path(directory);run=out/'runs'/c.RUN_ID
            initial={'path':'/fixture/admission','sha256':'a'*64}
            prepared={'script_sha256':'pin','inputs_sha256':'pin','inherited_helpers_sha256':'pin',
                'registration_sha256':'pin','host':c.HOST,'resource_admission':initial}
            lease={'resource_admission':initial,'valid_from_unix':0,'valid_until_unix':4000,
                'cpu_affinity':[0,1,2,3],'gpu_uuids_by_host':{c.HOST:['gpu-a','gpu-b']}}
            values={'PREPARED.json':prepared,'RESOURCE_LEASE.json':lease,'ENVIRONMENT.json':{'fixture':True}}
            native={'status':'PAUSED','source_sha':c.SOURCE,'completed':completed,'exposure':exposure}
            if resume:
                run.mkdir(parents=True);(run/'RESULT.json').write_text(json.dumps(native))
                (out/'LAUNCH_initial.json').write_text('{}')
                (run/'checkpoints'/'synthetic-slot').mkdir(parents=True)
            writes={};locks=[];archive=[]
            def read(path):
                return native if Path(path).name=='RESULT.json' else values[Path(path).name]
            def write(path,value,immutable=False):
                self.assertTrue(immutable);writes[Path(path).name]=deepcopy(value)
            def lock(path):
                locks.append(str(path));return nullcontext(SimpleNamespace(fileno=lambda:999))
            verify=Mock()
            legacy=SimpleNamespace(modules=lambda:(lambda:{'fixture':True},None,None,None,verify,write,None),
                lock=lock,proc_start=lambda pid:1,archive_run_metadata=lambda *a:archive.append(a[2]))
            owner=SimpleNamespace()
            monitor=SimpleNamespace(lifetime_scope=lambda:nullcontext({'signal':None}),
                pidfd_open=lambda pid:None,OwnedSession=lambda child:owner)
            def spawn(*args,**kwargs):
                run.mkdir(parents=True,exist_ok=True);(run/'RESULT.json').write_text(json.dumps(native))
                self.assertEqual(kwargs['pass_fds'],(999,999))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                return SimpleNamespace(pid=123)
            approval={'initial_resource_admission':initial,'registration_sha256':'pin','allowed_actions':['resume']}
            with patch.object(c,'OUT',out),patch.object(c,'legacy',return_value=legacy),patch.object(c,'read',side_effect=read), \
                 patch.object(c,'digest',return_value='pin'),patch.object(c.time,'time',return_value=1000), \
                 patch.object(c.os,'sched_getaffinity',return_value={0,1,2,3}),patch.object(c.os,'sched_setaffinity'), \
                 patch.object(c,'gpu_record',return_value=[{'uuid':'gpu-a'},{'uuid':'gpu-b'}]), \
                 patch.object(c,'storage',return_value={}),patch.object(c,'monitor_helpers',return_value=(monitor,None)), \
                 patch.object(c.subprocess,'Popen',side_effect=spawn) as popen, \
                 patch.object(c,'monitor_training',return_value=(0,{'reason':reason})) as monitored,redirect_stdout(io.StringIO()):
                caught=None
                try:c.execute(approval,initial,resume=resume)
                except RuntimeError as exc:caught=str(exc)
            return caught,writes,popen.call_count,locks,archive,verify.call_count,monitored.call_count

    def test_actual_execute_terminal_resume_rejected_before_launch(self):
        caught,writes,spawns,locks,archive,verifies,monitors=self.exercise_execute(resume=True)
        self.assertIn('completed64 cannot be extended',caught)
        self.assertEqual(spawns,0);self.assertEqual(monitors,0);self.assertEqual(archive,[])
        self.assertEqual(writes,{})
        self.assertEqual(locks[-2:],[str(x) for x in c.GPU_LOCKS]);self.assertEqual(verifies,1)

    def test_actual_execute_resource_failure_not_erased_by_native_rc_zero(self):
        caught,writes,spawns,locks,archive,verifies,monitors=self.exercise_execute(reason='RSS_LIMIT')
        self.assertIn('Monitored failure preserved',caught)
        self.assertEqual(writes['EXIT_initial.json']['status'],'FAIL')
        self.assertEqual(writes['EXIT_initial.json']['returncode'],0)
        self.assertEqual(writes['EXIT_initial.json']['result']['completed'],64)
        self.assertEqual(writes['MONITOR_RESULT_initial.json']['reason'],'RSS_LIMIT')
        self.assertEqual(spawns,1);self.assertEqual(monitors,1);self.assertEqual(verifies,1)
        self.assertEqual(locks[-2:],[str(x) for x in c.GPU_LOCKS])
        self.assertFalse(writes['LAUNCH_initial.json']['formal'])
        self.assertEqual(writes['LAUNCH_initial.json']['max_total_scene_exposures'],128)

    def test_actual_execute_refuses_wrong_consumed_exposure(self):
        caught,writes,spawns,*_=self.exercise_execute(completed=64,exposure=64)
        self.assertIn('incorrect terminal result',caught)
        self.assertEqual(spawns,1)
        self.assertEqual(writes['EXIT_initial.json']['result']['exposure'],64)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.ControlTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(Independent)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    with open('/proc/self/io') as f:final={k:int(v) for k,v in (line.split(':') for line in f)}
    print('COST_JSON='+json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'wall_seconds':time.monotonic()-START,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'logical_read_bytes':final['rchar']-BOOT['rchar'],'logical_written_bytes':final['wchar']-BOOT['wchar'],
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_imported':'torch' in sys.modules,
        'gpu_queries':0,'real_checkpoint_or_database_reads':0,'actual_launches':0},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
