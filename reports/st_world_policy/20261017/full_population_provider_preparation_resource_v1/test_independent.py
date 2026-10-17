"""Independent thin-control fixtures; no actual metadata, PCA or ledger reads."""
import time
START=time.monotonic()
with open('/proc/self/io') as f: BOOT={k:int(v) for k,v in (x.split(':') for x in f)}
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import socket
import sys
import tempfile
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/full_population_provider_preparation_v1')
EXPECTED='9ef4b038705613d38fee96a26ff5798d21ad7c4daa50009243685bd6a940ce24'
raw=(SOURCE/'prepare_once.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
spec=importlib.util.spec_from_file_location('independent_subject',SOURCE/'prepare_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PLAN=json.loads((SOURCE/'PLAN.json').read_bytes())
fixture_written=0

class Reads:
    def raw(self,path,sha=None):
        p=Path(path)
        if p.is_relative_to(SOURCE):
            b=p.read_bytes()
            if sha is not None and hashlib.sha256(b).hexdigest()!=sha:raise ValueError('pin')
            return b
        if p.is_relative_to(m.ROOT):return p.read_bytes()
        return b''
    def document(self,pin):
        if pin['path']==str(SOURCE/'PLAN.json'):return copy.deepcopy(PLAN)
        if pin['path']=='/fixture/admission':return admission()
        return {'status':'COMPLETE','stage':'fit'}

def admission():
    now=time.time()
    return dict(schema='st_full_population_provider_preparation_admission_v1',
      status='APPROVED_PROVIDER_METADATA_PREPARATION_ONLY',reviewer='/root/st_owned_descendants_review_v2',
      hostname=socket.gethostname(),limits=m.LIMITS,cpu_affinity=[0,1],output_root=str(m.ROOT),
      max_output_bytes=m.OUTPUT_LIMIT,control_reservation=PLAN['control_reservation'],gpu=False,
      automatic_retry=False,ledger_written=False,checkpoint_reservation_created=False,
      plan={'path':str(SOURCE/'PLAN.json'),'sha256':m.PLAN_SHA},issued_at_unix=now-1,
      latest_start_unix=now+100,valid_until_unix=now+2000,
      controller_sources=[{'path':str(SOURCE/n),'sha256':hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()}
                          for n in ('prepare_once.py','launch_once.sh')])

def write_json(path,obj,**kw):
    global fixture_written
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj).encode();fixture_written+=len(raw)
    assert fixture_written<8<<20
    path.write_bytes(raw)

class Review(unittest.TestCase):
    def test_fixed_source_commands_reservation(self):
        self.assertEqual(hashlib.sha256((SOURCE/'PLAN.json').read_bytes()).hexdigest(),m.PLAN_SHA)
        calls=m.commands(PLAN)
        self.assertEqual([x[0] for x in calls],['tools.st_world_policy.qualify_indexed_provider','tools.st_world_policy.teacher_provider'])
        self.assertEqual(calls[0][1][0],'prepare-profile-migration')
        self.assertIn('--profile-full-population',calls[1][1])
        self.assertIn(PLAN['data_manifest']['path'],calls[1][1])
        r=PLAN['control_reservation']
        self.assertEqual(r['total_reserved_bytes'],r['prior_reserved_bytes']+(32<<20))
        self.assertEqual(r['remaining_bytes'],r['metadata_envelope_bytes']-r['total_reserved_bytes'])
        self.assertEqual(r['original_admin_reservation_bytes'],128<<20)
        self.assertFalse(r['typed_claims_changed']);self.assertFalse(r['ledger_written'])
        self.assertIn('timeout --signal=TERM --kill-after=40s 1500s taskset -c 0,1',(SOURCE/'launch_once.sh').read_text())
        self.assertEqual(m.LIMITS,dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30))

    def test_admission_negative_bindings(self):
        good=admission()
        with patch.object(m.os,'sched_getaffinity',return_value={0,1}):
            m.validate(good,PLAN,None,Reads(),time.time())
            cases=[('reviewer','/root/st_indexprep_boundary_fix'),('gpu',True),('ledger_written',True),
                   ('automatic_retry',True),('checkpoint_reservation_created',True),('latest_start_unix',0),
                   ('valid_until_unix',time.time()+10),('issued_at_unix',float('nan'))]
            for key,value in cases:
                with self.subTest(key=key):
                    bad=copy.deepcopy(good);bad[key]=value
                    with self.assertRaises(ValueError):m.validate(bad,PLAN,None,Reads(),time.time())
            bad=copy.deepcopy(good);bad['controller_sources'][0]['sha256']=None
            with self.assertRaises(ValueError):m.validate(bad,PLAN,None,Reads(),time.time())
            bad=copy.deepcopy(good);bad['control_reservation']['total_reserved_bytes']+=1
            with self.assertRaises(ValueError):m.validate(bad,PLAN,None,Reads(),time.time())

    def test_real_prints_form_one_monitor_envelope(self):
        bootstrap=m.bootstrap()
        core=bootstrap.load(m.LIB/'full_future_metadata_preparation_v1/run_build_once.py',bootstrap.CORE_SHA,'independent_core_decode')
        out=io.StringIO();err=io.StringIO()
        def cli(*args,**kwargs):
            print(json.dumps({'original_cli':args[0]}));print('diagnostic',file=sys.stderr)
        final={'schema':'st_full_future_child_result_v2','cli_result':{},'error':None,
          'counters':{'proc_self_io_baseline':{'rchar':1},'proc_self_io_final':{'rchar':3},'proc_self_io_delta':{'rchar':2}}}
        with redirect_stdout(out),redirect_stderr(err),patch.object(m.runpy,'run_module',side_effect=cli):
            for module,argv in m.commands(PLAN):m.original_cli(module,argv)
            print(json.dumps(final))
        self.assertEqual(core.decode_final(out.getvalue()),final)
        self.assertEqual(err.getvalue().count('original_cli'),2)
        with self.assertRaises(ValueError):core.decode_final('{"original":1}\n'+out.getvalue())

    def test_fifo_bootstrap_fails_without_wait(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            fifo=Path(directory)/'source';os.mkfifo(fifo)
            start=time.monotonic()
            with patch.object(m,'BOOTSTRAP',fifo),self.assertRaises(ValueError):m.bootstrap()
            self.assertLess(time.monotonic()-start,.5)

    def child(self,kind):
        global fixture_written
        with tempfile.TemporaryDirectory(dir=HERE) as directory,ExitStack() as stack:
            root=Path(directory);calls=[]
            stack.enter_context(patch.object(m,'ROOT',root));stack.enter_context(patch.object(m,'CONTROL',root/'controls'))
            # Only the two original CLI bodies and tensor-importing atomic writer
            # are replaced. The actual thin child loop, accounting and census run.
            sys.path.insert(0,str(m.RUNTIME))
            from tools.st_world_policy import metadata_storage
            obs=ModuleType('observed_cli')
            def proc_io():
                with open('/proc/self/io') as f:return {k:int(v) for k,v in (x.split(':') for x in f)}
            obs.proc_io=proc_io
            atom=ModuleType('starVLA.dataloader.st_world_targets');atom.atomic_json=write_json
            stack.enter_context(patch.dict(sys.modules,{'supervise_once':ModuleType('supervise_once'),
                'observed_cli':obs,'starVLA.dataloader.st_world_targets':atom}))
            stack.enter_context(patch.object(m,'validate',return_value=None))
            stack.enter_context(patch.object(m.os,'getpgrp',side_effect=os.getpid))
            fd=os.open(root,os.O_RDONLY);stack.callback(os.close,fd)
            stack.enter_context(patch.dict(os.environ,{'ST_PROBE_SUPERVISOR_LOCK_FD':str(fd),'CUDA_VISIBLE_DEVICES':''}))
            deadline=time.monotonic()+5
            if kind=='deadline':deadline=time.monotonic()+.03
            if kind=='output':stack.enter_context(patch.object(m,'OUTPUT_LIMIT',2<<20))
            def cli(module,argv):
                global fixture_written
                calls.append(module)
                print('original result',file=sys.stderr)
                if len(calls)==1:
                    write_json(root/'SOURCE_MIGRATION.json',{'stage':1})
                    if kind=='failure':raise RuntimeError('fixture first stage failed')
                    if kind=='deadline':time.sleep(.06)
                    if kind=='output':
                        data=b'x'*((1<<20)+1);fixture_written+=len(data);(root/'retained.tmp').write_bytes(data)
                else:
                    for n in ('PROVIDER.json','configured/training_config.json','configured/PROVIDER_CHANNEL_SCOPE.json','configured/PROVIDER_CONFIG_LINEAGE.json'):
                        write_json(root/n,{'stage':2})
            stack.enter_context(patch.object(m,'original_cli',side_effect=cli))
            core=SimpleNamespace();b=SimpleNamespace(CORE_SHA='fixture',load=lambda *a:core,read_source=lambda *a:None)
            stdout=io.StringIO();stderr=io.StringIO()
            with redirect_stdout(stdout),redirect_stderr(stderr):
                rc=m.run(SimpleNamespace(admission='/fixture/admission',admission_sha256='fixture',child=True),b,None,SimpleNamespace(Reads=Reads),deadline)
            result=json.loads(stdout.getvalue())
            self.assertEqual(result['schema'],'st_full_future_child_result_v2')
            self.assertTrue((root/'SOURCE_MIGRATION.json').exists())
            self.assertFalse('torch' in sys.modules)
            return rc,result,calls

    def test_child_success_exactly_two_stages(self):
        rc,result,calls=self.child('success');self.assertEqual(rc,0);self.assertEqual(len(calls),2)
        self.assertEqual(result['cli_result']['status'],'PROVIDER_PREPARATION_COMPLETE')
        self.assertEqual(len(result['cli_result']['outputs']),5)
        self.assertFalse(result['cli_result']['training_started'])

    def test_first_stage_failure_preserves_partial_no_second_stage(self):
        rc,result,calls=self.child('failure');self.assertEqual(rc,2);self.assertEqual(len(calls),1)
        self.assertIn('first stage failed',result['error']['message'])

    def test_shared_deadline_stops_before_second_stage(self):
        rc,result,calls=self.child('deadline');self.assertEqual(rc,2);self.assertEqual(len(calls),1)
        self.assertEqual(result['error']['type'],'TimeoutError')

    def test_output_census_includes_retained_partial_and_terminal_reserve(self):
        rc,result,calls=self.child('output');self.assertEqual(rc,2);self.assertEqual(len(calls),1)
        self.assertIn('output slice exceeded',result['error']['message'])

if __name__=='__main__':
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Review))
    with open('/proc/self/io') as f:final={k:int(v) for k,v in (x.split(':') for x in f)}
    print('COST_JSON='+json.dumps({'tests':outcome.testsRun,'failures':len(outcome.failures),'errors':len(outcome.errors),
      'wall_seconds':time.monotonic()-START,'rchar_delta':final['rchar']-BOOT['rchar'],'rchar_final':final['rchar'],
      'fixture_written_bytes':fixture_written,'cpu_affinity':sorted(os.sched_getaffinity(0)),
      'ru_maxrss_bytes_diagnostic_only':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'torch_imported': 'torch' in sys.modules,'real_payload_or_ledger_read':False}))
    raise SystemExit(not outcome.wasSuccessful())
