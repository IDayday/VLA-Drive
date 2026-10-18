import copy,hashlib,importlib.util,json,os,socket,sys,tempfile,time,unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock
HERE=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('control',HERE/'control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def admission():
    return dict(schema='st_teacher_channel_diagnostic_admission_v1',status='APPROVED_DIAGNOSTIC_ONLY',reviewer='independent_fixture',
        hostname=socket.gethostname(),limits=c.LIMITS.copy(),output_root=str(c.ROOT),control_root=str(c.CONTROL),
        output_bytes=c.OUTPUT_CAP,control_bytes=c.CONTROL_CAP,diagnostic={'path':str(HERE/'diagnose.py'),'sha256':c.DIAGNOSTIC_SHA},
        binding={'path':str(HERE/'BINDING.json'),'sha256':c.BINDING_SHA},automatic_retry=False,training=False,
        new_checkpoint_reservation=False,ledger_writes=False,minimum_gpu_free_bytes=c.GPU_FREE,
        cpu_affinity=sorted(os.sched_getaffinity(0))[:2],gpu_index=0,gpu_uuid='GPU-fixture',issued_at_unix=time.time()-1,
        latest_start_unix=time.time()+10,valid_until_unix=time.time()+2000,
        controller_sources=[dict(path=str(HERE/name),sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest()) for name in ('control.py','launch_once.sh')])

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.f=c.bootstrap(time.monotonic()+30)
        cls.core=cls.b.load(c.HELPER_ROOT/'full_future_metadata_preparation_v1/run_build_once.py',cls.b.CORE_SHA,'fixture_original_monitor')
        cls.s,cls.obs,cls.m=c.parent_helpers(cls.b)

    def test_all_actual_small_pins_and_parent_helpers(self):
        self.b.read_source(HERE/'diagnose.py',c.DIAGNOSTIC_SHA);self.b.read_source(HERE/'BINDING.json',c.BINDING_SHA)
        self.assertEqual(Path(self.m.__file__).resolve(),c.PARENT_RUNTIME/'tools/st_world_policy/metadata_storage.py')
        self.assertEqual(hashlib.sha256(Path(self.core.__file__).read_bytes()).hexdigest(),self.b.CORE_SHA)
        self.assertNotIn(str(c.PARENT_RUNTIME),Path(c.HERE/'launch_once.sh').read_text())

    def test_admission_exact_limits_lease_and_available_cpus(self):
        a=admission();c.validate(a,time.time(),os.sched_getaffinity(0),socket.gethostname())
        for key,value in [('limits',dict(c.LIMITS,read_bytes=321<<30)),('output_bytes',16<<20),('status','PROPOSAL'),
            ('training',True),('new_checkpoint_reservation',True),('automatic_retry',True),('gpu_uuid','unknown'),
            ('cpu_affinity',[a['cpu_affinity'][0]]),('valid_until_unix',time.time()+100),('minimum_gpu_free_bytes',1)]:
            bad=copy.deepcopy(a);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate(bad,time.time(),os.sched_getaffinity(0),socket.gethostname())

    def test_fresh_gpu_uuid_free_and_ram(self):
        a=admission()
        with mock.patch.object(c.subprocess,'check_output',return_value='GPU-fixture, NVIDIA A800 80GB PCIe, 81920, 70000\n'),\
            mock.patch('builtins.open',mock.mock_open(read_data='MemAvailable: 104857600 kB\n')):
            self.assertFalse(c.fresh_capacity(a)['exclusive_performance_claim'])
        for row in ('GPU-wrong, A800, 81920, 80000','GPU-fixture, A800, 81920, 69500'):
            with mock.patch.object(c.subprocess,'check_output',return_value=row),self.assertRaises(ValueError):c.fresh_capacity(a)
        with mock.patch.object(c.subprocess,'check_output',return_value='GPU-fixture, A800, 81920, 80000'),\
            mock.patch('builtins.open',mock.mock_open(read_data='MemAvailable: 1024 kB\n')),self.assertRaises(ValueError):c.fresh_capacity(a)

    def test_separate_output_caps_and_no_parent_claim(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(c,'ROOT',Path(tmp)/'child'),mock.patch.object(c,'CONTROL',Path(tmp)/'controls'):
            c.CONTROL.mkdir();c.census(self.m)
            self.assertFalse(c.ROOT.exists())
            c.ROOT.mkdir();(c.ROOT/'large').write_bytes(b'x'*33)
            with mock.patch.object(c,'OUTPUT_CAP',1),self.assertRaises(ValueError):c.census(self.m)
            (c.ROOT/'large').unlink();(c.ROOT/'escape').symlink_to(c.CONTROL)
            with self.assertRaises(Exception):c.census(self.m)

    def test_two_pipe_logs_compacted_but_error_preserved(self):
        output=c.compact_resources(dict(stdout='x'*(1<<20),stderr='y'*(1<<20),failure='STDERR_LIMIT',cleanup={'live_owned_remaining':0}))
        self.assertEqual(output['stderr']['bytes'],1<<20);self.assertEqual(output['failure'],'STDERR_LIMIT')
        self.assertLess(len(json.dumps(output)),12<<10)

    def test_failure_receipt_survives_child_overrun_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(c,'ROOT',Path(tmp)/'child'),mock.patch.object(c,'CONTROL',Path(tmp)/'controls'):
            a=admission();path=Path(tmp)/'admission.json';raw=json.dumps(a).encode();path.write_bytes(raw)
            args=SimpleNamespace(admission=str(path),admission_sha256=hashlib.sha256(raw).hexdigest(),child=False)
            def exceeded(*_args,**kwargs):
                c.ROOT.mkdir();(c.ROOT/'partial').write_text('preserve failed child')
                raise ValueError('Diagnostic/control output allocation exceeded')
            fakecore=SimpleNamespace(monitor=exceeded)
            s=SimpleNamespace(LOCK=Path(tmp)/'global.lock',shared_mount=lambda:None,
                acquire=lambda path,stack:self.s.acquire(Path(tmp)/Path(path).name,stack))
            with mock.patch.object(self.b,'load',return_value=fakecore),mock.patch.object(c,'parent_helpers',return_value=(s,self.obs,self.m)),\
                mock.patch.object(c,'fresh_capacity',return_value={'fixture':True}):
                self.assertEqual(c.run(args,self.b,self.f,time.monotonic()+10),2)
                self.assertEqual(json.loads((c.CONTROL/'RESULT.json').read_text())['status'],'FAILED_NO_AUTO_RETRY')
                self.assertTrue((c.ROOT/'partial').exists())
                with self.assertRaisesRegex(ValueError,'One shot only'):c.run(args,self.b,self.f,time.monotonic()+10)

    def _real_monitor(self,exitcode):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);fixture=root/'cli.py';fixture.write_text('print("first JSON")\nprint("second JSON")\nraise SystemExit('+str(exitcode)+')\n')
            runner=root/'child.py';runner.write_text(
                'import runpy,json,os\n'
                'm=runpy.run_path('+repr(str(HERE/'control.py'))+')\n'
                'def action():\n'
                ' m["original_cli"]('+repr(str(fixture))+')\n'
                ' return {"status":"fixture_complete"}\n'
                'e=m["child_envelope"](action,m["BOOT_IO"])\n'
                'print(json.dumps(e))\n'
                'raise SystemExit(2 if e["error"] else 0)\n')
            lock=self.s.acquire(root/'lock',stack);self.core.BOOT_IO=c.proc_io();self.core.START=time.monotonic()
            cpus=tuple(sorted(os.sched_getaffinity(0))[:2])
            value=self.core.monitor([sys.executable,'-B',str(runner)],runtime=str(root),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
                fd=lock.fileno(),root=root,s=self.s,m=self.m,obs=self.obs,
                budget=dict(wall_seconds=10,read_bytes=64<<20,rss_bytes=1<<30),cpu_affinity=cpus)
            self.assertEqual(value['cleanup']['live_owned_remaining'],0)
            self.assertIsNone(value['failure']);self.assertEqual(value['stdout'].count('st_full_future_child_result_v2'),1)
            self.assertIn('first JSON',value['stderr']);self.assertIn('second JSON',value['stderr'])
            self.assertEqual(value['returncode'],2 if exitcode else 0)
            self.assertEqual(value['child_accounting']['error'] is not None,bool(exitcode))
            self.assertEqual(value['read_accounting']['status'],'COMPLETE')

    def test_real_monitor_multiple_prints_and_systemexit_zero(self):self._real_monitor(0)
    def test_real_monitor_nonzero_failure_accounted_and_clean(self):self._real_monitor(3)

if __name__=='__main__':unittest.main(verbosity=2)
