"""Pure control fixtures; no metadata writer, ledger, GPU or real payload access."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('finalization_adapter_test', HERE/'run_finalize_once.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Controls(unittest.TestCase):
    def admission(self):
        now = time.time()
        return dict(schema='st_future_finalization_owned_admission_v1',
            status='APPROVED_FUTURE_FINALIZATION_ONLY', reviewer='independent-test-reviewer',
            hostname=socket.gethostname(), limits=adapter.LIMITS,
            checkpoint_receipt=adapter.CP, output_root=str(adapter.ROOT),
            runtime=str(adapter.RUNTIME), control_root=str(adapter.CONTROL_ROOT),
            control_reserved_bytes=adapter.CONTROL_CAP, issued_at_unix=now-1,
            latest_start_unix=now+20, valid_until_unix=now+4000,
            registration={'path':'fixture-storage.json','sha256':'a'*64},
            finalization_registration={'path':'fixture-finalizer.json','sha256':'b'*64},
            source_terminal={'path':'fixture-old-result.json','sha256':'c'*64})

    def test_admission_bounds(self):
        a=self.admission(); adapter.validate(a,time.time())
        for key,value in [('status','DRAFT_NOT_APPROVED'),('reviewer','/root/st_future_finalization_cost'),
                ('valid_until_unix',time.time()+3600),('issued_at_unix',float('nan')),
                ('limits',{**adapter.LIMITS,'cpu_affinity':[6,7]})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate({**a,key:value},time.time())

    def test_draft_is_rejected(self):
        a=json.loads((HERE/'DRAFT_ADMISSION.json').read_text())
        with self.assertRaises(ValueError):adapter.validate(a,time.time())

    def test_control_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='finalization-control-') as directory:
            root=Path(directory); adapter.control_once(root,'RESULT.json',{'status':'FAILED_NO_AUTO_RETRY'})
            first=(root/'RESULT.json').read_bytes()
            with self.assertRaises(FileExistsError):adapter.control_once(root,'RESULT.json',{'status':'COMPLETE'})
            self.assertEqual((root/'RESULT.json').read_bytes(),first)
            self.assertTrue((root/'RESULT.json.tmp').exists())

    def test_control_lock_nonblocking(self):
        with tempfile.TemporaryDirectory(prefix='finalization-control-') as directory:
            with adapter.control_lock(directory):
                with self.assertRaises(BlockingIOError):
                    with adapter.control_lock(directory):pass

    def run_fixture(self, *, bad_schema=False, overflow=False, fail_monitor=False, fail_publish=False):
        # Load only the reviewed pure arithmetic function, never monitor or launch code.
        core_path=adapter.LIBRARY/'full_future_metadata_preparation_v1/run_build_once.py'
        tree=ast.parse(core_path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='charge_after_reap')
        namespace={}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(core_path),'exec'),namespace)
        calls=[]
        def publish(_m,_b,_guard,path,value):
            calls.append(str(path))
            if fail_publish:raise BlockingIOError('fixture locked ledger')
            path.write_text(json.dumps(value))
        def monitor(*args,**kwargs):
            self.assertEqual(kwargs['cpu_affinity'],(0,1))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            if fail_monitor:raise RuntimeError('fixture monitor failure')
            artifact={'schema':'wrong' if bad_schema else 'st_future_metadata_finalization_complete_v1','status':'COMPLETE'}
            return {'returncode':0,'failure':None,'cleanup':{'live_owned_remaining':0},
                'read_accounting':{'parent_post_reap':{'rchar':120},'charged_parent_child_rchar_bytes':100},
                'child_accounting':{'cli_result':{'status':'GUARDED_METADATA_RETURN',
                    'registration_sha256':'a'*64,'artifact':artifact}}}
        core=SimpleNamespace(publish_guarded=publish,monitor=monitor,charge_after_reap=namespace['charge_after_reap'])
        obs=SimpleNamespace(proc_io=lambda:{'rchar':121+(adapter.LIMITS['read_bytes'] if overflow else 0)})
        with tempfile.TemporaryDirectory(prefix='finalization-control-') as directory:
            base=Path(directory); output=base/'output'; output.mkdir(); controls=base/'controls'; controls.mkdir()
            with patch.object(adapter,'ROOT',output), patch.object(adapter,'CONTROL_ROOT',controls), redirect_stdout(io.StringIO()):
                a=self.admission(); args=SimpleNamespace(admission='fixture-admission.json',admission_sha256='d'*64)
                code=adapter.run_parent(args,a,{}, {},None,obs,None,SimpleNamespace(fcntl=adapter.fcntl),
                    core,lambda:1<<50,work_deadline=time.monotonic()+30)
                result=json.loads((controls/'RESULT.json').read_text())
                marker=output/'CONTROL_FINALIZATION_RESULT.json'
                interim=json.loads(marker.read_text()) if marker.exists() else None
                return code,result,interim,calls

    def test_success_and_once_only_read_charge(self):
        code,result,interim,_=self.run_fixture()
        self.assertEqual(code,0); self.assertEqual(result['status'],'FINALIZATION_COMPLETE')
        self.assertEqual(result['charged_rchar_bytes'],101)
        self.assertEqual(result['cleanup']['live_owned_remaining'],0)
        self.assertEqual(interim['status'],'FINALIZATION_RETURNED_PENDING_TERMINAL_ACCOUNTING')
        self.assertEqual(result['source_terminal']['sha256'],'c'*64)

    def test_wrong_artifact_cannot_complete(self):
        code,result,_,_=self.run_fixture(bad_schema=True)
        self.assertEqual(code,2); self.assertEqual(result['status'],'FAILED_NO_AUTO_RETRY')

    def test_terminal_read_budget(self):
        code,result,_,_=self.run_fixture(overflow=True)
        self.assertEqual(code,2); self.assertGreater(result['charged_rchar_bytes'],adapter.LIMITS['read_bytes'])

    def test_monitor_exception_retains_unknown_cleanup(self):
        code,result,_,_=self.run_fixture(fail_monitor=True)
        self.assertEqual(code,2); self.assertEqual(result['cleanup']['status'],'UNKNOWN_MONITOR_EXCEPTION')
        self.assertEqual(result['accounting'],'CONSERVATIVE_FULL_ALLOWANCE_ON_EXCEPTION')

    def test_busy_ledger_fallback_does_not_retry(self):
        code,result,_,calls=self.run_fixture(fail_publish=True)
        self.assertEqual(code,2); self.assertEqual(len(calls),1)
        self.assertEqual(result['cleanup']['status'],'NOT_STARTED')
        self.assertFalse(result['ledger_result_write_confirmed'])


if __name__=='__main__': unittest.main(verbosity=2)
