"""Pure control fixtures; no metadata writer, ledger, GPU or real payload access."""
import ast
from contextlib import contextmanager, redirect_stdout
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
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

    def test_actual_draft_allows_real_writer_lock(self):
        # Exact draft allowlist and real guard; only the output root is relocated.
        # Compile the real stdlib lock functions to avoid Torch's unrelated import
        # reads under this control-only 32 MiB budget. The path expression itself
        # comes from the frozen finalizer.run AST, never a test-created filename.
        sys.path.insert(0,str(adapter.RUNTIME))
        from tools.st_world_policy import metadata_storage as storage
        reg=json.loads((HERE/'DRAFT_STORAGE_REGISTRATION.json').read_text())
        writer=ast.parse((adapter.RUNTIME/'tools/st_world_policy/finalize_future_metadata.py').read_text())
        run=next(n for n in writer.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        lock=next(n.context_expr for w in ast.walk(run) if isinstance(w,ast.With) for n in w.items
            if isinstance(n.context_expr,ast.Call) and isinstance(n.context_expr.func,ast.Attribute)
            and n.context_expr.func.attr=='metadata_lock')
        preparation=adapter.RUNTIME/'tools/st_world_policy/prepare_future_metadata.py'
        source=ast.parse(preparation.read_text())
        names={'metadata_path','metadata_parent','metadata_lock'}
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
        with tempfile.TemporaryDirectory(prefix='finalization-lock-draft-') as directory:
            output=Path(directory); reg['output_root']=str(output)
            reservation=SimpleNamespace(_total=lambda claim:2<<30,
                _check_disk=lambda *args,**kwargs:None,core={'free_reserve_bytes':0})
            guard=storage.MetadataWriteContext(reservation,{},output,{},reg,'unused-fixture-pin','0'*64)
            namespace=dict(Path=Path,os=os,stat=stat,fcntl=fcntl,contextmanager=contextmanager,
                _storage_check=guard.check)
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(preparation),'exec'),namespace)
            def check_path(path,missing=False):
                namespace['metadata_path'](path,missing=missing); guard.check(path=path)
            budget=SimpleNamespace(scan=guard.check,path=check_path)
            with patch.object(storage,'_checkpoint_reserve',return_value=0), \
                    patch.object(storage,'process_tree_rss',return_value=32<<20):
                context=eval(compile(ast.Expression(lock),'<actual finalizer lock expression>','eval'),
                    {'f':SimpleNamespace(metadata_lock=namespace['metadata_lock']),'output':output,'budget':budget})
                with context:
                    observed=guard.check(before_commit=True)
                    self.assertGreaterEqual(observed['charged_disk_bytes'],0)
                    self.assertEqual([p.name for p in output.iterdir()],['FINALIZE.lock'])

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
