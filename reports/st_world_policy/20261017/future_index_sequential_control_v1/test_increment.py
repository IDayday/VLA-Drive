"""Narrow stdlib source/control checks; no dispatch, ledger or real payload."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
m=load(HERE.parent/'full_future_index_execution_v1/run_index_once.py','new_index_controller')
prep=load(HERE/'prepare_once.py','bounded_index_preparation')
fixture_path=HERE.parent/'full_future_index_execution_v1/test_run_index_once.py'
tree=ast.parse(fixture_path.read_text());namespace={'m':m}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('preparation','index_result')],type_ignores=[]),str(fixture_path),'exec'),namespace)


class Increment(unittest.TestCase):
    def test_exact_train_new_path_and_legacy_dev(self):
        for role in ('dev','train'):
            p,reg,done=namespace['preparation'](role);m.validate_preparation(p,role,reg,done)
            suffix='train_finalized_v1' if role=='train' else 'dev_v1'
            self.assertEqual(m.expected_command(role,'a'*64)[5],str(m.BASE/('caches/full_future_metadata_'+suffix+'/TEACHER_REQUESTS.json')))
        p,reg,done=namespace['preparation']('train');done['schema']='st_future_metadata_complete_v1'
        with self.assertRaises(ValueError):m.validate_preparation(p,'train',reg,done)

    def test_train_runtime_exact_commit_and_dev_preserved(self):
        source={'schema':'st_future_finalized_index_runtime_pins_v1','runtime':str(m.FINALIZER_RUNTIME),'runtime_commit':m.FINALIZER_COMMIT}
        self.assertEqual(m.runtime_binding('train',source),(str(m.FINALIZER_RUNTIME),m.FINALIZER_COMMIT))
        source['runtime_commit']='838dd21f'
        with self.assertRaises(ValueError):m.runtime_binding('train',source)
        runtime,commit=m.core.RUNTIMES['dev']
        self.assertEqual(m.runtime_binding('dev',{'runtime':runtime,'source_commit':commit}),(runtime,commit))

    def test_new_request_dependency_validated_in_index_complete(self):
        p,contract,done,result=namespace['index_result']('train')
        self.assertEqual(m.validate_result('train',p,result,done,contract),done['identity'])
        done['dependencies']['requests']['path']=str(m.BASE/'caches/full_future_metadata_train_v1/TEACHER_REQUESTS.json')
        done['identity']=m.digest({k:v for k,v in done.items() if k!='identity'})
        result['artifact']={k:done[k] for k in m.FIELDS}
        with self.assertRaises(ValueError):m.validate_result('train',p,result,done,contract)

    def test_frozen_monitor_and_strict_binding_sources_are_exact(self):
        self.assertEqual(hashlib.sha256(m.CORE_PATH.read_bytes()).hexdigest(),m.CORE_SHA)
        self.assertEqual(hashlib.sha256(m.PREPARER_PATH.read_bytes()).hexdigest(),m.PREPARER_SHA)
        for name,sha in m.MONITORS.items():
            path=m.FINALIZER_RUNTIME/'reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1'/name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sha)
        source=ast.parse((HERE.parent/'full_future_index_execution_v1/run_index_once.py').read_text())
        binding=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='validate_finalized_input')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bind_finalized_train' for n in ast.walk(binding)))
        self.assertNotIn('torch',sys.modules)

    def test_preparation_source_pin_and_bounds(self):
        self.assertEqual(prep.LIMITS,{'wall_seconds':120,'cleanup_seconds':40,'read_bytes':384<<20,'rss_bytes':1<<30})
        self.assertEqual(hashlib.sha256(prep.ADAPTER.read_bytes()).hexdigest(),prep.ADAPTER_SHA)
        with tempfile.TemporaryDirectory(prefix='index-source-control-') as directory:
            path=Path(directory)/'fixture.py';path.write_bytes(b'x=1\n')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(prep.read_source(path,sha),b'x=1\n')
            with self.assertRaises(ValueError):prep.read_source(path,'0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
