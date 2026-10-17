"""Narrow synthetic controller/worker-boundary checks; no real dispatch or ledger."""
import copy,hashlib,importlib.util,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
owner=load(HERE.parent/'future_index_sequential_control_v1/test_increment.py','owner_increment')
m=owner.m;prep=owner.prep
route_tests=load(HERE.parent/'future_finalized_index_route_v1/test_index_route.py','owner_binding_fixtures')

def pin(path):return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

class NarrowController(unittest.TestCase):
 def test_train_wrong_root_schema_and_source_flags(self):
  for key in ['old_root','schema','request_path','raw_read','not_reused']:
   with self.subTest(key=key):
    p,reg,done=owner.namespace['preparation']('train')
    if key=='old_root':p['source_inputs']['roles']['train']['future_complete']['path']='/old/COMPLETE.json'
    elif key=='schema':done['schema']='st_future_metadata_complete_v1'
    elif key=='request_path':p['actual_completion_bindings']['train']['requests']='/old/TEACHER_REQUESTS.json'
    elif key=='raw_read':done['raw_log_deserialized']=True
    else:done['source_shards_reused']=False
    with self.assertRaises(ValueError):m.validate_preparation(p,'train',reg,done)
 def test_runtime_wrong_schema_path_commit(self):
  good={'schema':'st_future_finalized_index_runtime_pins_v1','runtime':str(m.FINALIZER_RUNTIME),'runtime_commit':m.FINALIZER_COMMIT}
  for key in good:
   broken=dict(good);broken[key]='wrong'
   with self.assertRaises(ValueError):m.runtime_binding('train',broken)
 def exercise_binding(self,change=None):
  with tempfile.TemporaryDirectory(prefix='st-index-bind-',dir='/tmp') as d:
   folder=Path(d);fixture=list(route_tests.RouteTests().fixture());helper,plan,done,done_pin,result,result_pin,admission,storage=fixture
   storage['command'][0]=m.core.PYTHON
   storage_path=folder/'storage.json';storage_path.write_text(json.dumps(storage));storage_pin=pin(storage_path)
   admission['registration']=storage_pin;admission_path=folder/'admission.json';admission_path.write_text(json.dumps(admission))
   result.update(admission=pin(admission_path),registration=storage_pin);result['cli_result']['registration_sha256']=storage_pin['sha256']
   p={'source_inputs':{'roles':{'train':{'plan':plan,'future_complete':done_pin,'original_source_code_root':'/synthetic-original-runtime'}}},'actual_completion_bindings':{'train':{'build_result':result_pin}}}
   if change=='terminal':result['status']='FAILED_NO_AUTO_RETRY'
   elif change=='cleanup':result['cleanup']['live_owned_remaining']=1
   elif change=='source_root':p['source_inputs']['roles']['train']['original_source_code_root']='/changed-original'
   modules=[];strict=Mock(return_value={'source_code_root':'/synthetic-original-runtime'})
   original_pinned=m.core.pinned
   def module_factory(name):
    module=ModuleType(name);modules.append(module);return module
   def pinned(value,decode=True):
    if value==result_pin:
     modules[-1].strict_finalized_receipt=strict
     return result
    return original_pinned(value,decode=decode)
   with patch.object(m,'ModuleType',module_factory),patch.object(m.core,'pinned',pinned):m.validate_finalized_input(p,done)
   self.assertEqual(strict.call_count,1)
 def test_real_binder_routing_reuses_strict_validator(self):self.exercise_binding()
 def test_real_binder_wrong_terminal_cleanup_and_source(self):
  for key in ['terminal','cleanup','source_root']:
   with self.subTest(key=key):
    with self.assertRaises(ValueError):self.exercise_binding(key)
 def test_preparer_wrong_hash_rejected_before_receipt_use(self):
  with patch.object(m,'PREPARER_SHA','0'*64):
   with self.assertRaises(ValueError):m.validate_finalized_input({}, {})

class PreparationWrapper(unittest.TestCase):
 def run_fixture(self,change=None):
  with tempfile.TemporaryDirectory(prefix='st-index-prep-wrapper-',dir='/tmp') as d:
   base=Path(d);output=base/'proposal';control=base/'proposal_controls'
   args=SimpleNamespace(wrapper_sha256=hashlib.sha256(Path(prep.__file__).read_bytes()).hexdigest(),preparer=str(m.PREPARER_PATH),preparer_sha256=m.PREPARER_SHA,output=str(output),child=False)
   finite=prep.load(prep.ADAPTER,prep.ADAPTER_SHA,'actual_finite_fixture')
   resources={'returncode':0,'failure':None,'cleanup':{'live_owned_remaining':0},'read_accounting':{'fixture':True},'child_accounting':{'cli_result':{'status':'PREPARED_NOT_MIGRATED_NOT_AUTHORIZED'}}}
   charge=100;calls=[]
   if change=='read':charge=prep.LIMITS['read_bytes']+1
   elif change=='cleanup':resources['cleanup']['live_owned_remaining']=1
   elif change=='result':resources['child_accounting']['cli_result']['status']='WAITING_ACTUAL_COMPLETE'
   def monitor(argv,**kwargs):
    calls.append((argv,kwargs));self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kwargs['cpu_affinity'],(0,1));self.assertEqual(argv.count('--child'),1)
    if change=='monitor':raise RuntimeError('synthetic monitor failure')
    if change=='disk':
     output.mkdir();(output/'overflow.txt').write_bytes(b'x'*(1<<20))
    return copy.deepcopy(resources)
   core=SimpleNamespace(monitor=monitor,charge_after_reap=lambda *_:charge)
   original_load=prep.load
   def selected_load(path,sha,name):
    if name=='reviewed_index_prepare_monitor':return core
    return original_load(path,sha,name)
   if change=='existing':control.mkdir()
   with patch.object(prep,'ALLOWED',base),patch.object(prep,'load',selected_load),patch.object(sys,'argv',['prepare_once.py','--fixture']) ,patch.object(prep,'START',time.monotonic()):
    if change in ['existing','disk']:
     with self.assertRaises(ValueError):prep.run(args,finite)
     return calls,None
    code=prep.run(args,finite);receipt=json.loads((control/'RESULT.json').read_text())
    self.assertEqual(code,0 if change is None else 2)
   return calls,receipt
 def test_one_cpu_child_then_complete(self):
  calls,r=self.run_fixture();self.assertEqual(len(calls),1);self.assertEqual(r['status'],'PREPARATION_COMPLETE');self.assertFalse(r['ledger_written']);self.assertFalse(r['index_built'])
 def test_budget_cleanup_waiting_fail_closed(self):
  for change in ['read','cleanup','result']:
   with self.subTest(change=change):
    calls,r=self.run_fixture(change);self.assertEqual(len(calls),1);self.assertEqual(r['status'],'FAILED_NO_AUTO_RETRY');self.assertFalse(r['automatic_retry'])
 def test_monitor_exception_never_claims_cleanup_zero(self):
  calls,r=self.run_fixture('monitor');self.assertEqual(len(calls),1);self.assertEqual(r['cleanup'],'UNKNOWN_OR_NOT_STARTED');self.assertEqual(r['charged_rchar_bytes'],prep.LIMITS['read_bytes'])
 def test_spent_output_and_disk_cap_block(self):
  calls,_=self.run_fixture('existing');self.assertEqual(len(calls),0)
  calls,_=self.run_fixture('disk');self.assertEqual(len(calls),1)
