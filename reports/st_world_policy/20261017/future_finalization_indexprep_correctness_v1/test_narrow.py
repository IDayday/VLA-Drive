"""Independent adapter checks; small synthetic metadata only, never live inputs."""
import copy,hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tests.st_world.test_finalize_future_metadata import make_case
from tools.st_world_policy import finalize_future_metadata as final
from tools.st_world_policy import prepare_future_metadata as f
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('owner_index_tests',HERE.parent/'future_finalized_index_route_v1/test_index_route.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);route=owner.route

class IndependentRouteTests(unittest.TestCase):
 def test_reviewed_helper_and_runtime_callback_pins(self):
  helper=route.load_helper();source=json.loads((HERE.parent/'future_finalized_index_route_v1/TRAIN_INDEX_RUNTIME_SOURCE_PINS.json').read_bytes())
  self.assertEqual(source['runtime_commit'],'3df052ea422c4ba042ef3be33dea19438c5702dc')
  for name,pin in source['files'].items():
   self.assertEqual(Path(pin['path']),route.FINALIZER_RUNTIME/name);helper.read(pin)
  pins=helper.callback_pins(source,'tools.st_world_policy.index_teacher_metadata')
  self.assertEqual(len(pins),3)
  damaged=copy.deepcopy(source);damaged['files']['starVLA/dataloader/st_world_metadata_index.py']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'Pinned input changed'):helper.callback_pins(damaged,'tools.st_world_policy.index_teacher_metadata')
 def test_more_terminal_failures_rejected_before_shared_validator(self):
  for key in ['pending','source','inner_cleanup','read_accounting','returncode','failure','auto_retry','science','self_review']:
   with self.subTest(key=key):
    fixture=list(owner.RouteTests().fixture());done,result,admission=fixture[2],fixture[4],fixture[6]
    if key=='pending':result['status']='FINALIZATION_RETURNED_PENDING_TERMINAL_ACCOUNTING'
    elif key=='source':admission['source_terminal']=owner.pin('other-source')
    elif key=='inner_cleanup':result['resources']['cleanup']['live_owned_remaining']=2
    elif key=='read_accounting':result['resources']['read_accounting']['status']='INCOMPLETE'
    elif key=='returncode':result['resources']['returncode']=1
    elif key=='failure':result['resources']['failure']='LIFECYCLE_FAILURE'
    elif key=='auto_retry':result['automatic_retry']=True
    elif key=='science':result['scientific_result']=True
    elif key=='self_review':admission['reviewer']='/root/st_future_finalization_cost'
    with patch.object(route,'strict_finalized_receipt',side_effect=AssertionError('invalid terminal reached validator')):
     with self.assertRaises(ValueError):route.bind_finalized_train(*fixture[:6])
 def test_missing_new_runtime_pins_fail_before_output(self):
  args=SimpleNamespace(check_only=False,policy='/synthetic-policy',policy_sha256='a'*64,ledger_sha256='b'*64,output='/synthetic-unused-output',train_index_source_pins=None,train_index_source_pins_sha256=None)
  with patch.object(route,'completion_inputs',return_value=({}, {'status':'READY_FOR_INDEX_CONTROL_PREPARATION'})), patch.object(Path,'mkdir',side_effect=AssertionError('no output before runtime pins')):
   with self.assertRaisesRegex(ValueError,'New immutable TRAIN'):route.prepare(args)
 def test_waiting_receipt_creates_no_proposal(self):
  state={'status':'WAITING_ACTUAL_COMPLETE'}
  with patch.object(route,'completion_inputs',return_value=({},state)), patch.object(Path,'mkdir',side_effect=AssertionError('must not mkdir')):
   self.assertIs(route.prepare(SimpleNamespace(check_only=False)),state)
 def test_metadata_baseline_mutation_rejected_and_budget_not_written(self):
  helper=SimpleNamespace(MiB=1<<20,GiB=1<<30)
  policy=json.loads((HERE.parent/'future_finalization_actual_proposal_v1/DRAFT_PROPOSED_POLICY.json').read_bytes())
  self.assertEqual(route.planned_metadata_bytes(helper,policy,2176<<20),7701<<20)
  bad=copy.deepcopy(policy);root=next(iter(bad['metadata_package']['baseline_claims']))
  next(c for c in bad['claims'] if c['root']==root)['output_bytes']+=1
  with self.assertRaisesRegex(ValueError,'baseline claim changed'):route.planned_metadata_bytes(helper,bad,2176<<20)

class RealSharedValidatorAdapterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='st-indexprep-independent-',dir='/tmp')
  cls.mp=pytest.MonkeyPatch();cls.oldcwd=Path.cwd();os.chdir(route.FINALIZER_RUNTIME)
  cls.case=make_case(Path(cls.temp.name),cls.mp,full=True)
  cls.done=final.run(cls.case.args)
  cls.helper=route.load_helper()
  cls.pin=staticmethod(lambda path:{'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})
 @classmethod
 def tearDownClass(cls):
  cls.mp.undo();os.chdir(cls.oldcwd);cls.temp.cleanup()
 def test_actual_shared_validator_accepts_original_ordered_two_scene_receipt(self):
  with patch.dict(route.COUNTS,train=2):
   reg=route.strict_finalized_receipt(self.helper,self.done,self.pin(self.case.output/'FINALIZATION_COMPLETE.json'))
  self.assertEqual(reg,self.case.reg)
  manifest=json.loads((self.case.output/'TEACHER_REQUESTS.json').read_bytes())
  self.assertEqual([v['token'] for v in manifest['scenes']],[v['token'] for v in self.case.plan['selected_rows']])
  self.assertEqual([v['log'] for v in manifest['scenes']],[v['log'] for v in self.case.plan['selected_rows']])
  self.assertEqual(manifest['times_s'],[0.,.5,1.,2.,4.]);self.assertEqual(self.done['scenes'],2)
 def test_actual_shared_validator_rejects_changed_failed_source(self):
  path=self.case.terminal;old=path.read_bytes()
  try:
   broken=json.loads(old);broken['status']='COMPLETE';path.write_text(json.dumps(broken))
   with patch.dict(route.COUNTS,train=2):
    with self.assertRaises(ValueError):route.strict_finalized_receipt(self.helper,self.done,self.pin(self.case.output/'FINALIZATION_COMPLETE.json'))
  finally:path.write_bytes(old)
 def test_actual_shared_validator_rejects_self_consistent_wrong_time(self):
  changed=copy.deepcopy(self.done);changed['target_times_s']=[0.,1.,2.,3.,4.];changed.pop('identity');changed=f.signed(changed)
  with patch.dict(route.COUNTS,train=2):
   with self.assertRaisesRegex(ValueError,'population/provenance'):route.strict_finalized_receipt(self.helper,changed,self.pin(self.case.output/'FINALIZATION_COMPLETE.json'))
 def test_actual_bind_route_calls_unmocked_validator(self):
  fixture=list(owner.RouteTests().fixture());done=copy.deepcopy(self.done);base=Path(self.temp.name)
  storage_path=base/'NEW_INDEX_TEST_STORAGE.json';admission_path=base/'NEW_INDEX_TEST_ADMISSION.json'
  storage=fixture[7];storage.update(output_root=str(self.case.output));storage['command']=[self.helper.PYTHON,'-m','tools.st_world_policy.finalize_future_metadata','finalize-existing','--registration',self.case.args.registration,'--registration-sha256',self.case.args.registration_sha256,'--output',str(self.case.output)]
  storage_path.write_text(json.dumps(storage));storage_pin=self.pin(storage_path)
  admission=fixture[6];admission.update(registration=storage_pin,finalization_registration=done['finalization_registration'],source_terminal=done['source_terminal'],output_root=str(self.case.output),control_root=str(base/'controls'))
  admission_path.write_text(json.dumps(admission));result=fixture[4];result.update(registration=storage_pin,admission=self.pin(admission_path),finalization_registration=done['finalization_registration'],source_terminal=done['source_terminal'],ledger_result=str(self.case.output/'CONTROL_FINALIZATION_RESULT.json'))
  result['cli_result'].update(registration_sha256=storage_pin['sha256'],artifact=done)
  with patch.dict(route.COUNTS,train=2),patch.object(route,'FINALIZED_TRAIN',self.case.output),patch.object(route,'FINALIZATION_RESULT',base/'controls/RESULT.json'):
   bound=route.bind_finalized_train(self.helper,done['source_plan'],done,self.pin(self.case.output/'FINALIZATION_COMPLETE.json'),result,owner.pin(str(base/'controls/RESULT.json')))
  self.assertEqual(bound['source_registration'],self.case.reg['source_registration'])
  self.assertEqual(json.loads(self.case.terminal.read_bytes())['status'],'FAILED_NO_AUTO_RETRY')
