"""Actual immutable control drafts plus in-memory/small-temp rejection tests only."""
import copy,hashlib,importlib.util,json,os,re,socket,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
BASE=Path('/mnt/project/VLA-Drive-st-future-finalization-fix-20261017/reports/st_world_policy/20261017/future_finalization_actual_proposal_v1')
spec=importlib.util.spec_from_file_location('owner_finalization_controls',BASE/'test_control_adapter.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);a=owner.adapter

def read(name):return json.loads((BASE/name).read_bytes())

class ActualDraftWiring(unittest.TestCase):
 def test_all_actual_draft_pins_and_worker_fields(self):
  d=read('DRAFT_ADMISSION.json');storage=read('DRAFT_STORAGE_REGISTRATION.json');internal=read('DRAFT_FINALIZATION_REGISTRATION.json');cmd=read('DRAFT_COMMAND.json')
  for pin in [d['registration'],d['policy'],d['finalization_registration'],*d['controller_sources'],*internal['implementation']]:
   p=Path(pin['path']);self.assertLess(p.stat().st_size,1<<20);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),pin['sha256'])
  self.assertEqual(d['registration']['sha256'],'1b42bc7bd3294444e111e0b4049fa9ab29dd884780663d720eaedafa8774705c')
  self.assertEqual(d['policy']['sha256'],'266a253d266fecf20057b2a79988c28cd3de1fbcb7d9293a7daa90b21300cfdb')
  self.assertEqual(storage['command'],cmd['finalizer_worker_argv']);self.assertEqual(storage['source_pins'],internal['implementation'])
  self.assertEqual(cmd['controller_argv'][-1],hashlib.sha256((BASE/'DRAFT_ADMISSION.json').read_bytes()).hexdigest())
  self.assertFalse(cmd['direct_worker_invocation_authorized']);self.assertEqual(d['status'],'DRAFT_NOT_APPROVED')
  self.assertEqual(storage['writer_kind'],'future_finalization');self.assertEqual(storage['command'][1:4],['-m','tools.st_world_policy.finalize_future_metadata','finalize-existing'])
  self.assertEqual(internal['source_terminal'],d['source_terminal']);self.assertEqual(internal['source_shards']['count'],1176)
  self.assertEqual(internal['source_shards']['bytes'],984972623)
  self.assertTrue(all(internal[k] is False for k in ['gpu_authorized','teacher_extraction_authorized','training_authorized']))

 def test_policy_is_exact_addition_not_live_migration(self):
  d=read('DRAFT_ADMISSION.json');old=json.loads(Path(d['old_policy']['path']).read_bytes());new=read('DRAFT_PROPOSED_POLICY.json')
  self.assertEqual({k:v for k,v in old.items() if k!='claims'},{k:v for k,v in new.items() if k!='claims'})
  self.assertEqual(new['claims'][:-1],old['claims']);self.assertEqual(len(old['claims']),11)
  last=new['claims'][-1];self.assertEqual(last['root'],d['output_root']);self.assertEqual(last['registration_sha256'],d['registration']['sha256'])
  self.assertEqual(sum(last[k] for k in ['output_bytes','staging_bytes','metadata_bytes']),2<<30)

 def test_actual_allowlist_covers_only_expected_flat_publications(self):
  patterns=read('DRAFT_STORAGE_REGISTRATION.json')['file_patterns']
  names=['FINALIZE.lock','METADATA_ADMISSION.json','RAW_LOG_PROVENANCE.json','TEACHER_REQUESTS.json','SHARD_RECEIPTS.json','FINALIZATION_COMPLETE.json','CONTROL_FINALIZATION_STARTED.json','CONTROL_FINALIZATION_RESULT.json','RAW_LOG_PROVENANCE.json.abcdef.partial']
  for name in names:self.assertTrue(any(re.fullmatch(p,name) for p in patterns),name)
  for name in ['BUILD.lock','old/SHARD_RECEIPTS.json','weights.pt','foo.json','TEACHER_REQUESTS.json.INVALID.partial']:
   self.assertFalse(any(re.fullmatch(p,name) for p in patterns),name)

 def exercise_controls(self,mutation=None):
  d=read('DRAFT_ADMISSION.json');storage=read('DRAFT_STORAGE_REGISTRATION.json');internal=read('DRAFT_FINALIZATION_REGISTRATION.json')
  now=time.time();d.update(status='APPROVED_FUTURE_FINALIZATION_ONLY',reviewer='synthetic-independent-reviewer-NOT-REAL-APPROVAL',hostname=socket.gethostname(),issued_at_unix=now-1,latest_start_unix=now+10,valid_until_unix=now+4000)
  if mutation:mutation(d,storage,internal)
  original=a.read
  def fixture_read(pin):
   if pin['path']=='/SYNTHETIC_NO_LAUNCH.json':return d
   if pin==a.CP:return dict(status='CURRENT',remaining_bytes=2303<<30,free_floor_bytes=1<<40,valid_until_unix=now+5000)
   if pin==d['registration']:return storage
   if pin==d['finalization_registration']:return internal
   return original(pin)
  with patch.object(a,'read',fixture_read),patch.object(a.os,'statvfs',return_value=SimpleNamespace(f_bavail=1<<50,f_frsize=1)):
   result=a.controls(SimpleNamespace(admission='/SYNTHETIC_NO_LAUNCH.json',admission_sha256='0'*64,child=False))
  return d,result

 def test_actual_controls_to_reviewed_guard_without_invocation(self):
  d,r=self.exercise_controls();self.assertEqual(r[2]['registration_sha256'],d['registration']['sha256']);self.assertEqual(r[2]['policy_sha256'],d['policy']['sha256'])
  self.assertEqual(r[2]['checkpoint_sha256'],a.CP['sha256']);self.assertEqual(Path(r[5].__file__).resolve(),a.RUNTIME/'tools/st_world_policy/metadata_storage.py')
  self.assertEqual(Path(r[7].__file__).resolve(),a.LIBRARY/'full_future_metadata_preparation_v1/run_build_once.py')

 def test_wrong_command_path_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'command'):self.exercise_controls(lambda d,s,i:s['command'].__setitem__(2,'tools.st_world_policy.prepare_future_metadata'))
 def test_internal_terminal_mismatch_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'population/lineage'):self.exercise_controls(lambda d,s,i:i['source_terminal'].update(sha256='0'*64))
 def test_changed_row_limit_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'population/lineage'):self.exercise_controls(lambda d,s,i:i.update(max_row_bytes=1<<20))
 def test_changed_internal_workers_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'population/lineage'):self.exercise_controls(lambda d,s,i:i['limits'].update(workers=2))
 def test_wrong_storage_budget_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'Exact finalizer limits'):self.exercise_controls(lambda d,s,i:s['limits'].update(max_disk_bytes=3<<30))
 def test_missing_monitor_pin_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'monitor dependencies'):self.exercise_controls(lambda d,s,i:d['controller_sources'].pop())
 def test_actual_pin_tampering_is_rejected(self):
  pin=read('DRAFT_ADMISSION.json')['registration']
  with self.assertRaisesRegex(ValueError,'Control changed'):a.read(dict(pin,sha256='0'*64))
 def test_short_CP_lease_is_rejected(self):
  # Reuse actual controls, only the synthetic CP expiry becomes shorter than required.
  original=a.read
  def cp_short(pin):
   if pin==a.CP:return dict(status='CURRENT',remaining_bytes=2303<<30,free_floor_bytes=1<<40,valid_until_unix=0)
   return original(pin)
  # Existing path is not opened/leased; controls' separate guard rejection is
  # directly exercised by pinning the in-memory admission expiry beyond our fake CP.
  with self.assertRaisesRegex(ValueError,'Checkpoint obligations'):
   d=read('DRAFT_ADMISSION.json');s=read('DRAFT_STORAGE_REGISTRATION.json');i=read('DRAFT_FINALIZATION_REGISTRATION.json');now=time.time()
   d.update(status='APPROVED_FUTURE_FINALIZATION_ONLY',reviewer='synthetic-NOT-APPROVAL',hostname=socket.gethostname(),issued_at_unix=now-1,latest_start_unix=now+10,valid_until_unix=now+4000)
   def rd(pin):
    if pin['path']=='/SYNTHETIC_NO_LAUNCH.json':return d
    if pin==a.CP:return cp_short(pin)
    if pin==d['registration']:return s
    if pin==d['finalization_registration']:return i
    return original(pin)
   with patch.object(a,'read',rd):a.controls(SimpleNamespace(admission='/SYNTHETIC_NO_LAUNCH.json',admission_sha256='0'*64,child=False))

 def test_guarded_runtime_source_inheritance(self):
  expected={'finalize_future_metadata.py':'e3443e703e40f16747a0d0bf307dd895045b4406d0e69a9a0b06e7300415e98d','metadata_storage.py':'aaa36f5383451eadcd92c160c6f3ffb8704110ab44eb12ee433b3207995e71ba','teacher_provider.py':'2ba5003f05366c97e0091e48e8a271f5e7ec771319341acdc236249f11dd36d1'}
  for name,sha in expected.items():self.assertEqual(hashlib.sha256((a.RUNTIME/'tools/st_world_policy'/name).read_bytes()).hexdigest(),sha)
  src=(a.RUNTIME/'tools/st_world_policy/teacher_provider.py').read_text();self.assertIn("if future.get('schema') == 'st_future_metadata_finalization_complete_v1':",src);self.assertIn('validate_completion(future,',src)
