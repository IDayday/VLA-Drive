"""Finite scheduler fixtures; no nvidia-smi, launcher or model process runs."""
import importlib.util,json,unittest,tempfile,types,sys,hashlib,os
from pathlib import Path
from unittest.mock import patch
SOURCE=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/teacher_channel256_actual_v1/wait_evaluate_once.py')
SPEC=(SOURCE.parent/'EVALUATION_SPEC.json').read_bytes(); DOC=json.loads(SPEC)
class WaiterTests(unittest.TestCase):
 def exercise(self,mode):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); copy=root/'wait_evaluate_once.py';copy.write_bytes(SOURCE.read_bytes());(root/'EVALUATION_SPEC.json').write_bytes(SPEC)
   assert hashlib.sha256(copy.read_bytes()).hexdigest()=='d34ab3d00c138170efcca4b5829541c2767fa9c669c9023a03e29ea71d819d04'
   spec=importlib.util.spec_from_file_location('tested_waiter',copy);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
   realP=Path; mappings={DOC['registration_root']:root/'control',DOC['output_root']:root/'out','/proc':root/'proc'}
   m.Path=lambda p:mappings.get(str(p),realP(p))
   clock={'now':0}; calls=[]
   def query(*a,**kw):
    calls.append(('query',kw.get('timeout')))
    if mode in ('late_free','busy_timeout'):clock['now']=3601
    return DOC['gpu_uuid']+(', 0, 0' if mode!='busy_timeout' else ', 1000, 99')
   def launch(cmd,**kw):
    stage=cmd[cmd.index('--stage')+1];calls.append(('launch',stage))
    if mode=='launch_refused':return types.SimpleNamespace(returncode=2,stderr='fixture refusal',stdout='')
    dest=root/'out'/('export/COMPLETE_rank0.json' if stage=='export' else 'official_score/COMPLETE.json')
    if mode!='missing_complete':
     dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps({'status':'PARTIAL' if mode=='incomplete_score' and stage=='score' else 'COMPLETE'}))
    return types.SimpleNamespace(returncode=0,stderr='',stdout=json.dumps({'pid':999999999,'stage':stage}))
   m.subprocess=types.SimpleNamespace(check_output=query,run=launch)
   m.time=types.SimpleNamespace(time=lambda:1792280000.,monotonic=lambda:clock['now'],sleep=lambda x:clock.__setitem__('now',clock['now']+x))
   with patch.object(sys,'argv',[str(copy),'--review','/SYNTHETIC_REVIEW','--review-sha256','0'*64]):
    error=None
    try:ret=m.main()
    except Exception as e:error=e;ret=None
   state=json.loads((root/'control/WAIT_EVALUATE.json').read_text())
   return ret,error,state,calls
 def test_complete_export_then_actual_score_receipt(self):
  ret,error,state,calls=self.exercise('success');self.assertEqual(ret,0);self.assertIsNone(error);self.assertEqual(state['status'],'COMPLETE');self.assertEqual([c for c in calls if c[0]=='launch'],[('launch','export'),('launch','score')])
 def test_launch_refusal_never_retries(self):
  ret,error,state,calls=self.exercise('launch_refused');self.assertIsInstance(error,RuntimeError);self.assertEqual(state['status'],'FAILED_NO_AUTO_RETRY');self.assertEqual([c for c in calls if c[0]=='launch'],[('launch','export')])
 def test_missing_export_stops_before_score(self):
  ret,error,state,calls=self.exercise('missing_complete');self.assertIsInstance(error,RuntimeError);self.assertEqual([c for c in calls if c[0]=='launch'],[('launch','export')])
 def test_partial_official_receipt_is_not_complete(self):
  ret,error,state,calls=self.exercise('incomplete_score');self.assertIsInstance(error,RuntimeError);self.assertEqual(state['status'],'FAILED_NO_AUTO_RETRY')
 def test_busy_timeout_never_launches(self):
  ret,error,state,calls=self.exercise('busy_timeout');self.assertEqual(ret,3);self.assertEqual(state['status'],'BLOCKED_RESOURCE');self.assertFalse(any(c[0]=='launch' for c in calls))
 def test_query_returning_free_after_deadline_never_launches(self):
  ret,error,state,calls=self.exercise('late_free');self.assertEqual(ret,3);self.assertEqual(state['status'],'BLOCKED_RESOURCE');self.assertFalse(any(c[0]=='launch' for c in calls))
if __name__=='__main__':
 os.sched_setaffinity(0,{8,9});unittest.main(verbosity=2)
