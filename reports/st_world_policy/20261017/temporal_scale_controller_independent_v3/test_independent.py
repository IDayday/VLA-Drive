"""Bounded resource/env review only; no model weights, data, CUDA, or dispatch."""
import hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
FROZEN=Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v3-20261017')
HERE=FROZEN/'reports/st_world_policy/20261017'
spec=importlib.util.spec_from_file_location('independent_v3_controller',HERE/'temporal_scale_calibration_v1/run_once.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def test_environment_exists_before_child_python_imports():
 parent=dict(os.environ);parent.pop('CUBLAS_WORKSPACE_CONFIG',None)
 env=c.child_runtime_environment(parent)
 assert 'CUBLAS_WORKSPACE_CONFIG' not in parent
 out=subprocess.check_output([sys.executable,'-c',"import os,sys,json; print(json.dumps({'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'torch_imported':'torch' in sys.modules,'cuda_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))"],env=env,text=True,timeout=5)
 value=json.loads(out)
 assert value=={'cublas':':4096:8','torch_imported':False,'cuda_visible':''}


def test_actual_failed_attempt_plus_new_caps_stays_inside_original_budget():
 p=Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1/RESULT.json')
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='72bcfe8e153d5ccc2e7a993dde6817e3f48d0ccff9be27928420804c9c20f582'
 first=json.loads(raw)
 assert first['status']=='FAILED_NO_AUTO_RETRY' and first['resources']['cleanup']['live_owned_remaining']==0
 assert first['charged_parent_child_rchar_bytes']+c.LIMITS['read_bytes']<=256<<30
 assert first['resources']['invocation_seconds']+c.LIMITS['wall_seconds']+c.LIMITS['cleanup_seconds']<=3640
 assert c.ROOT.name=='loss_scale_calibration_real4_v2' and c.LIMITS['max_invocations']==1
 assert c.LIMITS['optimizer_updates']==c.LIMITS['new_tensor_bytes']==0


def differences(a,b,prefix=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(a.keys()|b.keys()):
   key=prefix+'.'+k if prefix else k
   if k not in a or k not in b:out.append(key)
   else:out.extend(differences(a[k],b[k],key))
  return out
 return [] if a==b else [prefix]


def test_same_four_scenes_math_and_recipe_only_output_time_environment_change():
 old=json.loads(Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1/result/RESOLVED_CONFIG.json').read_text())
 new=json.loads((HERE/'temporal_scale_cublas_preflight_fix_v1/PROPOSED_CALIBRATION_CONFIG.json').read_text())
 assert len(new['batch']['scene_indices'])==4 and new['batch']==old['batch']
 assert set(differences(old,new))=={'runtime_environment','output_root','training_config.path','training_config.sha256'}
 def readpin(pin):
  raw=Path(pin['path']).read_bytes(); assert hashlib.sha256(raw).hexdigest()==pin['sha256'];return json.loads(raw)
 oldtrain,newtrain=readpin(old['training_config']),readpin(new['training_config'])
 assert set(differences(oldtrain,newtrain))=={'runtime.max_seconds','runtime.output_root'}
 assert newtrain['runtime']['deterministic'] is True
 oldroot=Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v2-20261017')
 for rel in ['tools/st_world_policy/calibration_measurement.py','starVLA/model/modules/st_world/losses.py','reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py']:
  assert (FROZEN/rel).read_bytes()==(oldroot/rel).read_bytes()


def test_fixed_new_runtime_identity_has_no_old_review_or_automatic_retry():
 assert c.RUNTIME==FROZEN
 assert c.LIMITS==dict(wall_seconds=3000,cleanup_seconds=40,read_bytes=208<<30,rss_bytes=96<<30,cpu_affinity=[2,3],max_disk_bytes=16<<20,max_invocations=1,optimizer_updates=0,new_tensor_bytes=0)
 text=Path(c.__file__).read_text()
 assert "review.get('controller_sources') != control['sources']" in text
 assert "review.get('limits') != LIMITS" in text
 assert "if ROOT.exists():" in text
 assert "'scientific_calibration_selected': False" in text
 assert 'dict(child_runtime_environment(os.environ)' in text
