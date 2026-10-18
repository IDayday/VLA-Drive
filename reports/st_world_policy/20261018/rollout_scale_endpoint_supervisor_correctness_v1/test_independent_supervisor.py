"""Independent narrow CPU counterexamples; no real weights, DEV or scoring."""
from copy import deepcopy
import importlib.util
import json,os
from pathlib import Path
import subprocess,sys,time
import pytest
ROOT=Path(os.environ['ST_ENDPOINT_IMPLEMENTATION_ROOT'])
SPEC=importlib.util.spec_from_file_location('endpoint_author_fixture',ROOT/'tests/st_world/test_endpoint_supervisor.py')
t=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(t)
s=t.s


def isolated(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'OUTPUT',tmp_path/'job');monkeypatch.setattr(s,'ANALYSIS',tmp_path/'analysis')
 monkeypatch.setattr(s,'immutable',t.immutable)
 (s.OUTPUT/'control').mkdir(parents=True)


def test_all_claimed_attempt_directories_count_even_without_admission(tmp_path,monkeypatch):
 isolated(tmp_path,monkeypatch)
 for ordinal in ('first_unknown','second_unknown'):
  (s.OUTPUT/'control/invocations/score'/ordinal).mkdir(parents=True)
 with pytest.raises(ValueError,match='exhausted'):
  s.claim_invocation(s.OUTPUT/'control','score',resume=True,binding_sha='fixture',argv=['fixture'])
 assert len(list((s.OUTPUT/'control/invocations/score').iterdir()))==2


def test_storage_and_authority_refresh_do_not_redefine_science_or_lease():
 original={'checkpoint':{'tag':'step_004096_123'},'route':{'valid_until_unix':10000.,'gpu_uuids':['fixed']},
   'source_sha':s.SOURCE,'storage_reservations':{'sha256':'a'},'dispatch_policy':{'sha256':'b'},'authorization':{'sha256':'c'}}
 refresh={**original,'storage_reservations':{'sha256':'new-storage'},'dispatch_policy':{'sha256':'new-policy'},'authorization':{'sha256':'new-auth'}}
 assert s.digest(s.stable_binding(original))==s.digest(s.stable_binding(refresh))
 for key,value in (('checkpoint',{'tag':'step_004095_123'}),('source_sha','0'*40),('route',{'valid_until_unix':20000.,'gpu_uuids':['other']})):
  assert s.digest(s.stable_binding({**refresh,key:value}))!=s.digest(s.stable_binding(original))


def make_load(isolated_root):
 checkpoint={'completed':4096,'tag':'step_004096_123','run_identity':'a'*64,'receipt_identity':'b'*64,
  'native_receipt_sha256':'c'*64,'native_checkpoint_sha256':'d'*64,'parent_checkpoint_sha256':'f'*64,'scope':'FORMAL','weights_usable_for_formal_init':True}
 checkpoint['sha256']=s.digest(checkpoint)
 signature={'checkpoint':checkpoint,'source_sha':s.SOURCE,'world_size':4,'seed':42,
  'precision':'FP32_master_eval_TF32_off','solver':'original_Euler_10','candidates_per_scene':1,
  'requested_scenes':1696,'targets_read':False,'teacher_cache_read':False,'scorer':None,
  'selected_scenes_sha256':'1'*64,'observations':{'manifest_sha256':'2'*64}}
 signature['identity']=s.digest(signature)
 t.put(s.OUTPUT/'export/IDENTITY.json',signature)
 folder=s.OUTPUT/'export/attempts/fixture'
 invocation={'pid':123,'started_unix':10.,'identity':signature['identity'],'arguments':{'rank':0,'world_size':4,
   'output':str(s.OUTPUT/'export'),'registration':'/fixture/REGISTRATION.json','checkpoint_tag':checkpoint['tag'],
   'device':'cuda:0','seed':42,'max_scenes':1696}}
 t.put(folder/'INVOCATION.json',invocation)
 audit={'schema':'st_flow_fp32_restoration_audit_v1','checkpoint_record_sha256':checkpoint['sha256'],
  'parent_checkpoint_sha256':checkpoint['parent_checkpoint_sha256'],'native_checkpoint_sha256':checkpoint['native_checkpoint_sha256'],
  'frozen_encoder_source':'unchanged_original_bound_S0_FP32_masters','training_BF16_encoder_reused':False,
  'original_parent_modified':False,'optimizer_resume_claim':False,'training_feedback_required':False,
  'all_parameters_frozen':True,'strict_action_load':True,'master_dtype':'float32',
  'action_buffers':'unchanged_bound_parent_values_verified_against_native'}
 t.put(folder/'LOAD_AUDIT.json',audit)
 expected={'registration':'/fixture/REGISTRATION.json','tag':checkpoint['tag'],
  **{k:checkpoint[k] for k in ('run_identity','receipt_identity','native_receipt_sha256','native_checkpoint_sha256')},
  'selected_scenes_sha256':'1'*64,'manifest_sha256':'2'*64}
 return folder,invocation,audit,expected


@pytest.mark.parametrize('change',['identity','master','stale'])
def test_wrong_or_stale_load_audit_never_frees_heavy_loader(tmp_path,monkeypatch,change):
 isolated(tmp_path,monkeypatch);folder,invocation,audit,expected=make_load(tmp_path)
 assert s.load_finished(0,{123:{}},after=0.,expected=expected)
 if change=='identity':
  invocation['identity']='0'*64;t.put(folder/'INVOCATION.json',invocation)
 elif change=='master':
  audit['master_dtype']='bfloat16';t.put(folder/'LOAD_AUDIT.json',audit)
 else:
  assert not s.load_finished(0,{123:{}},after=11.,expected=expected);return
 with pytest.raises(ValueError):s.load_finished(0,{123:{}},after=0.,expected=expected)


def test_post_popen_receipt_failure_cleans_actual_owned_cpu_child(tmp_path,monkeypatch):
 isolated(tmp_path,monkeypatch)
 binding,documents,_=t.endpoint_fixture(tmp_path/'fixture',monkeypatch)
 binding['route']={'gpus':[0,1,2,3],'gpu_uuids':['fixture0','fixture1','fixture2','fixture3']}
 helpers={'check_route_time_v3':lambda *args:None,'fresh_gpu_v3':lambda *args:None}
 monkeypatch.setattr(s,'mature_helpers',lambda:helpers)
 monkeypatch.setattr(s,'route_checks',lambda *args:binding['route'])
 monkeypatch.setattr(s,'gpu_monitor',lambda *args:{})
 monkeypatch.setattr(s,'argv_for',lambda *args,**kwargs:[sys.executable,'-c','import time;time.sleep(60)'])
 original_put=t.immutable
 def receipt_failure(path,value):
  if Path(path).name=='LAUNCH.json':raise OSError('Independent injected receipt write failure')
  return original_put(path,value)
 monkeypatch.setattr(s,'immutable',receipt_failure)
 children=[];popen=subprocess.Popen
 def capture(*args,**kwargs):
  child=popen(*args,**kwargs);children.append(child);return child
 monkeypatch.setattr(s.subprocess,'Popen',capture)
 try:
  result=s.execute('export',{'fixture':'admission'},binding,documents,resume=False)
  assert result['status']=='PARTIAL_OR_FAILED'
  assert len(children)==1 and children[0].poll() is not None
  attempts=list((s.OUTPUT/'control/invocations/export_rank0').glob('*'))
  assert len(attempts)==1 and (attempts[0]/'ADMISSION.json').exists()
  exit_record=json.loads((attempts[0]/'EXIT.json').read_text())
  assert exit_record['owned_cleanup_complete']
 finally:
  for child in children:
   if child.poll() is None:
    os.killpg(child.pid,9);child.wait()


@pytest.mark.parametrize('key,value',[('termination_reason','UNDECLARED_REASON'),('completed',4095),('policy_version',4095),('pending_feedback',{'synthetic':'pending'})])
def test_rehashed_native_metadata_cannot_promote_nonterminal_boundary(tmp_path,monkeypatch,key,value):
 binding,documents,folder=t.endpoint_fixture(tmp_path,monkeypatch)
 record=json.loads((folder/'COMPLETE.json').read_text());record['state'][key]=value
 record['receipt_identity']=s.digest({k:v for k,v in record.items() if k!='receipt_identity'})
 binding['checkpoint']['complete']=t.put(folder/'COMPLETE.json',record)
 binding['checkpoint']['receipt_identity']=record['receipt_identity']
 with pytest.raises(ValueError,match='optimizer endpoint'):
  s.validate_endpoint(binding,documents)
