"""Independent metadata-only issuance. Never invokes common/prepare/run/GPU/SSH."""
import copy,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWNER=Path('/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261018/temporal_T3_profile4_binding_v1')
PROPOSAL=Path('/mnt/project/st-world-policy-20261016/resources/T3_profile4_actual_20261018T0053/RESOURCE_ADMISSION_PROPOSAL.json')
EXPECTED='57c8a1a59331ea160466213dcbe67bc9ddef6e248ec9235964d5fa269956073e'
REVIEWER='/root/temporal_narrow_review'
def read(path):
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.stat().st_size < 4<<20
 raw=p.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()
def check_refs(obj,out):
 if isinstance(obj,dict):
  if set(('path','sha256'))<=set(obj):
   q=Path(obj['path']);assert q.is_file() and not q.is_symlink() and q.stat().st_size<4<<20
   sha=hashlib.sha256(q.read_bytes()).hexdigest();assert sha==obj['sha256'],obj['path'];out[obj['path']]=sha
  for x in obj.values():check_refs(x,out)
 elif isinstance(obj,list):
  for x in obj:check_refs(x,out)
def equality(candidate,original):
 target=copy.deepcopy(original);target['status']='ADMITTED';target['reviewer_agent']=REVIEWER
 assert candidate==target,'Signed output differs from the freshly verified proposal'
def main():
 a,sha=read(PROPOSAL);assert sha==EXPECTED
 pins={};check_refs(a,pins)
 plan,plan_sha=read(OWNER/'PLAN.json');assert plan_sha=='e8438c8361ea7a3c5965e8acdb7a07ff7b0488945bcfab0c17e2458be5f59143';check_refs(plan,pins)
 assert hashlib.sha256((OWNER/'controller.py').read_bytes()).hexdigest()==a['controller_sha256']
 spec=importlib.util.spec_from_file_location('independent_T3_control',OWNER/'controller.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);c=w.controller()
 inputs,inputs_sha=read(OWNER/'INPUTS.json');assert inputs_sha==a['inputs_sha256'];recorded=copy.deepcopy(inputs);recorded.pop('native_argv_for_existing_supervisor');assert recorded==w.b.actual_inputs()
 check_refs(inputs,pins)
 cp,_=read(a['checkpoint_commitments']['path']);assignment,_=read(a['checkpoint_hold_assignment']['path']);assert cp['status']=='CURRENT' and assignment['status']=='ASSIGNED'
 assert assignment['aggregate_bytes']==2703<<30 and assignment['additional_bytes']==0
 snapshot,_=read(a['host_snapshot']['path']);now=time.time()
 assert 0<=now-snapshot['observed_unix']<900
 assert snapshot['host']==a['host'] and snapshot['protected_external_processes']==a['protected_external_processes']==[]
 uuid,total,used,free,util=[x.strip() for x in snapshot['gpu_row'].split(',')]
 assert uuid==a['gpu_uuid'] and int(free)>=68*1024 and set(a['cpu_affinity'])<=set(snapshot['cpu_affinity'])
 assert snapshot['no_signals_sent'] is True and snapshot['training_started'] is False
 policy,_=read(a['dispatch_policy']['path']);auth,_=read(a['authorization']['path']);key='training-vlawm-tsinghua-zt2'
 assert a['host'] in policy['allowed_hosts'][key]['aliases'] and a['host'] in policy['allowed_aliases'] and key in auth['allowed_hosts']
 for k,v in policy.get('global_dispatch_prohibitions',{}).items():
  if k in (a['host'],key) or a['host'] in v.get('aliases',[]):assert v.get('new_task_dispatch_allowed') is True
 assert now+2100<=a['valid_until_unix']<=cp['valid_until_unix'] and a['valid_from_unix']<=now
 stat=os.statvfs('/mnt/project');free_disk=stat.f_bavail*stat.f_frsize;required=(2703+1024+768+8)<<30;assert free_disk>=required
 signed=copy.deepcopy(a);signed['status']='ADMITTED';signed['reviewer_agent']=REVIEWER;equality(signed,a)
 # A nested mutation must be caught and must not contaminate source or signature.
 bad=copy.deepcopy(signed);bad['checkpoint_hold_assignment']['sha256']='0'*64
 try:equality(bad,a)
 except AssertionError:pass
 else:raise AssertionError('Nested-output mismatch negative failed')
 assert read(PROPOSAL)[1]==EXPECTED and read(PROPOSAL)[0]==a
 candidate=HERE/'CANDIDATE_ADMISSION.json';candidate.write_text(json.dumps(signed,sort_keys=True,indent=2)+'\n')
 candidate_sha=read(candidate)[1]
 for action in ('prepare','run'):c.admission(str(candidate),candidate_sha,action)
 fresh,fresh_sha=read(PROPOSAL);assert fresh_sha==EXPECTED and fresh==a
 before={};check_refs(fresh,before);assert all(pins[k]==v for k,v in before.items())
 target=HERE/'ACTUAL_ADMISSION.json'
 with target.open('x') as f:json.dump(signed,f,sort_keys=True,indent=2);f.write('\n')
 final,final_sha=read(target);equality(final,read(PROPOSAL)[0]);assert read(PROPOSAL)[1]==EXPECTED
 after={};check_refs(final,after);assert before==after
 for action in ('prepare','run'):c.admission(str(target),final_sha,action)
 result={'status':'PASS_METADATA_ONLY_ADMISSION','reviewer_agent':REVIEWER,'checked_at_unix':now,'proposal':{'path':str(PROPOSAL),'sha256':EXPECTED},'actual_admission':{'path':str(target),'sha256':final_sha},'signed_changes':['status','reviewer_agent'],'nested_mutation_negative':'PASS','pins_before':before,'pins_after':after,'fresh_free_shared_bytes':free_disk,'conservative_protected_bytes':required,'host_snapshot_age_seconds':now-snapshot['observed_unix'],'remote_snapshot_supplied_by':'/root','fresh_host_gpu_capacity_and_processes_rechecked_by_controller_before_prepare_and_run':True,'actual_gpu_query_by_reviewer':False,'actual_dispatch_by_reviewer':False,'ledger_access':False,'formal_training_approved':False}
 (HERE/'SIGNING_AUDIT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':result['status'],'actual_admission':result['actual_admission'],'latest_start_unix':a['valid_until_unix']-2100}))
if __name__=='__main__':main()
