"""Finite CPU metadata checks; no owner construction, probes, dispatch or live writes."""
import copy,hashlib,json,os,tempfile,time
from pathlib import Path
import yaml
from tools.qwen_world_safety_exploration import revival_campaign as m
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');G=m.ROOT_PATH
m.configure(G)
owner=m.owner_class()  # Real qualified relative imports; constructor deliberately not called.
from tools.qwen_world_safety_exploration.eval_owner import EvaluationOwner
from tools.qwen_world_safety_exploration.review import Review,compatible
F=json.loads((m.CONTROL_ROOT/'RUNS.json').read_text());template=next(x for x in F['runs']if x['id']=='SAFETY_G8_CONTROL')
checks=[]
def rejected(value):
 try:m.validate_registry(value,m.ROOT_PATH)
 except ValueError:return
 raise AssertionError('Invalid metadata accepted')
with tempfile.TemporaryDirectory(prefix='revival_cpu_')as directory:
 root=Path(directory);m.ROOT_PATH=root
 reg=copy.deepcopy(F);reg.update(root=str(root),continuation_max_updates=50000,external_control=dict(registry=str(m.CONTROL_ROOT/'RUNS.json'),arm='SAFETY_G8_CONTROL',root=str(m.CONTROL_ROOT),read_only=True));reg['runs']=[]
 for arm,role in m.ROLES.items():
  run=copy.deepcopy(template);p=root/(arm+'.yaml');cfg=yaml.safe_load(Path(template['config']).read_text());cfg['research']['revival_training']='world_legacy_algorithms_fixed50k_revival_v1';cfg['research']['standard_role']=role;cfg['research']['campaign_root']=str(root);p.write_text(yaml.safe_dump(cfg,sort_keys=False));run.update(id=arm,revival_role=role,root=str(root/'runs'/arm),owner_state=str(root/'owners/REVIVAL_NATIVE/STATE.json'),group_size=16 if arm=='REVIVAL_G16'else 8,config=str(p),config_sha256=m.sha(p));reg['runs'].append(run)
 assert m.validate_registry(reg,root)is reg
 three=copy.deepcopy(reg);three['runs']=three['runs'][:3];assert m.validate_registry(three,root)is three
 bad=copy.deepcopy(reg);bad['runs'][0]['group_size']=8;rejected(bad)
 bad=copy.deepcopy(reg);bad['external_control']['read_only']=False;rejected(bad)
 bad=copy.deepcopy(reg);bad['runs'][0]['deployment_candidates']=[dict(host='training-vla-zt',lease_slot='primary',minimum_free_MiB=32768)];rejected(bad)
 bad=copy.deepcopy(reg);bad['runs'][0]['target_updates']=20000;rejected(bad)  # Gates still10k.
 checks.append(dict(name='three_or_four_registry_exact_roles_external_readonly_and_protected_placement',status='PASS',temporary_files_only=True))
 m.ROOT_PATH=G
 assert m.gates(20000)==[2000,5000,10000,12000,14000,16000,18000,20000]
 assert m.decision(10000,'g16')['next_cap']==20000
 weak=dict(PDMS_delta=-1.2,PDMS_CI95=[-2,-.6],new_zero=60,repaired_zero=5)
 assert m.decision(20000,'dense',weak,weak,dict(PDMS_delta=0,PDMS_CI95=[-.1,.1]))['next_cap']==30000
 assert m.decision(20000,'dense',weak,weak,dict(PDMS_delta=-.1,PDMS_CI95=[-.2,-.01]))['action']=='RESEARCH_RESERVE_REVIEW'
 # Reuse actual sealed protocol identities without loading any scene/weight.
 a=Path('/mnt/project/qwen-world-safety-exploration-20261031/evaluations/SAFETY_G16_u001000/NAVTEST/identity.json');b=Path('/mnt/project/qwen-world-safety-exploration-20261031/evaluations/SAFETY_G8_CONTROL_u001000/NAVTEST/identity.json')
 compatible(json.loads(a.read_text()),json.loads(b.read_text()))
 checks.append(dict(name='real_owner_eval_review_imports_gates_late_policy_and_existing_identity_compatibility',status='PASS',owner_class=owner.__name__,qualified_eval=EvaluationOwner.__name__,qualified_review=Review.__name__))
actual=None
if(G/'RUNS.json').exists():
 actual=m.validate_registry(json.loads((G/'RUNS.json').read_text()),G)
result=dict(schema='world_revival_campaign_metadata_cpu_check_v1',status='PASS_ACTUAL_REGISTRY'if actual else'PASS_METADATA_FIXTURES_REAL_IMPORTS_ACTUAL_G_REGISTRATION_PENDING',created_unix=time.time(),source=str(Path(m.__file__).resolve()),source_sha256=m.sha(m.__file__),checks=checks,actual_registry=str(G/'RUNS.json')if actual else None,actual_arms=[r['id']for r in actual['runs']]if actual else[],GPU_calls=0,live_state_registry_or_sources_mutated=False,real_owner_instantiated=False)
p=E/'analysis/REVIVAL_CAMPAIGN_CPU_CHECK.json';q=Path(__file__).with_name(p.name)
for dst in(p,q):dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
