"""Pure source/control delta check, no torch/CUDA/model/scoring."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

START=time.monotonic()
ROOT=Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
REL='reports/st_world_policy/20261018/gradient_balance_remote_v3/supervise_gradient_once.py'
new_path=ROOT/REL
old_path=Path('/mnt/project/VLA-Drive-st-road-direction-systems-20261018/reports/st_world_policy/20261018/gradient_balance_systems_v1/supervise_gradient_once.py')
old=old_path.read_bytes();new=new_path.read_bytes()
assert hashlib.sha256(old).hexdigest()=='29556ded05f532cabc7f4fa8255668f7ab232341f7ce34a6f5fa5fa5650f76ae'
assert hashlib.sha256(new).hexdigest()=='763c753ef28f8959d0f99b0e2693a50dfe25f9f421c469519e74f682a4912e11'
assert new==subprocess.check_output(['git','show','0792bc28:'+REL],cwd=ROOT)
a,b=ast.parse(old),ast.parse(new)
def assignments(tree):return {n.targets[0].id:ast.dump(n.value,include_attributes=False) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
x,y=assignments(a),assignments(b)
changes=sorted(k for k in x if x[k]!=y[k])
assert changes==['GPU_LOCK','REGISTRATION','REG_SHA']
def stripped(tree):return ast.dump(ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in changes)],type_ignores=[]),include_attributes=False)
assert stripped(a)==stripped(b)
base=Path('/mnt/project/st-world-policy-20261016/audits/consequence')
p2=base/'gradient_balance_boundary3584_v2_controls/REGISTRATION.json';p3=base/'gradient_balance_boundary3584_v3_controls/REGISTRATION.json'
b2,b3=p2.read_bytes(),p3.read_bytes();r2,r3=json.loads(b2),json.loads(b3)
assert hashlib.sha256(b3).hexdigest()=='6a48f942e8b57f568d74be0bd92cab389daa6140d0a0307896acd40da3358770'
assert sorted(k for k in r2 if r2[k]!=r3[k])==['output','resource']
assert r2['resource']['cpu_threads']==r3['resource']['cpu_threads']==2
assert r3['resource']['host']=='training-vlawm-tsinghua-zt2-worker-0'
assert r3['resource']['gpu_uuid']=='GPU-c0a3d651-8691-b507-5df6-6c7f0cc6c897'
assert r3['source_sha']=='83679937fe9702be39f0184154fdf9ebdc4eb98a'
assert r3['diagnostic']['optimizer_steps']==r3['diagnostic']['new_score_calls']==0
assert r3['diagnostic']['global_scenes']==32
proc=subprocess.run([sys.executable,str(new_path)],capture_output=True,text=True,timeout=5)
assert proc.returncode==0
cli=json.loads(proc.stdout)
assert cli['status']=='NOT_RUN' and cli['registration_sha256']==hashlib.sha256(b3).hexdigest()
assert cli['runtime_environment']=={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
result={'status':'SOURCE_AND_NAMESPACE_DELTA_PASS','wrapper_path':str(new_path),'wrapper_sha256':hashlib.sha256(new).hexdigest(),'wrapper_source_sha':'0792bc2853c7a01914627ad7232b27fcf01299d9','changed_top_level_assignments':changes,'all_other_AST_unchanged':True,'registration_sha256':hashlib.sha256(b3).hexdigest(),'registration_changed_fields':['output','resource'],'source_sha':r3['source_sha'],'resource':r3['resource'],'budget':r3['budget'],'runtime_environment':cli['runtime_environment'],'metadata_cli':cli,'v2_output_exists':Path(r2['output']).exists(),'v2_attempt_exists':(p2.parent/'RUNTIME_ATTEMPT').exists(),'direct_source_control_bytes':len(old)+len(new)+len(b2)+len(b3),'wall_seconds':time.monotonic()-START,'real_assets_read':False,'torch_loaded':False,'GPU_calls':0,'new_score_calls':0,'dispatch':False}
Path(__file__).with_name('REMOTE_V3_REASSIGNMENT_CHECK.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
