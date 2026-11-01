"""Small metadata-only cooperative-return and finite-worker dispatch check.

No host connections, resource probes, GPU children, signals or live owner writes.
"""
import copy,hashlib,json,os,tempfile,time
from pathlib import Path
from unittest.mock import patch
from tools.qwen_world_safety_exploration import resource_boundary as B
from tools.qwen_world_safety_exploration import dispatch as D
from starVLA.rl.qwen_world_safety_exploration import contracts as C

ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031');W=Path(__file__).resolve().parents[2]
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
with tempfile.TemporaryDirectory(prefix='finite_resource_cpu_')as tmp:
 study=Path(tmp)/'study';runroot=study/'runs/ARM';job=study/'jobs/ARM/attempt001';job.mkdir(parents=True);cp=runroot/'checkpoints/update010001_inner1';cp.mkdir(parents=True)
 cfg=dict(adaptation='ATTN_LORA_MEMORY',max_steps=50000,world_size=8);config=study/'configs/ARM.yaml';config.parent.mkdir(parents=True);config.write_text('adaptation: ATTN_LORA_MEMORY\nmax_steps: 50000\nworld_size: 8\n')
 ident=dict(purpose='formal',world=8,recipe=cfg,sha256='own-fixture');write(runroot/'identity.json',ident);write(cp/'identity.json',ident)
 write(cp/'transaction.json',dict(schema='simwam_checkpoint_transaction_v2'))
 for n in ['optimizer.pt']+[f'rank{r}.pt'for r in range(8)]:(cp/n).write_bytes(b'CPU-metadata-fixture; never restored or loaded')
 files={p.name:sha(p)for p in cp.iterdir()};write(cp/'files.json',files);write(cp/'COMPLETE',dict(schema=2,phase='INNER_COMMITTED',world=8,updates=10001,next_inner=1,files_sha256=sha(cp/'files.json')))
 for rank in range(8):write(runroot/f'result_rank{rank}.json',dict(updates=10001,checkpoint=str(cp)))
 run=dict(id='ARM',root=str(runroot),config=str(config));node=dict(host='finite-host',job=str(job));latest=dict(updates=10001,checkpoint=str(cp));grant=dict(temporary_hosts=['finite-host'],permanent_hosts=['permanent-host'],expires_unix=20000,checkpoint_guard_seconds=1200)
 with patch.object(C,'authorization',return_value=(grant,{})),patch.object(C,'allowed_hosts',return_value=[]),patch.object(B.time,'time',return_value=18800):
  assert B.cooperative_return(run,node,dict(returncode=1),latest,12000)is None
  got=B.cooperative_return(run,node,dict(returncode=0,temporary=False,migration_required=False),latest,12000)
  assert got['updates']==10001 and got['next_inner']==1 and got['checkpoint']==str(cp)
  assert not(job/'STOP_REQUESTED').exists()
  checks.append(dict(name='nonzero_never_retried_and_owned8rank_deadline_return_accepts_old_temporary_false',status='PASS',tensor_payloads_loaded=0))
 with patch.object(C,'authorization',return_value=(grant,{})),patch.object(C,'allowed_hosts',return_value=[]),patch.object(B.time,'time',return_value=17000):
  assert B.cooperative_return(run,dict(node,host='withdrawn_without_stop_reason'),dict(returncode=0),latest,12000)is None
  checks.append(dict(name='unauthorized_host_without_guard_stop_or_migration_receipt_not_auto_resumed',status='PASS'))
 with patch.object(C,'authorization',return_value=(grant,{})),patch.object(C,'allowed_hosts',return_value=[]),patch.object(B.time,'time',return_value=18800):
  def rejected(fn):
   try:fn()
   except ValueError:return
   raise AssertionError('Unsafe metadata accepted')
  foreign=dict(ident,sha256='foreign-same-recipe');write(cp/'identity.json',foreign);rejected(lambda:B.cooperative_return(run,node,dict(returncode=0),latest,12000));write(cp/'identity.json',ident)
  outside=study/'jobs/OTHER/attempt001';outside.mkdir(parents=True);rejected(lambda:B.cooperative_return(run,dict(node,job=str(outside)),dict(returncode=0),latest,12000))
  original=copy.deepcopy(files);files.pop('optimizer.pt');write(cp/'files.json',files);rejected(lambda:B.cooperative_return(run,node,dict(returncode=0),latest,12000));files=original
  files['../../outside']='untrusted';write(cp/'files.json',files);rejected(lambda:B.cooperative_return(run,node,dict(returncode=0),latest,12000))
  checks.append(dict(name='same_run_identity_complete_inventory_no_path_escape_and_exact_job_scope',status='PASS'))
 actual,policy=C.authorization();worker=Path(actual['worker_execution_source'])/'tools/qwen_world_safety_exploration/job_worker.py';assert sha(worker)==actual['worker_execution_worker_sha256']
 science=ROOT/'standard_training_20261101/runtime/standard_v1';seen=[]
 def fake_remote(host,args):seen.append(dict(host=host,args=args));return json.dumps(dict(worker_pid=99999,worker_start_ticks=111))
 with patch.object(D,'ROOT',study),patch.object(D,'authorization',return_value=(actual,policy)),patch.object(D,'remote',side_effect=fake_remote):
  out=D.launch('CPU-MOCK-ONLY',science,['python','-m','known.science'],'cpu_dispatch/mock001');receipt=json.loads((out/'DISPATCH.json').read_text())
  assert receipt['executor']==str(worker) and receipt['source']==str(science) and receipt['executor_sha256']==actual['worker_execution_worker_sha256']
  assert str(science)in seen[0]['args'][-1] and '--cwd'in seen[0]['args'][-1]
  bad=dict(actual,worker_execution_worker_sha256='wrong')
  with patch.object(D,'authorization',return_value=(bad,policy)):
   rejected(lambda:D.launch('CPU-MOCK-ONLY',science,['python'],'cpu_dispatch/wrongpin'))
  assert len(seen)==1
 checks.append(dict(name='new_dispatch_uses_pinned_finite_worker_and_retains_old_scientific_cwd_PIN_mismatch_refused',status='PASS',remote_calls='one mocked call, zero real connections'))
result=dict(schema='world_finite_resource_boundary_CPU_check_v1',status='PASS_METADATA_ONLY_NO_OWNER_OR_GPU_ACTION',created_unix=time.time(),checks=checks,sources={str(p.relative_to(W)):sha(p)for p in[W/'tools/qwen_world_safety_exploration/resource_boundary.py',W/'tools/qwen_world_safety_exploration/standard_campaign.py',W/'tools/qwen_world_safety_exploration/revival_campaign.py',W/'tools/qwen_world_safety_exploration/dispatch.py']},real_grant_worker=dict(source=actual['worker_execution_source'],sha256=actual['worker_execution_worker_sha256']),large_tensor_or_model_loads=0,owner_instantiation=0,real_remote_connections=0,GPU_signals_or_dispatch=0,live_state_or_registry_writes=0)
p=ROOT/'resources/FINITE_OPS_BOUNDARY_CPU_CHECK_20261101.json';write(p,result);write(Path(__file__).with_suffix('.json'),result);print(json.dumps(result,indent=2))
