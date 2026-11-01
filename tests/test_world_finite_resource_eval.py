"""Two metadata-only eval owner cases, no owner constructor or external jobs."""
import json
from pathlib import Path
from unittest.mock import patch

from tools.qwen_world_safety_exploration import eval_owner as E
from starVLA.rl.qwen_world_safety_exploration import contracts as C


def fixture(tmp_path, *, rank0_code=0):
 root=tmp_path/'study';output=root/'evaluations/ARM_u012000/NAVTEST';output.mkdir(parents=True)
 source=root/'SCIENCE_UNCHANGED';source.mkdir()
 def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,sort_keys=True)+'\n')
 def seal(v):return dict(v,sha256=E.fingerprint(v))
 ident=seal(dict(cohort='NAVTEST',seed=2101,inference=dict(sampler='ODE',steps=10,shift=5.0,seed=2101,scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)))
 spec=seal(dict(world=2,seed=2101,cohort='NAVTEST',arm='ARM',source=str(source)))
 write(output/'identity.json',ident);write(output/'SPEC.json',spec)
 cache=output/'scenes/unchanged.json';cache.parent.mkdir();cache.write_text('existing-scene-cache')
 entry=dict(world=2,cohort='NAVTEST',arm='ARM',label='ARM_u012000',evaluation=str(output),spec_sha256=spec['sha256'],ranks={},status='RUNNING')
 for rank in range(2):
  job=root/f'jobs/evaluation/ARM/NAVTEST/rank{rank:02}_attempt001';job.mkdir(parents=True)
  command=['MODEL','-u','-m','tools.qwen_world_safety_exploration.evaluate','--output',str(output),'--rank',str(rank),'--world','2','--seed','2101','--cohort','NAVTEST']
  receipt=dict(returncode=rank0_code if rank==0 else 0,status='EXITED'if(rank or rank0_code==0)else'FAILED',host='finite-host',temporary=False,migration_required=False,command=command,cwd=str(source))
  write(job/'RESULT.json',receipt)
  node=dict(job=str(job),host='finite-host',source=str(source),command=command,status='RUNNING')
  entry['ranks'][str(rank)]=node
  write(output/f'rank{rank}.json',dict(status='STOPPED_AT_SCENE_BOUNDARY'if rank==0 else'COMPLETE',world=2,rank=rank,identity=ident['sha256'],host='finite-host'))
 owner=E.EvaluationOwner.__new__(E.EvaluationOwner);owner.registry=dict(runs=[dict(id='ARM')]);owner.state=dict(runs={},evaluations={'entry':entry});owner.helper_active=lambda:False;owner.flush=lambda:None
 return root,owner,entry,cache


def test_owned_resource_exit0_requeues_same_shard_without_touching_cache_or_sibling(tmp_path):
 root,owner,entry,cache=fixture(tmp_path);spec=(Path(entry['evaluation'])/'SPEC.json').read_bytes();command=list(entry['ranks']['0']['command']);oldjob=entry['ranks']['0']['job']
 grant=dict(temporary_hosts=['finite-host'],expires_unix=20000,checkpoint_guard_seconds=1200)
 with patch.object(E,'ROOT',root),patch.object(E,'allowed_hosts',return_value=[]),patch.object(C,'authorization',return_value=(grant,{})),patch.object(E.time,'time',return_value=18800):owner.advance([])
 node=entry['ranks']['0'];assert node['status']=='RESOURCE_REQUEUE'and'job'not in node and'result'not in node
 assert node['command']==command and node['resource_returns'][0]['job']==oldjob and node['resource_returns'][0]['result']['temporary']is False
 assert entry['ranks']['1']['status']=='COMPLETE'and entry['status']!='FAILED_REVIEW_REQUIRED'
 assert cache.read_text()=='existing-scene-cache'and(Path(entry['evaluation'])/'SPEC.json').read_bytes()==spec
 with patch.object(E,'allowed_hosts',return_value=[])as hosts:
  assert owner.samples()==[]
  hosts.assert_called_once_with(admit_long_job=True)


def test_nonresource_stop_and_child_error_keep_review_but_harvest_complete_sibling(tmp_path):
 grant=dict(temporary_hosts=['finite-host'],expires_unix=20000,checkpoint_guard_seconds=1200)
 # Allowed host + ordinary manual scene stop is not a resource migration.
 root,owner,entry,cache=fixture(tmp_path/'manual')
 with patch.object(E,'ROOT',root),patch.object(E,'allowed_hosts',return_value=['finite-host']),patch.object(C,'authorization',return_value=(grant,{})),patch.object(E.time,'time',return_value=17000):owner.advance([])
 assert entry['status']=='FAILED_REVIEW_REQUIRED'and entry['ranks']['0']['status']=='FAILED_REVIEW_REQUIRED'and'job'in entry['ranks']['0']
 assert entry['ranks']['1']['status']=='COMPLETE'and'resource_returns'not in entry['ranks']['0']
 # A real nonzero child failure never enters the resource EXIT0 path.
 root,owner,entry,cache=fixture(tmp_path/'failure',rank0_code=1)
 with patch.object(E,'ROOT',root),patch.object(E,'allowed_hosts',return_value=[]),patch.object(C,'authorization',return_value=(grant,{})),patch.object(E.time,'time',return_value=18800):owner.advance([])
 assert entry['status']=='FAILED_REVIEW_REQUIRED'and entry['ranks']['0']['status']=='FAILED_REVIEW_REQUIRED'and'job'in entry['ranks']['0']
 assert entry['ranks']['1']['status']=='COMPLETE'and'resource_returns'not in entry['ranks']['0']
