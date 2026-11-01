"""Advisory live-waiter and exact priorityyield contracts; CPU metadata only."""
import json,os,socket,time
from pathlib import Path
from unittest.mock import patch
from tools.qwen_world_safety_exploration import native_priority as P,eval_owner as E
from starVLA.rl.qwen_world_safety_exploration import contracts as C
from test_world_finite_resource_eval import fixture


def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d)+'\n')


def test_only_true_waiter_reserves_lane_and_releases_when_native_started(tmp_path):
 root=tmp_path/'H';state=root/'owners/NATIVE/STATE.json';now=time.time()
 run=dict(id='native',status='REGISTERED',owner_state=str(state),gates=[12000],deployment_candidates=[dict(host='h1',minimum_free_MiB=32768,lease_slot='primary'),dict(host='h2',minimum_free_MiB=32768,lease_slot='research')])
 write(root/'RUNS.json',dict(runs=[run]));n=dict(status='WAITING_AUTHORIZED_RESOURCE',updates=10000,stage_target_updates=12000,heartbeat_unix=now,stages={'12000':{}});write(state,dict(runs={'native':n}));write(state.parent/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname()))
 samples=[(h,dict(gpus=[dict(uuid=h+str(i),free_MiB=50000,total_MiB=80000)for i in range(8)],owners=[]))for h in ['h1','h2']];routes=[dict(host=h,gpu=0)for h in ['h1','h2']]
 with patch.object(P,'STUDIES',(root,)),patch.object(P,'ADVISORY',tmp_path/'advisory.json'),patch.object(P,'allowed_hosts',return_value=['h1','h2']):
  assert len(P.collect_waiters())==1
  assert P.filter_eval_routes(routes,samples)==[routes[1]]
  P.reserve_yield_lane(P.collect_waiters()[0]['key'],'h2')
  assert P.filter_eval_routes(routes,samples)==[routes[0]]
  n['status']='TRAINING';n['stages']['12000']['job']='live-job';write(state,dict(runs={'native':n}))
  assert P.filter_eval_routes(routes,samples)==routes and not P.read(P.ADVISORY)['reservations']


def test_explicit_owned_yield_requeues_allowed_host_and_preserves_cache(tmp_path):
 root,owner,entry,cache=fixture(tmp_path);node=entry['ranks']['0'];job=Path(node['job']);out=Path(entry['evaluation']);now=time.time()
 with patch.object(P,'STUDIES',(root,)),patch.object(P,'allowed_hosts',return_value=['finite-host']):intent=P.make_yield_receipt(entry,0,node,dict(arm='retention',target_updates=5000))
 write(job/'OWNED_NATIVE_PRIORITY_YIELD.json',intent);(job/'STOP_REQUESTED').touch()
 for path in [job/'RESULT.json',out/'rank0.json']:
  d=json.loads(path.read_text());d['end_unix']=now+1;write(path,d)
 with patch.object(E,'ROOT',root),patch.object(E,'allowed_hosts',return_value=['finite-host']),patch.object(C,'authorization',return_value=(dict(temporary_hosts=[],expires_unix=now+1000,checkpoint_guard_seconds=200),{})):owner.advance([])
 assert node['status']=='RESOURCE_REQUEUE'and'job'not in node and node['resource_returns'][0]['reason']=='OWNED_NATIVE_PRIORITY_YIELD'
 assert node['resource_returns'][0]['priority_yield_receipt']['sha256']==intent['sha256']and cache.read_text()=='existing-scene-cache'
 assert entry['ranks']['1']['status']=='COMPLETE'
