"""P opts into existing exact owned-eval yield; no live owner or STOP writes."""
import json,os,socket,time
from pathlib import Path
from tools.qwen_world_performance import launch as L
from tools.qwen_world_safety_exploration import native_priority as N,eval_owner as E
from starVLA.rl.qwen_world_safety_exploration import contracts as C
from test_world_finite_resource_eval import fixture


def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n')
def waiter(root,arm,host):
 state=root/'owners/NATIVE/STATE.json';now=time.time()
 run=dict(id=arm,status='REGISTERED',owner_state=str(state),gates=[1616],deployment_candidates=[dict(host=host,minimum_free_MiB=32768,lease_slot='shared')])
 n=dict(status='WAITING_RESOURCE',updates=808,stage_target_updates=1616,heartbeat_unix=now,stages={'1616':{}})
 write(root/'RUNS.json',dict(runs=[run]));write(state,dict(runs={arm:n}));write(state.parent/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname()))
 return state,n


def test_P_waiter_and_owned_yield_use_existing_routes_receipts_and_keep_old_projects(tmp_path,monkeypatch):
 root,owner,entry,cache=fixture(tmp_path);old=tmp_path/'old_F';state,n=waiter(root,'S1_NOMINAL_RESCUE','finite-host');oldstate,oldn=waiter(old,'old_control','old-host')
 monkeypatch.setattr(L,'ROOT',root);monkeypatch.setattr(N,'STUDIES',(old,));monkeypatch.setattr(N,'ADVISORY',tmp_path/'advisory.json');monkeypatch.setattr(N,'allowed_hosts',lambda **kw:['finite-host','old-host'])
 L.register_native_priority_study();L.register_native_priority_study()
 assert N.STUDIES==(old,root)  # Old defaults/order retained, no duplicate P.
 demands=N.collect_waiters();assert {x['arm']for x in demands}=={'old_control','S1_NOMINAL_RESCUE'}
 demand=next(x for x in demands if x['arm']=='S1_NOMINAL_RESCUE')
 samples=[('finite-host',dict(gpus=[dict(uuid=str(i),free_MiB=50000,total_MiB=81920)for i in range(8)],owners=[]))]
 assert N.filter_eval_routes([dict(host='finite-host',gpu=0)],samples)==[]
 N.reserve_yield_lane(demand['key'],'finite-host')
 node=entry['ranks']['0'];job=Path(node['job']);spec=(Path(entry['evaluation'])/'SPEC.json').read_bytes();command=list(node['command'])
 intent=N.make_yield_receipt(entry,0,node,demand);write(job/'OWNED_NATIVE_PRIORITY_YIELD.json',intent);(job/'STOP_REQUESTED').touch()
 for path in(job/'RESULT.json',Path(entry['evaluation'])/'rank0.json'):
  data=json.loads(path.read_text());data['end_unix']=time.time()+1;write(path,data)
 monkeypatch.setattr(E,'ROOT',root);monkeypatch.setattr(E,'allowed_hosts',lambda **kw:['finite-host']);monkeypatch.setattr(C,'authorization',lambda:({},None))
 owner.advance([])
 assert node['status']=='RESOURCE_REQUEUE'and'job'not in node
 assert node['resource_returns'][0]['priority_yield_receipt']['sha256']==intent['sha256']
 assert node['command']==command and entry['ranks']['1']['status']=='COMPLETE'
 assert cache.read_text()=='existing-scene-cache'and(Path(entry['evaluation'])/'SPEC.json').read_bytes()==spec
 n['status']='TRAINING';n['stages']['1616']['job']='existing-native';write(state,dict(runs={'S1_NOMINAL_RESCUE':n}))
 assert {x['arm']for x in N.collect_waiters()}=={'old_control'}


def test_both_performance_owner_factories_register_P_without_changing_default_projects(monkeypatch,tmp_path):
 from tools.qwen_world_performance import evaluate as V
 old=N.STUDIES
 monkeypatch.setattr(N,'STUDIES',old);monkeypatch.setattr(L,'ROOT',tmp_path);monkeypatch.setattr(L,'bootstrap',lambda:None);monkeypatch.setattr(V,'bootstrap',lambda:None)
 monkeypatch.setattr(E,'KNOWN_ROOTS',E.KNOWN_ROOTS);monkeypatch.setattr(E,'completion_priority',E.completion_priority)
 L.owner_class();assert N.STUDIES==old+(tmp_path,)
 N.STUDIES=old;V.owner_class();assert N.STUDIES==old+(tmp_path,)


def test_pinned_waiter_only_reserves_overlap_not_unrelated_native_hosts(monkeypatch,tmp_path):
 from starVLA.rl.qwen_world_safety_exploration import resources as R
 from tools.qwen_world_safety_exploration import known_peers as K
 monkeypatch.setattr(L,'bootstrap',lambda:None);monkeypatch.setattr(L,'ROOT',tmp_path);monkeypatch.setattr(N,'STUDIES',N.STUDIES)
 monkeypatch.setattr(C,'allowed_hosts',lambda **kw:['TSH5','TSH2','WM4'])
 monkeypatch.setattr(E,'lease_free',lambda p:True);monkeypatch.setattr(K,'paths_for_root',lambda root:[])
 def route(h):return dict(host=h,lease_slot='primary',minimum_free_MiB=32768)
 high=dict(id='S1',priority=0,deployment_candidates=[route('TSH5')]);lower=dict(id='S0',priority=3,root=str(tmp_path/'runs/S0'),deployment_candidates=[route('TSH5'),route('TSH2')])
 seen=[]
 def sample(host):
  seen.append(host)
  return dict(host=host,gpus=[dict(index=i,uuid=str(i),free_MiB=81920,total_MiB=81920,utilization=0)for i in range(8)],owners=[],worker_python_exists=True,MemAvailable_bytes=200*2**30,load1=1,CPU_affinity=128)
 monkeypatch.setattr(R,'probe',sample)
 Owner=L.owner_class();o=Owner.__new__(Owner);o.runs=[high,lower];o.registry={'runs':o.runs};o.state=dict(runs={'S1':dict(status='WAITING_RESOURCE')})
 assert L.priority_reserved_hosts(lower,o.runs,o.state['runs'])=={'TSH5'}
 assert o.candidate(lower)[0]['host']=='TSH2'and seen==['TSH2']
 seen.clear();fast=dict(lower,id='FAST',root=str(tmp_path/'runs/FAST'),deployment_candidates=[route('WM4')]);assert o.candidate(fast)[0]['host']=='WM4'and seen==['WM4']
 seen.clear();blocked=dict(lower,deployment_candidates=[route('TSH5')]);assert o.candidate(blocked)is None and seen==[]
 # A failed higher arm does not reserve any route indefinitely.
 o.state['runs']['S1']['status']='FAILED_REVIEW_REQUIRED'
 assert L.priority_reserved_hosts(lower,o.runs,o.state['runs'])==set()
