"""E-local advisory training lanes and exact owned-eval priority-yield receipts.

No process signals, admission relaxation or foreign-owner mutations. A lane is
only advisory; UUID/visible-owner/headroom/lease checks still decide admission.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

BASE=Path('/mnt/project/qwen-world-safety-exploration-20261031')
STUDIES=tuple(BASE/p for p in('standard_training_20261101','revival_training_20261101','improvement_training_20261101'))
ADVISORY=BASE/'resources/NATIVE_TRAINING_DEMAND.json'
PROTECTED={'training-vla-zt','training-vlawm-tsinghua-zt'}
WAITING={'WAITING_RESOURCE','WAITING_AUTHORIZED_RESOURCE','WAITING_RESOURCE_CAPACITY','WAITING_AUTHORIZED_MIGRATION','RESOURCE_REQUEUE','WAITING_RESOURCE_REQUEUE'}
TTL=180
YIELD_SCHEMA='world_owned_native_priority_yield_v1'

def read(p,default=None):
 p=Path(p)
 return json.loads(p.read_text())if p.exists()else default

def collect_waiters(*,now=None):
 now=time.time()if now is None else now;legal=set(allowed_hosts(admit_long_job=True))-PROTECTED;out=[]
 for root in STUDIES:
  if(root/'STOP_REQUESTED').exists():continue
  registry=read(root/'RUNS.json',{})
  for run in registry.get('runs',[]):
   if run.get('new_training_authorized')is False or run.get('status','').startswith(('STOPPED','PAUSED','CANCELLED'))or any(run.get(k)for k in('standard_disposition','revival_disposition','improvement_disposition')):continue
   sp=Path(run['owner_state']);n=read(sp,{}).get('runs',{}).get(run['id'],{});target=n.get('stage_target_updates');stage=n.get('stages',{}).get(str(target),{})
   if(n.get('status')not in WAITING or now-n.get('heartbeat_unix',0)>TTL or target not in run['gates'] or n.get('updates',0)>=target or stage.get('job')or stage.get('complete')):continue
   owner=read(sp.parent/'OWNER.json',{})
   if owner.get('host')in(None,socket.gethostname()):
    try:
     p=Path('/proc')/str(owner['pid']);v=(p/'stat').read_text().split()
     if int(v[21])!=owner['start_ticks']or v[2]in('Z','X'):continue
    except(FileNotFoundError,KeyError):continue
   routes=[dict(r)for r in run['deployment_candidates']if r['host']in legal]
   if not routes:continue
   last=read(Path(n.get('job','/nonexistent'))/'RESULT.json',{})
   since=last.get('end_unix',n.get('heartbeat_unix',now))
   out.append(dict(key=str(root)+'::'+run['id']+'::'+str(target),study=str(root),arm=run['id'],target_updates=target,updates=n.get('updates',0),waiting_since_unix=since,routes=routes,world=8,priority=int(run.get('native_resource_priority',0))))
 return sorted(out,key=lambda x:(-x['priority'],x['waiting_since_unix'],x['key']))

def refresh_advisory(samples=()):
 """Stable whole-host lanes only for actual live8rank waiting demands."""
 now=time.time();demands=collect_waiters(now=now);legal=set(allowed_hosts(admit_long_job=True))-PROTECTED;byhost=dict(samples)
 ADVISORY.parent.mkdir(parents=True,exist_ok=True)
 with ADVISORY.with_suffix('.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old=read(ADVISORY,{});previous=old.get('reservations',{});reservations={};used=set()
  for d in demands:
   prior=previous.get(d['key']);hosts={r['host']for r in d['routes']}
   if prior and prior['host']in legal&hosts and(prior['expires_unix']>now or not byhost):
    reservations[d['key']]=dict(prior,expires_unix=now+TTL);used.add(prior['host'])
  for d in demands:
   if d['key']in reservations:continue
   choices=[]
   for host,s in byhost.items():
    routes=[r for r in d['routes']if r['host']==host]
    if host in used or host not in legal or not routes or len(s.get('gpus',[]))!=8:continue
    potential=[]
    for gpu in s['gpus']:
     # Known evals mayfinish naturally; do not claimtheir memory is free now.
     released=sum(o.get('used_MiB',0)or 0 for o in s.get('owners',[])if o.get('uuid')==gpu['uuid']and'evaluat'in o.get('entry','').lower())
     potential.append(min(gpu['total_MiB'],gpu['free_MiB']+released))
    if min(potential)<min(r['minimum_free_MiB']for r in routes):continue
    choices.append((-min(potential),-min(g['free_MiB']for g in s['gpus']),host))
   if choices:
    host=min(choices)[2];reservations[d['key']]=dict(host=host,arm=d['arm'],study=d['study'],target_updates=d['target_updates'],expires_unix=now+TTL);used.add(host)
  payload=dict(schema='world_E_native_training_priority_advisory_v1',at_unix=now,demands=demands,reservations=reservations,scope='E/F/G/H evaluation dispatch only; foreigndispatchers not modified',no_preemption=True,no_admission_relaxation=True)
  payload['sha256']=fingerprint(payload);atomic_json(ADVISORY,payload);return payload

def reserve_yield_lane(demand_key,host):
 """Explicit Root-only advisory pin before yielding own evals; never dispatches."""
 demands=collect_waiters();match=next((d for d in demands if d['key']==demand_key),None)
 if match is None or host not in {r['host']for r in match['routes']}:raise ValueError('Yield lane requires a currentlylive authorized native demand')
 now=time.time();ADVISORY.parent.mkdir(parents=True,exist_ok=True)
 with ADVISORY.with_suffix('.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);old=read(ADVISORY,{})
  reservations={k:v for k,v in old.get('reservations',{}).items()if k!=demand_key and v['host']!=host}
  reservations[demand_key]=dict(host=host,arm=match['arm'],study=match['study'],target_updates=match['target_updates'],expires_unix=now+TTL,explicit_operator_yield=True)
  payload=dict(schema='world_E_native_training_priority_advisory_v1',at_unix=now,demands=demands,reservations=reservations,scope='E/F/G/H evaluation dispatch only; foreign dispatchers not modified',no_preemption=True,no_admission_relaxation=True)
  payload['sha256']=fingerprint(payload);atomic_json(ADVISORY,payload);return payload

def filter_eval_routes(routes,samples):
 p=refresh_advisory(samples);held={v['host']for v in p['reservations'].values()}
 return [r for r in routes if r['host']not in held]

def _sealed(value):
 b=dict(value);saved=b.pop('sha256',None)
 if not saved or fingerprint(b)!=saved:raise ValueError('Priorityyield metadata seal changed')
 return value

def make_yield_receipt(entry,rank,node,requested_for):
 """Pure Root helper: returns intent; caller writesintent thenown STOP_REQUESTED."""
 output=Path(entry['evaluation']).resolve();study=output.parents[2];job=Path(node['job']).resolve()
 if study not in(STUDIES+(BASE,))or not job.is_relative_to(study/'jobs/evaluation'):raise ValueError('Only owned E/F/G/H evaluation jobs')
 spec=_sealed(read(output/'SPEC.json'));identity=_sealed(read(output/'identity.json'))
 if spec['sha256']!=entry['spec_sha256']or spec['world']!=entry['world']or node['source']!=spec['source']:raise ValueError('Changed immutable evaluation binding')
 if node['host']not in allowed_hosts(admit_long_job=True):raise ValueError('Explicitpriorityyield requires a currentlyauthorizedhost')
 command=node['command']
 def arg(flag):return command[command.index(flag)+1]if command.count(flag)==1 and command.index(flag)+1<len(command)else None
 if not 0<=rank<entry['world']or arg('--rank')!=str(rank)or arg('--world')!=str(entry['world'])or arg('--seed')!='2101'or arg('--cohort')!=entry['cohort']or arg('--output')is None or Path(arg('--output')).resolve()!=output:raise ValueError('Priorityyield command/rank/seed/output mismatch')
 body=dict(schema=YIELD_SCHEMA,id=uuid.uuid4().hex,job=str(job),study_root=str(study),evaluation=str(output),spec_sha256=spec['sha256'],identity_sha256=identity['sha256'],rank=int(rank),world=entry['world'],source=node['source'],host=node['host'],seed=2101,command_sha256=fingerprint(node['command']),requested_unix=time.time(),requested_for=requested_for,reason='OWNED_NATIVE_PRIORITY_YIELD',requester='root_operator',same_scene_cache_must_be_reused=True)
 return dict(body,sha256=fingerprint(body))

def verified_yield(entry,rank,node,result,rank_receipt):
 job=Path(node['job']).resolve();p=job/'OWNED_NATIVE_PRIORITY_YIELD.json'
 if not p.exists()or not(job/'STOP_REQUESTED').exists():return None
 try:
  v=_sealed(read(p));output=Path(entry['evaluation']).resolve();identity=_sealed(read(output/'identity.json'))
  expected=dict(schema=YIELD_SCHEMA,job=str(job),study_root=str(output.parents[2]),evaluation=str(output),spec_sha256=entry['spec_sha256'],identity_sha256=identity['sha256'],rank=rank,world=entry['world'],source=node['source'],host=node['host'],seed=2101,command_sha256=fingerprint(node['command']),reason='OWNED_NATIVE_PRIORITY_YIELD',requester='root_operator')
  if any(v.get(k)!=x for k,x in expected.items()):return None
  if result.get('returncode')!=0 or rank_receipt.get('status')!='STOPPED_AT_SCENE_BOUNDARY':return None
  requested=v['requested_unix']
  if(not isinstance(requested,(int,float))or requested>time.time()+5 or result.get('end_unix',0)<requested-5 or rank_receipt.get('end_unix',0)<requested-5):return None
  return v
 except(KeyError,TypeError,ValueError):return None
