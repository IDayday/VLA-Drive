"""Finite CPU waiter. No signal/dispatch: one exact S7 placement amendment only.

Wait for old I's natural10000 completion AND actual worker/native release.
Then preserve every current registry field except append-only S7 routes and the
registry's derived registration hash. Native admission/resume remains owned by
PerformanceOwner, including S1 priority/max2/UUID/32GiB/unknown-owner checks.
"""
import argparse,copy,fcntl,hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
P=Path('/mnt/project/qwen-world-performance-rl-20261103')
OLD=Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102')
OLD_ID='INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC';NEW_ID='S7_STRUCTURED_PROGRESS'
HOST='training-vlawm-tsinghua-zt5';TASK='qwen_world_safety_exploration_20261031'
D=P/'operations/S7_PERMANENT_CAPACITY_WAIT';SCHEMA='world_S7_after_natural_I10000_permanent_placement_v1'
class Pending(Exception):pass
class Review(Exception):pass
class Done(Exception):pass

def read(p,default=None):return json.loads(Path(p).read_text())if Path(p).exists()else default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);v=dict(x);v['sha256']=fp(v);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(q,p);return v
def same(a,b):
 try:return Path(a).resolve()==Path(b).resolve()
 except(TypeError,ValueError,OSError):return False
def arg(c,n):return c[c.index(n)+1]if c.count(n)==1 and c.index(n)+1<len(c)else None

def inspect_natural_I(root=OLD):
 """Publication/metadata checks only; no optimizer/rank tensor loads/hashes."""
 root=Path(root).resolve();reg=read(root/'RUNS.json',{});run=next((x for x in reg.get('runs',[])if x.get('id')==OLD_ID),None)
 if not run:raise Review('Old I registered run missing')
 if reg.get('task')!=TASK or run.get('target_updates')!=10000:raise Review('Old I approved target/task changed; do not assume capacity release')
 if any(run.get(k)for k in('disposition','standard_disposition','improvement_disposition'))or(root/'STOP_REQUESTED').exists():raise Review('Old I explicitly stopped/reserved; natural-completion condition absent')
 state=read(run['owner_state'],{}).get('runs',{}).get(OLD_ID,{})
 if state.get('status','').startswith(('FAILED','CANCELLED','STOPPED')):raise Review('Old I current native status requires review: '+state['status'])
 stage=state.get('stages',{}).get('10000',{})
 if not stage.get('complete'):
  target=state.get('stage_target_updates');current=state.get('stages',{}).get(str(target),{});result=read(Path(current.get('job','/nonexistent'))/'RESULT.json',{})
  if result.get('returncode')not in(None,0,75):raise Review('Old I actual current child failed; no automatic capacity amendment')
  raise Pending('Old I10000 stage not complete; sealed/latest='+str(state.get('updates'))+' target='+str(target))
 job=Path(stage.get('job','/nonexistent')).resolve();rr=Path(run['root']).resolve();cp=Path(stage.get('checkpoint','/nonexistent')).resolve()
 if not job.is_relative_to(root/'jobs'/OLD_ID)or not cp.is_relative_to(rr/'checkpoints')or'.incomplete'in cp.name:raise Review('Old I stage job/checkpoint escaped registered run')
 result=read(job/'RESULT.json');dispatch=read(job/'DISPATCH.json');running=read(job/'RUNNING.json')
 if result is None:raise Pending('Old I stage marker published before actual RESULT')
 if result.get('returncode')!=0 or result.get('status')!='EXITED'or result.get('forced_expiry')or result.get('migration_required'):raise Review('Old I did not naturally EXIT0')
 if(job/'STOP_REQUESTED').exists()or(rr/'STOP_REQUESTED').exists():raise Review('Old I ownSTOP is present; do not call completion natural')
 command=result.get('command',[]);effective=result.get('cwd');host=result.get('host')
 if(effective not in(run['source'],run.get('execution_source',run['source']))or dispatch.get('source')!=effective
    or dispatch.get('command')!=command or running.get('command')!=command or running.get('cwd')!=effective
    or host!=stage.get('host')or running.get('host')!=host or dispatch.get('host')!=host
    or any(dispatch.get(k)!=result.get(k)or running.get(k)!=result.get(k)for k in('worker_pid','worker_start_ticks'))
    or any(running.get(k)!=result.get(k)for k in('child_pid','child_start_ticks'))
    or any(not isinstance(result.get(k),int)for k in('worker_pid','worker_start_ticks','child_pid','child_start_ticks'))
    or run.get('trainer_module')not in command
    or arg(command,'--stop-after-update')!='10000'or not same(arg(command,'--output'),rr)
    or not same(arg(command,'--config'),run['config'])or not same(arg(command,'--il-checkpoint'),run['parent_checkpoint'])):
  raise Review('Old I exact job/command/execution/worker binding changed')
 if sha(run['config'])!=run['config_sha256']:raise Review('Old I scientific config changed')
 seal=read(cp/'COMPLETE',{});files=read(cp/'files.json',{});identity=read(cp/'identity.json',{});tx=read(cp/'transaction.json',{})
 required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{r}.pt'for r in range(8)}
 if(seal.get('schema')!=2 or seal.get('updates')!=10000 or seal.get('next_inner')!=0 or seal.get('world')!=8 or seal.get('phase')not in('READY','COMPLETE')or seal.get('files_sha256')!=sha(cp/'files.json')):raise Review('Old I10000 is not an atomic complete-window world8 publication')
 if not required<=set(files)or any(Path(n).is_absolute()or not(cp/n).resolve().is_relative_to(cp)or not(cp/n).is_file()or(cp/n).stat().st_size<=0 for n in files):raise Review('Old I native inventory missing/escaping')
 body=dict(identity);digest=body.pop('sha256',None)
 import yaml
 if(digest!=fp(body)or identity!=read(rr/'identity.json')or identity.get('purpose')!='formal'or identity.get('world')!=8 or identity.get('recipe')!=yaml.safe_load(Path(run['config']).read_text())):raise Review('Old I native identity/config changed')
 for n in('identity.json','transaction.json'):
  if sha(cp/n)!=files[n]:raise Review('Old I native metadata inventory changed')
 ranks=tx.get('ranks',[])
 if len(ranks)!=8 or [r.get('rank')for r in ranks]!=list(range(8))or any(r.get('run')!=digest or r.get('updates')!=10000 or r.get('next_inner')!=0 or r.get('phase')!=seal['phase']for r in ranks):raise Review('Old I actual8rank transaction mismatch')
 latest=read(rr/'latest.json',{})
 if latest.get('updates')!=10000 or not same(latest.get('checkpoint'),cp)or latest.get('files_sha256')!=seal['files_sha256']:raise Review('Old I latest moved beyond/differs from natural10000 stage')
 result_files={}
 for rank in range(8):
  f=rr/f'result_rank{rank}.json';r=read(f,{})
  if r.get('rank')!=rank or r.get('updates')!=10000 or r.get('next_inner')!=0 or r.get('host')!=host or r.get('qualification')is not False or not same(r.get('checkpoint'),cp):raise Review('Old I8rank actual results do not match sealed10000')
  result_files[str(f)]=sha(f)
 uuids=result.get('gpu_uuids',[])
 if len(uuids)!=8 or len(set(uuids))!=8 or uuids!=running.get('gpu_uuids'):raise Review('Old I physical8UUID result changed')
 return dict(old_registry=str(root/'RUNS.json'),old_registry_sha256=sha(root/'RUNS.json'),run_root=str(rr),job=str(job),host=host,result=result,
  checkpoint=dict(path=str(cp),updates=10000,world=8,files_sha256=seal['files_sha256'],identity_sha256=digest),gpu_uuids=uuids,
  file_receipts={str(job/n):sha(job/n)for n in('RESULT.json','DISPATCH.json','RUNNING.json')},native_metadata={str(cp/n):sha(cp/n)for n in('COMPLETE','files.json','identity.json','transaction.json')},rank_result_files=result_files)

def remote_release(proof,remote,python):
 """One final administrative read on a still-authorized permanent host only."""
 result=proof['result'];known=[dict(pid=result['worker_pid'],start_ticks=result['worker_start_ticks']),dict(pid=result['child_pid'],start_ticks=result['child_start_ticks'])]
 known+=result.get('known_rank_processes',[])+result.get('guard_rank_processes',[])
 code='''import json,subprocess,socket
from pathlib import Path
saved=SAVED;output=OUTPUT;alive=[]
for r in saved:
 try:
  s=(Path('/proc')/str(r['pid'])/'stat').read_text();v=s[s.rfind(')')+2:].split()
  if int(v[19])==r['start_ticks'] and v[0]not in('Z','X'):alive.append(dict(pid=r['pid'],start_ticks=r['start_ticks']))
 except FileNotFoundError:pass
uuids=[u.strip()for u in subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()]
remaining=[]
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
 pid,uuid=row.split(',');p=Path('/proc')/pid.strip()
 try:
  args=[a for a in(p/'cmdline').read_bytes().decode().split('\\0')if a]
  if '--output'in args and Path(args[args.index('--output')+1]).resolve()==Path(output).resolve():remaining.append(dict(pid=int(pid),uuid=uuid.strip()))
 except (OSError,ValueError,IndexError):pass
print(json.dumps(dict(host=socket.gethostname().removesuffix('-worker-0'),alive_exact_processes=alive,remaining_old_I_GPU_outputs=remaining,gpu_uuids=uuids)))
'''.replace('SAVED',repr(known)).replace('OUTPUT',repr(proof['run_root']))
 value=json.loads(remote(proof['host'],[python,'-c',code]))
 if value.get('host')!=proof['host']or value.get('gpu_uuids')!=proof['gpu_uuids']:raise Review('Old I physical release host/UUID mapping changed')
 if value.get('alive_exact_processes')or value.get('remaining_old_I_GPU_outputs'):raise Pending('Old I worker/child/native still naturally exiting')
 lock=Path(proof['job'])/'worker.lock'
 if not lock.is_file():raise Review('Old I exact worker lock absent')
 with lock.open('r')as handle:
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise Pending('Old I worker still holds exact invocation lock')
 value['exact_invocation_worker_lock_released']=True;value['verified_unix']=time.time();return value

def eligible_S7(registry,native):
 if registry.get('task')!=TASK or not same(registry.get('root'),P):raise Review('Current P registry root/task changed')
 run=next((r for r in registry.get('runs',[])if r.get('id')==NEW_ID),None)
 if run is None:raise Pending('S7 not formally registered yet')
 state=native.get('runs',{}).get(NEW_ID,{})
 if run.get('new_training_authorized')is False or any(run.get(k)for k in('disposition','standard_disposition','revival_disposition','improvement_disposition'))or run.get('status','').startswith(('STOPPED','PAUSED','CANCELLED','FAILED'))or state.get('status','').startswith(('STOPPED','PAUSED','CANCELLED','FAILED')):raise Done('S7 cancelled/reserved/requires review; no route mutation')
 if state.get('status')=='COMPLETED_AT_REGISTERED_CAP'or state.get('updates',0)>=run['target_updates']:raise Done('S7 already completed its approved cap')
 protocol=run.get('resource_protocol',{})
 if protocol.get('schema')!='world_performance_measured_native_sharing_v2'or protocol.get('maximum_known_native_per_host')!=2 or protocol.get('maximum_campaign_native_per_host')!=2 or protocol.get('minimum_free_MiB',0)<32768:raise Review('S7 exact max2 measured-sharing protocol changed')
 if not same(run['root'],P/'runs'/NEW_ID)or not same(run.get('execution_output_root'),run['root']):raise Review('S7 logical/transparent output path changed')
 return run

def append_routes(registry,native):
 """Pure, preserve all other rows/science/optimizer/budget metadata verbatim."""
 old=copy.deepcopy(registry);run=eligible_S7(old,native);routes=run['deployment_candidates'];added=[]
 for slot in('primary','shared','research'):
  found=[r for r in routes if r.get('host')==HOST and r.get('lease_slot')==slot]
  if found:
   if len(found)!=1 or found[0].get('minimum_free_MiB',0)<32768:raise Review('Existing permanent route conflicts with strict admission')
   continue
  route=dict(host=HOST,lease_slot=slot,minimum_free_MiB=32768);routes.append(route);added.append(route)
 if added:old['registration_sha256']=fp({k:v for k,v in old.items()if k!='registration_sha256'})
 check=copy.deepcopy(old);original=next(r for r in registry['runs']if r['id']==NEW_ID);next(r for r in check['runs']if r['id']==NEW_ID)['deployment_candidates']=copy.deepcopy(original['deployment_candidates'])
 check['registration_sha256']=registry.get('registration_sha256')
 if check!=registry:raise Review('Unexpected non-placement registry delta')
 return old,added

def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run',action='store_true');ap.add_argument('--interval',type=float,default=60);ap.add_argument('--max-hours',type=float,default=72);ap.add_argument('--operations-source',type=Path,default=P/'runtime/ops_v6');a=ap.parse_args()
 if not a.run:ap.error('Explicit --run required; preparation does not launch this waiter')
 if not 30<=a.interval<=60 or not 0<a.max_hours<=72:ap.error('30..60s metadata polling and maximum72hours only')
 os.environ.update(WORLD_COMPARE_ROOT=str(P),WORLD_COMPARE_TASK=TASK,CUDA_VISIBLE_DEVICES='');sys.path.insert(0,str(a.operations_source.resolve()))
 from starVLA.rl.qwen_world_safety_exploration.contracts import authorization,allowed_hosts
 from starVLA.rl.qwen_world_safety_exploration.resources import remote,OPERATOR_PYTHON
 D.mkdir(parents=True,exist_ok=True);lock=(D/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(D/'APPLIED.json').exists():print('ALREADY_APPLIED');return
 started=time.time();deadline=started+a.max_hours*3600
 write(D/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),host=socket.gethostname(),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),started_unix=started,deadline_unix=deadline,command=sys.argv))
 expected=None
 try:
  while time.time()<deadline:
   if(D/'STOP_OWNER').exists():write(D/'STATE.json',dict(status='STOPPED_CPU_ONLY',at_unix=time.time(),RUNS_mutated=False));return
   try:
    if(P/'STOP_REQUESTED').exists():raise Done('P global STOP; no placement amendment')
    registry=read(P/'RUNS.json',{});native=read(P/'owners/PERFORMANCE_NATIVE/STATE.json',{});run=eligible_S7(registry,native)
    binding={k:run[k]for k in('id','root','source','source_commit','config','config_sha256','parent_checkpoint')}
    if expected is None:expected=binding
    elif binding!=expected:raise Review('S7 scientific/parent/source binding changed while waiter active')
    grant,_=authorization();legal=allowed_hosts(admit_long_job=True)
    if HOST not in grant.get('permanent_hosts',[])or HOST not in legal:raise Pending('TSH5 not currently legal permanent host; no probe/amendment')
    proof=inspect_natural_I()
    if proof['host']not in {HOST,'training-vlawm-tsinghua-zt3'}or proof['host']not in grant.get('permanent_hosts',[])or proof['host']not in legal:raise Review('Old I release host no longer a legal permanent administrative read target')
    release=remote_release(proof,remote,OPERATOR_PYTHON)
    # Only the current registry is modified under its own lock. No native/eval
    # CPU is stopped, signalled or handed off and no GPU job is dispatched.
    with(P/'RUNS.lock').open('a')as guard:
     fcntl.flock(guard,fcntl.LOCK_EX)
     grant,_=authorization()
     if HOST not in grant.get('permanent_hosts',[])or HOST not in allowed_hosts(admit_long_job=True):raise Pending('Permanent authorization changed before amendment')
     if(P/'STOP_REQUESTED').exists()or(D/'STOP_OWNER').exists():raise Done('STOP arrived before mutation')
     if sha(OLD/'RUNS.json')!=proof['old_registry_sha256']:raise Review('Old I budget/registration changed after release proof')
     current=read(P/'RUNS.json',{});native=read(P/'owners/PERFORMANCE_NATIVE/STATE.json',{});run=eligible_S7(current,native)
     if{k:run[k]for k in expected}!=expected:raise Review('S7 binding changed under RUNS lock')
     updated,added=append_routes(current,native);before=sha(P/'RUNS.json')
     intent=write(D/'READY_TO_APPLY.json',dict(schema=SCHEMA,status='VERIFIED_RELEASE_METADATA_READY',old_I=proof,actual_release=release,registered_S7_binding=expected,added_routes=added,registry_before_sha256=before,at_unix=time.time(),capacity_claim='No promise of free8GPUs. Existing strict native owner must recheck max2/32GiB/UUID/leases/known-owner and S1 priority.'))
     if added:
      tmp=P/('RUNS.json.S7-placement-'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(updated,indent=2)+'\n');os.replace(tmp,P/'RUNS.json')
     write(D/'APPLIED.json',dict(schema=SCHEMA,status='APPLIED_PLACEMENT_ONLY'if added else'ALREADY_PRESENT_AFTER_VERIFIED_RELEASE',release_receipt_sha256=intent['sha256'],registry_before_sha256=before,registry_after_sha256=sha(P/'RUNS.json'),added_routes=added,at_unix=time.time(),new_GPU=False,signals=False,I_changed=False,S1_changed=False))
    write(D/'STATE.json',dict(status='COMPLETE_PLACEMENT_ONLY',applied=str(D/'APPLIED.json'),at_unix=time.time()));return
   except Pending as e:
    write(D/'STATE.json',dict(status='WAITING_READONLY',reason=str(e),heartbeat_unix=time.time(),deadline_unix=deadline,RUNS_mutated=False))
    time.sleep(min(a.interval,max(0,deadline-time.time())))
  write(D/'STATE.json',dict(status='EXPIRED_REVIEW_NO_MUTATION',at_unix=time.time(),RUNS_mutated=False))
 except Done as e:write(D/'STATE.json',dict(status='ENDED_NO_MUTATION',reason=str(e),at_unix=time.time(),RUNS_mutated=False))
 except Exception:
  write(D/'FAILURE.json',dict(status='REVIEW_REQUIRED_NO_AUTOMATIC_RETRY',error=traceback.format_exc(),at_unix=time.time()));raise
if __name__=='__main__':main()
