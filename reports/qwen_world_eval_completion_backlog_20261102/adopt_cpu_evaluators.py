"""Exact CPU parent takeover, no signals; same-spec manual failed-rank recovery."""
import json,os,fcntl,time,subprocess,hashlib,shutil,copy
from pathlib import Path
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');O=E/'operations/EVALUATION_BACKLOG_COMPLETION_20261102';SOURCE=O/'runtime/completion_v1';PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
AUTH=E/'operations/MANUAL_SIGKILL_EVAL_RETRY_20261102/REVIEW_AUTHORIZATION.json'
CASES=[('F','standard_training_20261101','RESEARCH_EVALUATION',3494449,1502080151,'STANDARD_LAUNCH.json'),('G','revival_training_20261101','RESEARCH_EVALUATION',3494295,1502079867,'STANDARD_LAUNCH.json'),('H','improvement_training_20261101','IMPROVEMENT_EVAL',3491961,1502057046,'HELPER_LAUNCH.json')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.new.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n');os.replace(tmp,p)
def alive(pid,start):
 try:v=(Path('/proc')/str(pid)/'stat').read_text().split();return int(v[21])==start and v[2]not in('Z','X')
 except FileNotFoundError:return False
lock=(O/'adoption.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for tag,study,dirname,pid,start,helper in CASES:
 d=E/study/'owners'/dirname;owner=read(d/'OWNER.json');assert owner['pid']==pid and owner['start_ticks']==start and alive(pid,start),(study,owner)
 assert not(d/'STOP_OWNER').exists(),('foreignstopmarker',study)
 b=O/'BEFORE'/tag;b.mkdir(parents=True,exist_ok=True)
 for name in ('OWNER.json','STATE.json',helper):
  if(d/name).exists():shutil.copyfile(d/name,b/name)
 atomic(d/'STOP_OWNER',dict(reason='ROOT_DELEGATED_CPU_ONLY_COMPLETION_FIRST_ADOPTION',operation=str(O),pid=pid,start_ticks=start,at=time.time()))
atomic(O/'QUIESCE_REQUESTED.json',dict(at=time.time(),cases=CASES,no_signals=True))
end=time.time()+100
while any(alive(pid,start)for tag,study,dirname,pid,start,helper in CASES):
 if time.time()>end:raise RuntimeError('OwnedCPUdidnotnaturallyexit; noPIDsignals sent')
 time.sleep(1)
print('EXACT_THREE_OLD_CPU_PARENTS_NATURALLY_EXITED',flush=True)
from sys import path
path.insert(0,str(SOURCE))
os.environ['WORLD_COMPARE_ROOT']=str(E);os.environ['WORLD_COMPARE_TASK']='qwen_world_safety_exploration_20261031'
from starVLA.rl.qwen_world_safety_exploration.contracts import fingerprint
assert read(AUTH)['status']=='ROOT_MANUAL_SAME_SPEC_RETRY_AUTHORIZED'
records=[];launches=[]
for tag,study,dirname,pid,start,helper in CASES:
 root=E/study;d=root/'owners'/dirname
 with(d/'owner.lock').open('a')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(d/'STATE.json');b=O/'QUIESCED'/tag;b.mkdir(parents=True,exist_ok=True);shutil.copyfile(d/'STATE.json',b/'STATE.json')
  for case in read(AUTH)['exact_cases']:
   if case['study']!=study:continue
   key=case['evaluation_key'];rank=str(case['rank']);entry=state['evaluations'][key];node=entry['ranks'][rank];job=Path(case['job']);spec=read(Path(entry['evaluation'])/'SPEC.json');identity=read(Path(entry['evaluation'])/'identity.json');result=read(job/'RESULT.json');text=(job/'stdout.log').read_text(errors='replace')
   assert node['job']==str(job)and node['status']=='FAILED_REVIEW_REQUIRED'and entry['spec_sha256']==case['spec_sha256']==spec['sha256']
   assert result['returncode']==-9 and result['host']=='training-vlawm-zt5'and not result.get('forced_expiry')and result.get('child_pid')and result.get('child_start_ticks')
   assert result['command']==node['command']and result['cwd']==node['source']==spec['source']
   for v in (spec,identity):assert fingerprint({k:x for k,x in v.items()if k!='sha256'})==v['sha256']
   assert spec['world']==entry['world']==32 and spec['seed']==2101 and identity['seed']==2101
   assert 'Traceback'not in text and 'FloatingPointError'not in text and 'ValueError'not in text
   command=node['command'];assert command[command.index('--rank')+1]==rank and command[command.index('--world')+1]=='32'and command[command.index('--seed')+1]=='2101'
   a=O/'FAILURE_EVIDENCE'/tag/key/('rank'+rank);a.mkdir(parents=True,exist_ok=True)
   for n in ('RESULT.json','RUNNING.json','DISPATCH.json','stdout.log','worker.log'):
    if(job/n).exists():shutil.copyfile(job/n,a/n)
   atomic(a/'NODE_BEFORE.json',node);atomic(a/'SPEC.json',spec);atomic(a/'IDENTITY.json',identity)
   receipt=dict(schema='world_manual_same_spec_external_SIGKILL_retry_v1',classification='HOST_UNAVAILABLE_EXTERNAL_SIGKILL_CAUSE_UNRESOLVED',authorization=str(AUTH),authorization_sha256=sha(AUTH),case=case,original_result=result,stdout_sha256=sha(job/'stdout.log'),source=node['source'],command=list(command),spec_sha256=spec['sha256'],identity_sha256=identity['sha256'],same_scene_cache_preserved=True,new_success_not_assumed=True,at=time.time())
   atomic(a/'REVIEW.json',receipt)
   node.setdefault('operator_failures',[]).append(receipt);node.pop('job');node.pop('result',None);node.pop('error',None)
   node.update(status='RESOURCE_REQUEUE',operator_retry_hosts=['training-rl-zt2','training-vlawm-tsinghua-zt2','training-vlawm-zt'])
   entry.update(status='RESOURCE_REQUEUE');entry.pop('failure_rank',None);entry.pop('error',None)
   records.append(dict(study=study,key=key,rank=int(rank),old_job=str(job),attempt_preserved=node['attempt'],evidence=str(a)))
  state.update(status='CPU_ADOPTION_READY',error=None,completion_first_adoption=dict(operation=str(O),at=time.time(),previous_pid=pid,source=str(SOURCE)))
  atomic(d/'STATE.json',state)
 #STOPmarkerkeepsmaincampaign fromrestartingoldhelperwhilewewrite itsnewreceipt.
 command=[PYTHON,'-u','-m','tools.qwen_world_safety_exploration.eval_completion_owner','--study',tag,'--registry',str(root/'RUNS.json'),'--interval','30']
 with(O/(tag+'_cpu.log')).open('ab',buffering=0)as log:
  p=subprocess.Popen(command,cwd=SOURCE,env=dict(os.environ,WORLD_COMPARE_ROOT=str(root),WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',LC_ALL='C.UTF-8'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]);launch=dict(pid=p.pid,start_ticks=ticks,host=os.uname().nodename,source=str(SOURCE),command=command,at_unix=time.time(),operation=str(O),replaces=dict(pid=pid,start_ticks=start))
 old=read(d/helper);atomic(d/(helper.removesuffix('.json')+f'_{old["pid"]}_{old["start_ticks"]}.json'),old);atomic(d/helper,launch);atomic(O/(tag+'_LAUNCH.json'),launch)
 d.joinpath('STOP_OWNER').unlink();launches.append(dict(study=study,directory=str(d),**launch));print('LAUNCHED',tag,p.pid,ticks,flush=True)
atomic(O/'APPLIED.json',dict(status='APPLIED_CPU_ONLY_EVALUATION_COMPLETION_FIRST',at=time.time(),source=str(SOURCE),source_freeze=str(O/'SOURCE_FREEZE.json'),launches=launches,requeued_exact_cases=records,failures_requeued=len(records),scientific_specs_models_sources_seeds_shards_unchanged=True,existing_workers_unsignalled=True,training_registry_and_owners_untouched=True,I_untouched=True))
