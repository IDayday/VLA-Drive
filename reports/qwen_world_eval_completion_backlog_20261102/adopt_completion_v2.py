import json,os,time,fcntl,subprocess,shutil
from pathlib import Path
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');O=E/'operations/EVALUATION_BACKLOG_COMPLETION_20261102';SOURCE=O/'runtime/completion_v2';PYTHON='/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.new.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n');os.replace(tmp,p)
def alive(p,s):
 try:v=(Path('/proc')/str(p)/'stat').read_text().split();return int(v[21])==s and v[2]not in('Z','X')
 except FileNotFoundError:return False
lock=(O/'adoption.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
CASES=[('G','revival_training_20261101','RESEARCH_EVALUATION',3704487,1506656918,'STANDARD_LAUNCH.json'),('H','improvement_training_20261101','IMPROVEMENT_EVAL',3704513,1506657050,'HELPER_LAUNCH.json')]
for tag,study,dirname,pid,start,helper in CASES:
 d=E/study/'owners'/dirname;old=read(d/'OWNER.json');assert(old['pid'],old['start_ticks'])==(pid,start)and alive(pid,start);assert not(d/'STOP_OWNER').exists()
 b=O/'BEFORE_V2'/tag;b.mkdir(parents=True,exist_ok=True)
 for name in ('OWNER.json','STATE.json',helper):shutil.copyfile(d/name,b/name)
 atomic(d/'STOP_OWNER',dict(reason='CPU_SCHEDULER_FAIR_ADMISSION_AND_OLDEST_RETENTION',operation=str(O),pid=pid,start_ticks=start))
end=time.time()+100
while any(alive(p,s)for t,st,d,p,s,h in CASES):
 if time.time()>end:raise RuntimeError('NoCPUsignals; naturalexitwaittimeout')
 time.sleep(1)
launches=[]
for tag,study,dirname,pid,start,helper in CASES:
 d=E/study/'owners'/dirname
 with(d/'owner.lock').open('a')as g:
  fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);shutil.copyfile(d/'STATE.json',O/'BEFORE_V2'/tag/'QUIESCED_STATE.json')
 command=[PYTHON,'-u','-m','tools.qwen_world_safety_exploration.eval_completion_owner','--study',tag,'--registry',str(E/study/'RUNS.json'),'--interval','30']
 with(O/(tag+'_cpu_v2.log')).open('ab',buffering=0)as log:
  p=subprocess.Popen(command,cwd=SOURCE,env=dict(os.environ,WORLD_COMPARE_ROOT=str(E/study),WORLD_COMPARE_TASK='qwen_world_safety_exploration_20261031',PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUTF8='1',LC_ALL='C.UTF-8'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]);value=dict(pid=p.pid,start_ticks=ticks,host=os.uname().nodename,source=str(SOURCE),command=command,at_unix=time.time(),operation=str(O),replaces=dict(pid=pid,start_ticks=start))
 old=read(d/helper);atomic(d/(helper.removesuffix('.json')+f'_{old["pid"]}_{old["start_ticks"]}.json'),old);atomic(d/helper,value);atomic(O/(tag+'_LAUNCH_V2.json'),value);(d/'STOP_OWNER').unlink();launches.append(dict(study=study,directory=str(d),**value));print(tag,p.pid,ticks,flush=True)
atomic(O/'ADOPTION_V2_APPLIED.json',dict(status='APPLIED_CPU_ONLY',at=time.time(),launches=launches,source=str(SOURCE),F='unchanged V1 CPU3704486',reason='G newadmission32; H currentRetention oldest10/12/14; allcurrentGPUworkersfinishnaturally',no_policy_training_science_or_worker_signal_changes=True))
