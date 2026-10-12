from pathlib import Path
import json,statistics,time,datetime,os,hashlib,fcntl
ROOT=Path(__file__).resolve().parent
DIS=ROOT.parent/'navsim_navtest_dispatch_20261011T0225'
POLICY=DIS/'COHOST_COST_POLICY.json'
BASE=json.loads((ROOT/'TRAINING_BASELINE_BEFORE_EVAL.json').read_text())['rows']
HOSTS={'G1_FULL_UNIFORM':'training-vlawm-zt2','G3_EVENT_LOCAL':'training-vla-zt'}
last_changed={};started=time.monotonic()
def atomic(path,v):
 tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(path)
def window(path):
 with path.open('rb') as s:
  s.seek(max(0,s.seek(0,2)-300000));r=[]
  for line in s.read().splitlines():
   try:r.append(json.loads(line))
   except ValueError:pass
 return r[-32:]
while time.monotonic()-started<43200:
 complete=DIS/'completed/G3_EVENT_LOCAL_0090000.json'
 state=json.loads((DIS/'STATE.json').read_text())
 active=state.get('status')=='INFERENCE_RUNNING' and state.get('request',{}).get('key') in ('G1_FULL_UNIFORM_0060000','G3_EVENT_LOCAL_0090000')
 rows=[];changes=[]
 with (ROOT/'pacing_policy.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  before=POLICY.read_bytes();policy=json.loads(before)
  for base in BASE:
   file=Path(base['run'])/'metrics.jsonl';r=window(file);median=statistics.median(v['seconds'] for v in r)
   ratio=median/base['median_seconds'];host=HOSTS.get(base['group'])
   row={'group':base['group'],'updates':[r[0]['updates'],r[-1]['updates']],'median_seconds':median,'baseline_median_seconds':base['median_seconds'],'ratio':ratio,'metric_age_seconds':time.time()-file.stat().st_mtime}
   if host:
    duty=policy.get('owned_training_duty_by_host',{}).get(host,policy['owned_training_inference_duty_cycle']);row['duty_cycle']=duty
    if active and ratio>1.25 and duty>.05 and row['metric_age_seconds']<30 and time.monotonic()-last_changed.get(host,started)>=120:
     new=max(.05,duty/2);policy.setdefault('owned_training_duty_by_host',{})[host]=new;last_changed[host]=time.monotonic();changes.append({'host':host,'old':duty,'new':new,'reason':'last32-update median exceeds measured baseline by25percent','ratio':ratio});row['new_duty_cycle']=new
   rows.append(row)
  if changes:
   assert POLICY.read_bytes()==before,'Concurrent cost policy update; preserve other writer'
   policy['last_pacing_adjustment_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(POLICY,policy)
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluation_status':state.get('status'),'active_full_inference':active,'rows':rows,'pacing_changes':changes,'workers_signaled':False,'model_or_training_changes':False}
 with (ROOT/'TRAINING_INTERFERENCE.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
 atomic(ROOT/'TRAINING_INTERFERENCE_LATEST.json',record)
 if complete.exists() and json.loads(complete.read_text()).get('valid') is True and (DIS/'completed/G1_FULL_UNIFORM_0060000.json').exists():
  atomic(ROOT/'INTERFERENCE_MONITOR_COMPLETE.json',{'utc':record['utc'],'reason':'G1 60k and G3 90k recovered queue complete','training_signaled':False});break
 if (ROOT/'STOP_INTERFERENCE_MONITOR').exists():break
 time.sleep(60)
