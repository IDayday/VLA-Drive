"""Two read-only /proc observations, ten seconds apart; no payload opens."""
import concurrent.futures,json,pathlib,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent
CODE=r'''
import json,os,pathlib,subprocess,socket,time
root=ROOTPID
seen=set();todo=[root];rows=[]
while todo:
 pid=todo.pop()
 if pid in seen:continue
 seen.add(pid);p=pathlib.Path('/proc')/str(pid)
 try:
  ss=(p/'stat').read_text();tail=ss[ss.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  io={l.split(':')[0]:int(l.split(':')[1]) for l in (p/'io').read_text().splitlines()}
  kids=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()];todo.extend(kids)
  status={l.split(':')[0]:l.split(':')[1].strip() for l in (p/'status').read_text().splitlines() if ':' in l}
  env=(p/'environ').read_bytes().split(b'\0');rank=next((x.decode().split('=',1)[1] for x in env if x.startswith(b'RANK=')),None)
  fdclasses={}
  for f in (p/'fd').iterdir():
   try:t=os.readlink(f)
   except OSError:continue
   cat='parent_checkpoint' if '/checkpoints/step_005000_' in t else ('other_checkpoint' if t.endswith(('.pt','.bin','.safetensors')) else ('source_log' if '/navsim' in t or '/logs/' in t else ('causal_history_output' if 'full_causal_population_v1/train/' in t else 'other')))
   fdclasses[cat]=fdclasses.get(cat,0)+1
  rows.append({'pid':pid,'ppid':int(tail[1]),'state':tail[0],'startticks':int(tail[19]),'utime_ticks':int(tail[11]),'stime_ticks':int(tail[12]),'rss_kib':int(status.get('VmRSS','0 kB').split()[0]),'wchan':(p/'wchan').read_text().strip(),'io':io,'rank':rank,'role':('world_rank' if 'tools.st_world_policy.train_world' in cmd and rank is not None else 'owned_controller_or_worker'),'open_fd_categories':fdclasses})
 except (OSError,ValueError):continue
out={'time_unix':time.time(),'host':socket.gethostname(),'root_pid':root,'tick_hz':os.sysconf('SC_CLK_TCK'),'processes':rows}
if GPU:
 out['gpus']=subprocess.check_output(['nvidia-smi','--id=0,1,2,3','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(out))
'''
def shot(host,pid,gpu):
 code=CODE.replace('ROOTPID',str(pid)).replace('if GPU:',f'if {gpu}:')
 cmd=['python','-'] if host=='local' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python -']
 r=subprocess.run(cmd,input=code,text=True,capture_output=True,check=True,timeout=30);return json.loads(r.stdout)
specs=[('training-vlawm-tsinghua-zt6',482742,True),('training-vlawm-tsinghua-zt2',1625778,True),('local',786421,False)]
observations=[]
for n in range(2):
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:observations.append(list(ex.map(lambda s:shot(*s),specs)))
 if not n:time.sleep(10)
summary=[]
for a,b in zip(*observations):
 by={(p['pid'],p['startticks']):p for p in a['processes']};deltas=[]
 for p in b['processes']:
  old=by.get((p['pid'],p['startticks']))
  if old:
   deltas.append({'pid':p['pid'],'rank':p['rank'],'state':p['state'],'wchan':p['wchan'],'rss_kib':p['rss_kib'],'cpu_seconds':((p['utime_ticks']+p['stime_ticks'])-(old['utime_ticks']+old['stime_ticks']))/b['tick_hz'],'logical_read_delta':p['io']['rchar']-old['io']['rchar'],'storage_read_delta':p['io']['read_bytes']-old['io']['read_bytes'],'storage_write_delta':p['io']['write_bytes']-old['io']['write_bytes'],'open_fd_categories':p['open_fd_categories']})
 summary.append({'host':b['host'],'seconds':b['time_unix']-a['time_unix'],'rows':deltas,'gpus':b.get('gpus')})
result={'schema':'st_readonly_startup_observation_v1','observations':observations,'deltas':summary,'payloads_read':False,'signals_sent':False,'new_jobs_launched':False,'process_commands_or_environment_saved':False}
pathlib.Path('/tmp/st_spatial20k_startup_observation_20261018.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'OBSERVED','samples_per_host':2,'raw_inventory_saved_outside_repository':True}))
