"""CPU-only reservation for this D invocation; never signals a native job."""
from contextlib import ExitStack
import fcntl,json,os,signal,time
from pathlib import Path
R=Path(__file__).resolve().parent
stop=False
def halt(*_):
 global stop
 stop=True
for s in(signal.SIGTERM,signal.SIGINT):signal.signal(s,halt)
def save(status,**kw):
 p=R/'RESERVATION_OWNER.json';t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(dict(status=status,pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),at=time.time(),host='training-vla-zt',native_signals=0,**kw),indent=2));os.replace(t,p)
entry=json.loads((R/'NEW_INVOCATION.json').read_text())['entry'];job=Path(entry['receipt'])
save('WAITING_FOR_MEASUREMENT_LEASE_HANDOFF',invocation=str(job))
with ExitStack()as held:
 for folder in('runtime_gpu_shared_workers','runtime_gpu_research_slots'):
  for uuid in entry['uuids']:
   stream=held.enter_context((Path('/mnt/project')/folder/(uuid+'.lock')).open('a'))
   while not stop:
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     if(job/'RESULT.json').exists()or(R/'FAILURE.json').exists():stop=True;break
     time.sleep(.2)
   if stop:break
  if stop:break
 if not stop:
  # Preserve only an alive exact invocation, never a permanent idle reservation.
  live=json.loads((job/'RUNNING.json').read_text());proc=Path('/proc')/str(live['worker_pid'])
  stat=proc.joinpath('stat').read_text().split()
  if int(stat[21])!=live['worker_start_ticks']or stat[2]=='Z':raise RuntimeError('D worker identity changed')
  save('HELD_FOR_D_INVOCATION',invocation=str(job),release_file=str(R/'RELEASE_RESERVATION'))
  while not stop and not(job/'RESULT.json').exists()and not(R/'RELEASE_RESERVATION').exists():
   p=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
   if p.get('global_dispatch_prohibitions',{}).get('training-vla-zt',{}).get('new_task_dispatch_allowed')is False:break
   try:
    s=proc.joinpath('stat').read_text().split()
    if int(s[21])!=live['worker_start_ticks']or s[2]=='Z':break
   except FileNotFoundError:break
   time.sleep(30)
save('RELEASED',invocation=str(job))
