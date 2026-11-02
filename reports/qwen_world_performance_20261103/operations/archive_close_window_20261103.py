"""Reinstate only this exact six-host window's bans at its actual hard expiry."""
import fcntl,json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent
POLICY=Path('/mnt/project/server_dispatch_policy.json')
REQUEST=Path('/mnt/project/qwen-world-performance-rl-20261103/operations/RESOURCES_UNTIL_20261103T000000Z/REQUEST.json')
def write(p,x):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def main():
 r=json.loads(REQUEST.read_text());write(D/'CLOSE_WINDOW_OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),deadline_unix=r['deadline_unix']))
 while time.time()<r['deadline_unix']:time.sleep(min(10,r['deadline_unix']-time.time()))
 with Path('/mnt/project/server_dispatch_policy.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=json.loads(POLICY.read_text());own=p['task_authorizations']['qwen_world_safety_exploration_20261031'];changed=[];skipped=[]
  for h in r['hosts']:
   current=p['global_dispatch_prohibitions'].get(h,{})
   if own['expires_unix']!=r['deadline_unix']or current.get('record')!=str(D/'WINDOW_CORRECTION.json'):
    skipped.append(h);continue
   p['global_dispatch_prohibitions'][h]=dict(new_task_dispatch_allowed=False,utc=r['deadline_UTC'],record=str(REQUEST),reason='Actual user-authorized six-host window has ended at Beijing2026-11-03 08:00; no new tasks or retries')
   changed.append(h)
  write(POLICY,p);write(D/'ACTUAL_HARD_WINDOW_CLOSED.json',dict(at_unix=time.time(),hosts=changed,skipped_due_to_later_policy=skipped,signals_sent=False,other_task_grants_changed=False))
if __name__=='__main__':main()
