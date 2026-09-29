"""Finite P0 queue. Never starts effect training or adopts profile weights."""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.foresight.run_allocated import occupants


def main():
 p=argparse.ArgumentParser(__doc__)
 for key in ('campaign-root','local-root','index','qwen','sources','host-tag','mode','pressure-python','pressure-script'):p.add_argument('--'+key,required=True)
 p.add_argument('--candidates',default='C0,C1,C2,C3,C4,C5');a=p.parse_args()
 if a.mode not in ('single4','single8','paired4'):raise ValueError('Unknown registered P0 execution mode')
 candidates=a.candidates.split(',')
 if len(set(candidates))!=len(candidates) or any(c not in ('C0','C1','C2','C3','C4','C5') for c in candidates):raise ValueError('Invalid queue')
 if a.mode=='paired4' and len(candidates)%2:raise ValueError('Pairing requires even candidate count')
 root=Path(a.campaign_root);source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();work=str(Path.cwd())
 if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source before queue')
 name=f'p0_{a.host_tag}_{a.mode}_{source[:7]}';statepath=root/(name+'_queue.json');lock=(root/(name+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if statepath.exists():raise FileExistsError('Queue already exists; inspect progress rather than relaunch')
 # Both machines must have completed the common C0/C5 actual-model start check.
 for host in ('local','zt2'):
  for c in ('C0','C5'):
   state=json.loads((root/'students'/f'p0_{host}_{c}_4gpu_mb8_startup_v2'/'status.json').read_text())
   if state['status']!='COMPLETE' or state['completed']!=4:raise ValueError('Cross-host startup gate incomplete')
 for c in candidates:
  cached=json.loads((Path(a.local_root)/'targets'/c/'local_replica.json').read_text())
  if cached['images']<11520 or not cached['verified_sha256']:raise ValueError('Local P0 cache incomplete')
 state={'source':source,'mode':a.mode,'host':socket.gethostname(),'candidates':candidates,'status':'RUNNING','completed':[],'runs':[]};atomic_json(statepath,state)
 children=[];stop_requested=[False]
 def stop(*_):
  stop_requested[0]=True
  for child in children:
   if child.poll() is None:child.send_signal(signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 waves=[candidates[i:i+2] for i in range(0,len(candidates),2)] if a.mode=='paired4' else [[c] for c in candidates]
 try:
  for wave in waves:
   if stop_requested[0]:state['status']='STOPPED';break
   children=[]
   for i,c in enumerate(wave):
    ngpu=8 if a.mode=='single8' else 4;gpus=list(range(i*4,i*4+4)) if a.mode=='paired4' else list(range(ngpu))
    # Wait for previous wrapper's restored pressure to become visible before borrowing it.
    deadline=time.time()+60
    while True:
     occupied=occupants(Path(a.pressure_script).resolve());own=[(g,parent) for g,_,parent in occupied if g in gpus]
     if {g for g,p in own if p is not None}==set(gpus):break
     if any(p is None for g,p in own):raise RuntimeError('Unrelated workload on queued GPU')
     if time.time()>deadline:raise RuntimeError('Pressure handover incomplete; no blind launch')
     time.sleep(1)
    run_id=f'{name}_{c}';record=root/'allocations'/(run_id+'.json')
    cmd=[sys.executable,'-m','tools.foresight.run_allocated','--gpus',','.join(map(str,gpus)),'--worktree',work,'--record',str(record),'--pressure-script',a.pressure_script,'--pressure-python',a.pressure_python,'--',sys.executable,'-m','tools.dino_tradeoff.run_profile','--candidate',c,'--campaign-root',str(root),'--local-root',a.local_root,'--index',a.index,'--run-id',run_id,'--qwen',a.qwen,'--sources',a.sources,'--gpus',str(ngpu),'--updates','120','--micro-batch',str(32//ngpu),'--master-port',str(29901+i)]
    with (root/(run_id+'.log')).open('x') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    children.append(child);state['runs'].append({'candidate':c,'run_id':run_id,'command':cmd,'pid':child.pid,'gpus':gpus,'started_unix':time.time()});atomic_json(statepath,state)
   codes=[c.wait() for c in children]
   if any(code!=0 for code in codes):raise RuntimeError('Profile failed; evidence retained and queue halted: '+str(codes))
   state['completed'].extend(wave);atomic_json(statepath,state)
  else:state['status']='COMPLETE'
 except BaseException as error:
  state.update(status='FAILED',error=repr(error));stop();raise
 finally:
  state['ended_unix']=time.time();atomic_json(statepath,state)
if __name__=='__main__':main()
