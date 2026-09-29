import json,subprocess,socket,time
from pathlib import Path

def run(args):
 p=subprocess.run(args,text=True,capture_output=True);return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
commands=[['nvidia-smi','--query-gpu=index,name,memory.total,uuid','--format=csv'],['nvidia-smi','topo','-m'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],['lscpu'],['free','-b'],['df','-B1','-T','/','/mnt/project','/mnt/navsim'],['ps','-eo','pid,ppid,etime,args']]
rows=[run(c) for c in commands]
# Include only workload/pressure rows, never the complete unrelated process list.
rows[-1]['stdout']='\n'.join(x for x in rows[-1]['stdout'].splitlines() if any(k in x for k in ('gpu_stress.py','train_student','train_trajectory','torchrun','cache_targets','cache_dinov3')))
print(json.dumps({'host':socket.gethostname(),'time_unix':time.time(),'authorized_gpus':'local and training-vla-zt2; pressure only may be released','results':rows},indent=2))
