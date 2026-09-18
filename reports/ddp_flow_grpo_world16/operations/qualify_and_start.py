from pathlib import Path
import subprocess,os,json,time
from scripts.cluster_flow_grpo.cluster import base_env,write_json
root=Path("runs/resource_reallocation_v2")
python="/root/miniconda3/envs/ddp/bin/python"
state=root/"formal_pipeline_state.json"
def stage(name,**kw):
 write_json(state,{"stage":name,"time":time.time(),"pid":os.getpid(),**kw})
 print(name,kw,flush=True)
try:
 stage("WAITING_FOR_RUNNING_RELOCATION")
 deadline=time.monotonic()+600
 marker=root/"f16_relocated_control/result.json"
 while not marker.exists():
  if time.monotonic()>deadline:raise TimeoutError("bounded relocation completion deadline")
  time.sleep(5)
 assert json.loads(marker.read_text())["status"]=="PASS","relocation failed"
 stage("EXACT_PLACEMENT_COMPARISON")
 command=[python,"scripts/cluster_flow_grpo/boundary_evidence.py","--continuous",str(root/"f16_full_cont/checkpoints/update_000002"),"--resumed",str(root/"f16_relocated/checkpoints/update_000002"),"--output",str(root/"f16_relocation_comparison.json")]
 with (root/"f16_relocation_comparison.log").open("x") as log:
  subprocess.run(command,env=base_env(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
 stage("PUBLISHING_NATIVE_RELEASES")
 children=[]
 for v in ["f","u"]:
  log=(root/(v+"16_relocated_release_publication.log")).open("x")
  child=subprocess.Popen([python,"scripts/cluster_flow_grpo/release.py","--variant",v,"--root",str(root)],env=base_env(),stdout=log,stderr=subprocess.STDOUT)
  children.append((v,child,log))
 codes={}
 for v,child,log in children:
  codes[v]=child.wait(timeout=2400);log.close()
 stage("NATIVE_RELEASE_RESULTS",exit_codes=codes)
 if any(codes.values()):raise RuntimeError("native semantic release failed")
 stage("STARTING_PAIRED_CONTROLLER")
 command=[python,"scripts/cluster_flow_grpo/paired.py","--plan","configs/cluster_flow_grpo/paired_world16.json"]
 write_json(root/"formal_launch.json",{"pid":os.getpid(),"command":command,"time":time.time(),"log":str(root/"formal_pipeline.log")})
 os.execvpe(python,command,base_env())
except BaseException as e:
 stage("FAILED",error=repr(e))
 raise
