"""Bounded draft ownership reproductions; no model/GPU/official-score access."""
import hashlib,importlib.util,json,os,pathlib,signal,subprocess,sys,tempfile,time,types
source=pathlib.Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018/resources/st_endpoint_supervisor_v1.py')
raw=source.read_bytes();module=types.ModuleType('draft_endpoint');module.__file__=str(source)
exec(compile(raw,str(source),'exec'),module.__dict__)
root=pathlib.Path(__file__).resolve().parent
result={'source_sha256':hashlib.sha256(raw).hexdigest(),'new_model_or_GPU_or_official_calls':0}
with tempfile.TemporaryDirectory(prefix='endpoint-resource-review-') as temp:
 temp=pathlib.Path(temp)
 # The source only inspects main-task children. Keep the spawning thread alive
 # so Linux exposes its child under that task rather than the main task.
 script="import subprocess,threading,time,pathlib,sys\ndef run():\n p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)\nthreading.Thread(target=run).start();time.sleep(60)"
 parent=subprocess.Popen([sys.executable,'-c',script,str(temp/'child.pid')],start_new_session=True)
 try:
  end=time.monotonic()+5
  while not (temp/'child.pid').exists() and time.monotonic()<end:time.sleep(.02)
  child=int((temp/'child.pid').read_text());tree=module.process_tree(parent.pid)
  result['thread_created_child']={'actual_child_discovered':child in tree,'observed_tree_process_count':len(tree),'expected_at_least':2}
 finally:
  os.killpg(parent.pid,signal.SIGKILL);parent.wait()
 # Exercise the exact post-Popen / before-LAUNCH publication failure window.
 module.OUTPUT=temp/'job';module.ANALYSIS=temp/'analysis';module.OUTPUT.mkdir()
 (module.OUTPUT/'official_score').mkdir();(module.OUTPUT/'official_score'/'COMPLETE.json').write_text('{}')
 fake=types.ModuleType('tools.st_world_policy.score_world')
 fake.read_complete_export=lambda p:({'world_size':4,'source_sha':module.SOURCE,'checkpoint':{'completed':4096}},[{}]*1696)
 sys.modules['tools.st_world_policy.score_world']=fake
 helpers={'check_route_time_v3':lambda *args:None};module.mature_helpers=lambda:helpers;module.route_checks=lambda *args:{}
 module.argv_for=lambda *args:[sys.executable,'-c','import time;time.sleep(60)']
 def write(path,value):
  path=pathlib.Path(path)
  if path.name=='LAUNCH.json':raise OSError('synthetic LAUNCH publication failure after Popen')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
 module.immutable=write
 original=subprocess.Popen;spawned=[]
 def tracked(*args,**kwargs):
  child=original(*args,**kwargs);spawned.append(child);return child
 module.subprocess=types.SimpleNamespace(Popen=tracked,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT)
 try:
  outcome=module.execute('analysis',{'fixture':'only'},{'route':{}},{},resume=False)
  result['post_popen_publication_failure']={'returned_status':outcome['status'],'stop_reason':outcome['stop_reason'],'spawned_count':len(spawned),'owned_child_left_alive':any(p.poll() is None for p in spawned),'reported_invocations_started':outcome['invocations_started']}
 finally:
  for child in spawned:
   if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
   child.wait()
result['source_sha256_after']=hashlib.sha256(source.read_bytes()).hexdigest()
(root/'DRAFT_REPRODUCTIONS_ATTEMPT2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
