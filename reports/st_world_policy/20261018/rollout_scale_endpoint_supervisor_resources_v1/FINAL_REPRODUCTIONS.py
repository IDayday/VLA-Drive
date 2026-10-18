"""Independent real-CPU ownership checks; synthetic inference and score stubs."""
import hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,time,types
source=pathlib.Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018/resources/st_endpoint_supervisor_v1.py')
raw=source.read_bytes();module=types.ModuleType('final_endpoint');module.__file__=str(source)
exec(compile(raw,str(source),'exec'),module.__dict__)
root=pathlib.Path(__file__).resolve().parent
result={'source_sha256':hashlib.sha256(raw).hexdigest(),'new_model_or_GPU_or_official_calls':0}
with tempfile.TemporaryDirectory(prefix='endpoint-resource-review-final-') as tmp:
 tmp=pathlib.Path(tmp)
 script="import subprocess,threading,time,pathlib,sys\ndef run():\n p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)\nthreading.Thread(target=run).start();time.sleep(60)"
 parent=subprocess.Popen([sys.executable,'-c',script,str(tmp/'child.pid')],start_new_session=True)
 try:
  end=time.monotonic()+5
  while not (tmp/'child.pid').exists() and time.monotonic()<end:time.sleep(.02)
  child=int((tmp/'child.pid').read_text());tree=module.process_tree(parent.pid)
  result['thread_created_child']={'actual_child_discovered':child in tree,'observed_tree_process_count':len(tree)}
  assert child in tree
 finally:
  os.killpg(parent.pid,signal.SIGKILL);parent.wait()
 module.OUTPUT=tmp/'job';module.ANALYSIS=tmp/'analysis';module.OUTPUT.mkdir()
 (module.OUTPUT/'official_score').mkdir();(module.OUTPUT/'official_score'/'COMPLETE.json').write_text('{}')
 fake=types.ModuleType('tools.st_world_policy.score_world')
 fake.read_complete_export=lambda p:({'world_size':4,'source_sha':module.SOURCE,'checkpoint':{'completed':4096}},[{}]*1696)
 sys.modules['tools.st_world_policy.score_world']=fake
 helpers={'check_route_time_v3':lambda *args:None};module.mature_helpers=lambda:helpers;module.route_checks=lambda *args:{}
 module.argv_for=lambda *args:[sys.executable,'-c','import time;time.sleep(60)']
 module.pinned=lambda ref:({'receipt_identity':'fixture_native','files':{}} if ref['path']=='/fixture_native' else ({'scenes':[{}]*1696} if ref['path']=='/fixture_manifest' else {'identity':'fixture_run'}))
 # These checks are covered by the 47 source tests. Here every source/data
 # field is an explicit fixture so only the real process ownership window runs.
 module.require_export_identity=lambda *args:None
 def write(path,value):
  path=pathlib.Path(path)
  if path.name=='LAUNCH.json':raise OSError('synthetic LAUNCH publication failure after Popen')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
 module.immutable=write
 original=subprocess.Popen;spawned=[]
 def tracked(*args,**kwargs):
  child=original(*args,**kwargs);spawned.append(child);return child
 module.subprocess=types.SimpleNamespace(Popen=tracked,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)
 binding={'route':{},'checkpoint':{'tag':'fixture','complete':{'path':'/fixture_native','sha256':'f'*64}},'training':{'registration':{'path':'/fixture_registration'},'identity':{'path':'/fixture_identity'}}}
 try:
  documents={'fixed_inputs':{'pins':{'DEV_manifest':{'path':'/fixture_manifest','sha256':'f'*64}}}}
  outcome=module.execute('analysis',{'fixture':'only'},binding,documents,resume=False)
  result['post_popen_publication_failure']={'returned_status':outcome['status'],'stop_reason':outcome['stop_reason'],'spawned_count':len(spawned),'owned_child_left_alive':any(p.poll() is None for p in spawned),'reported_invocations_started':outcome['invocations_started']}
  assert len(spawned)==1 and spawned[0].poll() is not None and outcome['invocations_started']==1
 finally:
  for child in spawned:
   if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
   child.wait()
 # A mkdir-only unknown attempt is not a free replacement or ignorable record.
 p=tmp/'partial';(p/'invocations'/'score'/'attempt1').mkdir(parents=True);(p/'invocations'/'score'/'attempt2').mkdir()
 try:module.claim_invocation(p,'score',resume=True,binding_sha='fixture',argv=['fixture'])
 except ValueError as error:result['two_partial_attempts_refused']='exhausted' in str(error)
 else:raise AssertionError('Two partial attempt directories were not charged')
 assert result['two_partial_attempts_refused']
result['source_sha256_after']=hashlib.sha256(source.read_bytes()).hexdigest();assert result['source_sha256_after']==result['source_sha256']
(root/'FINAL_REPRODUCTIONS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
