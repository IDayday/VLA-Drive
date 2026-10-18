"""Only self SIGTERM after a real fixture Popen, before caller registration."""
import hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,time,types
path=pathlib.Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018/resources/st_endpoint_supervisor_v1.py');raw=path.read_bytes();m=types.ModuleType('signal_window_endpoint');m.__file__=str(path);exec(compile(raw,str(path),'exec'),m.__dict__)
out=pathlib.Path(__file__).resolve().parent;result={'controller_sha256':hashlib.sha256(raw).hexdigest(),'only_synthetic_CPU_processes':True,'model_GPU_DEV_metriccache_official_calls':0}
with tempfile.TemporaryDirectory(prefix='endpoint-sigterm-review-') as temp:
 temp=pathlib.Path(temp);m.OUTPUT=temp/'job';m.ANALYSIS=temp/'analysis';m.OUTPUT.mkdir();(m.OUTPUT/'official_score').mkdir();(m.OUTPUT/'official_score'/'COMPLETE.json').write_text('{}')
 fake=types.ModuleType('tools.st_world_policy.score_world');fake.read_complete_export=lambda p:({'world_size':4,'source_sha':m.SOURCE,'checkpoint':{'completed':4096}},[{}]*1696);sys.modules['tools.st_world_policy.score_world']=fake
 m.mature_helpers=lambda:{'check_route_time_v3':lambda *args:None};m.route_checks=lambda *args:{};m.argv_for=lambda *args:[sys.executable,'-c','import time;time.sleep(60)'];m.require_export_identity=lambda *args:None
 m.pinned=lambda ref:({'receipt_identity':'fixture_native','files':{}} if ref['path']=='/fixture_native' else ({'scenes':[{}]*1696} if ref['path']=='/fixture_manifest' else {'identity':'fixture_run'}))
 def immutable(path,value):
  path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
 m.immutable=immutable;real=subprocess.Popen;spawned=[]
 def popen(*args,**kwargs):
  child=real(*args,**kwargs);spawned.append(child)
  # The actual execute-installed handler raises while launch still has no
  # returned child handle. This is SIGTERM, not uncatchable SIGKILL.
  os.kill(os.getpid(),signal.SIGTERM)
  return child
 m.subprocess=types.SimpleNamespace(Popen=popen,DEVNULL=subprocess.DEVNULL,STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)
 binding={'route':{},'checkpoint':{'tag':'fixture','complete':{'path':'/fixture_native','sha256':'f'*64}},'training':{'registration':{'path':'/fixture_registration'},'identity':{'path':'/fixture_identity'}}};documents={'fixed_inputs':{'pins':{'DEV_manifest':{'path':'/fixture_manifest','sha256':'f'*64}}}}
 started=time.monotonic()
 try:
  answer=m.execute('analysis',{'fixture':'only'},binding,documents,resume=False)
  result.update(returned_status=answer['status'],stop_reason=answer['stop_reason'],spawned_count=len(spawned),child_left_alive=any(p.poll() is None for p in spawned),reported_invocations_started=answer['invocations_started'],durable_admission_count=len(list((m.OUTPUT/'control/invocations/analysis').glob('*/ADMISSION.json'))),wall_seconds=time.monotonic()-started)
 finally:
  for child in spawned:
   if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
   child.wait()
  result['fixture_cleanup_confirmed']=all(p.poll() is not None for p in spawned)
result['controller_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest();assert result['controller_sha256']==result['controller_sha256_after'];(out/'SIGTERM_WINDOW_REPRODUCTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
