"""Single bounded independent CPU review window, only after clean source seal."""
import argparse,datetime,hashlib,json,os,pathlib,resource,signal,subprocess,sys,time
parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True);parser.add_argument('--controller-sha256',required=True);args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parent
runtime=pathlib.Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018')
controller=runtime/'resources/st_endpoint_supervisor_v1.py'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==args.source_commit
assert not subprocess.check_output(['git','status','--porcelain'],cwd=runtime).strip()
assert sha(controller)==args.controller_sha256
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(runtime)}
commands=[[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(runtime/'tests/st_world/test_endpoint_supervisor.py')],[sys.executable,str(root/'FINAL_REPRODUCTIONS.py')]]
def preexec():
 os.setsid();os.sched_setaffinity(0,{0,1});resource.setrlimit(resource.RLIMIT_CPU,(120,120))
started=time.monotonic();initial=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;max_live_waited_cpu=0.;outputs=[];reason=None
for n,command in enumerate(commands):
 start=time.monotonic()
 with (root/f'FINAL_TEST_{n}_OUTPUT.txt').open('x') as log:
  child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=preexec)
  while child.poll() is None:
   table={}
   for path in pathlib.Path('/proc').glob('[0-9]*/stat'):
    try:
     text=path.read_text().rsplit(')',1)[1].split();table[int(path.parent.name)]=(int(text[1]),int(text[21])*os.sysconf('SC_PAGE_SIZE'),sum(int(text[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK'))
    except (OSError,ValueError,IndexError):continue
   owned={child.pid};changed=True
   while changed:
    discovered={pid for pid,(parent,_,__) in table.items() if parent in owned};changed=bool(discovered-owned);owned|=discovered
   rss=sum(table.get(pid,(0,0,0))[1] for pid in owned);peak=max(peak,rss)
   cpu=sum(table.get(pid,(0,0,0))[2] for pid in owned);max_live_waited_cpu=max(max_live_waited_cpu,cpu)
   sofar=resource.getrusage(resource.RUSAGE_CHILDREN);waited=sofar.ru_utime+sofar.ru_stime-initial.ru_utime-initial.ru_stime
   size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
   if time.monotonic()-started>600:reason='WALL_CAP'
   elif rss>2*2**30:reason='RSS_CAP'
   elif waited+cpu>120:reason='CPU_CAP'
   elif size>32*2**20:reason='OUTPUT_CAP'
   if reason:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    break
   time.sleep(.05)
  child.wait()
 outputs.append({'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start})
 if child.returncode or reason:break
final=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'status':'PASS' if len(outputs)==2 and all(o['exit_code']==0 for o in outputs) and reason is None else 'FAIL','source_commit':args.source_commit,'controller_sha256':args.controller_sha256,'controller_sha256_after':sha(controller),'commands':outputs,'wall_seconds':time.monotonic()-started,'waited_child_CPU_seconds':final.ru_utime+final.ru_stime-initial.ru_utime-initial.ru_stime,'sampled_tree_peak_RSS_bytes':peak,'stop_reason':reason,'limits':{'wall_seconds':600,'CPU_seconds':120,'RSS_bytes':2*2**30,'output_bytes':32*2**20},'actual_GPU_contexts':0,'actual_model_loads':0,'actual_DEV_payloads_read':0,'actual_metric_caches_read':0,'actual_official_calls':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
assert result['controller_sha256_after']==args.controller_sha256
(root/'FINAL_TEST_EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='PASS' else 2)
