"""Measure only the bounded independent resource-control fixture process tree."""
import datetime,json,os,pathlib,resource,signal,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
runtime=pathlib.Path('/mnt/project/VLA-Drive-st-road-failure-direction-20261018')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(runtime)+':'+str(runtime/'tests/st_world')}
tests=[str(root/'test_resource_delta.py'),
 str(runtime/'tests/st_world/test_road_failure_selection.py')+'::test_per_case_and_global_new_call_guards_no_original_or_retry',
 str(runtime/'tests/st_world/test_road_failure_selection.py')+'::test_new_mode_exact_approval_and_same_path_replacement_rejected',
 str(runtime/'tests/st_world/test_run_road_direction.py')+'::test_bounded_decompression_refuses_bomb_and_trailing',
 str(runtime/'tests/st_world/test_run_road_direction.py')+'::test_budget_hash_change_and_read_cap',
 str(runtime/'tests/st_world/test_run_road_direction.py')+'::test_bounded_read_growth_cannot_exceed_reserved_probe',
 str(runtime/'tests/st_world/test_run_road_direction.py')+'::test_output_scan_ignores_only_vanished_file',
 str(runtime/'tests/st_world/test_run_road_direction.py')+'::test_real_spawn_cleanup_and_full_denominator']
cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests]
def before():
 os.setsid();os.sched_setaffinity(0,{0,1});resource.setrlimit(resource.RLIMIT_CPU,(120,120))
start=time.monotonic();before_cpu=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;reason=None
with (root/'TEST_OUTPUT.txt').open('x') as log:
 p=subprocess.Popen(cmd,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=before)
 while p.poll() is None:
  table={}
  for stat in pathlib.Path('/proc').glob('[0-9]*/stat'):
   try:
    text=stat.read_text();rest=text[text.rfind(')')+2:].split();table[int(stat.parent.name)]=(int(rest[1]),int(rest[21])*os.sysconf('SC_PAGE_SIZE'))
   except (OSError,ValueError,IndexError):continue
  owned={p.pid};change=True
  while change:
   new={pid for pid,(ppid,rss) in table.items() if ppid in owned};change=bool(new-owned);owned|=new
  rss=sum(table.get(pid,(0,0))[1] for pid in owned);peak=max(peak,rss)
  size=sum(q.stat().st_size for q in root.rglob('*') if q.is_file())
  if time.monotonic()-start>600:reason='WALL_CAP'
  elif rss>2*2**30:reason='AGGREGATE_RSS_CAP'
  elif size>32*2**20:reason='OUTPUT_CAP'
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
   break
  time.sleep(.1)
 p.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'command':cmd,'status':'PASS' if p.returncode==0 and reason is None else 'FAIL','exit_code':p.returncode,'stop_reason':reason,'wall_seconds':time.monotonic()-start,'waited_child_cpu_seconds':after.ru_utime+after.ru_stime-before_cpu.ru_utime-before_cpu.ru_stime,'sampled_tree_peak_rss_bytes':peak,'real_proxy_calls':0,'real_official_scores':0,'model_loads':0,'CUDA_VISIBLE_DEVICES':'','limits':{'wall_seconds':600,'aggregate_tree_rss_bytes':2*2**30,'cpu_seconds':120,'output_bytes':32*2**20},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'TEST_EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='PASS' else 2)
