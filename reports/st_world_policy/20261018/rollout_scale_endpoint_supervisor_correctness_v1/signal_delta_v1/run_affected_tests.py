"""One bounded affected-only execution; never launch model/evaluation work."""
import json,os,resource,signal,subprocess,sys,time
from pathlib import Path
folder=Path(__file__).resolve().parent
owner=Path('/mnt/project/VLA-Drive-st-endpoint-supervisor-20261018')
expected='82e8f9dcba7d3323ca15491fdfe7f6e5481922ff'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner,text=True).strip()==expected
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=owner)
base=str(owner/'tests/st_world/test_endpoint_supervisor.py')
argv=[sys.executable,'-m','pytest',base+'::test_log_and_post_Popen_receipt_failure_preserves_admission_and_cleanup',base+'::test_complete_owned_session_cleanup_handles_thread_and_root_exit',base+'::test_real_owned_cpu_process_cleanup_on_monitor_fault',str(folder/'test_signal_first_stop.py'),'-q']
env={**os.environ,'ST_ENDPOINT_IMPLEMENTATION_ROOT':str(owner),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
def tree(pid):
 values={};queue=[pid]
 while queue:
  current=queue.pop()
  try:
   parts=Path(f'/proc/{current}/stat').read_text().rsplit(')',1)[1].split()
   values[current]=int(parts[21])*os.sysconf('SC_PAGE_SIZE')
   queue.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
  except (FileNotFoundError,ProcessLookupError):pass
 return values
limits={'wall_seconds':90,'aggregate_sampled_RSS_bytes':2*2**30,'CPU_seconds':120}
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;violation=None
with (folder/'TEST_OUTPUT.txt').open('xb') as stream:
 child=subprocess.Popen(argv,cwd=owner,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 try:
  while child.poll() is None:
   peak=max(peak,sum(tree(child.pid).values()))
   if peak>limits['aggregate_sampled_RSS_bytes']:violation='RSS_CAP'
   if time.monotonic()-start>limits['wall_seconds']:violation=violation or 'WALL_CAP'
   if violation:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(.05)
 finally:
  if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
if cpu>limits['CPU_seconds']:violation=violation or 'CPU_CAP'
record={'schema':'endpoint_signal_delta_independent_execution_v1','source_commit':expected,'argv':argv,
 'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'waited_child_CPU_seconds':cpu,
 'sampled_aggregate_RSS_peak_bytes':peak,'max_child_RSS_bytes':after.ru_maxrss*1024,'limits':limits,'violation':violation,
 'weights_images_DEV_payload_metriccache_reads':0,'GPU_or_official_score_calls':0,
 'source_unchanged':not subprocess.check_output(['git','status','--porcelain=v1'],cwd=owner)}
(folder/'TEST_EXECUTION.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,sort_keys=True));raise SystemExit(0 if child.returncode==0 and violation is None else 2)
