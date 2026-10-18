"""Run only the named CPU fixture tests under an owned-child resource guard."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
folder=Path(__file__).resolve().parent
owner=Path(os.environ['ST_ENDPOINT_IMPLEMENTATION_ROOT'])
expected=os.environ['ST_ENDPOINT_EXPECTED_COMMIT']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner,text=True).strip()==expected
assert not subprocess.check_output(['git','status','--porcelain=v1'],cwd=owner)
selected=[str(folder/'test_independent_supervisor.py')] if os.environ.get('ST_ENDPOINT_ONLY_INDEPENDENT')=='1' else [str(owner/'tests/st_world/test_endpoint_supervisor.py'),str(folder/'test_independent_supervisor.py')]
argv=[sys.executable,'-m','pytest',*selected,'-q']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
limits={'wall_seconds':90,'sampled_aggregate_RSS_bytes':2*2**30,'CPU_seconds':120}
def tree(root):
 result={};pending=[root]
 while pending:
  pid=pending.pop()
  try:
   parts=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
   result[pid]={'rss':int(parts[21])*os.sysconf('SC_PAGE_SIZE'),'ticks':int(parts[19])}
   pending.extend(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
  except (FileNotFoundError,ProcessLookupError):pass
 return result
peak=0;violation=None
with (folder/'TEST_OUTPUT.txt').open('xb') as output:
 child=subprocess.Popen(argv,cwd=owner,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while child.poll() is None:
   memory=sum(row['rss'] for row in tree(child.pid).values());peak=max(peak,memory)
   if time.monotonic()-started>limits['wall_seconds']:violation='WALL_CAP'
   if memory>limits['sampled_aggregate_RSS_bytes']:violation='RSS_CAP'
   if violation:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(.05)
 finally:
  if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
cpu=usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime
if cpu>limits['CPU_seconds']:violation=violation or 'CPU_CAP'
result={'schema':'endpoint_supervisor_independent_fixture_execution_v1','source_commit':expected,'argv':argv,'exit_code':child.returncode,
 'wall_seconds':time.monotonic()-started,'waited_child_CPU_seconds':cpu,'sampled_aggregate_RSS_peak_bytes':peak,
 'max_waited_child_RSS_bytes':usage.ru_maxrss*1024,'limits':limits,'violation':violation,
 'real_weights_images_DEV_payload_metric_cache_reads':0,'official_score_calls':0,'GPU_calls':0,
 'source_unchanged_after_tests':not subprocess.check_output(['git','status','--porcelain=v1'],cwd=owner)}
assert result['source_unchanged_after_tests']
(folder/'TEST_EXECUTION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True));raise SystemExit(0 if child.returncode==0 and violation is None else 2)
