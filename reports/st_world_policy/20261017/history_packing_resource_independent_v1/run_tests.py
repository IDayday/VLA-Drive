import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).parent
os.sched_setaffinity(0,{8,9})

def io():
 return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}

def snapshot():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit(): continue
  try:
   fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
   out[int(p.name)]={'ppid':int(fields[1]),'start':int(fields[19]),'rss':int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
  except (FileNotFoundError,ProcessLookupError): continue
 return out

commands=[sys.executable,'-m','pytest','-p','no:cacheprovider','-q',str(HERE/'test_resources.py'),str(HERE.parent/'history_metadata_scale_v1/test_scale.py')]
start=time.monotonic(); baseline=io(); peak=0; issue=None
with (HERE/'PYTEST.txt').open('w') as log:
 child=subprocess.Popen(commands,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
 while child.poll() is None:
  rows=snapshot(); members={child.pid}
  changed=True
  while changed:
   new={pid for pid,r in rows.items() if r['ppid'] in members}; changed=bool(new-members); members|=new
  peak=max(peak,sum(rows[p]['rss'] for p in members|{os.getpid()} if p in rows))
  if peak>2<<30 or time.monotonic()-start>120:
   issue='RESOURCE_LIMIT'; child.terminate(); child.wait(timeout=10); break
  time.sleep(.05)
 code=child.wait()
post=io(); read=post['rchar']-baseline['rchar']
value={'schema':'st_independent_process_review_tests_v1','source_commit':'fc76b5f075f0f3ed2fbc0cc35825bd557d3fa006','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'/root/st_owned_descendants_review_v2','command':commands,'exit_code':code,'elapsed_seconds':time.monotonic()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'cuda_visible_devices':'','peak_sampled_rss_bytes':peak,'post_wait_parent_rchar_delta':read,'read_cap_bytes':2<<30,'read_within_budget':read<=2<<30,'rss_cap_bytes':2<<30,'wall_cap_seconds':120,'resource_issue':issue,'gpu_or_real_task_launched':False,'signals_only_fixture':True,'output_bytes':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),'read_accounting':'post-wait parent delta includes reaped fixture reads exactly once; imports and /proc snapshots included','rss_scope':'50ms PPID descendant sampling including cross-SID fixture tasks, plus harness; not kernel hard guarantee'}
(HERE/'TEST_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
raise SystemExit(code)
