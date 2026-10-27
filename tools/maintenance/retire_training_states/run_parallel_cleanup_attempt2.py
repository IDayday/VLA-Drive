import json,subprocess,os,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT/'attempt1_evidence/PARALLEL_EXIT.json').read_text());assert old['exit_codes']['3']==0
workers={}
for rank in [0,1,2]:
 cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-u',str(ROOT/'retire_states_parallel.py'),'--inventory',str(ROOT/'CHECKPOINT_INVENTORY.json'),'--apply','--shard-rank',str(rank),'--shard-world','4','--attempt','parallel2']
 with(ROOT/f'cleanup_attempt2_rank{rank}.log').open('x') as log:
  workers[rank]=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/mnt/project/VLA-Drive-qwenflow-rl77-selected-navtest-20261027',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'PARALLEL_LAUNCH_attempt2.json').write_text(json.dumps(dict(at=time.time(),workers={str(k):dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21])) for k,p in workers.items()},reused_completed_rank3=True),indent=2)+'\n')
results={'3':0}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures={pool.submit(p.wait):rank for rank,p in workers.items()}
 for f in concurrent.futures.as_completed(futures):results[str(futures[f])]=f.result()
out=dict(status='COMPLETE' if all(x==0 for x in results.values()) else 'FAILED_PRESERVE_EVIDENCE',exit_codes=results,at=time.time(),attempt=2,prior_failure_preserved='attempt1_evidence')
t=ROOT/'PARALLEL_EXIT.json.tmp';t.write_text(json.dumps(out,indent=2)+'\n');os.replace(t,ROOT/'PARALLEL_EXIT.json')
