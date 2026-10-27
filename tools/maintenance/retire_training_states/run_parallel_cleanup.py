import json,subprocess,time,os,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if(ROOT/'PARALLEL_LAUNCH.json').exists():raise FileExistsError('Already launched')
workers={}
for rank in range(4):
 cmd=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-u',str(ROOT/'retire_states_parallel.py'),'--inventory',str(ROOT/'CHECKPOINT_INVENTORY.json'),'--apply','--shard-rank',str(rank),'--shard-world','4']
 with(ROOT/f'cleanup_rank{rank}.log').open('x') as log:
  p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/mnt/project/VLA-Drive-qwenflow-rl77-selected-navtest-20261027',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 workers[rank]=p
(ROOT/'PARALLEL_LAUNCH.json').write_text(json.dumps(dict(at=time.time(),workers={str(k):dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21])) for k,p in workers.items()}),indent=2)+'\n')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 future={pool.submit(p.wait):rank for rank,p in workers.items()}
 results={}
 for f in concurrent.futures.as_completed(future):results[str(future[f])]=f.result()
(ROOT/'PARALLEL_EXIT.json').write_text(json.dumps(dict(status='COMPLETE' if all(x==0 for x in results.values()) else 'FAILED_PRESERVE_EVIDENCE',exit_codes=results,at=time.time()),indent=2)+'\n')
