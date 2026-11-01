"""Read-only exact own-eval research-slot preview; never signals or writes job intent."""
import hashlib,json,time
from pathlib import Path
from starVLA.rl.qwen_world_safety_exploration.resources import probe,remote,OPERATOR_PYTHON
from tools.qwen_world_safety_exploration.native_priority import STUDIES,BASE,read,make_yield_receipt
from starVLA.rl.qwen_world_safety_exploration.contracts import allowed_hosts,fingerprint
host='training-vlawm-tsinghua-zt5'
if host not in allowed_hosts(admit_long_job=True):raise RuntimeError('Host not authorized')
s=probe(host)
code=r'''import json,os
from pathlib import Path
paths={}
for u in UU:
 p=Path('/mnt/project/runtime_gpu_research_slots')/(u+'.lock')
 if p.exists():
  st=p.stat();paths[(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino)]=(u,str(p))
held=[]
for line in Path('/proc/locks').read_text().splitlines():
 v=line.replace('->','').split()
 try:
  a,b,c=v[5].split(':');key=(int(a,16),int(b,16),int(c));pid=int(v[4])
 except(ValueError,IndexError):continue
 if key not in paths:continue
 u,path=paths[key];row=dict(uuid=u,lease=path,pid=pid,kernel_line=line)
 try:
  p=Path('/proc')/str(pid);args=p.joinpath('cmdline').read_bytes().decode().split('\0');row.update(start_ticks=int(p.joinpath('stat').read_text().split()[21]),argv=args,cwd=str(p.joinpath('cwd').resolve()),receipt_root=args[args.index('--receipt-root')+1]if'--receipt-root'in args else None)
 except(OSError,ValueError,IndexError)as e:row['inspection_error']=repr(e)
 held.append(row)
print(json.dumps(held))'''.replace('UU',repr([g['uuid']for g in s['gpus']]))
locks=json.loads(remote(host,[OPERATOR_PYTHON,'-c',code]));entries={}
for study in STUDIES:
 for sp in(study/'owners').glob('*/STATE.json'):
  for key,entry in read(sp,{}).get('evaluations',{}).items():
   for rank,node in entry.get('ranks',{}).items():
    if node.get('host')==host and node.get('job'):
     entries[str(Path(node['job']).resolve())]=(study,sp,key,entry,int(rank),node)
rows=[]
for lock in locks:
 job=lock.get('receipt_root');hit=entries.get(str(Path(job).resolve()))if job else None
 row=dict(lock)
 if hit:
  study,sp,key,entry,rank,node=hit;output=Path(entry['evaluation']);spec=read(output/'SPEC.json',{});identity=read(output/'identity.json',{})
  row.update(owned_candidate=True,campaignroot=str(study),owner_state=str(sp),entry_key=key,rank=rank,node=node,job=job,spec=str(output/'SPEC.json'),spec_sha256=spec.get('sha256'),identity=str(output/'identity.json'),identity_sha256=identity.get('sha256'),result=read(Path(job)/'RESULT.json'),rank_receipt=read(output/f'rank{rank}.json'),actual_running_receipt=read(Path(job)/'RUNNING.json'),actual_dispatch_receipt=read(Path(job)/'DISPATCH.json'))
  try:row['uncommitted_intent_preview']=make_yield_receipt(entry,rank,node,'IMPROVEMENT_VERIFIED_RETENTION_G8 native continuation; Root must recheck exact worker and current native demand at execution')
  except Exception as e:row['not_yield_eligible']=repr(e)
 else:row.update(owned_candidate=False,not_yield_eligible='No exact current E/F/G/H evaluation state binding; never touch')
 rows.append(row)
out=dict(schema='world_tsh5_owned_eval_priority_yield_preview_v1',status='READ_ONLY_PREVIEW_NOT_EXECUTED',created_unix=time.time(),host=host,gpus=[dict(uuid=g['uuid'],index=g.get('index'),free_MiB=g['free_MiB'])for g in s['gpus']],research_locks=rows,owned_candidate_count=sum(r.get('owned_candidate',False)and not r.get('not_yield_eligible')for r in rows),no_signals=True,no_job_files_written=True,requires_fresh_exact_owner_and_spec_check_at_execution=True,method='Read current allowed-host NVML inventory, /proc/locks and /proc worker identity; read shared filesystem receipts only')
out['sha256']=fingerprint(out)
p=BASE/'resources/TSH5_OWNED_EVAL_PRIORITY_YIELD_PREVIEW.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
Path(__file__).with_name('TSH5_OWNED_EVAL_PRIORITY_YIELD_PREVIEW.json').write_bytes(p.read_bytes())
print(json.dumps(dict(path=str(p),sha256=out['sha256'],research_locks=len(rows),owned_candidate_count=out['owned_candidate_count'],jobs=[dict(pid=r['pid'],start_ticks=r.get('start_ticks'),job=r.get('job'),rank=r.get('rank'),uuid=r['uuid'],eligible=not r.get('not_yield_eligible'))for r in rows]),indent=2))
