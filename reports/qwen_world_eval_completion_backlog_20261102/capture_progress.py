"""Finite local metadata proof; never scans scoringcache or trainingjournals."""
from pathlib import Path
import json,time,collections,hashlib,os
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');O=E/'operations/EVALUATION_BACKLOG_COMPLETION_20261102'
def read(p):return json.loads(Path(p).read_text())
def alive(v):
 try:s=(Path('/proc')/str(v['pid'])/'stat').read_text().split();return int(s[21])==v['start_ticks']and s[2]not in('Z','X')
 except FileNotFoundError:return False
out=[];since=read(O/'APPLIED.json')['at']-15
for tag,study,dirname in [('F','standard_training_20261101','RESEARCH_EVALUATION'),('G','revival_training_20261101','RESEARCH_EVALUATION'),('H','improvement_training_20261101','IMPROVEMENT_EVAL')]:
 root=E/study;d=root/'owners'/dirname;owner=read(d/'OWNER.json');state=read(d/'STATE.json');before=read(O/'BEFORE'/tag/'STATE.json');old_complete={k for k,v in before['evaluations'].items()if v.get('status')=='COMPLETE'}
 changed=[]
 for key,b in before['evaluations'].items():
  current=state['evaluations'].get(key)
  if current and any(current.get(k)!=b.get(k)for k in('evaluation','arm','architecture_id','updates','cohort','spec_sha256','world')):changed.append(key)
 assert not changed,changed
 new=[];priority=[]
 for key,entry in state['evaluations'].items():
  if entry['arm']in ('REVIVAL_G16','IMPROVEMENT_VERIFIED_RETENTION_G8')and entry.get('status')!='COMPLETE':priority.append(dict(key=key,status=entry.get('status'),rank_counts=dict(collections.Counter(n.get('status','PENDING')for n in entry['ranks'].values()))))
  for rank,node in entry['ranks'].items():
   if not node.get('job')or node.get('dispatched_unix',0)<since:continue
   j=Path(node['job']);v={}
   for name in ('RESULT.json','RUNNING.json','DISPATCH.json'):
    if(j/name).exists():v=read(j/name);break
   p=Path(entry['evaluation'])/f'rank{rank}.json';progress=read(p)if p.exists()else None
   if progress and(progress.get('host')!=node.get('host')or progress.get('start_unix',0)<node.get('dispatched_unix',0)-5):progress=None
   new.append(dict(key=key,rank=int(rank),world=entry['world'],source=node.get('source'),spec_sha256=entry['spec_sha256'],job=str(j),host=node.get('host'),actual=v,rank_progress=progress))
 newfull=[dict(key=k,summary=v.get('summary'))for k,v in state['evaluations'].items()if v.get('status')=='COMPLETE'and k not in old_complete]
 out.append(dict(study=study,tag=tag,owner=owner,owner_alive=alive(owner),state_status=state.get('status'),state_error=state.get('error'),state_heartbeat=state.get('heartbeat_unix'),max_workers=state.get('max_workers'),total_evaluations=len(state['evaluations']),counts=dict(collections.Counter(v.get('status')for v in state['evaluations'].values())),original_evaluation_bindings_unchanged=not changed,new_full_summaries=newfull,new_worker_count=len(new),actual_new_inferences=sum((n['rank_progress']or{}).get('physical_inference',0)for n in new),new_workers=new,priority_backlog=priority))
v=dict(status='ACTUAL_CPU_OWNERS_AND_SCIENTIFIC_CHILD_PROGRESS',at=time.time(),studies=out,manual_retry_authority=str(E/'operations/MANUAL_SIGKILL_EVAL_RETRY_20261102/REVIEW_AUTHORIZATION.json'),source_freezes=[str(O/'SOURCE_FREEZE.json'),str(O/'SOURCE_FREEZE_V2.json')],claims='Existing EVALUATION_DISPATCH.lock/CLAIMS/physicalleases retained; nontterminal orunknownclaims neverstolen',no_training_signals_or_policy_changes=True,no_partial_mean_scores=True)
(O/'ACTUAL_PROGRESS.json').write_text(json.dumps(v,ensure_ascii=True,indent=2)+'\n');(O/'CURRENT_OWNERS.json').write_text(json.dumps([dict(study=s['study'],**s['owner'])for s in out],indent=2)+'\n')
print(json.dumps([dict(study=s['study'],pid=s['owner']['pid'],start_ticks=s['owner']['start_ticks'],alive=s['owner_alive'],counts=s['counts'],new_workers=s['new_worker_count'],actual_new_inferences=s['actual_new_inferences'],new_full=s['new_full_summaries'],retention10=[dict(rank=n['rank'],host=n['host'],status=n['actual'].get('status'),progress=n['rank_progress'])for n in s['new_workers']if n['key']=='IMPROVEMENT_VERIFIED_RETENTION_G8_u010000_NAVTEST'])for s in out],indent=2))
