#!/usr/bin/env python3
"""Archive bounded evidence into this report only. Never stages Git or changes runs."""
from datetime import datetime,timezone
import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
entries=[]
def copy(source,dest):
 source=pathlib.Path(source);dest=P/dest;dest.parent.mkdir(parents=True,exist_ok=True)
 if not source.is_file():raise FileNotFoundError(source)
 s=sha(source)
 if not dest.exists()or sha(dest)!=s:shutil.copyfile(source,dest)
 assert sha(dest)==s
 entries.append(dict(source=str(source),archive=str(dest.relative_to(P)),file_sha256=s,bytes=source.stat().st_size,disposition='exact_copy'))
def write(dest,value):
 p=P/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return p
plan=read(R/'operations/EARLY_EVAL_20261103/PLAN.json');state=read(R/'operations/EARLY_EVAL_20261103/STATE.json')
for case in plan['cases']:
 s=state['cases'][case['id']]
 if s['status']!='COMPLETE' or s['completed_shards']!=32:raise ValueError('Complete results required')
 for field,name in [('summary','SUMMARY'),('paired_analysis','PAIRED_ANALYSIS')]:
  value=read(s[field])
  if value['status']!='COMPLETE'or value['scenes']!=12146 or value['invalid_scenes']!=0 or value['physical_logs']!=44:raise ValueError('Population changed')
  copy(s[field],f"results/early_full/{case['id']}_{name}.json")
copy(pathlib.Path(plan['baseline']).parent/'SUMMARY.json','results/early_full/IL46_SUMMARY.json')
copy(R/'operations/EARLY_EVAL_20261103/PLAN.json','analysis/PLAN.json')
for name in ('builder.py','D46_EARLY_FULL_FACTS.json','D46_EARLY_FULL_READOUT_zh.txt','D46_zero_repair_and_gain_loss.svg','D46_zero_repair_and_gain_loss.png','D46_safety_and_progress_delta.svg','D46_safety_and_progress_delta.png'):
 copy(R/'analysis/plots'/name,'plots/'+name)
copy(R/'qualification/ACTUAL_WORLD16_PENDING_RESUME_PASS.json','analysis/ACTUAL_WORLD16_PENDING_RESUME_PASS.json')
REC=R/'operations/EVAL_DIRECTORY_RECOVERY_20261103'
for name in ('D2_STARTUP_TIMEOUT_REVIEW.json','D2_RESUMED_RESCUE_ACTIVE.json','DIRECTORIES_AND_GATE_BACKFILL.json','RECOVERY_AUDIT.json','ORPHAN_PROCESS_AUDIT.json'):
 copy(REC/name,'results/recovery/'+name)
copy(R/'operations/EVALUATION_REFERENCE_DEPLOYMENT_20261103/OWNER_HANDOFF.json','results/operations_v6/OWNER_HANDOFF.json')
copy(R/'operations/EVALUATION_EQUIVALENCE.json','results/operations_v6/EVALUATION_EQUIVALENCE.json')
copy(R/'operations/OLD_I_RESOURCE_HOLD_20261103/RESOURCE_PRIORITY_PAUSE.json','results/old_I_resource_hold/RESOURCE_PRIORITY_PAUSE.json')
copy(R/'operations/OLD_I_RESOURCE_HOLD_20261103/NATIVE_SAVE_EXIT0.json','results/old_I_resource_hold/NATIVE_SAVE_EXIT0.json')
compact=dict(schema='D46_early_complete_evaluation_state_compact_v1',snapshot_utc=datetime.now(timezone.utc).isoformat(),
 source=str(R/'operations/EARLY_EVAL_20261103/STATE.json'),source_file_sha256=sha(R/'operations/EARLY_EVAL_20261103/STATE.json'),
 plan_sha256=state['plan_sha256'],complete=state.get('complete'),terminal=state.get('terminal'),cases={})
for case in plan['cases']:
 v=state['cases'][case['id']];compact['cases'][case['id']]={k:v[k]for k in('status','completed_shards','active_shards','PDMS','output','summary','paired_analysis')if k in v}
compact['sha256']=fp(compact);write('analysis/EARLY_FULL_STATE_COMPACT.json',compact)
# Actual shared-filesystem-only run snapshot. No SSH or process manipulation.
c=read(R/'CAMPAIGN.json');snap=dict(schema='D46_completed_result_archive_run_state_snapshot_v1',snapshot_utc=datetime.now(timezone.utc).isoformat(),
 source_campaign=str(R/'CAMPAIGN.json'),source_campaign_body_sha256=fp(c),source_campaign_at=c.get('at'),
 no_remote_process_probe=True,runs={})
for arm,r in c['runs'].items():
 root=pathlib.Path(r['output']);j=r.get('jobs',[])[-1]if r.get('jobs')else{};jp=pathlib.Path(j.get('job','/notthere'))
 result=read(jp/'RESULT.json')if(jp/'RESULT.json').exists()else None;running=read(jp/'RUNNING.json')if(jp/'RUNNING.json').exists()else None
 latest=read(root/'latest.json')if(root/'latest.json').exists()else None;counts=[];times=[]
 for rank in range(8):
  file=root/f'metrics_rank{rank}.jsonl'
  if not file.exists():continue
  with file.open('rb')as f:f.seek(max(0,file.stat().st_size-20000));lines=f.read().splitlines()
  rows=[]
  for line in lines[1:]:
   try:rows.append(json.loads(line))
   except ValueError:pass
  if rows:counts.append(rows[-1]['updates']);times.append(rows[-1].get('at'))
 snap['runs'][arm]=dict(campaign_status=r.get('status'),execution_hold=r.get('execution_hold'),
  scientific_source=r['source'],scientific_source_sha=r['source_sha'],latest_published=latest,
  current_job=j.get('job'),current_job_host=j.get('host'),current_job_returncode=result.get('returncode')if result else None,
  current_RUNNING_receipt_status=running.get('status')if running else None,
  all8_journal_files_read=len(counts),all8_latest_journal_updates_min=min(counts)if counts else None,
  all8_latest_journal_updates_max=max(counts)if counts else None,journal_timestamp_min=min(x for x in times if x is not None)if times else None,
  note='RUNNING receipt is not a remote liveness probe; journal timestamps/counts are actual committed rows. Held routes are not scientific failures.')
snap['sha256']=fp(snap);write('analysis/COMPLETED_RESULTS_RUN_STATE_SNAPSHOT.json',snap)
refs=[]
for p in [R/'CAMPAIGN.json',R/'qualification/ACTUAL_QUALIFICATION_AND_FRESH_FORMAL_PROOF.json',R/'analysis/EARLY_OPTIMIZATION_HEALTH_20261103.json',R/'operations/EARLY_EVAL_20261103/STATE.json']:
 refs.append(dict(source=str(p),file_sha256=sha(p),bytes=p.stat().st_size,disposition='reference_only_to_avoid_repeated_large_payload'))
archive=dict(schema='D46_completed_full_results_source_archive_v1',snapshot_utc=datetime.now(timezone.utc).isoformat(),
 root=str(R),report_directory=str(P),files=entries,references_only=refs,
 generated_compact_files=['analysis/EARLY_FULL_STATE_COMPACT.json','analysis/COMPLETED_RESULTS_RUN_STATE_SNAPSHOT.json'],
 no_model_tensor_scene_ledger_or_full_worker_log_archived=True,
 index_not_modified=True,archive_builder=dict(path='archive_completed_review.py',file_sha256=sha(__file__)))
archive['report_documents']=[dict(archive=str(q.relative_to(P)),file_sha256=sha(q),bytes=q.stat().st_size)for q in (P/'README.md',P/'EARLY_EVALUATION.md',P/'archive_completed_review.py')if q.is_file()]
archive['sha256']=fp(archive);write('SOURCE_ARCHIVE_COMPLETED_20261103.json',archive)
print(json.dumps(dict(report=str(P),files_copied=len(entries),bytes_copied=sum(x['bytes']for x in entries),snapshot=snap['snapshot_utc'],runs=snap['runs']),ensure_ascii=False))
