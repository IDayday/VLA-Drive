import csv,gzip,hashlib,json,pathlib,shutil,subprocess,time
r=pathlib.Path('/mnt/project/minimal-repair-artifacts/20261015')
w=pathlib.Path('/mnt/project/VLA-Drive-minimal-repair-20261015')
out=w/'reports/minimal_repair/20261015'; ev=out/'evidence'
def read(p):return json.loads(p.read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
files=['STAGE_B_DISPOSITION_v15.json','FINAL_COSTS_v15.json','SUPERSEDED_EXPORT_COST_BOUNDS_v15.json',
 'BALANCED_COMMON_ENDPOINT_AUDIT_v14.json','BALANCED_MATCHING_AUDIT_v14.json','TRAINING_CONDITIONAL_FIT_balanced_v14.json',
 'BALANCED_V14_POSTCHECKS_COMPLETE.json','MIN_NONE_BALANCED_ONLINE_ACCEPTANCE_v14.json','BALANCED_ONLINE_ACCEPTANCE_v14.json',
 'FORMAL_ONLINE_ACCEPTANCE_v14.json','LOG_NONE_FORMAL_ONLINE_ACCEPTANCE_v14.json','MIN_NONE_FORMAL_ONLINE_ACCEPTANCE_v14.json',
 'FINAL_NAVTEST_V15_REGISTRY.json','CPU_WRAPPER_REAL_CHECK_v15.json','ABSENT_TIME_RECOVERY_v15.json',
 'REGISTRY_METADATA_ERRATA_v14.json','BALANCED_V15_SCORING_RUNNERS.json','FOLDA_CLOSURE_RESULT_v5.json','TARGET_QUALITY_FOLDA_v5.json']
files += [p.name for pat in ['BALANCED_V15_EVAL_*.json','BALANCED_V15_MECHANISM_*.json','TEMP_RAM_RELEASE_*_v15.json','RESOURCE_CLOSURE_*_v15.json'] for p in sorted(r.glob(pat))]
for name in files:
 p=r/name;assert p.exists(),p;shutil.copy2(p,ev/name)
for name in ['FINAL_METHOD_LOCK.json','FINAL_READINESS.json','RESOURCE_PREFLIGHT.json','RUNNERS.json']:
 shutil.copy2(r/'final_navtest_v15'/name,ev/('NAVTEST_'+name.replace('.json','_v15.json')))
shutil.copy2(r/'resume_v14/ACCEPTANCE.json',ev/'BALANCED_EXACT_RESUME_ACCEPTANCE_v14.json')
for src,dst in [('BALANCED_RESULTS_v14','balanced_results'),('FINAL_NAVTEST_RESULTS_v15','final_navtest_results')]:
 for p in (r/src).iterdir():
  if p.is_file() and p.suffix in ('.json','.csv','.md'):
   (out/dst).mkdir(exist_ok=True);shutil.copy2(p,out/dst/p.name)
for name in ['archive_final_evidence_v15.py','audit_resource_closure_v15.py','release_temporary_ram_v15.py','recover_scoring_v15.py']:
 shutil.copy2(r/name,ev/'executed_validation'/name)
# Canonical numerical result rows, never images, maps, boxes or trajectory tensors.
manifest=[];baselines={}
registries=['STAGE_A_V5_EVAL_NATURAL.json','STAGE_A_V5_EVAL_RECOVERY.json',
 'STAGE_A_V5_MECHANISM_NATURAL.json','STAGE_A_V5_MECHANISM_RECOVERY.json',
 'BALANCED_V15_EVAL_NATURAL.json','BALANCED_V15_EVAL_RECOVERY.json',
 'BALANCED_V15_MECHANISM_NATURAL.json','BALANCED_V15_MECHANISM_RECOVERY.json','FINAL_NAVTEST_V15_REGISTRY.json']
for name in registries:
 reg=read(r/name)
 distribution=reg.get('distribution','NATURAL');population=reg.get('population','dev')
 for job in reg['jobs']:
  if job.get('epoch',24)!=24:continue
  p=pathlib.Path(job['scores']);complete=read(p/'COMPLETE.json')
  expected=12146 if population=='navtest' else 1696
  assert all(complete[k]['scenes']==expected and complete[k]['failed']==0 and complete[k]['valid'] for k in ['q0','q_final'])
  entry={'registry':name,'id':job['id'],'population':population,'distribution':distribution,
         'complete':complete,'scoring_identity':read(p/'identity.json'),
         'prediction_identity':read(pathlib.Path(job['predictions'])/'identity.json')}
  original_prediction_identity=pathlib.Path(job['predictions'])/'identity.json'
  idx=entry['prediction_identity'].pop('index')
  entry['prediction_identity_json_sha256']=h(original_prediction_identity)
  entry['prediction_index_summary']={'scenes':len(idx),'canonical_json_sha256':hashlib.sha256(json.dumps(idx,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'omitted_duplicate_token_log_list':'Same population is present in score_rows; full identity retained in shared artifacts'}
  if (p/'CPU_RESOURCE_USAGE.json').exists():entry['CPU_resource_usage']=read(p/'CPU_RESOURCE_USAGE.json')
  dump(ev/'canonical_endpoints'/f"{job['id']}_{distribution}.json",entry)
  for variant in ['q0','q_final']:
   source=p/variant/'scenes.csv'
   recorded_hash=complete[variant].get('scene_csv_sha256')
   if recorded_hash is not None:assert h(source)==recorded_hash
   rows=list(csv.DictReader(source.open()));assert len(rows)==expected and all(x['status']=='ok' for x in rows)
   for metric,value in complete[variant]['metrics'].items():
    assert abs(sum(float(x[metric]) for x in rows)/len(rows)-value)<1e-12,(job['id'],variant,metric)
   basekey=(population,distribution)
   if variant=='q0':
    if basekey in baselines:
     assert baselines[basekey]==h(source),(basekey,job['id']);continue
    baselines[basekey]=h(source);stem=f'{population}_{distribution}_input'
   else:stem=f'{population}_{distribution}_{job["id"]}'
   target=out/'score_rows'/(stem+'.csv.gz');target.parent.mkdir(exist_ok=True)
   target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
   manifest.append({'path':str(target.relative_to(out)),'source':str(source),'uncompressed_sha256':h(source),
                    'compressed_sha256':h(target),'compressed_bytes':target.stat().st_size,'scenes':expected})
dump(out/'score_rows/INDEX.json',{'content':'Numerical canonical result rows and scene/log identifiers only; no input images, GT or predictions',
 'common24_only':True,'interventions_separate':True,'baselines_deduplicated':True,'files':manifest})
# Native training metadata with complete source/initialization/fold identity and endpoint checksum.
runs=[]
for name,predroot in [('STAGE_A_V5_TASKS.json','development_natural_v5'),('STAGE_A_BALANCED_V14_TASKS.json','development_balanced_v14/natural')]:
 for job in read(r/name)['jobs']:
  path=pathlib.Path(job['output']);ident=read(path/'identity.json');complete=read(path/'COMPLETE.json')
  pred=read(r/predroot/'predictions'/(job['id']+'_epoch24')/'identity.json')
  assert complete['epoch']==24 and complete['completed']==9168
  logs=ident.pop('training_log_names',None)
  row={'run_id':job['id'],'run_path':str(path),'identity_json_sha256':h(path/'identity.json'),
       'identity_excluding_log_list':ident,'training_logs_count':len(logs) if logs else None,
       'complete':complete,'endpoint_checkpoint_sha256':pred['checkpoint_sha256'],'evaluation_source_sha':pred['source_sha']}
  runs.append(row)
dump(ev/'FINAL_TRAINING_IDENTITIES_v15.json',{'runs':runs,'heads':len(runs),'all_common24':True,'full_AB_training_per_head':False})
# Resource closure is observational: never kills or revives any other workload.
closures=[read(p) for p in r.glob('RESOURCE_CLOSURE_*_v15.json')]
assert len(closures)==8 and all(x['owned_queue_empty'] for x in closures)
ram=[read(p) for p in r.glob('TEMP_RAM_RELEASE_*_v15.json')];assert len(ram)==3 and all(x['status']=='RELEASED' for x in ram)
cost=read(r/'FINAL_COSTS_v15.json');bounds=read(r/'SUPERSEDED_EXPORT_COST_BOUNDS_v15.json')
closure={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'RESEARCH_QUEUES_FINISHED_CONDITIONAL_STAGE_B_NOT_RUN',
 'owned_research_processes_remaining':0,'audited_current_authorized_hosts':[x['host'] for x in closures],
 'temporary_RAM_logical_bytes_released':sum(z['logical_file_bytes'] for x in ram for z in x['removed']),
 'persistent_artifact_root':str(r),'persistent_root_logical_bytes_snapshot':31500445697,
 'storage_measurement':'Completed du -sb of study root after canonical completion; includes superseded outputs; shortly before small final receipts. Excludes reused S0 assets and Git worktrees.',
 'GPU_process_hours_recorded':cost['GPU_process_hours_recorded'],'superseded_export_additional_GPU_hour_upper_bound':bounds['sum_GPU_process_hour_upper_bound'],
 'CPU_process_hours_recorded_lower_bound':cost['CPU_process_hours_recorded_lower_bound'],'CPU_total_cap_certified':False,
 'CPU_accounting_gap':'Early uninstrumented scorer children, trainer CPU, generic tests/startup/metadata validation are missing; a lower bound below160 does not establish total CPU below160.',
 'shared_S0_historical_GPU_hours':483.481,'pressure_scripts_stopped':0,'pressure_scripts_restored':0,
 'canonical_data_checkpoints_and_superseded_results_retained':True,'paused_research_not_resumed':True,'G2_not_modified':True,
 'no_further_automatic_jobs':True}
dump(r/'RESOURCE_CLOSURE_FINAL_v15.json',closure);shutil.copy2(r/'RESOURCE_CLOSURE_FINAL_v15.json',ev/'RESOURCE_CLOSURE_FINAL_v15.json')
versions={'task_branch':'feature/scene-conditioned-minimal-repair-20261015',
 'archive_base':'e588fdc0b19a92922c1ec59aa5acba48a4683769','original_small_head_source':'3c08240a20709b285966ce5b1dfd6d7d49be0d6c',
 'S0_training':'1493deda247107efe076b9294863a6753391298a','primary_v5_training_and_model_evaluation':'0471af96e4f769eb2b31b787a68c57be24bebee0',
 'bounded_balance_training':'d2996601429044e3ef3b71055a4ca546a3ca928e',
 'analysis_group_identity':'85e2e7748ea8f1879c7c8fcab42a0f006f64ec23','RAM_label_helper':'8abc759c69a609f630f422b899b0211732653445',
 'endpoint_readiness_tool':'bedad382c67558df9d98bce48629f3f7c86ec8ba',
 'CPU_scoring_wrapper_and_by_run_summary':'3431960c7f5fcbcffb7249fbc2df5a9fe83e8ed6',
 'report_commit_is_not_training_source':True,'cost_closure_tool_sha256':h(w/'tools/decision_local/summarize_minimal_repair_costs.py'),
 'actual_model_file_hashes':'evidence/FINAL_TRAINING_IDENTITIES_v15.json','final_weight_hashes':'evidence/NAVTEST_FINAL_READINESS_v15.json',
 'final_push_receipt':'/mnt/project/minimal-repair-artifacts/20261015/FINAL_PUSH_RECEIPT_v15.json (written after final push; excluded from self-referential commit)'}
dump(out/'SOURCE_VERSIONS_FINAL.json',versions)
print(json.dumps({'archived_score_files':len(manifest),'score_compressed_bytes':sum(x['compressed_bytes'] for x in manifest),'formal_heads':len(runs),'closure':closure},ensure_ascii=False))
