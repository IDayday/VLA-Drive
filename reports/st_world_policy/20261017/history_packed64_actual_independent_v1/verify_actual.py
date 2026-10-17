"""Read only completed fixed64 metadata and small controls, never source payloads."""
import hashlib,json,math,os,resource,time
from pathlib import Path
start=time.monotonic();io0=int(Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0]);reads=0;inputs=[]
out=Path(__file__).resolve().parent
root=Path('/mnt/project/st-world-policy-20261016/audits/history_data/causal_history_cost64_v2');controls=root.with_name(root.name+'_controls')

def rd(p,sha=None,js=True):
 global reads
 p=Path(p);assert p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),str(p)
 st=p.stat();assert st.st_size<2<<20 and st.st_nlink==1
 b=p.read_bytes();reads+=len(b);assert reads<8<<20
 h=hashlib.sha256(b).hexdigest();assert sha is None or sha==h,(str(p),sha,h)
 inputs.append({'path':str(p),'sha256':h,'bytes':len(b)})
 return json.loads(b) if js else b

admission=rd(out.parent/'history_packed64_admission_precheck_v1/ACTUAL_ADMISSION.json','abe66a14cbfc9e617f876b145088b4c66b9df4b1f934e6d7d8dce37db4e1e9e8')
launch=rd('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/history64_actual_admission_v1/LAUNCH.json')
assert launch['admission']=={'path':str(out.parent/'history_packed64_admission_precheck_v1/ACTUAL_ADMISSION.json'),'sha256':'abe66a14cbfc9e617f876b145088b4c66b9df4b1f934e6d7d8dce37db4e1e9e8'}
assert launch['runtime']==admission['runtime'] and launch['hostname']==admission['hostname']
assert admission['issued_at_unix']<=launch['time_unix']<=admission['latest_start_unix']
assert not Path(f"/proc/{launch['pid']}").exists()
assert launch['new_teacher_tensor_budget']==0
inv=rd(controls/'INVOCATION.json');assert inv['admission']==launch['admission']['path'] and inv['sha256']==launch['admission']['sha256'] and inv['automatic_retry'] is False
final=rd(controls/'RESULT.json');inner=rd(root/'CONTROL_HISTORY_RESULT.json');complete=rd(root/'COMPLETE.json');cost=rd(root/'RESOURCE_COST.json')
assert final['status']=='COMPLETE' and final['ledger_result']==str(root/'CONTROL_HISTORY_RESULT.json')
assert inner['status']=='BUILD_RETURNED_PENDING_TERMINAL_ACCOUNTING' and inner['scientific_result'] is False
r=inner['resources'];cli=r['child_accounting']['cli_result'];artifact=cli['artifact'];assert r['returncode']==0 and r['failure'] is None and r['monitor_error'] is None
assert final['cleanup']['live_owned_remaining']==r['cleanup']['live_owned_remaining']==0 and final['cleanup']['verified_processes']==1
assert complete['status']==artifact['status']=='COMPLETE' and complete['scenes']==complete['logs']==64
assert complete['mode']=='registered' and complete['original_gt_or_scores_read'] is False and artifact['full_scale_qualified'] is False
for k,v in complete.items():assert artifact[k]==v,k
manifest=rd(root/'HISTORY_MANIFEST.json',complete['history_manifest_sha256']);requests=rd(root/'TEACHER_REQUESTS.json',complete['teacher_requests_sha256']);selected=rd(root/'SELECTED_SCENES.json');config=rd(root/'resolved_config.json');meta=rd(root/'METADATA_ADMISSION.json')
original=rd('/mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/SELECTED_SCENES.json','dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01')
assert config['selected']==selected['scenes']
assert {k:v for k,v in selected.items() if k!='scenes'}=={k:v for k,v in original.items() if k!='scenes'}
assert all(all(actual[k]==v for k,v in expected.items()) for actual,expected in zip(selected['scenes'],original['scenes']))
assert config['max_scenes']==64 and config['target_times']==requests['times_s']==[0.0] and config['record_layout']=='packed_log_v1'
assert config['split']==manifest['split']==requests['split']==selected['split']=='train'
assert manifest['identity']==complete['history_identity'] and requests['identity']==complete['teacher_requests_identity']
assert manifest['selection_identity']==requests['selection_identity']
ordered=[(x['token'],x['log']) for x in original['scenes']]
assert len(ordered)==len(set(x[0] for x in ordered))==len(set(x[1] for x in ordered))==64
assert ordered==[(x['token'],x['log']) for x in manifest['scenes']]==[(x['token'],x['log']) for x in requests['scenes']]
assert len(manifest['source_logs'])==len(requests['source_logs'])==64 and manifest['source_logs']==requests['source_logs']
assert manifest['view_names']==requests['view_names']==['CAM_F0','CAM_L0','CAM_R0']
shards=list((root/'history_shards').iterdir());assert len(shards)==64 and {p.name for p in shards}=={Path(x['record']).name for x in manifest['scenes']}
record_hashes=[];frame_count=0;valid_views=0;masked_views=0
for entry,sel,req in zip(manifest['scenes'],selected['scenes'],requests['scenes']):
 assert entry['record_layout']=='packed_log_v1' and entry['current_record_sha256']==sel['current_record_sha256']
 p=root/entry['record'];assert p.parent==root/'history_shards' and p.suffix=='.jsonseq'
 b=rd(p,js=False);offset=entry['record_offset'];length=entry['record_length'];assert offset==0 and length==len(b),('record bounds',entry)
 assert hashlib.sha256(b[offset:offset+length]).hexdigest()==entry['sha256']
 x=json.loads(b);assert x['token']==entry['token'] and x['log']==entry['log']
 assert x['schema']=='st_causal_history_v1' and x['offsets_s']==[-1.0,-0.5]
 assert len(x['current_record_identity'])==64 and x['missing_policy']=='masked_absent_no_current_or_future_substitution'
 assert x['view_names']==manifest['view_names']
 assert [f['requested_s'] for f in x['frames']]==[-1.0,-0.5]
 for f in x['frames']:
  assert f['timestamp_us']<=x['current_timestamp_us'] and f['actual_offset_s']<0
  assert abs(f['actual_offset_s']-f['requested_s'])<=x['timestamp_tolerance_s']
  assert [v['camera'] for v in f['views']]==x['view_names'];frame_count+=1
  for v in f['views']:
   valid_views+=int(v['valid']);masked_views+=int(not v['valid'])
 assert len(req['frames'])==1 and req['frames'][0]['requested_s']==0.0 and req['frames'][0]['timestamp_us']==x['current_timestamp_us']
 assert req['frames'][0]['frame_token']==x['token'] and len(req['frames'][0]['views'])==3
 record_hashes.append({'record':entry['record'],'sha256':entry['sha256'],'bytes':length})
reg=rd(admission['registration']['path'],admission['registration']['sha256']);assert cli['registration_sha256']==meta['registration_sha256']==admission['registration']['sha256']
assert meta['policy_sha256_at_admission']==admission['policy']['sha256'] and reg['record_layout']=='packed_log_v1'
ledger=rd('/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json');claim=ledger['claims'][str(root)]
assert claim['claim']==meta['claim'];assert claim['charged_bytes']==31<<20
assert 0<=claim['observed_bytes']<=claim['charged_bytes'] and claim['observed_bytes']==cli['peak_disk_bytes']
account=r['read_accounting'];assert account['status']=='COMPLETE'
assert account['charged_parent_child_rchar_bytes']==account['parent_post_reap_delta_rchar_bytes']
assert account['parent_post_reap_delta_rchar_bytes']==account['parent_post_reap']['rchar']-account['parent_baseline']['rchar']
assert account['parent_post_reap_delta_rchar_bytes']==account['parent_pre_reap_delta_rchar_bytes']+account['child_terminal_rchar_bytes']+account['parent_activity_during_wait_rchar_bytes']
assert inner['charged_rchar_bytes']>=account['charged_parent_child_rchar_bytes']
assert final['charged_rchar_bytes']>=inner['charged_rchar_bytes'] and final['charged_rchar_bytes']<admission['limits']['read_bytes']
assert r['peak_aggregate_rss_bytes']<admission['limits']['rss_bytes'] and r['invocation_seconds']<admission['limits']['wall_seconds']
assert cost['source_bytes_read']==artifact['history_resource_cost']['source_bytes_read']<final['charged_rchar_bytes']
# Enumerate only this ~1MiB pilot, not its data references or campaign tree.
stats=[]
for base in (root,controls):
 for p in base.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():
   st=p.stat();stats.append({'path':str(p),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512})
logical=sum(x['bytes'] for x in stats);allocated=sum(x['allocated_bytes'] for x in stats)
assert logical<32<<20 and allocated<32<<20
assert all(Path(x['path']).suffix in ('.json','.jsonseq','.lock') for x in stats)
summary={'schema':'st_history64_actual_independent_review_v1','status':'PASS_BOUNDED_ACTUAL_METADATA_PILOT','runtime_commit':'b847f5a99dd44ee5256362d380d9d96efb47d35b','reviewer':'/root/st_calibration_independent_correctness_v2','inputs':inputs,'records':record_hashes,'scenes':64,'logs':64,'selection_and_order_exact':True,'teacher_request_times_s':[0.0],'history_times_s':[-1.0,-0.5],'packed_shards':64,'history_frames':frame_count,'valid_history_views':valid_views,'masked_history_views':masked_views,'root_and_control_files':len(stats),'actual_total_logical_file_bytes':logical,'actual_total_allocated_file_bytes':allocated,'inventory':stats,'parent_pid':launch['pid'],'parent_absent':True,'control_status':final['status'],'intermediate_ledger_receipt_status':inner['status'],'status_interpretation':'BUILD_RETURNED_PENDING_TERMINAL_ACCOUNTING is an intentional intermediate receipt. Separate final control result COMPLETE resolves it after final budget/cleanup checks. Not an unresolved error.','cleanup':final['cleanup'],'actual_resources':{'final_charged_rchar_bytes':final['charged_rchar_bytes'],'inner_charged_rchar_bytes':inner['charged_rchar_bytes'],'source_meter_bytes':cost['source_bytes_read'],'child_block_read_bytes':r['counters']['proc_self_io_delta']['read_bytes'],'wrapper_invocation_seconds':r['invocation_seconds'],'builder_seconds':artifact['history_resource_cost']['elapsed_s'],'supervised_child_seconds':r['supervised_child_seconds'],'peak_aggregate_rss_bytes':r['peak_aggregate_rss_bytes'],'child_peak_rss_bytes':cli['peak_process_tree_rss_bytes'],'ledger_peak_observed_bytes':claim['observed_bytes'],'ledger_reserved_charged_bytes':claim['charged_bytes'],'resource_cost_snapshot':cost,'artifact_cost_snapshot':artifact['history_resource_cost']},'accounting_once_charged':True,'limitations':['Only64fixedscenes/64logs. Not full-scale qualification or teacher tensor build.','Referenced raw logs/images/teacher/checkpoints not reread. Metadata consistency verified; geometry not recomputed.','Source CPU/noGPU path inherits exact reviewed runtime; no new GPU activity sampler.','No output equivalence claim for changed source data outside these pinned metadata identities.','Original unrelated TRAIN FAILED_NO_AUTO_RETRY and observer resource failure remain untouched.'],'read_only':True,'rebuild':False,'tests_rerun':False,'cost':{'elapsed_seconds':time.monotonic()-start,'explicit_read_bytes':reads,'logical_read_delta':int(Path('/proc/self/io').read_text().split('rchar: ')[1].splitlines()[0])-io0,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}}
assert summary['cost']['elapsed_seconds']<60 and summary['cost']['logical_read_delta']<8<<20
p=out/'REVIEW.json';assert not p.exists();p.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');assert p.stat().st_size<1<<20
print(json.dumps({k:v for k,v in summary.items() if k in ['status','scenes','logs','packed_shards','history_frames','valid_history_views','masked_history_views','actual_total_logical_file_bytes','actual_total_allocated_file_bytes','cost','actual_resources']},indent=2));print('REVIEW_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
