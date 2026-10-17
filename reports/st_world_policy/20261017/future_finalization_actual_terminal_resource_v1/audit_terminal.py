"""Independent bounded terminal JSON/stat audit; no payload hashing or dispatch."""
import os,time
START=time.monotonic();os.sched_setaffinity(0,{8,9})
import hashlib,json,resource,stat
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20));resource.setrlimit(resource.RLIMIT_CPU,(55,55))
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_finalized_v1')
CONTROL=Path('/mnt/project/st-world-policy-20261016/audits/full_future_finalization_v1_controls')
READ=0;PINS=[];CHECKS=[]
def check(value,name):
    if not value:raise AssertionError(name)
    CHECKS.append(name)
def read(path,expected=None):
    global READ
    path=Path(path);before=path.stat()
    check(stat.S_ISREG(before.st_mode) and not path.is_symlink() and 0<before.st_size<=1<<20,'small_plain:'+path.name)
    READ+=before.st_size;check(READ<=4<<20,'json_budget')
    raw=path.read_bytes();after=path.stat();digest=hashlib.sha256(raw).hexdigest()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    check(stamp(before)==stamp(after) and len(raw)==before.st_size,'stable:'+path.name)
    if expected:check(digest==expected,'hash:'+path.name)
    PINS.append({'path':str(path),'sha256':digest,'bytes':len(raw)})
    return json.loads(raw)
done=read(ROOT/'FINALIZATION_COMPLETE.json','16c47d6603144dcabe0c764f85c23c231bb0edfaf347d869f93c34d79bdc5438')
root_result=read(ROOT/'CONTROL_FINALIZATION_RESULT.json','10566f3eab44dab4ab42e2fb9992a0621bf2fb21cc904b2d333d004bec7cf654')
result=read(CONTROL/'RESULT.json','db72d9418d390d66af7622a35999921f207d588c969ae017f42857572daec7dd')
invocation=read(CONTROL/'INVOCATION.json','04926c223a25ce33febb8929f0048e05aadead0a0e15972f20fd1dfedecb6d04')
check(result['admission']['sha256']=='eb15cea1e9df6f8671c7b3da056815422ac73218c45e664011a6bd8e944f6033','exact_admission')
admission=read(result['admission']['path'],result['admission']['sha256'])
storage=read(result['registration']['path'],result['registration']['sha256'])
freg=read(result['finalization_registration']['path'],result['finalization_registration']['sha256'])
old=read(done['source_terminal']['path'],done['source_terminal']['sha256'])
check(old['status']=='FAILED_NO_AUTO_RETRY','old_failure_preserved')
check(done['source_terminal']==result['source_terminal']==admission['source_terminal'],'failure_lineage')
check(done['schema']=='st_future_metadata_finalization_complete_v1' and done['status']=='COMPLETE','complete_schema')
check(done['scope']=='full_population' and done['scenes']==101592 and done['logs']==1176,'full_coverage_receipt')
check(done['source_shards']==freg['source_shards'] and done['source_shards']['count']==1176 and done['source_shards']['bytes']==984972623,'source_census_identity')
identity=hashlib.sha256(json.dumps({k:v for k,v in done.items() if k!='identity'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
check(identity==done['identity'],'complete_self_identity')
check(done['teacher_extractions']==0 and done['tensor_disk_bytes']==0 and done['training_admission'] is False and done['raw_log_deserialized'] is False and done['source_shards_reused'] is True,'metadata_only')
check(result['status']=='FINALIZATION_COMPLETE' and result['automatic_retry'] is False and result['scientific_result'] is False,'terminal_complete')
check(root_result['status']=='FINALIZATION_RETURNED_PENDING_TERMINAL_ACCOUNTING','root_marker_not_terminal')
check(invocation['admission']==result['admission']['path'] and invocation['sha256']==result['admission']['sha256'] and invocation['automatic_retry'] is False,'one_invocation_identity')
res=result['resources'];limits=admission['limits'];cli=result['cli_result'];account=res['read_accounting']
check(res['returncode']==0 and res['failure'] is None and res['monitor_error'] is None,'monitor_success')
check(result['cleanup']['live_owned_remaining']==res['cleanup']['live_owned_remaining']==0 and res['cleanup']['verified_processes']==1,'cleanup_receipts')
check(cli['artifact']==done and cli==res['child_accounting']['cli_result'] and cli==root_result['cli_result'],'artifact_matches_all_receipts')
check(cli['status']=='GUARDED_METADATA_RETURN' and cli['registration_sha256']==admission['registration']['sha256']==result['registration']['sha256'],'claim_registration_match')
check(cli['writer_callbacks']>0 and cli['writer_commit_callbacks']>0 and cli['checks']>0,'guard_callbacks_executed')
check(res['invocation_seconds']<=limits['wall_seconds'] and res['supervised_child_seconds']<=res['invocation_seconds'],'elapsed_bound')
check(res['peak_aggregate_rss_bytes']<=limits['rss_bytes'] and cli['peak_process_tree_rss_bytes']<=limits['rss_bytes'],'recorded_rss_bound')
check(cli['peak_disk_bytes']<=storage['limits']['max_disk_bytes']==2<<30,'claim_peak_disk_bound')
check(sum(storage['storage_requirements'].values())==2<<30,'claim_inclusive_reservation')
check(account['status']=='COMPLETE' and account['parent_baseline']==invocation['entry_io'],'accounting_baseline')
charge=account['parent_post_reap']['rchar']-account['parent_baseline']['rchar']
check(charge==account['charged_parent_child_rchar_bytes']==account['parent_post_reap_delta_rchar_bytes'],'single_postwait_charge')
check(account['parent_post_reap']['rchar']-account['parent_pre_reap']['rchar']-account['child_terminal_rchar_bytes']==account['parent_activity_during_wait_rchar_bytes']>=0,'child_absorbed_once')
check(charge<=root_result['charged_rchar_bytes']<=result['charged_rchar_bytes']<=limits['read_bytes']==16<<30,'later_publication_reads_included')
check(done['cost']['source_bytes_read']<=result['charged_rchar_bytes'] and done['cost']['metadata_bytes_reserved']<=2<<30,'writer_cost_below_outer_budget')
def census(path,allowed):
    entries=list(path.iterdir());check(len(entries)<=100,'bounded_entry_count')
    rows=[]
    for p in entries:
        s=p.lstat();check(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and p.name in allowed,'allowed_stat_only:'+p.name)
        rows.append({'name':p.name,'logical_bytes':s.st_size,'charged_bytes':max(s.st_size,s.st_blocks*512)})
    return {'files':rows,'logical_bytes':sum(x['logical_bytes'] for x in rows),'charged_bytes':sum(x['charged_bytes'] for x in rows)}
root_stat=census(ROOT,{'METADATA_ADMISSION.json','FINALIZE.lock','CONTROL_FINALIZATION_STARTED.json','CONTROL_FINALIZATION_RESULT.json','FINALIZATION_COMPLETE.json','RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json','TEACHER_REQUESTS.json'})
control_stat=census(CONTROL,{'SUPERVISOR.lock','INVOCATION.json','RESULT.json'})
check(root_stat['charged_bytes']<=2<<30 and control_stat['charged_bytes']<=1<<20,'terminal_stat_disk_bounds')
check(root_stat['charged_bytes']+control_stat['charged_bytes']<=limits['max_disk_bytes'],'combined_disk_bound')
processes=[]
for pid,start in ((540671,1368453455),(540678,1368453542)):
    path=Path('/proc')/str(pid)/'stat'
    try:
        fields=path.read_text().rsplit(') ',1)[1].split();same=int(fields[19])==start
        row={'pid':pid,'expected_start_ticks':start,'present':True,'same_original_process':same,'state':fields[0]}
    except FileNotFoundError:row={'pid':pid,'expected_start_ticks':start,'present':False,'same_original_process':False}
    check(not row['same_original_process'],'original_process_exited:'+str(pid));processes.append(row)
status={k:v.strip() for k,v in (line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines())}
cost={'elapsed_seconds':time.monotonic()-START,'explicit_JSON_read_bytes':READ,'actual_process_VmHWM_bytes':int(status['VmHWM'].split()[0])*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}
check(cost['elapsed_seconds']<60 and cost['actual_process_VmHWM_bytes']<256<<20,'review_budget')
report={'schema':'st_future_finalization_independent_actual_terminal_resource_review_v1','status':'PASS_ACTUAL_TERMINAL_RESOURCE_REVIEW_ONLY','reviewer':'/root/st_owned_descendants_review_v2','evidence':PINS,'checks':CHECKS,'actual_resources':{'invocation_seconds':res['invocation_seconds'],'supervised_child_seconds':res['supervised_child_seconds'],'peak_aggregate_RSS_bytes_recorded_by_original_monitor':res['peak_aggregate_rss_bytes'],'logical_read_terminal_bytes':result['charged_rchar_bytes'],'writer_read_bytes':done['cost']['source_bytes_read'],'claim_peak_disk_bytes':cli['peak_disk_bytes'],'compressed_scene_cache_bytes':done['compressed_cache_bytes'],'root_stat':root_stat,'control_stat':control_stat,'limits':limits},'wait_accounting':{'parent_postwait_minus_baseline':charge,'later_parent_publication_read_bytes':result['charged_rchar_bytes']-charge,'child_not_added_twice':True},'original_process_checks':processes,'coverage':{'scenes':done['scenes'],'logs':done['logs'],'request_identity':done['request_identity'],'request_sha256':done['files']['TEACHER_REQUESTS.json'],'basis':'pinned complete receipt and matching CLI artifact; no request-body rescan'},'old_source_status':old['status'],'fresh_old_failure_hash':done['source_terminal']['sha256'],'review_cost':cost,'actual_ledger_read_or_mutation':False,'payload_read_or_hash_scan':False,'dispatch':False,'scientific_or_training_PASS':False,'remaining_boundaries':['Current review independently checks original monitor receipts/counters, not a second continuous RSS history.','TEACHER_REQUESTS and source shards were not read/hashed; only bounded directory entries were statted.','The image_or_other read category is an inherited generic hash counter; metadata-only scope is bound to reviewed source/receipts, not inferred from category label.','Current storage claim accounting verified via pinned registration and executed guard receipts, not live ledger contents.','Any next index preparation requires fresh policy/ledger/checkpoint pins and its own finite whole-process admission.']}
(HERE/'REVIEW.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'review_sha256':hashlib.sha256((HERE/'REVIEW.json').read_bytes()).hexdigest(),'review_cost':cost,'root_charged_bytes':root_stat['charged_bytes'],'control_charged_bytes':control_stat['charged_bytes'],'actual_rchar':result['charged_rchar_bytes'],'actual_RSS':res['peak_aggregate_rss_bytes']}))
