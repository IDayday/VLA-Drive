from pathlib import Path
import hashlib,json,os,re,resource,signal,stat,time
signal.alarm(60);os.sched_setaffinity(0,{8,9});resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20));t=time.monotonic();nread=0;pins=[]
job=Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/train_extract');root=Path('/mnt/project/st-world-policy-20261016/caches/learned_w_probe_SP0_5000_v1/train')
def read(p,expected=None,keep=True):
 global nread
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<512<<10
 b=p.read_bytes();assert len(b)==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns;nread+=len(b);assert nread<=2<<20
 h=hashlib.sha256(b).hexdigest();assert expected is None or h==expected
 pin={'path':str(p),'sha256':h,'bytes':len(b)}
 if keep:pins.append(pin)
 return json.loads(b),pin
result,rpin=read(job/'RESULT.json','4d24e4ed9a97744fe0d977c21284cb5fd64ec14c1348d71f6edadeff61245b27')
cli,_=read(job/'CLI_RESULT.json','17a8e9dd09447a2f2166a1dc8b216ada72e87b3525cfb3228d940dbd64a3d350');done,_=read(root/'COMPLETE.json','17a8e9dd09447a2f2166a1dc8b216ada72e87b3525cfb3228d940dbd64a3d350')
sig,_=read(root/'identity.json','7c1f21f83295e50d22791d8664fb1f67fd3e2c5272d66ddbff64dbda56780b29')
child,childpin=read(job/'CHILD.json','ca68d7851f0ab6054185dd8906786ea989d8fc006d959f8819e63278ab111e09');c,_=read(job/'COUNTERS.json','8c9afd60bfe2e12c8d6769a98242c552f2f32c67ca22f65e0c8469bb58698e4b')
admitted,_=read(job/'ADMITTED.json','a826d340feb69fd7595961ac20dbd059bd1525c94fd8393a972091a1d4650aeb')
assert result['status']=='STOPPED_NO_AUTO_RETRY' and result['resources']['limit_reason']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
assert result['resources']['exit_code']==0 and result['resources']['first_monitor_error'] is None and result['resources']['process_cleanup']['live_owned_remaining']==0 and c==result['resources']['counters'] and c['limit_exceeded'] is None
assert done==cli==result['cli_result'] and done['completed']==done['requested']==64 and done['identity']==sig['identity'] and sig['targets_read'] is False and len(sig['rows'])==64
assert child['pid']==child['sid']==child['pgid']==c['pid']==544166 and child['start_ticks']==1368591802
assert admitted['admission']==result['admission'] and admitted['automatic_retry'] is False and admitted['consumed'] is True
assert child['argv'][1]=='/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/w_storage_bridge_v1/observed_cli.py' and child['argv'][-1]=='237302801889'
pidpath=Path('/proc')/str(child['pid'])/'stat';assert not pidpath.exists()
charged=max(c['hash_logical_bytes_admitted']+c['torch_load_nominal_bytes'],c['proc_self_io_delta']['rchar']);account=result['read_accounting'];cum=account['prior_read_bytes']+charged
assert charged==153889914639 and cum==191465019694 and cum<account['role_limit_bytes']
r=result['resources'];assert r['wall_seconds']<960 and r['sampled_tree_rss_peak_bytes']<64<<30 and c['gpu_peak_reserved_bytes']<24<<30
entries=list(root.iterdir());assert len(entries)==131
census=[];payload_count=0;allocated=0;logical=0
for p in entries:
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1;allocated+=s.st_blocks*512;logical+=s.st_size
 if p.suffix=='.safetensors':payload_count+=1
assert payload_count==64 and allocated<86254736
for row in sig['rows']:
 token=row['token'];meta,pin=read(root/(token+'.json'),keep=False)
 assert meta['identity']==sig['identity'] and meta['token']==token and meta['log']==row['log'] and re.fullmatch('[a-f0-9]{64}',meta['sha256'])
 payload=root/(token+'.safetensors');s=payload.stat();assert s.st_size==1179952
 census.append({'token':token,'receipt_sha256':pin['sha256'],'claimed_payload_sha256':meta['sha256'],'bytes':s.st_size})
body={'schema':'st_SP0_train_W_terminal_resource_review_v1','status':'TERMINAL_FAILURE_PRESERVED_METADATA_CONSISTENT','reviewer':'/root/st_owned_descendants_review_v2','original_result':rpin,'original_status':result['status'],'original_reason':r['limit_reason'],'cli_complete_and_signature_match':True,'completed':64,'requested':64,'signature_identity':sig['identity'],'resource_limits_satisfied':True,'resources':{'wall_seconds':r['wall_seconds'],'sampled_RSS_peak_bytes':r['sampled_tree_rss_peak_bytes'],'GPU_peak_allocated_bytes':c['gpu_peak_allocated_bytes'],'GPU_peak_reserved_bytes':c['gpu_peak_reserved_bytes'],'charged_extract_read_bytes':charged,'prior_plan_read_bytes':account['prior_read_bytes'],'cumulative_train_role_read_bytes':cum,'remaining_role_read_before_this_review_bytes':account['role_limit_bytes']-cum},'original_recorded_child':{'pid':child['pid'],'start_ticks':child['start_ticks'],'present_at_review':False,'argv_bound_to_actual_bridge':True},'cleanup_receipt':r['process_cleanup'],'process_verification_boundary':'Recordedchild absent; other5 owned process identities not exposed in these small receipts, no broad /proc census or new historical peak measurement.','cache_stat_only':{'entries':len(entries),'payload_files':payload_count,'logical_bytes':logical,'allocated_bytes':allocated,'claim_total_bytes':86254736},'record_metadata_census_sha256':hashlib.sha256(json.dumps(census,sort_keys=True).encode()).hexdigest(),'tensor_payload_verified':False,'tensor_files_opened':0,'sidecar_metadata_records_checked':64,'existing_DEV_reconciliation':'Same separation pattern is applicable in principle. Existing helper is pinned DEV16/specificfailedhash/consumertrain_plan and cannot be reused directly forTRAIN64/fit. Require a separate bounded64-record original hash/read_world/shape/finite verification and an explicit TRAIN reconciliation preserving original supervisor failure. No extraction retry needed or admitted.','fit_admitted':False,'extraction_retry_admitted':False,'ledger_access_or_mutation':False,'signals_or_dispatch':False,'input_pins':pins,'review_cost':{'explicit_metadata_bytes_read':nread,'wall_seconds':time.monotonic()-t,'actual_self_VmHWM_bytes':next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')),'cpu_affinity':sorted(os.sched_getaffinity(0))}}
out=Path(__file__).parent/'REVIEW.json';out.write_text(json.dumps(body,indent=2)+'\n');print(json.dumps({'status':body['status'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'cost':body['review_cost'],'cache':body['cache_stat_only']}))
