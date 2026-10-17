"""Read only fixed small result/control JSON; never import model or open tensors."""
import ast,collections,hashlib,json,math,os,resource,sys,time
from pathlib import Path
START=time.monotonic();OUT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
ROOT=Path('/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1')
RT=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017');PINS={};BLOBS={};READ=0;CHECKS=[]
def read(p,expected=None):
 global READ
 p=Path(p);assert p.suffix in ('.json','.py') and p.stat().st_size<1<<20
 if p not in BLOBS:
  before=p.stat();raw=p.read_bytes();after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);READ+=len(raw);assert READ<5<<20
  BLOBS[p]=raw;PINS[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 if expected:assert PINS[str(p)]['sha256']==expected
 return BLOBS[p]
def doc(p,sha=None):return json.loads(read(p,sha))
def check(name,value):
 assert value,name
 CHECKS.append(name)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def identity(v):return hashlib.sha256(canonical({k:x for k,x in v.items() if k!='identity'}).encode()).hexdigest()
stage=doc(ROOT/'stages/fit/RESULT.json','996176a20ec4d750e6687b0a7f273cfc136c4d6b3fd4a42c75a3807d0ca1498f')
result=doc(ROOT/'fit/RESULT.json','ef91d3f867c78e9bba018fdafbbe11c3210cd06b67e9daa225daaf1f92798431')
manifest=doc(ROOT/'fit/MANIFEST.json','694c4767d3a1110e4fed76ebb3ae96e813707752e603110199fdfe11b3d4892f')
state=doc(ROOT/'fit/STATE.json','ab70702e8ef504267e196095359ba9ef89354988d844e61600153cb82e2b6827')
check('manifest_identity',identity(manifest)==manifest['identity']);check('result_state_manifest_identity',result['identity']==state['identity']==manifest['identity']==stage['cli_result']['identity'])
reg=manifest['registration'];fitreg=doc(ROOT/'controls_581399eb_v2_metadata_only/FIT_REGISTRATION.json','7984d1ddedb9c769caa2649fd285af7782fe464f0c6a8b6c10f546fe64940d1e');check('original_registration',reg==fitreg and identity(reg)==reg['identity'])
protocol=doc(reg['protocol']['path'],reg['protocol']['sha256']);check('original_300_protocol',identity(protocol)==protocol['identity']==manifest['protocol_identity'] and protocol['recipe']['updates']==300 and protocol['recipe']['scenes_per_update']==2 and protocol['recipe']['patches_per_view']==32)
check('complete_terminal_endpoint',stage['status']==result['status']=='COMPLETE' and stage['stage']=='fit' and state['completed']==result['completed']==manifest['final_update']==300 and state['file'] in ('probe_a.pt','probe_b.pt'))
check('cli_matches_full_result_summary',stage['cli_result']=={k:v for k,v in result.items() if k not in ('metrics','exposure')})
res=stage['resources'];check('actual_cpu_clean_terminal',res['exit_code']==0 and res['limit_reason'] is None and res['first_monitor_error'] is None and res['process_cleanup']['live_owned_remaining']==0 and res['counters']['limit_exceeded'] is None and res['counters']['native_load_calls']==0 and res['counters']['torch_load_calls']==0 and res['counters']['gpu_peak_allocated_bytes']==res['counters']['gpu_peak_reserved_bytes']==0)
failed=doc(stage['previous_stage_result']['path'],stage['previous_stage_result']['sha256']);check('upstream_failure_retained',failed['status']=='STOPPED_NO_AUTO_RETRY' and failed['resources']['limit_reason']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT' and stage['read_accounting']['upstream_supervisor_status_preserved']==failed['status'])
reconpin=stage['read_accounting']['upstream_artifact_reconciliation'];recon=doc(reconpin['path'],reconpin['sha256']);check('reconciliation_is_artifact_acceptance_only',recon['status']=='ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED' and recon['eligible_consumer']=='fit' and recon['original_failed_result']==stage['previous_stage_result'] and recon['launch_approved'] is False)
labels=doc(protocol['labels']['path'],protocol['labels']['sha256']);check('raw_current_label_schema',labels['label_schema']==protocol['label_schema']=='current_raw_lidar_distorted_full_rgb_native384_patch_v1')
source=read(RT/'tools/st_world_policy/spatial_probe.py','e5d9e4306aae8311aefea9c8cc8ee777fd88424b412abcc6003128c6c7e0a20f').decode();check('same_validator_runtime',reg['source_sha']=='581399eb309677c20fe7464791b4f18685e947a6' and reg['device']=='cpu')
functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
check('frozen_assert_present',"frozen_hashes != {s: tensor_hash(data[s][0]) for s in data}" in functions['fit'] and result['frozen_W_unchanged'] is True)
check('global_task_valid_denominators_retained',"/ counts[task]" in functions['task_loss'] and "reduction='sum'" in functions['task_loss'])
check('finite_prediction_and_gradient_checks_present','torch.isfinite(prediction).all()' in functions['task_loss'] and 'error_if_nonfinite=True' in functions['update_probe'])
check('stats_train_only',"train_statistics(data['train'][2])" in functions['fit'])
exposure=result['exposure'];check('exact_300_exposure_records',[x['update'] for x in exposure]==list(range(1,301)))
counts=collections.Counter();valid=collections.Counter();zero=collections.Counter();losses={'depth':[],'small':[]}
for x in exposure:
 check('exposure:'+str(x['update']),len(x['scene_indices'])==2 and all(type(i)is int and 0<=i<64 for i in x['scene_indices']) and x['queried_patches']==192 and len(x['patch_sha256'])==64 and all(c in '0123456789abcdef' for c in x['patch_sha256']))
 counts.update(x['scene_indices']);check('update_valid:'+str(x['update']),set(x['valid'])=={'depth','small'} and all(type(v)is int and 0<=v<=192 for v in x['valid'].values()) and x['optimizer_step']==bool(sum(x['valid'].values())))
 for k in losses:
  v=x['losses'][k];check('finite_loss:'+k+str(x['update']),isinstance(v,(float,int)) and math.isfinite(v) and v>=0);losses[k].append(v);valid[k]+=x['valid'][k];zero[k]+=x['valid'][k]==0
check('600_train_exposures',sum(counts.values())==600 and len(counts)==64)
summary={}
for split,n in [('train',64),('dev',16)]:
 metrics=result['metrics'][split];overall=metrics['overall'];rows=metrics['scenes'];sig=doc(Path(reg['caches'][split]['root'])/'identity.json')
 check('signature_'+split,identity(sig)==sig['identity']==reg['caches'][split]['identity']);check('population_'+split,len(rows)==len(sig['rows'])==len(labels['roles'][split])==n and [(x['token'],x['log']) for x in rows]==[(x['token'],x['log']) for x in sig['rows']])
 labelmap={x['token']:x for x in labels['roles'][split]};check('raw_label_ids_'+split,{x['token'] for x in rows}==set(labelmap))
 for row in rows:
  lab=labelmap[row['token']];check('raw_denominators_'+split+row['token'],row['log']==lab['log'] and row['depth']['valid']==lab['depth_valid_patches'] and row['small']['positive']==lab['small_positive_patches'] and row['small']['negative']==lab['small_negative_patches'])
 def valid_metrics(m):
  check('metric_denominators',m['depth']['valid']+m['depth_unknown']==m['patches'] and m['small']['valid']+m['small_unknown']==m['patches'] and m['small']['positive']+m['small']['negative']==m['small']['valid'] and sum(x['valid'] for x in m['depth_ranges_m'].values())==m['depth']['valid'])
  for task in ('depth','small'):
   for k,v in m[task].items():
    if v is not None:check('finite_report_metric',isinstance(v,(int,float)) and math.isfinite(v))
  if m['small']['positive']==0 or m['small']['negative']==0:check('one_class_NA',all(m['small'][x] is None for x in ('PR_AUC','ROC_AUC','balanced_accuracy')))
 for m in [overall,*rows,*metrics['logs'].values(),*metrics['views'].values()]:valid_metrics(m)
 check('full_grid_count_'+split,overall['patches']==n*3*18*24 and len(metrics['views'])==3 and len(metrics['logs'])==len({x['log'] for x in rows}))
 for grouping in (rows,list(metrics['logs'].values()),list(metrics['views'].values())):
  check('group_counts_'+split,sum(x['depth']['valid'] for x in grouping)==overall['depth']['valid'] and sum(x['small']['valid'] for x in grouping)==overall['small']['valid'] and sum(x['small']['positive'] for x in grouping)==overall['small']['positive'])
  for name in ('MAE_m','AbsRel','RMSE_m'):
   power=2 if name=='RMSE_m' else 1;value=sum((x['depth'][name]**power)*x['depth']['valid'] for x in grouping)/overall['depth']['valid'];value=value**(1/power)
   check('depth_weighted_aggregate_'+split+name,math.isclose(value,overall['depth'][name],rel_tol=1e-12,abs_tol=1e-12))
 summary[split]={'scenes':n,'logs':len(metrics['logs']),'overall':overall,'single_class_logs':sum(m['small']['positive']==0 or m['small']['negative']==0 for m in metrics['logs'].values())}
check('train_stats_match_denominators',result['statistics']==manifest['statistics'] and result['statistics']['depth_valid']==summary['train']['overall']['depth']['valid'] and result['statistics']['positive']==summary['train']['overall']['small']['positive'] and result['statistics']['negative']==summary['train']['overall']['small']['negative'] and math.isclose(result['statistics']['pos_weight'],result['statistics']['negative']/result['statistics']['positive']))
check('not_scientific_or_full_PDMS',stage['scientific_result'] is False and result['scientific_result'] is False and manifest['scientific_result'] is False and result['interpretation']==protocol['interpretation'])
loss_summary={k:{'first':v[0],'last':v[-1],'first25_mean':sum(v[:25])/25,'last25_mean':sum(v[-25:])/25,'min':min(v),'max':max(v),'zero_valid_updates':zero[k],'valid_exposures':valid[k]} for k,v in losses.items()}
assert 'torch' not in sys.modules
review={'status':'PASS_SMALL_ARTIFACT_INTERNAL_CONSISTENCY_ONLY','reviewer':'/root/st_calibration_independent_correctness_v2','source_review_base':'f4764a949aa6ed08d576f4138339b4045a681dcb','inputs':PINS,'checks_count':len(CHECKS),'checks':CHECKS,'completed_updates':300,'optimizer_steps':sum(x['optimizer_step'] for x in exposure),'scene_exposures':600,'unique_train_scenes_exposed':len(counts),'scene_exposure_min':min(counts.values()),'scene_exposure_max':max(counts.values()),'queried_patch_exposures':sum(x['queried_patches'] for x in exposure),'loss_summary':loss_summary,'metrics':summary,'parameter_count':manifest['parameter_count'],'frozen_W_receipt_and_code_assertion_verified':True,'W_or_state_tensors_independently_reloaded':False,'next_rng_or_checkpoint_roundtrip_verified':False,'gradient_values_logged':False,'full_PDMS_or_generalization_claim':False,'upstream_extraction_status':failed['status'],'upstream_reason':failed['resources']['limit_reason'],'new_dispatch_or_admission':False,'cost':{'read_bytes':READ,'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_affinity':sorted(os.sched_getaffinity(0))}}
assert review['cost']['wall_seconds']<120 and review['cost']['peak_RSS_bytes']<512<<20
(OUT/'REVIEW.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in review.items() if k not in ('inputs','checks')},indent=2))
