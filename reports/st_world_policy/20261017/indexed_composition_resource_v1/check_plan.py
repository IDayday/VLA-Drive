"""Read exact bounded controls/source/stat only; no runtime imports or payload reads."""
from pathlib import Path
import hashlib,json,math,os,resource,shutil,subprocess,time
resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));os.sched_setaffinity(0,{0,1})
ROOT=Path('/mnt/project/st-world-policy-20261016');RT=Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017');HERE=Path(__file__).parent;pins=[];checks={};consumed=0
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p,expected=None,decode=True):
 global consumed
 p=Path(p);assert p.resolve()==p and p.stat().st_size<=2<<20
 b=p.read_bytes();consumed+=len(b);assert consumed<=16<<20
 h=sha(b);assert expected is None or h==expected,(p,h,expected)
 pins.append(dict(path=str(p),bytes=len(b),sha256=h));return json.loads(b) if decode else b
p=read(ROOT/'audits/indexed_provider_composition_plan_v1/PLAN.json','f96e5612c068aa4000172f7076ea37031b8fe09d03b2f60706d57300e540f0eb')
checks['plan_identity']=p['identity']=='df38f4997f888c868a7de3991768be065646907d3d30480a720147131683688a'==sha(canon({k:v for k,v in p.items() if k!='identity'}))
checks['immutable_runtime']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RT,text=True).strip()=='31c3dca2fbc697b419bf343ce05d10be0a4fe13f' and not subprocess.check_output(['git','status','--porcelain'],cwd=RT)
for ref in p['source_files']:read(ref['path'],ref['sha256'],False)
controls={}
for name in ['prior_result','prior_plan','teacher_config']:
 ref=p[name];controls[name]=read(ref['path'],ref['sha256'])
for name in ['request_index','source_index']:
 ref=p[name]['complete'];v=read(ref['path'],ref['sha256']);controls[name]=v
 checks[name+'_identity']=v['identity']==p[name]['index_identity'] and v['request_identity']==p[name]['request_identity'] and v['status']=='COMPLETE'
 checks[name+'_database_stat']=Path(ref['path']).with_name('metadata.sqlite').stat().st_size==v['database_bytes']
checks['prior_qualified']=controls['prior_result']['errors']==[] and controls['prior_result']['tensor_disk_bytes']==0 and controls['prior_result']['construction_rng_unchanged'] is True
checks['same_execution']=p['execution']==controls['prior_result']['execution'] and p['execution_identity']==sha(canon(p['execution']))
checks['population_fixed']=p['scenes']==32 and len(p['scene_ids'])==len(set(p['scene_ids']))==32 and p['unique_images']==len(p['keys'])==294 and len(p['hit_keys'])==189 and len(p['miss_keys'])==105 and not set(p['hit_keys'])&set(p['miss_keys']) and set(p['keys'])==set(p['hit_keys'])|set(p['miss_keys'])
checks['fixed_fp32_no_training']=p['contract']['compute_dtype']==p['contract']['storage_dtype']=='float32' and p['scope']=='SMOKE' and p['formal_approved'] is False and p['batch_policy']['compute_batch_images']==8
checks['compute_budget']=math.ceil(294/8)+math.ceil(294/4)+4*sum(math.ceil(len(set(keys))/8) for keys in p['scene_keys'].values())==p['budget']['max_batch_calls']==367 and p['budget']['max_image_forwards']==2936
for s in p['source_shards']:
 r=read(s['receipt']['path'],s['receipt']['sha256']);assert r['identity']==p['current_identity'] and r['sha256']==s['tensor_sha256'];assert Path(s['tensor_path']).stat().st_size==s['tensor_bytes']
checks['source_shards_finite']=len(p['source_shards'])==188
checks['resource_bounds']=all(p['budget'][k]==v for k,v in {'cpu_threads':2,'visible_gpus':1,'ram_peak_bytes':8<<30,'gpu_peak_bytes':16<<30,'max_logical_input_bytes':64<<30,'metadata_bytes':32<<20,'max_seconds':900,'tensor_disk_bytes':0}.items())
checks['planned_read_under_limit']=p['budget']['planned_input_upper_bytes']<64<<30
checks['reference_ram_under_limit']=p['budget']['online_reference_bytes']+p['budget']['composed_reference_extra_bytes']<p['budget']['reference_resident_limit_bytes']==2<<30
ledger=read(ROOT/'resources/cache_storage_ledger_v1/CLAIMS.json');ledger_sha=pins[-1]['sha256'];policy=read(ROOT/'resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json','b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831')
checks['unchanged_v7_core']=all(policy.get(k)==v for k,v in ledger['core'].items())
checks['cache_limits_preserved']=policy['cache_cap_bytes']==768<<30 and policy['free_reserve_bytes']==1024<<30 and policy['metadata_package']['maximum_bytes']==4<<30
metadata_existing=1136<<20;report_reserve=64<<20
checks['report_allocation_within_four_GiB']=metadata_existing+report_reserve==1200<<20 and metadata_existing+report_reserve<=4<<30
checks['report_roots_not_claimed']=all(str(ROOT/'audits'/n) not in ledger['claims'] for n in ['indexed_provider_composition_plan_v1','indexed_provider_composition_run_v1'])
checks['fresh_run_output']=not (ROOT/'audits/indexed_provider_composition_run_v1').exists()
checks['checkpoint_headroom']=shutil.disk_usage(ROOT).free>((768+1024+1983)<<30)+report_reserve
# Conservative serialization fixture: actual key/layout cardinalities, longest
# finite float spelling, complete result structure and an extra 64 KiB allowance.
f=1.7976931348623157e308
metric=dict(elements=442368,bitwise_equal=False,**{k:f for k in ['max_abs','sum_abs','sum_squared','reference_squared','rms','l2_relative','max_elementwise_relative']})
skeleton=dict(execution=p['execution'],reference_digests={k:dict(features='f'*64,valid='f'*64) for k in p['keys']},reference_reordered4=[dict(key=k,difference=metric) for k in p['keys']],legacy_cached_reference=[dict(key=k,difference=metric) for k in p['hit_keys']],cases={case:[dict(scene=token,comparisons=[dict(key=k,difference=metric,online_difference=metric) for k in p['scene_keys'][token]],all_masks_equal=True,sampler_ids=['f'*64]*2,counts={k:999999 for k in ['ram_hits','immutable_hits','computed_images','evictions']}) for token in p['scene_ids']] for case in p['cases']})
serialized_bound=len((json.dumps(skeleton,sort_keys=True,indent=2)+'\n').encode())+65536
checks['fixed_result_fits_4MiB']=serialized_bound<4<<20
policy_dispatch=read('/mnt/project/server_dispatch_policy.json');checks['local_allowed']='training-vla-zt' in policy_dispatch['allowed_hosts']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
g0=next(line for line in gpu.splitlines() if line.startswith('0,'));checks['gpu0_snapshot_free']=g0.split(',')[2].strip()=='0' and g0.split(',')[1].strip() not in apps
assert all(checks.values()),{k:v for k,v in checks.items() if not v}
result=dict(status='PASS_RESOURCE_PLAN_WITH_EXPLICIT_REPORT_BUDGET',checked_unix=time.time(),plan_identity=p['identity'],plan_sha256=sha((ROOT/'audits/indexed_provider_composition_plan_v1/PLAN.json').read_bytes()),runtime=str(RT),source_commit='31c3dca2fbc697b419bf343ce05d10be0a4fe13f',checks=checks,pins=pins,review_bytes_read=consumed,ledger_sha256=ledger_sha,metadata_existing_reserved_bytes=metadata_existing,qualification_report_reserve_bytes=report_reserve,effective_metadata_bytes=metadata_existing+report_reserve,report_reserve_is_cache_ledger_claim=False,result_json_structural_upper_bytes=serialized_bound,filesystem_free_bytes=shutil.disk_usage(ROOT).free,gpus=gpu,compute_apps=apps,new_teacher_tensor_bytes=0,actual_qualification_executed=False,formal_training_approved=False)
(HERE/'RESOURCE_REVIEW.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','checked_unix','review_bytes_read','ledger_sha256','effective_metadata_bytes','result_json_structural_upper_bytes','filesystem_free_bytes']}))
