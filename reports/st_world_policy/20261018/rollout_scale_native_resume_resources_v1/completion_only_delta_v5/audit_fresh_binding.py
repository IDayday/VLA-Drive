"""Small JSON/source/hash/stat only; never imports model-side inventory helpers."""
import hashlib, importlib.util, json, time
from pathlib import Path

OUT = Path(__file__).parent
SOURCE = Path('/mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018/resources/st_rollout_resume_completion_v5.py')
BINDING = Path('/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_completion_v5_admission/BINDING_DRAFT.json')
FILE_SHA = '49683f242b213ee16efd8c6d49251549f96e4cfe509bdfec4d5d6f300bc8c9dd'
BINDING_SHA = '8b0d3b8f997d514e76b322ff9a5afb06fd60f55e0775fad30a13af4f7013a313'
assert hashlib.sha256(BINDING.read_bytes()).hexdigest() == FILE_SHA
spec=importlib.util.spec_from_file_location('final_completion_resource_audit', SOURCE)
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c); v=c.legacy()
row=json.loads(BINDING.read_text()); b=row['binding']; r=b['references']
assert row['metadata_only'] and row['status']=='WAITING_TWO_INDEPENDENT_REVIEWS'
assert row['binding_sha256']==BINDING_SHA==v.digest(b)
assert b['coordinator_sha256']==v.sha(SOURCE)=='85713b7a2b1ad855e2c24b2dbd8beda71c215859a3dc99a77450878264ede4cb'
assert b['observer_sha256']==v.sha(c.OLD_FILE)==c.OLD_SHA=='468fe8f1ed20ef3abb2b2407413f92fb078f6b1d05f6cbfb2ed4e4b6e69f9692'
assert b['source_sha']==v.SOURCE=='751fcfa2b41549ae16f7b3204aef8dd18b912016'
assert b['root']==str(c.ROOT) and not c.ROOT.exists()
reads=0; pins=[]
def check_pins(value):
    global reads
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p=Path(value['path']); assert p.stat().st_size<2*1024**2
            raw=p.read_bytes(); reads+=len(raw)
            assert hashlib.sha256(raw).hexdigest()==value['sha256']
            pins.append(value)
        else:
            for x in value.values(): check_pins(x)
    elif isinstance(value,list):
        for x in value: check_pins(x)
check_pins(b)
assert reads<8*1024**2
route=b['route']; lease=v.pinned(b['lease']); layout=v.pinned(b['resource_layout']); group=layout['groups']['B_pause2_resume2']
assert route['gpus']==group['GPU_indices']==[4,5,6,7]
assert route['gpu_uuids']==group['GPU_UUIDs']==[
 'GPU-f1784442-c8f0-767b-a0cc-e3675643e318','GPU-2fb29b99-d8a1-0bae-9b16-2a0c97f0351a',
 'GPU-418ae714-b62c-9b41-070b-065281c57cbd','GPU-b41079a0-8114-e9b0-55d6-4ea93daea76f']
assert lease['world_size']==4 and lease['hostnames']==[route['hostname']]
assert lease['gpu_uuids_by_host']=={route['hostname']:route['gpu_uuids']}
assert lease['cuda_visible_devices_by_host']=={route['hostname']:','.join(route['gpu_uuids'])}
now=time.time(); assert route['valid_from_unix']<=now<route['valid_until_unix']
assert lease['valid_from_unix']<=now<lease['valid_until_unix']
assert lease['valid_until_unix']-lease['valid_from_unix']==7201
assert route['host']=='training-vlawm-tsinghua-zt6'
policy=v.pinned(b['dispatch_policy']); auth=v.pinned(b['authorization'])
assert route['hostname'] in policy['allowed_aliases'] and route['host'] in auth['allowed_hosts']
for name,x in policy['global_dispatch_prohibitions'].items():
    assert route['host']!=name and route['hostname']!=name and route['hostname'] not in x.get('aliases',[])
assert [r[k] for k in ('prior_invocations','remaining_invocations','campaign_invocation_cap',
    'completed_updates','remaining_updates','consumed_calls','remaining_calls',
    'prior_reserved_seconds','remaining_training_seconds')]==[4,2,6,4,4,512,512,660,2940]
assert r['prior_stats']['sampled_rchar']==294432654281 and r['prior_stats']['sampled_read_bytes']==87309496320
assert len(r['history'])==4 and len(r['A_native_receipts'])==3 and len(r['A_observer_completions'])==4
assert 660+2940+120+180==3900
old_start=v.pinned(r['old_expired_START']); old_b=v.pinned(r['old_binding'])
assert now>old_start['deadline_unix'] and now>old_b['route']['valid_until_unix']
assert b['calendar_gap_seconds_at_admission']>7200
prior=json.loads((OUT/'ACTUAL_A_B_PROGRESS_AUDIT.json').read_text())
for ref in prior['small_control_refs']+prior['A_four_rank_observer_complete']:
    assert v.sha(ref['path'])==ref['sha256']
assert sorted(x.name for x in Path(prior['B_run']).iterdir())==['RUN.lock','identity.json','resolved_config.json']
for native in prior['A_native_stat_only']:
    assert native['receipt'] in [dict(x,bytes=Path(x['path']).stat().st_size) for x in r['A_native_receipts']]
    for file in native['files']:
        assert (Path(native['folder'])/file['name']).stat().st_size==file['bytes']
actual_A=v.pinned(r['A_result'])
assert actual_A['completed']==actual_A['new_optimizer_updates']==4
assert actual_A['counts']['actual_rollouts']==512 and actual_A['frozen_encoder_unchanged']
for rank,ref in enumerate(r['A_observer_completions']):
    obs=v.pinned(ref); assert obs['rank']==rank and obs['completed']==4 and obs['observer_sha256']==c.OLD_SHA
assert v.pinned(r['last_result'])['stats']==r['prior_stats']
out={'status':'PASS_METADATA_ONLY_NOT_DISPATCHED','binding_file':v.pin(BINDING),'binding_sha256':BINDING_SHA,
 'source_commit':'8787b62bd4f897b991ef92a8882b146b9ff42082','coordinator_sha256':b['coordinator_sha256'],
 'original_observer_sha256':b['observer_sha256'],'runtime_source_sha':b['source_sha'],
 'new_root_absent':True,'fresh_lease':b['lease'],'lease_valid_from_unix':lease['valid_from_unix'],
 'lease_valid_until_unix':lease['valid_until_unix'],'reviewed_unix':now,
 'lease_seconds_remaining':lease['valid_until_unix']-now,'route_host':route['host'],
 'route_hostname':route['hostname'],'GPU_indices':route['gpus'],'GPU_UUIDs':route['gpu_uuids'],
 'CPU_affinity':'16-31 (source-bound taskset)','old_A_hash_refs_unchanged':True,
 'old_B_three_metadata_files_preserved':True,'old_time_bounds_expired':True,
 'calendar_gap_seconds_at_admission':b['calendar_gap_seconds_at_admission'],
 'prior_reserved_seconds':660,'remaining_controller_seconds':2940,'comparison_seconds':120,
 'cleanup_seconds':180,'total_active_envelope_seconds':3900,'prior_invocations':4,
 'remaining_invocations':2,'total_invocation_cap':6,'prior_updates':4,'remaining_updates':4,
 'total_update_cap':8,'prior_calls':512,'remaining_calls':512,'total_calls':1024,'retry_calls':0,
 'cumulative_sampled_rchar_inherited':294432654281,'cumulative_sampled_read_bytes_inherited':87309496320,
 'IO_counting':'Old v4 cumulative statistics inherited once; observed lower bounds, not exact physical IO.',
 'global_sampled_rchar_guard_bytes':v.LIMITS['monitor_read_bytes'],
 'global_output_guard_bytes':v.LIMITS['output_bytes'],'group_RSS_guard_bytes':v.LIMITS['group_RSS_bytes'],
 'GPU_driver_guard_bytes_per_card':v.LIMITS['GPU_driver_bytes'],
 'runtime_shared_free_guard_bytes':v.LIMITS['runtime_shared_free_bytes'],'native_directory_cap':8,
 'small_pins_verified_count':len(pins),'pinned_small_metadata_bytes_read':reads,
 'actual_GPU_model_or_official_calls_by_reviewer':0,'large_payload_reads':0,
 'native_payload_checksums_read_by_reviewer':False,
 'correctness_status':'Not asserted; independent Correctness required.'}
(OUT/'BINDING_AUDIT.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out,sort_keys=True))
