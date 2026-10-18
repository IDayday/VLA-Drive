"""Materialize this independent review for the two fixed, checked registrations."""
from pathlib import Path
import copy,hashlib,json,sys,time,shutil
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/project/st-world-policy-20261016')
RUNTIME=Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
LEAD=Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
sys.path.insert(0,str(RUNTIME))
from tools.st_world_policy.check_formal_approval import verify_registration,file_hash

def ref(p):return {'path':str(p),'sha256':file_hash(p)}
def read(p):return json.loads(Path(p).read_text())
def recipe(d):
 d=copy.deepcopy(d);d.pop('continuation',None);d['execution'].pop('stop_after');d['runtime'].pop('output_root');return d
launcher=LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/launch20k_once_v2.py'
assert file_hash(launcher)=='8bf6b61023bc00d72e11861155b783bc771791102ca89836c153316eddedde77'
prior=read(HERE/'INHERITED_EVIDENCE.json');storage=read(HERE/'STORAGE_CHECK.json');hosts=read(HERE/'HOST_OBSERVATIONS.json')
assert storage['status']=='PASS' and storage['payload_bytes_read']==0
expected={'SP2':'f344cafa2352f1824a7a0d865aa8aaf2f5d680e2e328eefd4826834ffa0953f6','SP3':'ab7a0ae39d5f7d6662d6b9f2b8f01b6738f4b53d6ae20048f56e3ffff5a847b8'}
for cell in expected:
 rp=ROOT/f'formal/spatial/registrations/{cell}_fulltrain_seed42_official_20k_v1_resource_v2/REGISTRATION.json'
 assert file_hash(rp)==expected[cell]
 fp=verify_registration(rp,repo=RUNTIME); reg=read(rp);cfg=read(reg['artifacts']['config']['path']);lease=read(reg['artifacts']['resource_lease']['path'])
 assert '/root/spatial20k_efficiency' not in reg['implementer_agents']
 assert cfg['execution']==dict(scope='FORMAL',global_batch=32,micro_batch=4,world_size=4,max_updates=20000,stop_after=20000,sample_limit=0,precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off',family='spatial',seed=42)
 assert reg['source_sha']=='fdf25f99b702754787904215bf1c2a2b7515ea75'
 parent=read(cfg['continuation']['parent_registration']['path']);pcfg=read(parent['artifacts']['config']['path']);assert recipe(cfg)==recipe(pcfg)
 assert cfg['runtime']['max_seconds']==28800 and cfg['runtime']['save_every']==1000
 for k in ['base_binding','teacher_manifest','data_manifest','target_contract','environment']:assert parent['artifacts'][k]==reg['artifacts'][k]
 cp=read(lease['checkpoint_commitments']['path']);assert file_hash(lease['checkpoint_commitments']['path'])==lease['checkpoint_commitments']['sha256']
 assert cp['remaining_bytes']==storage['remaining_bytes'] and cp['total_retained_allocation_bytes']==4143<<30
 assert lease['remaining_cache_growth_bytes']==768<<30 and lease['separate_conservative_metadata_buffer_bytes']==10<<30
 assert lease['max_run_output_bytes']==18*40*(1<<30) and lease['maximum_optimizer_updates_this_stage']==15000
 assert lease['first_continuation_only'] and lease['no_automatic_resume']
 assert lease['valid_until_unix']>time.time()+29580
 host=lease['hostnames'][0];obs=next(x for x in hosts['observations'] if x['host']==host)
 assert 'A800-SXM4-80GB' in obs['gpu'] and 'NV8' in obs['gpu_interconnect'] and obs['affinity_count']==128
 rows=[s.split(',') for s in obs['gpu'].strip().splitlines()];assert [r[1].strip() for r in rows[:4]]==lease['gpu_uuids_by_host'][host]
 env=read(LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/ENVIRONMENTS.json')
 assert next(x['environment'] for x in env['observations'] if x['host']==host.removesuffix('-worker-0'))==read(reg['artifacts']['environment']['path'])
 required=cp['remaining_bytes']+(1024+768+10)*(1<<30);free=shutil.disk_usage(ROOT).free;assert free>=required
 review={'schema':'st_world_review_v1','role':'efficiency','status':'PASS','reviewer_agent':'/root/spatial20k_efficiency','fingerprint':fp,'created_unix':time.time(),'decision':'PASS for one first native 5k-to20k continuation invocation of this exact registration on the leased four GPUs. No automatic second invocation, GPU reservation, model efficacy or scientific endpoint completion is asserted.','source_sha':reg['source_sha'],'independence':'Reviewer did not author trainer/backend/budget-extension/launcher/cache-budget or resource receipts and is absent from implementer_agents.','evidence_artifacts':[ref(HERE/'INHERITED_EVIDENCE.json'),ref(HERE/'HOST_OBSERVATIONS.json'),ref(HERE/'STORAGE_CHECK.json'),ref(HERE/'check_storage.py'),ref(HERE/'REVIEW_SCOPE.md'),ref(launcher),reg['artifacts']['resource_lease'],lease['checkpoint_commitments'],cp['credited_checkpoint_snapshot'],ref(LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/ENVIRONMENTS.json'),ref(Path(parent['artifacts']['config']['path']).parent/'reviews/EFFICIENCY.json')],'checks':{'actual_registration_verified':True,'same_full_recipe_and_runtime':True,'same_original_schedule':True,'same_parent_data_teacher_and_precision':True,'same_original_optimizer_native_state_path':True,'fresh_native_storage_guard_with_prospective_train':'PASS','independent_complete_checkpoints':20,'independent_unique_payload_stats':180,'retained_gross_bytes':4143<<30,'credited_already_occupied_bytes':747345660520,'remaining_checkpoint_bytes':cp['remaining_bytes'],'free_bytes_at_review':free,'conservative_required_free_bytes':required,'remaining_headroom_bytes':free-required,'native_cache_guard_code_changed':False,'cache_occupancy_credit_bytes':0,'new_model_or_profile_run':False,'payload_read_bytes':0},'requirements':['Lead must combine this role with distinct exact-fingerprint Evidence and Correctness reports using the native gate.','Use pinned launch20k_once_v2.py only. Recheck lease, source, exact GPU UUIDs and idleness under retained nonblocking locks immediately before dispatch. Protect unknown/real jobs.','Enforce 28,800s native + outer timeout/cleanup; save at optimizer boundary. More invocations need updated real lease/resource review and must restore the new run, not repeat the parent.','Retain all old checkpoints and failures. Fifteen periodic saves plus two nonperiodic/failed and one in-flight 40GiB slots remain 720GiB per new run; logs must fit that inclusive bound.','Record actual restored step5000/native master/RNG/cursor and subsequent progress; partial bounded completion is not 20k completion.'],'limitations':['Inherited profile is actual matched 240-update/two-window evidence, not a new exact-node timing or full cold101592-scene benchmark.','No cross-process FlashAttention2 bitwise trajectory equivalence claimed; original numerical errata remain.','Project quota and hidden ancestor resources remain unobservable; fresh filesystem capacity and safeguards apply.','Payload hashes are inherited from native receipts; reviewer stat census does not replace native full load/checksum verification.','At inherited3.2–3.4s/update, 15k further updates likely exceed8h; this approval preserves the finite pause rather than claiming completion.']}
 out=HERE/f'{cell}_EFFICIENCY.json';assert not out.exists();out.write_text(json.dumps(review,indent=2)+'\n');print(cell,file_hash(out),fp['registration_sha256'])
