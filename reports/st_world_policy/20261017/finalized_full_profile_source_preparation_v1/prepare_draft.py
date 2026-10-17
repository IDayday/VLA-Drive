"""Source/control-only preparation. No model imports, index opens or dispatch.

Run this once inside the owned report directory's sparse implementation tree.
The result is a draft, not a provider descriptor, run registration or approval.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

OUT=Path(__file__).resolve().parent
OLD=Path('/mnt/project/VLA-Drive-st-full-index-profile-20261017')
DONOR=Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017')
EXACT=Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
OLD_SHA='bccebb00319512049e93df58d18cb5ead49575f0'
DONOR_SHA='3df052ea422c4ba042ef3be33dea19438c5702dc'
FINAL=Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_finalized_v1/FINALIZATION_COMPLETE.json')
FINAL_SHA='16c47d6603144dcabe0c764f85c23c231bb0edfaf347d869f93c34d79bdc5438'
EXACT_CONFIG=Path('/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/CONFIG.json')
CURRENT_COMPLETE=Path('/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1/COMPLETE.json')
state={'source_bytes':0,'receipt_bytes':0}; started=time.monotonic(); blobs={}

def read(path,kind):
    path=Path(path)
    if path in blobs:return blobs[path]
    ceiling=16<<20 if kind=='source' else 4<<20
    before=path.stat()
    if state[kind+'_bytes']+before.st_size>ceiling:raise RuntimeError('Bounded '+kind+' read exceeded')
    with path.open('rb') as stream: data=stream.read(before.st_size+1)
    after=path.stat()
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stamp(before)!=stamp(after) or len(data)!=before.st_size:raise RuntimeError('Changed bounded input')
    if time.monotonic()-started>180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1<<30:
        raise RuntimeError('Wall/RSS budget exceeded')
    state[kind+'_bytes']+=len(data); blobs[path]=data
    return data

def sha(data):return hashlib.sha256(data).hexdigest()
def pin(path,kind='receipt'):return {'path':str(path),'sha256':sha(read(path,kind))}
def small(path):return json.loads(read(path,'receipt'))
def write(name,value):
    path=OUT/name
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()

assert git(OLD,'rev-parse','HEAD')==OLD_SHA and not git(OLD,'status','--porcelain=v1')
assert git(DONOR,'rev-parse','HEAD')==DONOR_SHA and not git(DONOR,'status','--porcelain=v1')
qualifier=ast.parse(read(DONOR/'tools/st_world_policy/qualify_indexed_provider.py','source'))
math_paths=next(ast.literal_eval(node.value) for node in qualifier.body if isinstance(node,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='MATH_PATHS' for t in node.targets))
rows=[]
for rel in math_paths:
    old,donor,exact=(sha(read(root/rel,'source')) for root in (OLD,DONOR,EXACT))
    rows.append(dict(path=rel,old_sha256=old,donor_sha256=donor,exact32_sha256=exact,
                     old_matches_exact32=old==exact,donor_matches_exact32=donor==exact))
assert all(r['old_matches_exact32'] for r in rows)
assert [r['path'] for r in rows if not r['donor_matches_exact32']]==['starVLA/dataloader/st_world_history.py']
helpers=['tools/st_world_policy/prepare_future_metadata.py','tools/st_world_policy/prepare_current_teacher.py',
 'starVLA/dataloader/st_world_metadata_index.py','tools/st_world_policy/train_world.py',
 'tools/st_world_policy/training_config.py','tools/st_world_policy/qualify_indexed_provider.py',
 'starVLA/dataloader/st_world_dataset.py','tools/st_world_policy/official_runtime.py',
 'tools/ddpolicy_vehicle/training_state.py','tools/foresight/student_state.py']
unchanged=[]
for rel in helpers:
    old,donor=(sha(read(root/rel,'source')) for root in (OLD,DONOR))
    assert old==donor
    unchanged.append(dict(path=rel,sha256=old))
changes=['tools/st_world_policy/teacher_provider.py','tools/st_world_policy/finalize_future_metadata.py']
new_files=[pin(DONOR/p,'source') for p in changes]
# Copy the exact reviewed files in patch form only; do not create a runtime.
patch=subprocess.check_output(['git','diff','--binary',OLD_SHA,DONOR_SHA,'--',*changes],cwd=DONOR)
(OUT/'MINIMAL_RUNTIME.patch').write_bytes(patch)
final=small(FINAL);assert sha(read(FINAL,'receipt'))==FINAL_SHA
assert final['schema']=='st_future_metadata_finalization_complete_v1' and final['status']=='COMPLETE'
assert final['scope']=='full_population' and final['scenes']==101592 and final['logs']==1176
assert final['tensor_disk_bytes']==0 and final['training_admission'] is False
config=small(EXACT_CONFIG);draft=deepcopy(config)
draft['execution'].update(scope='PROFILE',sample_limit=final['scenes'],stop_after=64,world_size=2,micro_batch=1,global_batch=2)
assert draft['execution']['max_updates']==20000 and draft['optimizer']['warmup']==1000
assert draft['runtime']['deterministic'] is True and draft['runtime']['gradient_checkpointing'] is False
assert draft['execution']['family']=='temporal' and draft['data']['history_root'] is None
assert draft['runtime']['official_execution']['preset']=='drivedreamer_f35fe783'
source=small(CURRENT_COMPLETE)
draft['data'].pop('teacher_provider',None);draft['data']['teacher_root']=source['cache_root']
draft['runtime'].update(loader_workers=0,profile_mode='none',output_root='__UNASSIGNED_REGISTERED_PROFILE_OUTPUT__',
                        max_seconds=None,save_every=None)
# Deliberately non-runnable until resource admission assigns wall/save/output;
# do not invent a resource lease from the small code-preparation budget.
policy=ast.parse(read(DONOR/'tools/st_world_policy/official_runtime.py','source'))
reference=next(ast.literal_eval(node.value) for node in policy.body if isinstance(node,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='REFERENCE_RECIPE' for t in node.targets))
effective={**{k:draft['execution'][k] for k in ('world_size','micro_batch','global_batch','max_updates')},
           **draft['optimizer'],'gradient_checkpointing':False}
draft['runtime']['official_execution'].update(adaptation_reason='Bounded full-index PROFILE only: world2/micro1/global2, stop64 total including resume; not formal global32 throughput.',
 declared_recipe_overrides=sorted(k for k,v in reference.items() if effective[k]!=v))
write('DRAFT_TRAINING_CONFIG.json',draft)
consumption={'schema':'st_full_index_profile_consumption_v1','initial_completed_updates':0,'initial_epoch':0,'initial_offset':0,
 'cumulative_optimizer_updates':64,'cumulative_scene_exposures':128,'world_size':2,'micro_batch':1,'global_batch':2,
 'gradient_accumulation_steps':1,'per_rank_scene_exposures':[64,64],
 'resume_budget':'same_run_cumulative_not_per_invocation','formal_global32_throughput_qualified':False}
write('DRAFT_CONSUMPTION.json',consumption)
write('SOURCE_DELTA.json',{'schema':'st_finalized_profile_source_preparation_v1',
 'implementation_base':DONOR_SHA,'proposed_runtime_base':OLD_SHA,'archived_exact32_runtime':str(EXACT),
 'archived_exact32_commit':git(EXACT,'rev-parse','HEAD'),'original_math_inventory':rows,'unchanged_runtime_helpers':unchanged,
 'exact_copied_files':new_files,'minimal_patch':pin(OUT/'MINIMAL_RUNTIME.patch','source'),
 'do_not_copy':['starVLA/dataloader/st_world_history.py','tools/st_world_policy/prepare_history.py','tools/st_world_policy/metadata_storage.py'],
 'direct_donor_runtime_compatible':False,'reason':'Protected history file changed; preserve original15 math SHAs.',
 'composition_review_required':True,'new_runtime_created':False})
write('INPUT_BINDINGS.json',{'status':'DRAFT_WAITING_ACTUAL_FULL_TRAIN_REQUEST_INDEX_AND_RESOURCE_ADMISSION',
 'finalization_receipt':pin(FINAL),'finalization_identity':final['identity'],'request_identity':final['request_identity'],
 'request_path':str(FINAL.parent/'TEACHER_REQUESTS.json'),'request_sha256_from_receipt':final['files']['TEACHER_REQUESTS.json'],
 'request_body_rehashed':False,'actual_full_future_index':None,'provider_descriptor':None,'pca_application_scope':None,
 'original_exact32_config':pin(EXACT_CONFIG),'full_current_index_receipt':pin(CURRENT_COMPLETE),
 'future_complete_flag_must_point_to':str(FINAL),'original_selection':final['original_selection'],
 'scientific_selection_changed':False,'new_teacher_tensor_disk_bytes':0,'formal_approved':False,
 'source_migration':None,'new_run_registration':None,'dispatch_performed':False,
 'unassigned_config_fields':['runtime.output_root','runtime.max_seconds','runtime.save_every']})
write('READ_COST.json',dict(state,elapsed_seconds=time.monotonic()-started,
 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cpu_affinity=sorted(os.sched_getaffinity(0)),
 source_limit_bytes=16<<20,receipt_limit_bytes=4<<20,wall_limit_seconds=180,rss_limit_bytes=1<<30,
 no_imported_torch=True,no_index_database_opened=True,no_model_image_teacher_payload_opened=True,
 earlier_manual_source_inspection_not_metered=True,
 note='Metered capture only; earlier bounded source diff/header inspections were under1MiB. Git sparse implementation checkout was administrative; no runtime created.'))
print(json.dumps(state))
