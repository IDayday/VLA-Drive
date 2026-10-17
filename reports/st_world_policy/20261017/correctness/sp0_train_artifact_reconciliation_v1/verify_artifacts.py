"""Bounded read-only reconciliation of exact committed SP0 train W artifacts."""
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time

RUNTIME=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017')
BASE=Path('/mnt/project/st-world-policy-20261016')
STAGE=BASE/'audits/spatial_probe_actual_SP0_5000_v1/stages/train_extract'
CACHE=BASE/'caches/learned_w_probe_SP0_5000_v1/train'
SOURCE='581399eb309677c20fe7464791b4f18685e947a6'
assert Path.cwd()==RUNTIME
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and len(os.sched_getaffinity(0))<=2
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','status','--porcelain=v1'],text=True).strip()
sys.path.insert(0,str(RUNTIME))
boot_seconds=time.monotonic()
def io():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
boot_io=io()
from tools.st_world_policy import spatial_probe as probe
startup_io=io(); startup_seconds=time.monotonic()-boot_seconds
started=time.monotonic()
pins={}
def get(path,expected=None):
    path=Path(path); raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    assert len(raw)<=1<<20
    if expected is not None: assert digest==expected
    pins[str(path)]={'sha256':digest,'bytes':len(raw)}
    return json.loads(raw)

def asset_guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).absolute()
    if not str(p).startswith('/mnt/project/'):return
    if args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):raise RuntimeError('No project artifact writes allowed')
    if p.suffix=='.safetensors':
        if p.parent!=CACHE or p.stat().st_size>2<<20:raise RuntimeError('Forbidden payload')
    elif p.suffix not in ('.py','.pyc','.json') or p.stat().st_size>1<<20:
        raise RuntimeError('Forbidden checkpoint/image/raw/large metadata read: '+str(p))
sys.addaudithook(asset_guard)

stage=get(STAGE/'RESULT.json','4d24e4ed9a97744fe0d977c21284cb5fd64ec14c1348d71f6edadeff61245b27')
complete=get(CACHE/'COMPLETE.json','17a8e9dd09447a2f2166a1dc8b216ada72e87b3525cfb3228d940dbd64a3d350')
cli=get(STAGE/'CLI_RESULT.json'); counters=get(STAGE/'COUNTERS.json');child=get(STAGE/'CHILD.json')
signature=get(CACHE/'identity.json'); plan_signature=get(STAGE.parent/'train_plan/SIGNATURE.json')
assert signature==plan_signature
probe.identity(signature);probe.validate_binding(signature['binding'])
prep=get(stage['preparation']['path'],stage['preparation']['sha256'])
binding=signature['binding']; manifest=get(binding['observations']['manifest']['path'],binding['observations']['manifest']['sha256'])
rows=probe.checked_rows(manifest,'train',64)
assert signature['rows']==rows and len(rows)==64
assert signature['schema']=='st_frozen_spatial_w_v1' and signature['targets_read'] is False
assert binding['arm']=='S-P0' and binding['source_sha']==SOURCE
assert signature['precision']=='FP32_native_master_TF32_off'
assert signature['checkpoint']['completed']==5000 and signature['checkpoint']['scope']=='FORMAL'
assert binding['checkpoint_identity']==signature['checkpoint']['sha256']
assert binding['hidden_dim']==2048 and binding['memory_tokens']==144
assert prep['runtime']=={'path':str(RUNTIME),'source_sha':SOURCE}
assert prep['caches']['train']=={'root':str(CACHE),'identity':signature['identity']}
assert complete==cli==stage['cli_result']
assert complete=={'completed':64,'requested':64,'identity':signature['identity'],'scientific_result':False,'status':'COMPLETE'}
assert counters==stage['resources']['counters']
res=stage['resources']
assert stage['status']=='STOPPED_NO_AUTO_RETRY' and stage['stage']=='train_extract'
assert res['exit_code']==0 and res['limit_reason']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
assert res['first_monitor_error'] is None and counters['limit_exceeded'] is None
assert res['process_cleanup']['live_owned_remaining']==0 and res['process_cleanup']['verified_processes']==6
assert res['wall_seconds']<960 and res['sampled_tree_rss_peak_bytes']<64<<30
assert counters['hash_calls_completed']==counters['hash_calls_started']
charged=max(counters['hash_logical_bytes_admitted']+counters['torch_load_nominal_bytes'],counters['proc_self_io_delta']['rchar'])
prior=stage['read_accounting']['prior_read_bytes']; cap=stage['read_accounting']['role_limit_bytes']
assert prior==37575105055 and cap==256<<30 and charged<stage['read_accounting']['remaining_read_bytes']

# Stat the complete exact root before payload access. Count nominal hash + mmap
# reads conservatively; one original validator call per record, no checkpoint.
expected_files={'identity.json','COMPLETE.json','RUN.lock'}|{r['token']+ext for r in rows for ext in ('.json','.safetensors')}
assert {p.name for p in CACHE.iterdir()}==expected_files
for p in [CACHE,*CACHE.iterdir()]:
    s=p.lstat();assert not stat.S_ISLNK(s.st_mode)
    assert stat.S_ISDIR(s.st_mode) or (stat.S_ISREG(s.st_mode) and s.st_nlink==1)
payload_bytes=sum((CACHE/(r['token']+'.safetensors')).stat().st_size for r in rows)
assert payload_bytes==75516928 and payload_bytes*2<=160<<20
assert startup_io['rchar']-boot_io['rchar']+payload_bytes*2+4*(1<<20)<512<<20
def version(p):
 s=p.stat();return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'nlink':s.st_nlink}
def budget():
 assert time.monotonic()-boot_seconds<120
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2<<30
 assert io()['rchar']-boot_io['rchar']<512<<20
records=[]
for row in rows:
    meta=get(CACHE/(row['token']+'.json'))
    budget();payload=CACHE/(row['token']+'.safetensors');before=version(payload)
    w,m=probe.read_world(CACHE,signature,row)
    assert before==version(payload)
    records.append({'token':row['token'],'log':row['log'],'payload_sha256':meta['sha256'],
        'receipt':{'path':str(CACHE/(row['token']+'.json')),'sha256':pins[str(CACHE/(row['token']+'.json'))]['sha256']},
        'payload_path':str(payload),'file_version':before,
        'shape':list(w.shape),'dtype':str(w.dtype),'mask_shape':list(m.shape),
        'finite':bool(probe.torch.isfinite(w).all()),'mask_all_true':bool(m.all()),
        'feature_elements':w.numel(),'mask_elements':m.numel()})
    del w,m

# Historical PID alone is not signal authority. Only read current stat metadata.
proc=Path('/proc');entries=list(proc.iterdir());assert len(entries)<32768
matching=[]
for p in entries:
    if not p.name.isdigit():continue
    try:
        f=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if int(f[3])==child['sid']:
            matching.append({'pid':int(p.name),'state':f[0],'sid':int(f[3]),'start_ticks':int(f[19])})
    except (FileNotFoundError,ProcessLookupError):pass
assert not matching, matching
assert not probe.torch.cuda.is_initialized()
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2<<30
assert time.monotonic()-boot_seconds<120
for path,pin in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin['sha256']
end_io=io();measured_rchar=end_io['rchar']-boot_io['rchar'];startup_read=startup_io['rchar']-boot_io['rchar']
small_source_json_bytes=sum(p['bytes'] for p in pins.values())*2+100000
review_charged=max(measured_rchar,startup_read+payload_bytes*2+small_source_json_bytes)
budget();assert review_charged<512<<20
result={'schema':'st_SP0_train_w_artifact_reconciliation_review_v1',
    'status':'COMMITTED_ARTIFACTS_INDEPENDENTLY_VALIDATED','original_stage_status_unchanged':stage['status'],
    'original_supervisor_reason':res['limit_reason'],'automatic_retry':False,'launch_approved':False,
    'downstream_transition_approved':False,'scientific_result':False,
    'validator_runtime':SOURCE,'validator_path':str(Path(probe.__file__)),
    'validator_sha256':hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
    'signature_identity':signature['identity'],'checkpoint_identity':binding['checkpoint_identity'],
    'checkpoint_tag':binding['checkpoint_tag'],'input_pins_unchanged':pins,
    'scenes':64,'records':records,'payload_bytes':payload_bytes,
    'payload_hash_plus_mmap_nominal_bytes':payload_bytes*2,'review_total_read_cap_bytes':512<<20,
    'compute_resources':res,'prior_role_read_bytes':prior,'extract_charged_read_bytes':charged,
    'remaining_role_read_bytes_before_review':cap-prior-charged,
    'remaining_role_read_bytes_less_review_payload_upper':cap-prior-charged-review_charged,
    'matching_old_session_processes_at_review':matching,'processes_signalled':0,
    'review_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'review_seconds_after_import':time.monotonic()-started,'gpu_initialized':False,
    'review_startup_seconds':startup_seconds,'review_startup_rchar_bytes':startup_read,
    'review_total_rchar_bytes':measured_rchar,'review_charged_read_bytes':review_charged,
    'review_read_accounting':'max(proc rchar, startup rchar + hash and mmap nominal payload bytes + double pinned JSON bytes + 100000 source/control allowance)',
    'review_cpu_affinity':sorted(os.sched_getaffinity(0)),
    'review_scope':'read_world plus binding/signature/manifest/completion/receipt checks; no model recomputation or labels',
    'cache_validity_does_not_rewrite_failed_launch':True}
print(json.dumps(result,indent=2))
