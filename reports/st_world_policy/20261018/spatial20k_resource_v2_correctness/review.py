"""Narrow resource-v2 re-signing; no new model/scientific tests or dispatch."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPO = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'spatial20k_independent_correctness_v1'
RESOURCE = Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/checkpoint_remaining_reconciliation_v1')
REG = Path('/mnt/project/st-world-policy-20261016/formal/spatial/registrations')
sys.path.insert(0, str(REPO))
from tools.st_world_policy.check_formal_approval import file_hash, verify_registration

def ref(path): return {'path':str(Path(path).resolve()),'sha256':file_hash(path)}
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream: stream.write(json.dumps(value,indent=2)+'\n')

old_launcher=LEAD/'reports/st_world_policy/20261018/spatial20k_actual_preparation_v1/launch20k_once.py'
launcher=old_launcher.with_name('launch20k_once_v2.py')
assert file_hash(old_launcher)=='e175da58b7e9f0b128c3059c13ab3a2dd7b0c65063daead80a6253165efe79e2'
assert file_hash(launcher)=='8bf6b61023bc00d72e11861155b783bc771791102ca89836c153316eddedde77'
oldtree=ast.parse(old_launcher.read_text()); newtree=ast.parse(launcher.read_text())
def interval(tree,first):
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assert) and first in ast.unparse(n))
    end=next(i for i,n in enumerate(tree.body[start:],start) if isinstance(n,ast.Assert) and ast.unparse(n).startswith('assert free >='))
    return start,end
oa,ob=interval(oldtree,"cp['remaining_bytes'] == 4143")
na,nb=interval(newtree,"cp['total_retained_allocation_bytes'] == 4143")
assert ast.dump(ast.Module(body=oldtree.body[:oa]+oldtree.body[ob+1:],type_ignores=[]))==ast.dump(ast.Module(body=newtree.body[:na]+newtree.body[nb+1:],type_ignores=[]))
guards=compile(ast.Module(body=newtree.body[na:nb+1],type_ignores=[]),str(launcher),'exec')
cp_path=RESOURCE/'CHECKPOINT_COMMITMENTS_REMAINING_V2.json'; cp=read(cp_path)
assert file_hash(cp_path)=='7b88fcbf0a4b4505aa251fdbf2e146c6b94dd3fcb15cef8269cee4d1989fa60c'
snapshot=read(cp['credited_checkpoint_snapshot']['path'])
assert ref(cp['credited_checkpoint_snapshot']['path'])==cp['credited_checkpoint_snapshot']
checkpoints=[c for r in snapshot['runs'] for c in r['checkpoints']]
files=[f for c in checkpoints for f in c['files']]
assert len(checkpoints)==20 and len(files)==180
assert len({(f['device'],f['inode']) for f in files})==180
assert sum(f['credited_bytes'] for f in files)==sum(c['credited_bytes'] for c in checkpoints)==747345660520
assert all(f['credited_bytes']==min(f['bytes'],f['allocated_bytes']) for f in files)
assert cp['commitment_release_bytes']==0 and cp['files_deleted_or_modified'] is False
lease={'remaining_cache_growth_bytes':768*2**30,'separate_conservative_metadata_buffer_bytes':10*2**30}
required=cp['remaining_bytes']+cp['free_floor_bytes']+sum(lease.values())
assert required==5636049483160
exec(guards,{'cp':deepcopy(cp),'lease':deepcopy(lease),'free':required})
negative=[]
for field in ('total_retained_allocation_bytes','credited_already_on_disk_bytes','remaining_bytes','free_floor_bytes'):
    for offset in (-1,1):
        c=deepcopy(cp); c[field]+=offset
        try: exec(guards,{'cp':c,'lease':deepcopy(lease),'free':required+2**40})
        except AssertionError: negative.append(field+str(offset))
        else: raise AssertionError('Guard did not reject '+field)
for field in lease:
    value=deepcopy(lease);value[field]-=1
    try: exec(guards,{'cp':deepcopy(cp),'lease':value,'free':required+2**40})
    except AssertionError: negative.append(field)
    else: raise AssertionError('Guard did not reject '+field)
try: exec(guards,{'cp':deepcopy(cp),'lease':deepcopy(lease),'free':required-1})
except AssertionError: negative.append('free_one_byte_below_required')
else: raise AssertionError('Insufficient space accepted')
common={'schema':'st_spatial20k_resource_v2_correctness_delta_v1','status':'PASS_CHANGED_SCOPE_ONLY',
    'reviewer':'/root/spatial20k_correctness','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
    'launcher':ref(launcher),'unchanged_launcher_ast_except_numeric_storage_guard':True,
    'positive_exact_boundary_passed':True,'negative_cases_passed':negative,
    'checkpoint_receipt':ref(cp_path),'credited_snapshot':cp['credited_checkpoint_snapshot'],
    'gross_bytes':4448512376832,'already_written_credit_bytes':747345660520,
    'remaining_bytes':3701166716312,'required_free_with_full_cache_metadata_floor_bytes':required,
    'credit_metadata_arithmetic':'20 complete checkpoints /180 distinct regular-file snapshot records; no new payload read',
    'native_source_math_config_and_budget_unchanged':True,'dispatches':0,'model_or_teacher_runs':0,
    'native_payload_bytes_read':0,'old_positive_and_25_negative_launcher_tests_inherited':ref(OLD/'LAUNCHER_TESTS.log')}
write(OUT/'DELTA_REVIEW.json',common)
pins={'SP2':'f344cafa2352f1824a7a0d865aa8aaf2f5d680e2e328eefd4826834ffa0953f6',
      'SP3':'ab7a0ae39d5f7d6662d6b9f2b8f01b6738f4b53d6ae20048f56e3ffff5a847b8'}
for arm in ('SP2','SP3'):
    oldpath=REG/f'{arm}_fulltrain_seed42_official_20k_v1/REGISTRATION.json'
    path=REG/f'{arm}_fulltrain_seed42_official_20k_v1_resource_v2/REGISTRATION.json'
    assert file_hash(path)==pins[arm]
    previous=read(oldpath); current=read(path)
    normalized=deepcopy(current); normalized['artifacts']['resource_lease']=previous['artifacts']['resource_lease']
    assert normalized==previous
    inherited=read(OLD/arm/'CORRECTNESS.json')
    assert inherited['status']=='PASS' and inherited['fingerprint']==verify_registration(oldpath,repo=REPO)
    fp=verify_registration(path,repo=REPO)
    assert '/root/spatial20k_correctness' not in current['implementer_agents']
    actual_lease=read(current['artifacts']['resource_lease']['path'])
    assert actual_lease['supersedes_unlaunched_lease']==previous['artifacts']['resource_lease']
    assert actual_lease['checkpoint_commitments']==ref(cp_path)
    assert actual_lease['first_continuation_only'] and actual_lease['no_automatic_resume']
    assert actual_lease['max_invocation_seconds']==28800 and actual_lease['maximum_optimizer_updates_this_stage']==15000
    assert actual_lease['valid_from_unix']<=time.time() and actual_lease['valid_until_unix']>=time.time()+29580
    assert cp['checked_at_unix']<=time.time() and cp['valid_until_unix']>=time.time()+29580
    cfg=read(current['artifacts']['config']['path']);output=Path(cfg['runtime']['output_root'])/current['run_id']
    assert not output.exists() and not (path.parent/'LAUNCH.json').exists() and not (oldpath.parent/'LAUNCH.json').exists()
    exec(guards,{'cp':cp,'lease':actual_lease,'free':required})
    result={'schema':'st_world_review_v1','status':'PASS','role':'correctness',
        'reviewer_agent':'/root/spatial20k_correctness','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
        'fingerprint':fp,'evidence_artifacts':[ref(OUT/'DELTA_REVIEW.json'),ref(OUT/'review.py'),ref(launcher),
            ref(OLD/arm/'CORRECTNESS.json'),ref(OLD/arm/'AUDIT.json'),ref(cp_path),cp['credited_checkpoint_snapshot']],
        'decision_scope':f'Only {arm} first native20k continuation with resource-v2 registration; old mathematical and native-resume evidence inherited unchanged.',
        'limits':['Independent evidence/efficiency role and fresh actual launch GPU/free-space checks remain required.',
                  'Resource credit uses actual metadata/stat snapshot, not new tensor verification; unchanged native trainer verifies payload on load.',
                  'No automatic resume, storage deletion, new model tests, or claim of scientific gain.']}
    dest=OUT/arm;dest.mkdir(exist_ok=False);write(dest/'CORRECTNESS.json',result)
    print(json.dumps({'status':'PASS_NOT_LAUNCHED','review':ref(dest/'CORRECTNESS.json')}))
