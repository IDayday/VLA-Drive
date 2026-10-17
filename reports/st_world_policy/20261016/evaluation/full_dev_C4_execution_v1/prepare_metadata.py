"""One-time metadata-only C4 binding, never invokes an evaluation/training CLI."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
PREVIOUS=Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_C3_execution_v1')
ROOT=Path('/mnt/project/st-world-policy-20261016')
RUNTIME=Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
SOURCE='6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
NOW=datetime.now(timezone.utc).isoformat()
READS={}

def data(path):
    path=Path(path)
    assert path.suffix in ('.json','.py') and path.stat().st_size<=2*1024*1024
    raw=path.read_bytes(); READS[str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return raw

def pin(path):
    path=Path(path); raw=data(path)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def read(path): return json.loads(data(path))
def ident(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(name,value):
    p=HERE/name
    with p.open('x') as f: f.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    return pin(p)

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()==SOURCE
assert not subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME).strip()
regdir=ROOT/'formal/consequence/registrations/C4_fulltrain_seed42_official_4096_v1'
reg=read(regdir/'REGISTRATION.json'); regpin=pin(regdir/'REGISTRATION.json')
assert regpin['sha256']=='e87165448c893d660b796eaa3029d839fc684ef8bdf5f3e4a18467b094d0f8fa'
assert reg['source_sha']==SOURCE and reg['run_id']=='C4_fulltrain_seed42_official_4096_v1'
config=read(regdir/'CONFIG.json'); configpin=pin(regdir/'CONFIG.json')
assert configpin==reg['artifacts']['config'] and config['cell']=='C4' and config['execution']==reg['execution']
approval=read(regdir/'RUN_APPROVAL.json'); apppin=pin(regdir/'RUN_APPROVAL.json')
fp=dict(registration_sha256=regpin['sha256'],run_id=reg['run_id'],source_fingerprint=reg['source_fingerprint'],execution=reg['execution'],artifacts={k:v['sha256'] for k,v in reg['artifacts'].items()})
assert approval['formal_launch_allowed'] is True and approval['fingerprint']==fp
reviews={}
for role in ('CORRECTNESS','EFFICIENCY','EVIDENCE'):
    p=regdir/'reviews'/f'{role}.json'; d=read(p)
    assert d['status']=='PASS' and d['fingerprint']==fp
    reviews[role.lower()]=pin(p)
launch=read(regdir/'LAUNCH.json'); launch_sha=pin(regdir/'LAUNCH.json')['sha256']
choices=list((regdir/'invocations').glob('*/LAUNCH.json')); assert 1<=len(choices)<=8
matches=[p for p in choices if pin(p)['sha256']==launch_sha]; assert len(matches)==1
launchpin=pin(matches[0])
assert launch['source_sha']==SOURCE and launch['registration_sha256']==regpin['sha256'] and launch['approval_sha256']==apppin['sha256']
assert launch['cwd']==str(RUNTIME) and launch['gate']==dict(formal_launch_allowed=True,identity=ident(fp),status='VERIFIED')
run=Path(config['runtime']['output_root'])/reg['run_id']; identity=read(run/'identity.json'); idpin=pin(run/'identity.json')
assert ident({k:v for k,v in identity.items() if k!='identity'})==identity['identity']
assert identity['registration_sha256']==regpin['sha256'] and identity['source_sha']==SOURCE and identity['config']==config
assert launch['output']==str(run)
statusraw=data(run/'status.json'); status=json.loads(statusraw)
assert status['identity']==identity['identity'] and status['source_sha']==SOURCE and status['status']=='RUNNING'
snapshot=write('C4_STATUS_SNAPSHOT.json',dict(schema='st_C4_status_metadata_snapshot_v1',recorded_utc=NOW,source=dict(path=str(run/'status.json'),sha256=hashlib.sha256(statusraw).hexdigest()),projection={k:status[k] for k in ('completed','exposure','identity','source_sha','status','tag','termination_reason','updated_unix')},scope='Historical status projection only; not a complete-checkpoint or available-resource claim'))
binding=dict(registration=regpin,config=configpin,approval=apppin,launch=launchpin,run_identity_file=idpin,runtime_identity=identity['identity'],registration_fingerprint=fp,source_sha=SOURCE,source_fingerprint=reg['source_fingerprint'],registered_execution=reg['execution'],registered_artifacts=reg['artifacts'],review_receipts=reviews,status='TRAINING_RUNNING',status_snapshot=snapshot,completed_at_snapshot=status['completed'],full_train_reverification_this_preparation=False,queue_boundary='No evaluation allocation is inherited from C4 training. Exact endpoint COMPLETE and fresh approved route/free slot remain required.',unresolved=['Exact native COMPLETE binding per 512/1024/4096 endpoint','Independent v5 helper review, Lead endpoint registration and fresh resource admission'])
bindpin=write('TRAINING_BINDINGS.json',dict(schema='st_C4_actual_training_binding_v1',created_at_utc=NOW,scope='Bounded actual control metadata only; no full training manifest or native/image/score payload read',source_runtime=str(RUNTIME),source_sha=SOURCE,bindings=dict(C4=binding)))
previous_plan=read(PREVIOUS/'EXECUTION_PLAN.json'); plan=deepcopy(previous_plan)
plan['arms']['C4']['status']='TRAINING_RUNNING'
plan['formal_runs']['C4']={**{k:binding[k] for k in ('registration','approval','launch','run_identity_file')},'status':'TRAINING_RUNNING','training_source_sha':SOURCE}
rev=plan['execution_binding_revision']; rev.update(version=7,actual_C4_metadata=bindpin,added_arms_this_version=['C4'],additive_preparation_directory=str(HERE),bound_arms=list(plan['formal_runs']),change_scope='Bind only actual C4 training execution; scientific rules and all existing W/C0/C1/C2/C3 bindings remain unchanged.',parent_execution_binding=pin(PREVIOUS/'EXECUTION_PLAN.json'),recorded_utc=NOW,status_scope='Only C4 status is refreshed. Other arm statuses remain historical observations, not new claims.',unbound_arms=[],unregistered_arm_execution={},pending_list_interpretation='Original preregistration checklist is historical. C4 now has actual approval, immutable initial LAUNCH and canonical runtime identity. Exact endpoint receipts and evaluation resource bindings remain pending.')
rev.pop('actual_C3_metadata',None)
plan['status']='C4_ACTUAL_EXECUTION_BOUND_PREPARATION_ONLY_NO_EVALUATION_AUTHORITY'
planpin=write('EXECUTION_PLAN.json',plan)
old_intents=read(PREVIOUS/'INTENTS.json'); jobs=[]
for old in old_intents['jobs']:
    # Replace only explicit C3 arm/path text; all per-endpoint scientific fields
    # and the control outputs are inherited without any score-dependent choice.
    job=json.loads(json.dumps(old).replace('C3','C4'))
    for k,key in [('training_registration','registration'),('training_approval','approval'),('training_launch','launch'),('runtime_identity_file','run_identity_file')]: job[k]=binding[key]
    job['runtime_identity']=binding['runtime_identity'];job['training_binding_ref']=bindpin
    assert all(job['checkpoint_binding'][key] is None for key in ('tag','complete_receipt','native_file_hashes','receipt_identity'))
    jobs.append(job)
intents=dict(schema='st_fixed_dev_C4_actual_binding_intents_v1',created_at_utc=NOW,scientific_plan=old_intents['scientific_plan'],parent_policy=old_intents['parent_policy'],execution_plan=planpin,jobs=jobs,additional_jobs=3,additional_scientific_arms=0,existing_control_jobs_reused=True,launch_authorized=False,scientific_results_created=False)
intentpin=write('INTENTS.json',intents)
old_commands=read(PREVIOUS/'COMMANDS.json'); commands=deepcopy(old_commands)
commands['created_at_utc']=NOW; commands['schema']='st_fixed_dev_C4_actual_binding_commands_v1'
commands.pop('C3_binding_behavior'); commands['C4_binding_behavior']='Actual training identity bound; exact endpoint COMPLETE and fresh route still required. Candidate v5 requires independent review and Lead deployment. No invocation is authorized by this file.'
commands['parent_commands']=pin(PREVIOUS/'COMMANDS.json')
commands['checkpoint_jobs']=[json.loads(json.dumps(j).replace('C3','C4')) for j in old_commands['checkpoint_jobs']]
for j in commands['checkpoint_jobs']:j['required_training_runtime_identity']=identity['identity']
commands['paired_analyses']=[json.loads(json.dumps(j).replace('C3','C4')) for j in old_commands['paired_analyses']]
for j in commands['paired_analyses']:
    j['analysis_plan']=planpin
    j['argv'][j['argv'].index('--plan')+1]=planpin['path']
commandpin=write('COMMANDS.json',commands)
old_policy=read(PREVIOUS/'POLICY.json'); policy=deepcopy(old_policy)
for ref in policy['execution_runtime']['source_files']: assert pin(ref['path'])==ref
policy.update(schema='st_fixed_dev_C4_actual_binding_policy_v1',created_at_utc=NOW,status='C4_PREPARATION_ONLY',commands=commandpin,intents=intentpin,training_bindings=bindpin,execution_plan=planpin)
policy.pop('previous_C3_pending_preparation')
policy['schedule']=dict(arms=['C4'],controls=dict(C4='C0'),diagnostic_early_elimination=False,new_scientific_arm_or_comparison=False,stages=old_policy['schedule']['stages'])
policy['unresolved']=dict(C4='Each exact 512/1024/4096 COMPLETE binding, independently reviewed v5 helper, endpoint evaluation registration, fresh evaluation resource and Lead dispatch')
policy['incremental_budget']=dict(C4_primary_score_admissions=5088,C4_maximum_total_score_admissions=5184,C4_output_reservation_GiB=3,additional_prepared_jobs=3,additional_scientific_arms=0,budget_scope='Three already preregistered C4 endpoints newly wired. Original sixteen prepared W/C0-C3 jobs and all old pins are unchanged; no duplicate C0 controls. Whole campaign now has nineteen fixed checkpoint jobs.',controller_budget=old_policy['incremental_budget']['controller_budget'],existing_C0_control_evaluations='Reuse same fixed C0 outputs; no duplicate jobs',global_limits_unchanged='At most 4 world1 exports and 2 scores with 2 workers each across the entire campaign; no new slots',maximum_export_invocations_per_job=2,maximum_score_invocations_per_job=2,hard_cap_per_job=1728,primary_per_job=1696,retry_attempts=1,native_copy_bytes=0,new_cache_payload_bytes=0,resume_call_budget='Same ledger and total admission cap across both invocations; no doubling of 1728')
policypin=write('POLICY.json',policy)
write('VALIDATION.json',dict(schema='st_C4_preparation_metadata_validation_v1',status='METADATA_VERIFIED_NOT_EXECUTION_APPROVAL',created_at_utc=NOW,exact_registration_fingerprint=fp,actual_runtime_identity=identity['identity'],completed_at_snapshot=status['completed'],policy=policypin,scientific_fields_unchanged=[k for k in ('statistics','endpoints','comparisons','populations','inference_seed','training_seed','shortlist_rule','optimizer','score_protocol_sha256') if plan[k]==previous_plan[k]],old_formal_bindings_unchanged=all(plan['formal_runs'][a]==v for a,v in previous_plan['formal_runs'].items()),old_arm_definitions_unchanged=all(plan['arms'][a]==v for a,v in previous_plan['arms'].items() if a!='C4'),metadata_reads=READS,read_model_or_native_weights=False,checkpoint_tags_bound=False,resource_or_launch_authority=False))
print(json.dumps(dict(status='PREPARED_ONLY',runtime_identity=identity['identity'],completed=status['completed'],policy=policypin)))
