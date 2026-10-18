"""Read-only constant/binding review; does not import or run the controller."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/full_causal_history_train_attempt2_preparation_v1')
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def readref(ref):
    assert pin(ref['path'])==ref
    return json.loads(Path(ref['path']).read_text())
handoff_path=ROOT/'HANDOFF.json'
assert sha(handoff_path)=='c95ad9c51b716ce10833a1c7fb67e280e1ca752008dace01cb9c0475d2bb3cb9'
h=json.loads(handoff_path.read_text())
old_wrapper=Path(h['source_changes']['original_wrapper']['path'])
new_wrapper=Path(h['source_changes']['wrapper']['path'])
assert pin(old_wrapper)==h['source_changes']['original_wrapper']
assert pin(new_wrapper)==h['source_changes']['wrapper']
old=old_wrapper.read_text();new=new_wrapper.read_text()
def plan_literal(text):
    tree=ast.parse(text)
    return next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='PLAN_SHA' for t in n.targets))
oldsha,newsha=plan_literal(old),plan_literal(new)
assert newsha==h['plan']['sha256']
assert old.replace(oldsha,newsha,1)==new
helper=Path(h['source_changes']['helper']['path'])
assert pin(helper)==h['source_changes']['helper']
assert helper.read_bytes()==(old_wrapper.parent/'storage_binding.py').read_bytes()
old_plan_path=old_wrapper.parent/'FULL_ROLE_PLAN.json'
assert sha(old_plan_path)==oldsha
old_plan=json.loads(old_plan_path.read_text());plan=readref(h['plan'])
assert plan['roles']['dev']==old_plan['roles']['dev']
normalized=deepcopy(plan)
normalized['roles']['train']['control_root']=old_plan['roles']['train']['control_root']
normalized['conservative_metadata_MiB']=old_plan['conservative_metadata_MiB']
attempt=normalized.pop('versioned_attempt')
assert attempt=={
    'additional_typed_artifact_allocation_bytes':0,'attempt':2,'automatic_retry':False,
    'failed_invocation':h['failed_invocation'],'failed_result':h['failed_result'],
    'global_lock_unchanged':old_plan['roles']['train']['control_root']+'/FULL_POPULATION.lock',
    'new_control_allocation_bytes':1<<20,'new_control_root':h['control_root_to_be_created_by_lead'],
    'new_read_allowance_bytes':512*(1<<30),'new_read_allowance_is_explicit_additional':True,
    'old_control_root_retained':old_plan['roles']['train']['control_root'],
    'old_fallback_charge_bytes_preserved':512*(1<<30),'old_fallback_is_measured_read_bytes':False,
    'previous_plan':pin(old_plan_path),'role':'train',
    'same_artifact_root':old_plan['roles']['train']['output_root'],
    'same_typed_registration':h['unchanged_claim_registration']}
assert normalized==old_plan
expected_accounting=deepcopy(old_plan['conservative_metadata_MiB'])
expected_accounting.update(failed_train_attempt1_control_retained=1,total=10098,remaining=142)
assert plan['conservative_metadata_MiB']==expected_accounting
spec=plan['roles']['train'];control=Path(spec['control_root'])
assert control==Path(h['control_root_to_be_created_by_lead'])
assert control.is_dir() and not control.is_symlink() and list(control.iterdir())==[]
assert not Path(spec['output_root']).exists()
assert spec['registration']==h['unchanged_claim_registration']
assert plan['append_only_policy_proposal']==h['unchanged_policy']
registration=readref(spec['registration'])
assert registration==spec['expected_registration']
assert registration['max_scenes']==101592
assert '--resume' not in registration['command']
assert registration['limits']['max_disk_bytes']==2047*(1<<20)
assert h['new_control_reserved_bytes']==h['old_control_reserved_bytes_retained']==1<<20
failed=readref(h['failed_result']);readref(h['failed_invocation'])
assert failed['status']=='FAILED_NO_AUTO_RETRY' and failed['phase']=='START_PUBLICATION'
assert failed['exception']=='StorageBudgetError' and failed['cleanup']=={'status':'NOT_STARTED'}
assert failed['charged_rchar_bytes']==512*(1<<30)
assert h['actual_previous_total_read_bytes'] is None
assert h['cumulative_old_fallback_plus_new_authorized_read_bytes']==1<<40
proposal=readref(h['admission_proposal'])
assert proposal['role']=='train' and proposal['status']=='PROPOSAL_NOT_APPROVED'
assert proposal['reviewer'] is None and proposal['plan']==h['plan']
assert proposal['control_root']==str(control) and proposal['output_root']==spec['output_root']
assert proposal['registration']==spec['registration'] and proposal['policy']==h['unchanged_policy']
assert proposal['checkpoint_receipt']==h['checkpoint_receipt']
assert proposal['limits']==dict(wall_seconds=14400,cleanup_seconds=40,
    read_bytes=512*(1<<30),rss_bytes=4*(1<<30),cpu_affinity=[10,11],
    max_disk_bytes=2048*(1<<20),max_invocations=1)
for source in proposal['controller_sources']:
    assert pin(source['path'])==source
assert "GLOBAL_CONTROL = BASE / 'audits/history_data/full_causal_population_v1/train_controls'" in new
assert "GLOBAL_LOCK = GLOBAL_CONTROL / 'FULL_POPULATION.lock'" in new
assert 'fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)' in new
runtime=Path(h['source_changes']['native_runtime'])
native_controller=runtime/'reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py'
native_monitor=runtime/'reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py'
native_builder=runtime/'tools/st_world_policy/prepare_history.py'
assert "if path.exists():raise ValueError('One-attempt marker already exists; no retry')" in native_monitor.read_text()
assert "os.link(tmp,path);tmp.unlink()" in native_controller.read_text()
assert "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)" in native_controller.read_text()
assert "if (out / 'resolved_config.json').exists() and not args.resume:" in native_builder.read_text()
inherited_path=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/temporal_closeout_independent_v1/full_history_role_review/REVIEW.json')
inherited=json.loads(inherited_path.read_text())
assert inherited['wrapper_sha256']==sha(old_wrapper) and inherited['plan_sha256']==oldsha
assert inherited['status']=='PASS_CODE_AND_BOUNDED_CPU_FIXTURES' and inherited['tests']['status']=='18 passed'
log=inherited_path.with_name('PYTEST.txt');assert sha(log)==inherited['tests']['log_sha256']
result={'schema':'st_full_history_attempt2_independent_correctness_v1',
    'status':'PASS_EXACT_BINDING_DELTA_NOT_RESOURCE_ADMISSION','reviewer':'/root/spatial20k_correctness',
    'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'handoff':pin(handoff_path),
    'old_wrapper':pin(old_wrapper),'new_wrapper':pin(new_wrapper),'new_plan':h['plan'],
    'helper':pin(helper),'inherited_18_fixture_review':pin(inherited_path),'inherited_test_log':pin(log),
    'checks':{'wrapper_only_PLAN_SHA_literal_change':True,'helper_byte_identical':True,
      'plan_only_train_control_root_one_MiB_accounting_and_explicit_attempt_lineage_change':True,
      'dev_role_unchanged_not_authorized_by_train_admission':True,'same_train_output_and_typed_claim':True,
      'old_global_nonblocking_lock_preserved':True,'same_supervisor_nonblocking_lock':True,
      'duplicate_invocation_and_start_marker_rejected':True,'existing_resolved_output_requires_resume_but_resume_false':True,
      'new_control_currently_plain_empty':True,'train_output_currently_absent':True,
      'prior_failed_receipts_unmodified':True,'original_native_disk_guard_and_monitor_unchanged':True,
      'no_automatic_retry':True,'metadata_total_MiB':10098,'metadata_cap_MiB':10240,
      'old_fallback_is_not_measured_total_reads':True,'cumulative_conservative_read_allowance_bytes':1<<40},
    'limits':['Actual fresh independent resource admission and launch-time CPU/lease/free-space checks remain with Lead and efficiency reviewer.',
      'No live ledger mutation, full population/index read, model/teacher/image loading, builder execution, dispatch or repeated18-test suite.',
      'The explicit attempt2 is separate authorization, not an automatic retry. Original failure and its1MiB control are retained.',
      'Native existing output protections cover immutable start marker/resolved preparation. No claim of rejecting the initially empty directory created by guarded publication.',
      'Four-hour cap is a stop bound; successful full-history completion is not yet established.']}
with (OUT/'REVIEW.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(pin(OUT/'REVIEW.json')))
