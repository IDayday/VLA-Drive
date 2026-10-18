"""Actual resource closeout: JSON/hash/stat only, no rerun or native payload read."""
import hashlib,json,os,time
from pathlib import Path

OUT=Path(__file__).parent
C=Path('/mnt/project/st-world-policy-20261016/smoke')
V5=C/'rollout_scale_native_resume_completion_v5'
AD=V5.with_name(V5.name+'_admission')
read_bytes=0
refs=[]
def load(path):
    global read_bytes
    p=Path(path); assert p.stat().st_size<256*1024
    data=p.read_bytes();read_bytes+=len(data);assert read_bytes<8*1024**2
    ref={'path':str(p),'sha256':hashlib.sha256(data).hexdigest()}
    refs.append(ref);return json.loads(data),ref
def physical_size(root):
    total=0;files=0
    if not root.exists():return 0,0
    for folder,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in names:
            p=Path(folder)/name
            if p.is_file() and not p.is_symlink():total+=p.stat().st_size;files+=1
    return total,files

prior=json.loads((OUT/'ACTUAL_A_B_PROGRESS_AUDIT.json').read_text())
bound,_=load(AD/'ADMISSION.json');binding=bound['binding']
assert bound['binding_sha256']=='8b0d3b8f997d514e76b322ff9a5afb06fd60f55e0775fad30a13af4f7013a313'
for ref in prior['small_control_refs']+prior['A_four_rank_observer_complete']:
    _,actual=load(ref['path']);assert actual['sha256']==ref['sha256']
for ref in binding['references']['old_B_metadata']:
    p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
assert sorted(p.name for p in Path(prior['B_run']).iterdir())==['RUN.lock','identity.json','resolved_config.json']
assert (V5/'continuous').is_symlink() and (V5/'continuous').resolve()==C/'rollout_scale_native_resume_v4/continuous'
initial,initial_ref=load(V5/'control_v5/initial_RESULT.json')
resume,resume_ref=load(V5/'control_v5/resume_RESULT.json')
assert initial['status']=='B2_COMPLETE_CLEAN' and resume['status']=='B4_COMPLETE_CLEAN'
assert not initial['A_reference_retrained'] and not resume['A_reference_retrained']
assert initial['new_invocations_started']==resume['new_invocations_started']==1
start,start_ref=load(V5/'control_v5/START.json')
assert abs(start['deadline_unix']-start['started_unix']-2940)<.001
assert (V5/'control_v5/resume_RESULT.json').stat().st_mtime<start['deadline_unix']
lease,_=load(AD/'B_RESOURCE_LEASE.json')
first_driver,_=load(AD/'INITIAL_DRIVER_LAUNCH.json');last_driver,_=load(AD/'RESUME_DRIVER_LAUNCH.json')
assert not first_driver['old_lease_or_deadline_reused']
assert last_driver['original_common_deadline_reused_not_reset']
assert lease['valid_from_unix']<=first_driver['started_unix']<last_driver['started_unix']<lease['valid_until_unix']
assert (V5/'control_v5/resume_RESULT.json').stat().st_mtime<lease['valid_until_unix']
roots=[C/f'rollout_scale_native_resume_v{i}' for i in range(1,5)]+[V5]
invocations=[];call_rows=[];native_rows=[];output_rows=[]
for root in roots:
    control='control_v5' if root==V5 else 'control'
    for admission in sorted((root/control/'invocations').glob('*/*/ADMISSION.json')):
        a,a_ref=load(admission);l,l_ref=load(admission.with_name('LAUNCH.json'));e,e_ref=load(admission.with_name('EXIT.json'))
        assert e['owned_cleanup_complete']
        invocations.append({'root':str(root),'unit':a['unit'],'admission':a_ref,'launch':l_ref,'exit':e_ref,
            'returncode':e['returncode'],'owned_cleanup_complete':True,'owned_ID_count':len(e.get('owned_IDs',{})),
            'admitted_unix':a['admitted_unix'],
            'admission_to_cleanup_receipt_upper_seconds':admission.with_name('EXIT.json').stat().st_mtime-a['admitted_unix'],
            'exit_receipt_mtime_unix':admission.with_name('EXIT.json').stat().st_mtime})
    # Deliberately skip the V5 A symlink; that physical data belongs only to V4.
    arms=['interrupted'] if root==V5 else ['continuous','interrupted']
    for arm in arms:
        for p in (root/arm).glob('runs/*/official_feedback/*/attempt_*/ADMITTED.json'):
            row,ref=load(p);assert p.parent.name=='attempt_00'
            call_rows.append({'root':str(root),'arm':arm,'calls':row['calls'],'receipt':ref})
        for p in (root/arm).glob('runs/*/checkpoints/step_*/COMPLETE.json'):
            row,ref=load(p);assert row['world_size']==4
            payload=0
            for name,file in row['files'].items():
                assert Path(name).name==name and (p.parent/name).stat().st_size==file['bytes']
                payload+=file['bytes']
            assert payload<12*1024**3
            native_rows.append({'root':str(root),'arm':arm,'completed':row['state']['completed'],
                'receipt':ref,'payload_bytes_stat_only':payload,'payload_checksums_reread':False})
    for folder in (root,root.with_name(root.name+'_admission')):
        size,count=physical_size(folder);output_rows.append({'root':str(folder),'physical_files':count,'bytes':size})
assert len(invocations)==6 and len([x for x in invocations if x['root']==str(V5)])==2
assert len(native_rows)==6 and sum(x['calls'] for x in call_rows)==1024
assert sum(x['bytes'] for x in output_rows)<104*1024**3
A,a_ref=load(binding['references']['A_result']['path'])
B2,b2_ref=load(V5/'control/interrupted_step2_RESULT_SNAPSHOT.json')
B4,b4_ref=load(V5/'control/interrupted_step4_RESULT_SNAPSHOT.json')
assert A['completed']==4 and A['new_optimizer_updates']==4
assert B2['completed']==2 and B4['completed']==4 and B2['new_optimizer_updates']==B4['new_optimizer_updates']==2
assert A['counts']['actual_rollouts']==B4['counts']['actual_rollouts']==512
assert all(x['frozen_encoder_unchanged'] and x['formal_launch_approval'] is False for x in (A,B2,B4))
for phase,expected in (('INITIAL',2),('RESUMED',4)):
    for rank in range(4):
        row,_=load(V5/'interrupted/observer'/f'{phase}_OBSERVER_COMPLETE_rank{rank}.json')
        assert row['completed']==expected and row['rank']==rank and row['observer_sha256']=='468fe8f1ed20ef3abb2b2407413f92fb078f6b1d05f6cbfb2ed4e4b6e69f9692'
comparison,comparison_ref=load(V5/'COMPARISON.json')
assert comparison_ref['sha256']=='4894507742bafd73ab951af0eab8a7203a63e7cf2b8c3dd99fa3271cc227fe89'
assert comparison['verdict']=='EXACT_REPLAY' and comparison['prefix_A_A_bitwise'] and comparison['resumed_suffix_bitwise'] and comparison['strict_native_load_and_RNG_match']
comparison_exec,comparison_exec_ref=load(AD/'COMPARE_EXECUTION.json')
assert comparison_exec['comparison']==comparison_ref and comparison_exec['exit_code']==0
assert comparison_exec['wall_seconds']<120 and comparison_exec['waited_child_max_RSS_bytes']<2*1024**3
stats=resume['stats'];assert prior['prior_cumulative_rchar_to_inherit']<initial['stats']['sampled_rchar']<stats['sampled_rchar']<1024**4
assert prior['prior_cumulative_read_bytes_to_inherit']<initial['stats']['sampled_read_bytes']<stats['sampled_read_bytes']
assert max(initial['stats']['peak_RSS_bytes'],stats['peak_RSS_bytes'])<256*1024**3
new_units=[x for x in invocations if x['root']==str(V5)]
for x in new_units: assert x['returncode']==0
wall=sum(x['admission_to_cleanup_receipt_upper_seconds'] for x in new_units)
elapsed=(V5/'control_v5/resume_RESULT.json').stat().st_mtime-start['started_unix']
observed_allocations={'A_continuous4':A['allocation_wall_seconds'],'B_initial2':B2['allocation_wall_seconds'],'B_resumed2':B4['allocation_wall_seconds']}
report={'schema':'st_native_resume_actual_resource_closeout_v5','status':'PASS_RECORDED_LIMITS_AND_COMPLETION',
 'reviewer':'/root/road_failure_resources_review_v1','observed_unix':time.time(),
 'binding_sha256':bound['binding_sha256'],'coordinator_commit':'8787b62bd4f897b991ef92a8882b146b9ff42082',
 'coordinator_sha256':bound['coordinator_sha256'],'runtime_source_sha':'751fcfa2b41549ae16f7b3204aef8dd18b912016',
 'comparison':comparison_ref,'comparison_verdict':'EXACT_REPLAY (read existing once; not rerun)',
 'comparison_execution':comparison_exec_ref,'comparison_wall_seconds':comparison_exec['wall_seconds'],
 'comparison_waited_child_CPU_seconds':comparison_exec['waited_child_CPU_seconds'],
 'comparison_waited_child_max_RSS_bytes':comparison_exec['waited_child_max_RSS_bytes'],
 'actual_trainer_invocations_including_failures':len(invocations),'invocation_rows':invocations,
 'actual_optimizer_updates':8,'A_updates':4,'B_updates':4,'A_retrained':False,
 'B_initial_new_updates':2,'B_resumed_new_updates':2,
 'official_admitted_calls_total':sum(x['calls'] for x in call_rows),'retry_calls':0,
 'official_admission_rows':call_rows,'native_directory_count':len(native_rows),'native_directory_cap':8,
 'native_rows':native_rows,'native_payload_bytes_stat_only':sum(x['payload_bytes_stat_only'] for x in native_rows),
 'all_output_bytes_including_prior_failures_and_admissions':sum(x['bytes'] for x in output_rows),
 'output_cap_bytes':104*1024**3,'physical_output_rows':output_rows,
 'A_symlink_counted_once':True,'old_A_and_old_B_three_files_unchanged':True,
 'old_expired_lease_reused':False,'fresh_B_lease_valid_for_both_invocations':True,
 'calendar_gap_seconds_at_admission':binding['calendar_gap_seconds_at_admission'],
 'v4_START_to_cleanup_receipt_upper_seconds':prior['v4_controller_START_to_cleanup_receipt_upper_seconds'],
 'prior_conservative_active_seconds':660,'new_B_admission_to_cleanup_sum_seconds':wall,
 'new_controller_START_to_final_result_receipt_upper_seconds':elapsed,
 'observed_conservative_charge_plus_new_elapsed_plus_comparison_seconds':660+elapsed+comparison_exec['wall_seconds'],
 'active_budget_seconds':3900,'budget_partition_seconds':{'prior':660,'new_controller':2940,'comparison':120,'cleanup':180},
 'wall_scope':'A/B old overlap is not added twice; new total elapsed includes between-invocation control gap. Filesystem mtime timing has one-second resolution. Calendar gap is not active execution.',
 'group_allocation_intervals_seconds':observed_allocations,
 'group_allocation_scope':A['allocation_wall_scope'],
 'selected_four_GPU_allocation_interval_GPU_seconds':4*sum(observed_allocations.values()),
 'GPU_allocation_interval_scope':'Four assigned GPUs times reported intervals only; excludes earlier failed launches/setup/drain, and is not measured GPU busy time or full campaign cost.',
 'exact_training_GPU_busy_seconds':'NOT_MEASURED','exact_full_campaign_GPU_allocation_seconds':'NOT_MEASURED',
 'exact_training_and_live_official_worker_CPU_seconds':'NOT_MEASURED',
 'cumulative_sampled_rchar':stats['sampled_rchar'],'cumulative_sampled_read_bytes':stats['sampled_read_bytes'],
 'sampled_rchar_limit_bytes':1024**4,'IO_scope':'Inherited v4 counters once, then B initial and resumed deltas. Sampling lower bounds omit unobserved short-lived/retired activity and some mmap; not exact physical IO or strict byte quota.',
 'new_initial_sampled_group_peak_RSS_bytes':initial['stats']['peak_RSS_bytes'],
 'new_resume_sampled_group_peak_RSS_bytes':stats['peak_RSS_bytes'],
 'prior_observed_aggregate_peak_RSS_bytes':prior['v4_control_result_preserved']['stats']['peak_aggregate_RSS'],
 'all_six_exit_receipts_confirm_owned_cleanup':True,
 'independent_current_GPU_and_PID_cleanup_check':'Lead fresh verification pending; this reviewer did not query GPU/remote process inventories.',
 'new_tests_or_comparison_runs':0,'large_payload_reads_or_hashes_by_reviewer':0,
 'small_JSON_and_source_read_bytes':read_bytes,'evidence_artifacts':refs,
 'formal_training_touched':False,'new_experiment_permission':False,
 'scientific_gain':'NOT_EVALUATED; this is recovery engineering evidence only.'}
(OUT/'ACTUAL_RESOURCE_CLOSEOUT.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','actual_trainer_invocations_including_failures','actual_optimizer_updates','official_admitted_calls_total','native_directory_count','all_output_bytes_including_prior_failures_and_admissions','new_B_admission_to_cleanup_sum_seconds','new_controller_START_to_final_result_receipt_upper_seconds','group_allocation_intervals_seconds','cumulative_sampled_rchar','cumulative_sampled_read_bytes','small_JSON_and_source_read_bytes')},sort_keys=True))
