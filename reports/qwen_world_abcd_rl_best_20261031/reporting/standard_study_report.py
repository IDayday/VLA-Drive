"""Read-only namespaces for related RL studies; local report artifacts only."""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import re
import socket
import time


E_ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031')
DEFAULT_ROOT=E_ROOT/'standard_training_20261101'


def is_revival(study):
    return study.get('registration',{}).get('study_kind')=='historical_revival'or Path(study['root']).name.startswith('revival_')


def is_improvement(study):
    return study.get('registration',{}).get('study_kind')=='configuration_improvement'or Path(study['root']).name.startswith('improvement_')


def section_id(study):
    if Path(study['root'])==DEFAULT_ROOT:return 'standard-training'
    name=re.sub('[^A-Za-z0-9_-]+','-',study.get('id',Path(study['root']).name))
    return 'related-study-'+name+'-'+hashlib.sha256(study['root'].encode()).hexdigest()[:8]


def section_links(studies):
    current=[(section_id(s),'当前H改进实验')for s in studies if is_improvement(s)]
    links=[(section_id(s),'历史方法复试'if is_revival(s)else '统一配方新研究'if Path(s['root'])==DEFAULT_ROOT else s.get('registration',{}).get('display_name',s['id']))for s in studies if not is_improvement(s)]
    if any(s.get('paired_reviews')for s in studies):links.insert(0,('rl-benefit','RL实质收益'))
    if any(s.get('phase_potential',{}).get('status','').startswith('COMPLETE')for s in studies):links.insert(0,('rl-potential','下一步提升潜力'))
    return [('rl-latest-evidence','最新完整证据与资源优先级')]+current+links


def _run_status(run,h):
    status=str(run.get('status','')).upper()
    if any(s in status for s in ['PAUSED','USER_PAUSE']):return '用户暂停，当前不推进'
    reserve=any('RESEARCH_RESERVE'in str(run.get(k,''))for k in ['standard_disposition','revival_disposition','disposition'])
    if reserve and(run.get('worker_returncode')==0 or status=='STOPPED_AT_COOPERATIVE_BOUNDARY'):
        return '已保存边界，资源优先级预留<small>资源转新研究；非方法族无效结论</small>'
    if reserve and run.get('worker_receipt_status')=='RUNNING':
        return '当前阶段仍在完成，之后预留资源<small>尚未退出，不记为已释放</small>'
    if 'WAITING_AUTHORIZED_MIGRATION'in status:return '已保存边界，等待授权host恢复'
    if run.get('progress_freshness',{}).get('stale'):
        fresh=run['progress_freshness'];return '<b>进度过期／当前运行未确认</b><small>最后日志 '+h.esc(fresh['last_observed_utc'])+'；距今 '+h.fmt(fresh['age_seconds']/3600,1)+' 小时。旧RUNNING回执不等于当前健康。</small>'
    if run.get('worker_returncode')==75:return '资源准入等待（非科学失败）'
    if run.get('worker_returncode')not in(None,0):return '执行失败待审，原件保留'
    if run.get('worker_receipt_status')=='RUNNING'and run.get('journal_ranks')==8 and h.finite(run.get('updates'))and run['updates']>0:return '实际八rank优化已见'
    if run.get('worker_receipt_status')=='RUNNING':return 'child启动，待实际更新'
    if run.get('worker_returncode')==0:return '本原生阶段完成；完整评测另列'
    if run.get('journal_ranks')==8 and h.finite(run.get('updates'))and run['updates']>0:return '已有八rank更新；当前派发待child确认'
    return '已配置，未见实际启动'


def _runtime_authorization(h,warnings,inputs):
    path=Path('/mnt/project/server_dispatch_policy.json');policy=h.read_json(path,warnings,inputs)or{}
    grant=policy.get('task_authorizations',{}).get('qwen_world_safety_exploration_20261031',{})
    record=h.read_json(Path(grant['record']),warnings,inputs)if grant.get('record')else None
    fields=['permanent_hosts','temporary_hosts','unbounded_temporary_hosts','starts_unix','expires_unix','checkpoint_guard_seconds','host_authorization_origin_unix']
    matches=bool(record)and all(record.get(k)==grant.get(k)for k in fields)
    selected=['training-vla-zt3','training-vlawm-zt2','training-vlawm-zt3','training-vlawm-zt4','training-vlawm-zt5','training-vlawm-tsinghua-zt4']
    bans={host:policy.get('global_dispatch_prohibitions',{}).get(host,{})for host in selected}
    excluded={host for host,value in bans.items()if value.get('new_task_dispatch_allowed')is False}
    permanent=set(grant.get('permanent_hosts',[]));unbounded=set(grant.get('unbounded_temporary_hosts',[]));temporary=set(grant.get('temporary_hosts',[]))
    expiry=grant.get('expires_unix');finite_window=bool(temporary)and h.finite(expiry)
    guard=expiry-grant.get('checkpoint_guard_seconds',0)if finite_window else None;now=time.time()
    within=finite_window and grant.get('starts_unix',0)<=now<expiry
    long_window=within and h.finite(guard)and now<guard
    authorized=(permanent|unbounded|(temporary if within else set()))-excluded
    admit=(permanent|unbounded|(temporary if long_window else set()))-excluded
    return dict(record=grant.get('record'),record_matches=matches,permanent_hosts=sorted(permanent-excluded),temporary_hosts=sorted(temporary-excluded),
        unbounded_temporary_hosts=sorted(unbounded-excluded),finite_deadline_hosts=sorted(temporary-excluded),
        authorized_now=sorted(authorized)if matches else[],long_admit_now=sorted(admit)if matches else[],global_bans=bans,
        grant_contains_banned=sorted((permanent|temporary|unbounded)&excluded),guard_utc=h.utc(guard)if guard is not None else None,
        deadline_utc=h.utc(expiry)if finite_window else None,checked_utc=h.utc(now),
        historical_expiry_receipt=str(E_ROOT/'resources'/'TEMPORARY_RESOURCE_EXPIRED_20261101T230000Z.json'))


def _comparison_summary(value):
    keys=['PDMS_delta','PDMS_CI95','repaired_zero','new_zero','baseline_zero_denominator','scene_denominator','bootstrap','bootstrap_seed','physical_groups']
    result={k:value[k]for k in keys if k in value}
    result['fixed_control_high']=value.get('fixed_0p9_cohorts',{}).get('fixed_cohorts',{}).get('high',{})
    return result


def _paired_reviews(root,arms,h,warnings,inputs):
    rows=[]
    for path in sorted((root/'analysis'/'PILOT_COMPARISONS').glob('*.json')):
        if not any(path.name.startswith(arm+'_')for arm in arms):continue
        value=h.read_json(path,warnings,inputs)or{}
        if value.get('status')!='COMPLETE':continue
        base=dict(arm=value.get('arm'),updates=value.get('updates'),evidence=str(path),
            fixed_IL70_high_advantage_PDMS=value.get('fixed_IL70_high_advantage_PDMS'),actual_collection_budget=value.get('actual_collection_budget'))
        if value.get('schema')in ['world_Safety_exploration_full_population_review_v1','world_revival_full_population_review_v1','world_improvement_full_population_review_v1']:
            for label in ['IL70','SafetyEP10k']:
                result=value.get('comparisons',{}).get(label,{}).get('result',{})
                if result.get('scene_denominator')==12146:rows.append(base|dict(kind='historical',reference=label,result=_comparison_summary(result)))
        elif value.get('reference_arm'):
            result=value.get('comparison',{}).get('result',{})
            if result.get('scene_denominator')==12146:rows.append(base|dict(kind='between_arms',reference=value['reference_arm'],reference_updates=value.get('reference_updates'),reference_root=value.get('external_control_root',str(root)),budget_contract=value.get('budget_contract'),result=_comparison_summary(result)))
    return rows


def _lr_values(value):
    if isinstance(value,list):
        values=[v.get('lr')if isinstance(v,dict)else v for v in value]
        if len(values)>=2 and all(isinstance(v,(float,int))for v in values[:2]):return values[:2]
    if isinstance(value,dict):
        values=[value.get('action_lora'),value.get('scene_predictive_memory')]
        if all(isinstance(v,(float,int))for v in values):return values
    return None


def _observed_lr(run,h):
    """Step-used values only; never substitute planned values for missing logs."""
    if not run.get('root')or not h.finite(run.get('updates')):return []
    rows=h.read_tail(Path(run['root'])/'metrics_rank0.jsonl')
    points=[];previous=None
    for row in rows:
        update=row.get('updates')
        if not h.finite(update)or update>run['updates']:continue
        used=None;source=None
        for name in ['optimizer_group_lrs_used','optimizer_used_group_lrs','used_optimizer_group_lrs','used_group_lrs']:
            used=_lr_values(row.get(name))
            if used is not None:source='saved update-before-step '+name;break
        if used is None and previous and update==previous.get('updates',-2)+1:
            prior_next=_lr_values(previous.get('optimizer_group_lrs'))
            if prior_next and h.finite(row.get('lr'))and math.isclose(prior_next[0],row['lr'],rel_tol=1e-10,abs_tol=1e-15):
                used=prior_next;source='previous row next-step groups; verified against current pre-step action lr'
        if used is not None and all(h.finite(v)for v in used):
            points.append(dict(optimizer_update=int(update),completed_before_update=int(update)-1,
                action_lr=used[0],memory_lr=used[1],source=source))
        elif h.finite(row.get('lr')):
            points.append(dict(optimizer_update=int(update),completed_before_update=int(update)-1,
                action_lr=row['lr'],memory_lr=None,source='pre-step action only; memory unobserved'))
        previous=row
    return points


def load_related(h,warnings,inputs):
    """An existing REGISTRATION or RELATED_STUDIES is the admission gate."""
    pointer=h.read_json(E_ROOT/'RELATED_STUDIES.json',warnings,inputs)or{}
    specs=[]
    if (DEFAULT_ROOT/'REGISTRATION.json').is_file():specs.append(dict(root=str(DEFAULT_ROOT)))
    for row in h.object_rows(pointer.get('studies',pointer.get('related_studies',[])),'studies'):
        if row.get('root'):specs.append(row)
    studies=[];seen=set()
    for spec in specs:
        root=Path(spec['root'])
        if root in seen or root==E_ROOT or E_ROOT not in root.parents:continue
        seen.add(root);reg_path=Path(spec.get('registration')or root/'REGISTRATION.json')
        if not reg_path.is_file():continue
        reg=h.read_json(reg_path,warnings,inputs)or{}
        if not reg:continue
        registry_path=Path(reg.get('registry')or root/'RUNS.json')
        registry=h.read_json(registry_path,warnings,inputs)or{}
        declared=registry.get('runs',reg.get('arms',[]))
        run_map={r['id']:r for r in h.object_rows(declared,'runs')if r.get('id')}
        state_paths=list(dict.fromkeys([Path(r['owner_state'])for r in run_map.values()if r.get('owner_state')]+sorted((root/'owners').glob('*/STATE.json'))))
        states={str(p):h.read_json(p,warnings,inputs)or{}for p in state_paths};runs=[]
        for arm,row in run_map.items():
            owner=states.get(row.get('owner_state'),{});node=owner.get('runs',{}).get(arm,{})if isinstance(owner.get('runs'),dict)else{}
            if not node:
                candidates=[(Path(p).stat().st_mtime,s['runs'][arm])for p,s in states.items()if isinstance(s.get('runs'),dict)and isinstance(s['runs'].get(arm),dict)]
                if candidates:node=max(candidates,key=lambda x:x[0])[1]
            merged={**row,**node,'id':arm,'root':node.get('root')or node.get('run_root')or row.get('root')}
            if not merged.get('root'):merged['root']=str(root/'runs'/arm)
            snapshot=h.run_snapshot(merged,warnings,inputs,registry_mtime=registry_path.stat().st_mtime if registry_path.exists()else 0)
            snapshot['status']=merged.get('status',reg.get('status'))
            if not snapshot.get('journal_ranks')or snapshot.get('journal_ranks')!=snapshot.get('expected_ranks'):snapshot['updates']=None
            identity_path=Path(merged['root'])/'identity.json';identity=h.read_json(identity_path,warnings,inputs)or{}
            snapshot['native_identity_sha256']=identity.get('scientific_identity',{}).get('sha256',identity.get('sha256'))
            snapshot['optimizer_parameter_groups']=identity.get('actual_optimizer',{}).get('parameter_groups',[])
            snapshot['observed_lr']=_observed_lr(snapshot,h)
            if h.finite(snapshot.get('evidence_mtime')):
                age=max(0,time.time()-snapshot['evidence_mtime'])
                snapshot['progress_freshness']=dict(last_observed_utc=h.utc(snapshot['evidence_mtime']),age_seconds=age,threshold_seconds=7200,
                    stale=age>7200 and snapshot.get('worker_receipt_status')=='RUNNING')
            runs.append(snapshot)
        entries=h.evaluation_entries(registry)+h.evaluation_entries(h.read_json(root/'RESULTS.json',warnings,inputs)or{})
        for state in states.values():
            entries+=h.evaluation_entries(state)
            for row in h.object_rows(state.get('evaluations',{}),'evaluations'):
                if row.get('cohort')not in('NAVTEST','TRAIN_DIAGNOSTIC','DEV')or not row.get('evaluation'):continue
                entry={k:row[k]for k in ['label','arm','updates','cohort','evaluation','status','expected_scenes']if k in row}
                entry['evidence']=row.get('summary')if isinstance(row.get('summary'),str)else str(Path(row['evaluation'])/'SUMMARY.json')
                if row['cohort']=='NAVTEST':entry['expected_scenes']=12146
                entries.append(entry)
        results={}
        for entry in entries:
            result=h.normalize_result(entry,warnings,inputs)
            if result.get('arm')not in run_map:continue
            key=result.get('evidence')or result.get('label')
            if key and not(results.get(key,{}).get('publishable_complete')and not result.get('publishable_complete')):results[key]=result
        controllers={}
        for path,state in states.items():
            folder=Path(path).parent;owner=h.read_json(folder/'OWNER.json',warnings,inputs)or{};launch=h.read_json(folder/'LAUNCH.json',warnings,inputs)or{}
            host=owner.get('host')or launch.get('host');identity=None
            if host==socket.gethostname():
                try:
                    process=Path('/proc',str(owner.get('pid',launch.get('pid'))),'stat').read_text().split()
                    identity=int(process[21])==int(owner.get('start_ticks',launch.get('start_ticks')))and process[2]not in('Z','X')
                except(OSError,ValueError,TypeError,IndexError):identity=False
            controllers[folder.name]=dict(state=state,state_path=path,owner=owner,launch=launch,local_pid_start_matches=identity)
        protocol_path=Path(reg.get('scheduler_protocol')or root/'SCHEDULER_PROTOCOL.json')
        exposure_path=Path(reg.get('exposure')or root/'TRAINING_EXPOSURE.json')
        protocol=h.read_json(protocol_path,warnings,inputs)or{}
        exposure=h.read_json(exposure_path,warnings,inputs)or{}
        prefix=h.read_json(root/'analysis'/'STANDARD_PRODUCTION_PREFIX_u000100.json',warnings,inputs)or{}
        if prefix:
            # Keep the compact actual-process result, never embed per-rank archives.
            prefix={k:v for k,v in prefix.items()if k!='results'}|{'results':[{k:v for k,v in r.items()if k!='receipts'}for r in prefix.get('results',[])if isinstance(r,dict)]}
        audit_path=reg.get('historical_audit');audit=h.read_json(Path(audit_path),warnings,inputs)if audit_path else None
        if audit:
            keep=['schema','status','main_conclusions_zh','E_complete_points','R_complete_points','R_summary_contract_errors','limits_zh']
            compact={k:audit[k]for k in keep if k in audit}
            for group in ['E_arms','R_arms']:
                compact[group]=[]
                for row in audit.get(group,[]):
                    item={k:row[k]for k in ['id','priority','action','assessment','reason_zh','completed_updates','candidate_cost_relative_to_G8']if k in row}
                    item['full_NAVTEST_points']=[{k:p[k]for k in ['updates','PDMS','source','path','same_budget_control']if k in p}for p in row.get('full_NAVTEST_points',[])]
                    if row.get('id')in ['PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS']:
                        item['schedule_matched_evidence']=[{k:p[k]for k in ['label','target_PDMS','baseline_PDMS','delta_PDMS','CI95_PDMS']if k in p}for p in row.get('matched_evidence',[])if p.get('label','').endswith('_minus_SAFETY_same_updates')]
                    compact[group].append(item)
            audit=compact
        deployment_path=root/'analysis'/'ACTUAL_REVIVAL_DEPLOYMENT.json'
        deployment=h.read_json(deployment_path,warnings,inputs)or{}
        if deployment:deployment={k:v for k,v in deployment.items()if k!='runs'}|{'runs':[{k:v for k,v in r.items()if k not in ['gpu_uuids','rank_updates']}for r in deployment.get('runs',[])]}
        recoveries=[]
        for folder in sorted((root/'operations').glob('*VALIDATOR_RECOVERY')):
            check=h.read_json(folder/'EXECUTION_EQUIVALENCE_PASS.json',warnings,inputs)or{}
            if check:recoveries.append(dict(name=folder.name,status=check.get('status'),scope=check.get('check_scope',check.get('scope')),evidence=str(folder/'EXECUTION_EQUIVALENCE_PASS.json'),migration=str(folder/'MIGRATION_RECEIPT.json')))
        studies.append(dict(id=reg.get('study_id',spec.get('id',root.name)),root=str(root),registration_path=str(reg_path),
            registration=reg,registry=registry,runs=runs,results=list(results.values()),controllers=controllers,
            scheduler_protocol=protocol,scheduler_protocol_path=str(protocol_path),exposure=exposure,exposure_path=str(exposure_path),production_prefix=prefix,
            historical_audit=audit or{},historical_audit_path=audit_path,deployment=deployment,deployment_path=str(deployment_path),validator_recoveries=recoveries,discovery_path=str(E_ROOT/'RELATED_STUDIES.json')))
        studies[-1]['paired_reviews']=_paired_reviews(root,run_map,h,warnings,inputs)
    by_root={s['root']:s for s in studies}
    for study in studies:
        shared=[];refs=h.object_rows(study['registration'].get('shared_controls',[]),'shared_controls')
        external=study['registration'].get('external_control',{})
        if external.get('read_only')is True:refs.append(dict(study_root=external.get('root'),arm=external.get('arm')))
        for ref in refs:
            source=by_root.get(ref.get('study_root'));arm=ref.get('arm')
            if not source or not arm:continue
            shared.append(dict(study_id=source['id'],study_root=source['root'],registration_path=source['registration_path'],arm=arm,
                results=[r for r in source['results']if r.get('arm')==arm and r.get('publishable_complete')]))
        study['shared_controls']=shared
        if is_improvement(study):
            folder=Path(study['root'])/'analysis'
            study['consolidation_handoff_path']=str(folder/'CONSOLIDATION_TRAINING_HANDOFF.json')
            study['consolidation_handoff']=h.read_json(Path(study['consolidation_handoff_path']),warnings,inputs)or{}
            study['multiscene_handoff_path']=str(folder/'MULTISCENE32_HANDOFF.json')
            study['multiscene_handoff']=h.read_json(Path(study['multiscene_handoff_path']),warnings,inputs)or{}
            study['retention_handoff_path']=str(folder/'VERIFIED_RETENTION_HANDOFF.json')
            study['retention_handoff']=h.read_json(Path(study['retention_handoff_path']),warnings,inputs)or{}
            study['common10k_audit_path']=str(folder/'COMMON_10K_REGRESSION_CONFIG_AUDIT.json')
            common=h.read_json(Path(study['common10k_audit_path']),warnings,inputs)or{}
            study['common10k_audit']={k:v for k,v in common.items()if k not in ['inputs','receipts','checkpoints','journal_windows','raw_rows']}
            study['algorithm_mapping_path']=str(folder/'ALGORITHM_PROBLEM_MAPPING_20261101.json')
            study['algorithm_mapping']=h.read_json(Path(study['algorithm_mapping_path']),warnings,inputs)or{}
            study['runtime_authorization']=_runtime_authorization(h,warnings,inputs)
            study['wm45_coordination_path']=str(E_ROOT/'operations'/'UNBOUNDED_WM45_COMBINATION_20261102'/'COORDINATION_DEFERRED_TO_EVAL_OPERATOR.json')
            coordination=h.read_json(Path(study['wm45_coordination_path']),warnings,inputs)or{}
            study['wm45_coordination']={k:v for k,v in coordination.items()if k in ['schema','status','utc','created_utc','at_unix','reason','wm4','wm5']}
            study['new_trials_literature_path']=str(folder/'PRIMARY_LITERATURE_NEW_TRIALS_20261101.json')
            study['new_trials_literature']=h.read_json(Path(study['new_trials_literature_path']),warnings,inputs)or{}
            study['fp32_check_path']=str(folder/'FP32_OUTPUT_CPU_CHECK_20261101.json')
            study['fp32_check']=h.read_json(Path(study['fp32_check_path']),warnings,inputs)or{}
            study['fp32_snapshot_check_path']=str(folder/'FP32_SNAPSHOT_RESTORE_CPU_CHECK.json')
            study['fp32_snapshot_check']=h.read_json(Path(study['fp32_snapshot_check_path']),warnings,inputs)or{}
            study['five_arm_deployment_path']=str(folder/'FIVE_ARM_ACTUAL_DEPLOYMENT_20261101.json')
            five=h.read_json(Path(study['five_arm_deployment_path']),warnings,inputs)or{}
            study['five_arm_deployment']={k:v for k,v in five.items()if k!='arms'}
            study['early_mixed_retention_path']=str(folder/'EARLY_MIXED_SIGNAL_AND_RETENTION_ACTIVATION_20261101.json')
            early=h.read_json(Path(study['early_mixed_retention_path']),warnings,inputs)or{}
            study['early_mixed_retention']={k:({a:b for a,b in v.items()if a!='source_windows'}if isinstance(v,dict)else v)for k,v in early.items()}
            study['fp32_gpu_parity_path']=str(Path(study['root'])/'evaluations'/'IMPROVEMENT_FP32_OUTPUT_G8_u000000'/'NAVTEST'/'native_export_parity.json')
            study['fp32_gpu_parity']=h.read_json(Path(study['fp32_gpu_parity_path']),warnings,inputs)or{}
            study['mixed_handoff_path']=str(folder/'MIXED_EXPLORATION_HANDOFF.json')
            study['mixed_handoff']=h.read_json(Path(study['mixed_handoff_path']),warnings,inputs)or{}
            study['mixed_support_path']=str(folder/'MIXED_EXPLORATION_EXISTING_SUPPORT_AUDIT.json')
            mixed=h.read_json(Path(study['mixed_support_path']),warnings,inputs)or{}
            study['mixed_support']={k:v for k,v in mixed.items()if k not in ['rows','tokens','receipts']}
        if Path(study['root'])==DEFAULT_ROOT:
            study['phase_potential_path']=str(E_ROOT/'analysis'/'LONG_RUN_PHASE_GAIN_RETENTION_AUDIT_20261101.json')
            study['phase_potential']=h.read_json(Path(study['phase_potential_path']),warnings,inputs)or{}
            study['phase_lr_path']=str(E_ROOT/'analysis'/'RL_PHASE_SCHEDULE_OBSERVATION_20261101.json')
            study['phase_lr_observation']=h.read_json(Path(study['phase_lr_path']),warnings,inputs)or{}
            study['four_route_oracle_path']=str(E_ROOT/'analysis'/'FOUR_ROUTE_STRICTSAFE_ORACLE_20261101'/'SUMMARY.json')
            oracle=h.read_json(Path(study['four_route_oracle_path']),warnings,inputs)or{}
            study['four_route_oracle']={k:v for k,v in oracle.items()if k in ['schema','status','population','policies','common','oracle','strict_high_quality_oracle','interpretation_zh']}
    return studies


def _lr_history(report_root,study,h):
    """Small observed-point archive, written only inside the report directory."""
    path=report_root/'reporting'/'standard_observed_lr.json';lock=path.with_suffix('.lock')
    key=hashlib.sha256(study['root'].encode()).hexdigest();out={}
    with lock.open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:state=json.loads(path.read_text())if path.exists()else {}
        except(ValueError,OSError):state={}
        bucket=state.setdefault('studies',{}).setdefault(key,{})
        for run in study.get('runs',[]):
            identity=[run.get('root'),run.get('source_commit'),run.get('native_identity_sha256')]
            old=bucket.get(run['id'],{})
            if old.get('identity')!=identity:old=dict(identity=identity,points={})
            for point in run.get('observed_lr',[]):old['points'][str(point['optimizer_update'])]=point
            bucket[run['id']]=old;out[run['id']]=sorted(old['points'].values(),key=lambda v:v['optimizer_update'])
        state['schema']='world_report_observed_standard_lr_v1';h.atomic_json(path,state)
    return out


def _production_prefix_block(study,h,label):
    prefix=study.get('production_prefix',{});results=prefix.get('results',[])
    if not prefix.get('status','').startswith('PASS_ACTUAL')or not results:return ''
    rows=[];names={'SAFETY_G8_CONTROL':'W3 Control','STANDARD_INVERSION_G8':'Inversion','STANDARD_COMBINED_G8':'Dense＋Sat＋Inversion','STANDARD_ENDPOINT_LR02_G8':'Endpoint 低LR'}
    for r in results:
        q=r.get('collection_quality',{});branches=r.get('combined_branches',{})
        activation='Dense '+h.number(branches.get('dense'))+'／Sat '+h.number(branches.get('saturated'))+'／Inv '+h.number(branches.get('inversion'))if r.get('arm')=='STANDARD_COMBINED_G8'else '独立反序抵消'if r.get('arm')=='STANDARD_INVERSION_G8'else '对照'if r.get('arm')=='SAFETY_G8_CONTROL'else '条件Gaussian端点更新'
        clipped=r.get('gradient_above_clip1_rows');updates=r.get('optimizer_updates');metric_rows=r.get('metric_rows')
        if prefix.get('gradient_rows_repeat_the_global_DDP_norm_across_ranks')and h.finite(clipped)and h.finite(updates)and h.finite(metric_rows)and updates>0:
            clip_text=h.number(clipped/(metric_rows/updates))+'/'+h.number(updates)+' 全局步'
        else:clip_text=h.number(clipped)+' rank行'
        blocked=r.get('ppo_sign_blocked_mean_by_inner',{}).get('4')
        rows.append([h.esc(names.get(r.get('arm'),r.get('arm'))),h.number(q.get('strict_high'))+'/'+h.number(q.get('candidates')),h.number(q.get('positive_high'))+'／'+h.number(q.get('negative_high')),activation,h.fmt(r.get('gradient_norm_mean'),4),clip_text+'<small>inner4符号阻塞 '+h.fmt(blocked,2,percent=True)+'%</small>'])
    body='<h3>机制是否生效：共同首100步TRAIN过程</h3><p>四臂各8rank×100更新，共3,200行；每臂200个fresh组、1,600候选。实际52,707,840参数参与优化，used／next LR均符合登记公式，loss与梯度有限。只按inner1计候选，DDP梯度范数跨rank重复不当作独立全局步。</p>'
    body+=h.table(['方法','严格高分候选','高分A正／负候选','干预实际触发','平均preclip梯度范数','clip1触发／内层阻塞'],rows)
    body+='<p>组合臂的信用分配已经改变，严格高分候选量与Control接近；Endpoint候选支持较多，同时梯度裁剪与内层符号阻塞明显。候选数、优势符号和保护触发都是过程观察，不能单独证明部署收益、学习率过大或clip有害。首100步尚不足以判断收敛。</p>'
    body+='<p class="muted">'+label(Path(study['root'])/'analysis'/'STANDARD_PRODUCTION_PREFIX_u000100.json','首100步正式过程回执')+'；本表不是NAVTEST成绩。</p>'
    return body


def _historical_reassessment_block(study,h,label):
    audit=study.get('historical_audit',{})
    if not audit.get('status','').startswith('COMPLETE'):return ''
    rows=[]
    for row in sorted((r for r in audit.get('E_arms',[])if h.finite(r.get('priority'))),key=lambda r:r['priority']):
        points=row.get('full_NAVTEST_points',[]);latest=max(points,key=lambda p:p.get('updates',0),default={});pair=latest.get('same_budget_control',{});ci=pair.get('PDMS_CI95')
        interval='['+', '.join(h.fmt(v,4)for v in ci)+']'if isinstance(ci,list)and len(ci)==2 else '既有区间未给'
        rows.append([h.esc(row['id']),h.number(latest.get('updates')),h.fmt(pair.get('PDMS_delta'),4)+'<small>'+interval+'</small>',h.number(row.get('candidate_cost_relative_to_G8'))+'×',h.esc(row.get('reason_zh',''))])
    body='<p>已逐项复核旧E '+h.number(audit.get('E_complete_points'))+'份与旧R '+h.number(audit.get('R_complete_points'))+'份完整NAVTEST；坏summary '+h.number(len(audit.get('R_summary_contract_errors',[])))+'。未重读场景或生成新评分。</p>'
    if rows:body+=h.table(['复试的历史方法','旧末次完整更新','对同预算Control Δ／配对95%CI','每更新候选成本','为何复试／保留的风险'],rows)
    body+='<p class="muted">同更新数不等于同候选预算，G16尤其如此；早期负面CI与高分损失仍有效，复试检验的是新统一配方下的较长曲线。旧对照继续保留在<a href="#exploration">旧E完整证据</a>。</p>'
    conclusions=audit.get('main_conclusions_zh',[])
    if conclusions:body+='<ul>'+''.join('<li>'+h.esc(v)+'</li>'for v in conclusions)+'</ul>'
    legacy=[]
    for row in audit.get('R_arms',[]):
        points=row.get('full_NAVTEST_points',[]);peak=max((p.get('PDMS')for p in points if h.finite(p.get('PDMS'))),default=None)
        legacy.append([h.esc(row['id']),h.number(row.get('completed_updates')),h.fmt(peak,4),h.esc(row.get('reason_zh',''))])
    if legacy:body+='<details><summary>旧R逐路线审计：已长训、具体负证据、用户终止与未启动分别判断</summary>'+h.table(['历史路线','实际完成更新','历史完整峰值','复核结论'],legacy)+'</details>'
    body+='<p>'+label(study.get('historical_audit_path'),'完整历史预算审计')+'；旧NAVTEST已参与方法选择，本轮披露此限制；NAVTEST场景不用于训练样本选择。</p>'
    return body


def _deployment_block(study,h,label):
    value=study.get('deployment',{})
    if value.get('status')!='ALL_FOUR_REVIVAL_ARMS_ACTUALLY_LEARNING':return ''
    body='<p>正式生产回执已确认复试四臂各8rank真实学习；首32步的实际used-LR、有限梯度与group size均已核对。部署回执记录 '+h.number(value.get('distinct_physical_GPU_UUIDs'))+' 个不同GPU UUID。'+label(study['deployment_path'],'真实部署回执')+'；这是生产过程检查，不是新评分。</p>'
    rows=[]
    for r in value.get('runs',[]):
        rows.append([h.esc(r.get('id')),h.number(r.get('common_updates')),h.number(r.get('group_size')),h.number(r.get('observed_scene_exposures')),h.number(r.get('observed_unique_scenes')),h.number(r.get('observed_repeated_scenes'))])
    body+='<details><summary>部署时固定快照：原始曝光、unique与重复分别计数</summary>'+h.table(['arm','回执更新','G','scene曝光','unique','重复'],rows)+'<p class="muted">此固定审计快照与上方持续更新的训练进度时间不同；Active在112步的224次曝光只有184个unique、40次重复，不能把raw曝光比例称为已遍历人口。</p></details>'
    return body


def _resource_block(study,h,label):
    value=study.get('runtime_authorization',{})
    if not value:return ''
    bans=[name for name,row in value.get('global_bans',{}).items()if row.get('new_task_dispatch_allowed')is False]
    unbounded=value.get('unbounded_temporary_hosts',[])
    body='<details><summary>当前授权与实际准入：无到期临时授权和旧截止记录分别看</summary><p>GLOBAL禁止新派发：'+h.esc('、'.join(bans))+'。现行E record与policy '+('一致'if value.get('record_matches')else '尚未核对一致，不能推断可派发')+'；当前授权范围 '+h.number(len(value.get('authorized_now',[])))+' host（固定 '+h.number(len(value.get('permanent_hosts',[])))+'／无到期临时 '+h.number(len(unbounded))+'／有限临时 '+h.number(len(value.get('temporary_hosts',[])))+'），不等于SSH可连、空闲或已准入。</p>'
    if unbounded:body+='<p><b>当前无到期临时授权：'+h.esc('、'.join(unbounded))+'。</b>这些host不受上一批23:00截止约束；无到期临时授权不等于永久资源，实际派发仍逐次验证连接、lease／headroom／可见owner。</p>'
    if value.get('finite_deadline_hosts'):body+='<p>本次有限临时host的提前保护 '+h.esc(value.get('guard_utc'))+'；最终截止 '+h.esc(value.get('deadline_utc'))+'。</p>'
    else:body+='<p>当前grant没有有限到期host；保留的兼容expires字段不作为wm4／wm5的到期限制。其余被撤回host不因本次两台新增授权而恢复。</p>'
    body+='<p>当前长期任务授权范围 '+h.number(len(value.get('long_admit_now',[])))+' host；授权数量不是已部署数量。</p>'
    if study.get('wm45_coordination'):
        body+='<p>本轮协调决定：wm4优先用于当前评测积压恢复；wm5本次SSH连接被拒，不能凭授权视为可实际部署。新组合登记与训练另按实际回执判断。'+label(study['wm45_coordination_path'],'评测优先与资源协调收据')+'</p>'
        wm4_use=Path(study['wm45_coordination_path']).parent/'WM4_ACTUAL_GPU_USE.json'
        if wm4_use.is_file():body+='<p>实际使用快照：wm4的8张GPU均已投入IL／RL并行分片评测，15个GPU进程；这不是新组合训练，也不代表空闲lane。'+label(wm4_use,'wm4实际GPU使用与评测entrypoint证据')+'</p>'
    if value.get('grant_contains_banned'):body+='<p class="warning">当前grant仍含GLOBAL禁派host：'+h.esc('、'.join(value['grant_contains_banned']))+'；本页不将其列为可派发。</p>'
    body+='<p>'+label(value.get('record'),'实际最新task grant')+' · 核对 '+h.esc(value.get('checked_utc'))+'</p>'
    historical=value.get('historical_expiry_receipt')
    if historical and Path(historical).is_file():body+='<details><summary>上一批临时资源的截止退出：历史记录</summary>'+label(historical,'2026-11-01 23:00到期及退出验证收据')+'；该记录保留，不覆盖当前wm4／wm5的重新授权。</details>'
    body+='</details>'
    return body


def _campaign_health_block(studies,h):
    rows=[]
    for study in studies:
        for name,value in study.get('controllers',{}).items():
            if not name.endswith('CAMPAIGN'):continue
            ok=value.get('local_pid_start_matches');health='本机pid／start匹配'if ok is True else '本机身份未匹配，待确认'if ok is False else '仅读回执，未验证异机进程'
            rows.append([h.esc(study['id']),h.number(value['owner'].get('pid',value['launch'].get('pid'))),h.esc(value['launch'].get('source_commit','未记录')),h.esc(value['state'].get('status')),health])
    return '<details><summary>F／G／H当前CPU协调身份：不替代模型更新或评测完成</summary>'+h.table(['研究','CPU PID','ops source','STATE','只读身份核验'],rows)+'</details>'if rows else ''


def _improvement_cards(study,h,label):
    descriptions={
        'early_decay':('P0 · 更早收敛','20k结束cos衰减后保持floor，检验共同后期回摆是否与调度阶段有关。','不是已证LR根因。'),
        'multiscene32':('P1 · 不同场景Batch32','每rank四个不同scene，G8不变；按source查询量对齐F Control。','Adam／计算成本另列。'),
        'verified_retention':('P2 · 质量门控历史RL保持','2k后通过实际ODE质量门槛入库；退化时分配原β预算25%给历史均值保持。','额外官方ODE查询最多25%。'),
        'fp32_output':('P3 · 真FP32动作投影','末层在FP32做matmul；rollout／replay／reference／export／eval一致。','先评native0，隔离初始数值函数变化。'),
        'mixed_noise':('P4 · 固定G8混合探索','4个候选在[7,8,9]加噪，4个在[3,6,8]加噪；每个仍三slots。','64个主评分不增，支持重组不是NAV增益。'),
        'retention_early_decay':('P5 · 保持＋更早衰减','从原IL70 fresh独立验证历史RL质量保持与20k衰减终点的组合。','不相加涨幅；非最高10k nativefork。'),
    }
    cards=[]
    for run in study['runs']:
        role=run.get('improvement_role')
        if role not in descriptions:continue
        title,mechanism,boundary=descriptions[role]
        cards.append('<article><h3>'+h.esc(title)+'</h3><p>'+h.esc(mechanism)+'</p><small>'+_run_status(run,h)+'；共同更新 '+h.number(run.get('updates'))+'</small><small>'+h.esc(boundary)+'</small></article>')
    return '<div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(210px,1fr))">'+''.join(cards)+'</div>'


def _benefit_front(report_root,studies,h):
    source={s['root']:s for s in studies};standard=source.get(str(DEFAULT_ROOT))
    if not standard:return ''
    scores={(s['root'],r['arm'],r.get('updates')):r for s in studies for r in s['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'and r.get('updates',0)>0}
    all_pairs=[r|{'study_root':s['root']}for s in studies for r in s.get('paired_reviews',[])if (s['root'],r['arm'],r['updates'])in scores]
    control10=scores.get((str(DEFAULT_ROOT),'SAFETY_G8_CONTROL',10000))
    if not control10:return ''
    link=lambda row:h.relative_link(report_root,row['evidence'],'完整配对')
    ci_text=lambda ci:'['+', '.join(h.fmt(v,4)for v in ci)+']'if isinstance(ci,list)and len(ci)==2 else '未给区间'
    historical={r['reference']:r for r in all_pairs if r['kind']=='historical'and r['study_root']==str(DEFAULT_ROOT)and r['arm']=='SAFETY_G8_CONTROL'and r['updates']==10000}
    series={}
    for r in all_pairs:
        if r['kind']=='between_arms'and r['reference']=='SAFETY_G8_CONTROL'and r.get('reference_updates')==r['updates']and r.get('reference_root')==str(DEFAULT_ROOT)and r['arm']!='REVIVAL_G16':series.setdefault((r['study_root'],r['arm']),[]).append(r)
    successive=any(len(values)>=2 and all(isinstance(r['result'].get('PDMS_CI95'),list)and r['result']['PDMS_CI95'][0]>0 for r in sorted(values,key=lambda r:r['updates'])[-2:])for values in series.values())
    verdict='已见连续阶段正向配对证据，仍须检验持续保持。'if successive else '有阶段性改善，尚无持续新算法胜出。'
    body='<div class="notice"><b>'+verdict+'</b>当前新研究完整峰值 '+h.fmt(max(r['PDMS']for r in scores.values()),5)+'；'+('尚未达到92。'if max(r['PDMS']for r in scores.values())<92 else '已达到92，继续检查保持性。')+'训练候选改善与部署收益分别判断；未完成全12146场景的评测不补分。</div>'
    if 'IL70'in historical:
        pair=historical['IL70'];r=pair['result']
        body+='<p><b>RL相对IL已有实质收益：</b>统一配方Control10k完整NAVTEST '+h.fmt(control10['PDMS'],5)+'，对原IL70 '+h.fmt(r.get('PDMS_delta'),5)+' PDMS点，配对95%CI '+ci_text(r.get('PDMS_CI95'))+'。'+link(pair)+'</p>'
    if 'SafetyEP10k'in historical:
        pair=historical['SafetyEP10k'];r=pair['result'];zero=r.get('baseline_zero_denominator');after=zero-r.get('repaired_zero',0)+r.get('new_zero',0)if h.finite(zero)else None
        body+='<p><b>对已有SafetyEP的增量仍不确定：</b>Control10k对历史SafetyEP10k '+h.fmt(r.get('PDMS_delta'),5)+'，CI '+ci_text(r.get('PDMS_CI95'))+'跨0；零分 '+h.number(zero)+'→'+h.number(after)+'。'+link(pair)
        later=scores.get((str(DEFAULT_ROOT),'SAFETY_G8_CONTROL',12000))
        if later:body+=' Control12k回落到 '+h.fmt(later['PDMS'],5)+'，10k高点没有在12k保持。'
        body+='</p>'
    requests=[('REVIVAL_SATURATED_G8',5000,5000),('STANDARD_INVERSION_G8',5000,5000),('REVIVAL_DENSE_G8',10000,10000),('REVIVAL_SATURATED_G8',10000,10000),('REVIVAL_G16',5000,10000),('REVIVAL_ACTIVE_DATA_G8',5000,5000),('REVIVAL_ACTIVE_DATA_G8',10000,10000),('STANDARD_INVERSION_G8',10000,10000),('STANDARD_ENDPOINT_LR02_G8',5000,5000),('STANDARD_COMBINED_G8',10000,10000)]
    names={'REVIVAL_SATURATED_G8':'饱和信用','STANDARD_INVERSION_G8':'纠偏信用','REVIVAL_DENSE_G8':'DenseDAC','REVIVAL_G16':'G16','REVIVAL_ACTIVE_DATA_G8':'Active课程','STANDARD_ENDPOINT_LR02_G8':'Endpoint低LR','STANDARD_COMBINED_G8':'三信用组合'}
    rows=[]
    for arm,u,ref_u in requests:
        pair=next((r for r in all_pairs if r['kind']=='between_arms'and r['arm']==arm and r['updates']==u and r['reference']=='SAFETY_G8_CONTROL'and r.get('reference_updates')==ref_u and r.get('reference_root')==str(DEFAULT_ROOT)),None)
        if not pair:continue
        target=scores.get((pair['study_root'],arm,u));baseline=scores.get((str(DEFAULT_ROOT),'SAFETY_G8_CONTROL',ref_u))
        if not target or not baseline:continue
        r=pair['result'];ci=r.get('PDMS_CI95');qualifier='阶段正向证据'if isinstance(ci,list)and ci[0]>0 else '该阶段负向证据'if isinstance(ci,list)and ci[1]<0 else '区间跨0'
        budget=h.number(u)+' vs '+h.number(ref_u)+('（同候选量）'if u!=ref_u else '（同更新）')
        rows.append([h.esc(names.get(arm,arm))+'<small>'+budget+'</small>',h.fmt(target['PDMS'],5)+'／'+h.fmt(baseline['PDMS'],5),h.fmt(r.get('PDMS_delta'),5)+'<small>'+ci_text(ci)+'</small>',h.number(r.get('repaired_zero'))+'／'+h.number(r.get('new_zero')),h.fmt(r.get('fixed_control_high',{}).get('mean_delta_PDMS'),4),qualifier+'<small>'+link(pair)+'</small>'])
    if rows:body+='<details><summary>早期5／10k关键完整配对：预算与高分保持一并看</summary>'+h.table(['方法／预算','方法／Control PDMS','Δ PDMS／配对95%CI','零分修复／新增','固定Control≥.9队列均值Δ','当前证据'],rows)+'</details>'
    body+='<p>5k的饱和信用与纠偏提供正向证据；持续性要结合后续同预算配对，而非只选择最好检查点。饱和信用到10k转为负向证据，Dense10k也未确认优于Control。减少零分不保证总分或高分队列改善。</p>'
    pending=[names[a]+h.number(u)for a,u in [('STANDARD_INVERSION_G8',10000),('STANDARD_ENDPOINT_LR02_G8',5000),('STANDARD_COMBINED_G8',10000)]if not any((s['root'],a,u)in scores for s in studies)]
    if pending:body+='<p class="muted">尚未完整：'+h.esc('、'.join(pending))+'，保留为空。评测执行恢复与数值训练失败分开记录，不以部分均值替代完整成绩。</p>'
    body+='<p class="muted">均为12146场景全量、单训练seed与44物理日志配对bootstrap；历史与多阶段结果参与方法选择，未作多重比较校正。高队列由各对照Control条件选出，是保持诊断，不等同无偏总体收益。</p>'
    return '<section id="rl-benefit"><h2>RL目前的实质收益与边界</h2>'+body+'</section>'


def _potential_front(report_root,studies,h):
    standard=next((s for s in studies if Path(s['root'])==DEFAULT_ROOT),{})
    audit=standard.get('phase_potential',{})
    if not audit.get('status','').startswith('COMPLETE'):return ''
    from research_report import figure
    link=lambda path,title:h.relative_link(report_root,str(path),title)
    ci_text=lambda v:'['+', '.join(h.fmt(x,4)for x in v)+']'if isinstance(v,list)and len(v)==2 else '未给区间'
    complete=[r for s in studies for r in s['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'and h.finite(r.get('PDMS'))and r.get('updates',0)>0]
    peak=max(complete,key=lambda r:r['PDMS'],default={})
    body='<div class="notice"><b>仍有具体可检验的提升空间，但不能把阶段性收益外推为继续加步数必然变好。</b>本次已收完整最高 '+h.fmt(peak.get('PDMS',audit.get('absolute_best_completed_PDMS')),5)+'（'+h.esc(peak.get('arm','Control'))+'，'+h.number(peak.get('updates',10000))+'更新）。完整审计 '+h.number(audit.get('snapshot_complete_points'))+' 个全量检查点／'+h.number(len(audit.get('paired_phase_gains',[])))+' 组既有配对，快照 '+h.utc(audit.get('created_unix'))+'。</div>'
    pairs=audit.get('paired_phase_gains',[]);phase_rows=[]
    selected=[('REVIVAL_SATURATED_G8',u)for u in [5000,10000,14000,20000]]+[('STANDARD_INVERSION_G8',u)for u in [5000,16000,20000]]
    for arm,u in selected:
        r=next((r for r in pairs if r['arm']==arm and r['updates']==u and r.get('reference_updates')==u),None)
        if not r:continue
        ci=r.get('CI95_PDMS');verdict='正向配对证据'if isinstance(ci,list)and ci[0]>0 else '负向配对证据'if isinstance(ci,list)and ci[1]<0 else '区间跨0'
        phase_rows.append(['饱和信用'if arm=='REVIVAL_SATURATED_G8'else '纠偏信用',h.number(u),h.fmt(r.get('absolute_PDMS'),5),h.fmt(r.get('delta_PDMS'),5)+'<small>'+ci_text(ci)+'</small>',verdict])
    body+='<h3>能学到局部改善，优势还不持续</h3>'+h.table(['方法','同预算更新','方法绝对PDMS','对当点Control Δ／95%CI','证据'],phase_rows)
    body+='<p>14k／16k的正向差是优于当点Control，绝对分数仍低于Control10k历史峰值。相邻阶段翻转说明需要识别共同阶段变化与方法独有影响，不能把单个正点判为稳定胜出，也不能把一个负点判为长期无效。</p>'
    points={}
    for s in studies:
        for r in s['results']:
            if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'and h.finite(r.get('PDMS')):points.setdefault(r['arm'],{})[r['updates']]=r['PDMS']
    def draw(plt):
        fig,axes=plt.subplots(1,2,figsize=(12.8,3.8),layout='constrained')
        labels={'SAFETY_G8_CONTROL':'Control','STANDARD_INVERSION_G8':'Inversion','STANDARD_COMBINED_G8':'Combined','STANDARD_ENDPOINT_LR02_G8':'Endpoint LR02','REVIVAL_DENSE_G8':'DenseDAC','REVIVAL_SATURATED_G8':'Saturated','REVIVAL_ACTIVE_DATA_G8':'Active','REVIVAL_G16':'G16 (2x candidates/update)'}
        panels=[['SAFETY_G8_CONTROL','REVIVAL_SATURATED_G8','STANDARD_INVERSION_G8','REVIVAL_DENSE_G8'],['SAFETY_G8_CONTROL','REVIVAL_ACTIVE_DATA_G8','REVIVAL_G16','STANDARD_COMBINED_G8','STANDARD_ENDPOINT_LR02_G8']]
        colors={'SAFETY_G8_CONTROL':'#1f2937','STANDARD_INVERSION_G8':'#1769aa','STANDARD_COMBINED_G8':'#8655bd','STANDARD_ENDPOINT_LR02_G8':'#8a8a8a','REVIVAL_DENSE_G8':'#a268aa','REVIVAL_SATURATED_G8':'#15957d','REVIVAL_ACTIVE_DATA_G8':'#d97919','REVIVAL_G16':'#37a9b0'}
        for ax,arms,title in zip(axes,panels,['Stage-gain methods and Control','Other routes; budget differs for G16']):
            values=[]
            for arm in arms:
                curve=sorted(points.get(arm,{}).items())
                if not curve:continue
                ax.plot([u/1000 for u,v in curve],[v for u,v in curve],marker='o',ms=3,lw=2 if arm=='SAFETY_G8_CONTROL'else 1.4,color=colors[arm],label=labels[arm]);values.extend(v for u,v in curve)
            if values:ax.set_ylim(min(values)-.09,max(values)+.1)
            ax.set(title=title,xlabel='Optimizer updates (k)',ylabel='Complete NAVTEST PDMS');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=7,frameon=False,ncol=2)
        return fig
    body+=figure(report_root,h,'long_run_complete_navtest_curves','完整部署曲线：阶段性优势与绝对峰值分别看',draw,'点只来自完整12146场景SUMMARY，线仅连接已观察点；两面板纵轴按各自数据范围显示。x轴相同更新数不代表G16相同候选成本；同预算配对CI见表，不给未完成评测补点。')
    active=next((r for r in pairs if r['arm']=='REVIVAL_ACTIVE_DATA_G8'and r['updates']==18000 and r.get('reference_updates')==18000),{})
    fixed=next((r for r in audit.get('fixed_IL70_same_cohort_between_arms',[])if r['arm']=='REVIVAL_ACTIVE_DATA_G8'and r['updates']==18000),{})
    curve_map={r['arm']:r for r in audit.get('curves',[])}
    if active and fixed:
        masses=active.get('cohort_net_mass_PDMS',{});positive=sum(masses.get(k,0)for k in ['zero','low','middle']);fixed_rows=[]
        for arm,name in [('REVIVAL_ACTIVE_DATA_G8','Active'),('SAFETY_G8_CONTROL','Control')]:
            trajectory=next((r for r in curve_map.get(arm,{}).get('own10k_fixed_IL70_high_trajectory',[])if r.get('to_updates')==18000),{})
            contribution=fixed.get('high_contribution_vs_IL70')if arm!='SAFETY_G8_CONTROL'else fixed.get('control_high_contribution_vs_IL70')
            fixed_rows.append([name,h.fmt(contribution,5),h.fmt(trajectory.get('high_population_contribution_change'),5),h.fmt(trajectory.get('total_PDMS_change'),5)])
        body+='<h3>修复低分与保持高分有净补偿；先用同一固定群体核对</h3><p>Active18k相对当点Control修复零分 '+h.number(active.get('repaired_zero'))+'、新增 '+h.number(active.get('new_zero'))+'；按当点Control分档的zero／low／middle贡献 '+h.fmt(positive,5)+'，被其high队列 '+h.fmt(masses.get('high'),5)+' 抵消，总差 '+h.fmt(active.get('delta_PDMS'),5)+'。这个high由当点Control条件选出，不能直接当作因果遗忘。</p>'
        body+=h.table(['固定同一IL70≥.9群体（8724）','18k高群体相对IL的总分贡献','自身10→18k高群体贡献变化','自身10→18k总体PDMS变化'],fixed_rows)
        body+='<p>在真正相同的8724个固定场景上，Active18k比Control18k的高群体贡献差 '+h.fmt(fixed.get('high_global_contribution_advantage'),5)+'；但两者自身10→18k高群体下降幅度相近。Active其他固定群体的补益抵消了共同高分回落，不能说这段是Active独有严重遗忘；数据阶段、优化状态与保持能力仍有混杂。</p>'
        oracle=active.get('oracle',{})
        body+='<p class="muted">只把选中high队列的<b>净缺口</b>设零的反事实为 '+h.fmt(oracle.get('PDMS_after_net_deficit_removed'),5)+'。这不是可实现收益，不是全部负损失修复的上界，也不能据此判92不可达。</p>'
    pool=standard.get('four_route_oracle',{});strict=pool.get('strict_high_quality_oracle',{})
    if pool.get('status','').startswith('COMPLETE')and strict.get('canonical_threshold')==.9:
        body+='<h3>有限策略池确有互补行为；oracle不是新的部署分</h3><p>四个既有策略逐场景核对后，Control10k的 '+h.number(strict.get('control_zero_denominator'))+' 个零分场景中，'+h.number(strict.get('control_zero_with_any_strict_high_witness'))+' 个有其它策略的<b>NC／DAC／TTC／Comfort全通过且canonical≥.9</b>见证。仍有 '+h.number(pool.get('common',{}).get('all_four_canonical_zero'))+' 个四策略都零分；这只表示有限pool尚未提供解，不等于场景不可解。</p>'
        body+='<details><summary>互补oracle诊断卡：不是实际排行榜或可部署策略</summary><p>只允许这些严格高质量改良、其余场景保留Control的未来标签oracle为 <b>'+h.fmt(strict.get('baseline_preserving_only_strict_high_improvements_PDMS'),5)+'</b>，零分为 '+h.number(strict.get('baseline_preserving_zero_after_oracle'))+'。该构造仍会保留部分原非strict轨迹；它不是可部署选择器、不是单策略分数、不是可实现收益承诺，<b>不能据此称已达到92</b>。不同更新数也不构成同预算算法排名；各单模型见证有重叠，不能相加。</p></details>'
        body+='<p class="muted">互补行为还需要信息、信用与保持机制才能学到同一个策略，本审计没有证明这一步。NAVTEST标签仅作诊断，不写TRAIN列表或teacher bank。'+link(standard['four_route_oracle_path'],'完整四策略互补审计')+'</p>'
    training=audit.get('training_evidence',{});exposure=training.get('active_cumulative_public_summary',{})
    body+='<p class="muted">过程证据是各阶段末64更新／128个TRAIN组的限定窗口，不是全训练census；所选窗口全8rank同时PG0为0，减少局部PG0并未稳定提高部署分。Active20k公开记录 '+h.number(exposure.get('main_exposures'))+' 次曝光／'+h.number(exposure.get('distinct_seen'))+' 个unique／'+h.number(exposure.get('priority_exposures'))+' 次优先重采；rawA方差不等于已验证学习价值。</p>'
    lr=standard.get('phase_lr_observation',{});observed=[r for r in lr.get('observed_rows',[])if r.get('optimizer_update')in [10000,20000]]
    if observed:
        body+='<h3>优先改进方向与实际执行状态</h3><p>已经启用warmup＋cosine；有cos不等于已进入强收敛阶段。下面是原F Control实际optimizer.step前使用值，action与memory分开列出，相关性尚不能证明LR导致回摆。新H执行状态单列。</p>'+h.table(['实际step','action used LR','memory used LR','各自峰值比例'],[[h.number(r['optimizer_update']),f"{r['used_action_lr']:.8g}",f"{r['used_memory_lr']:.8g}",h.fmt(r['fraction_of_peak'],2,percent=True)+'%']for r in observed])
    old_lr=[]
    for study in studies:
        for row in study.get('historical_audit',{}).get('R_arms',[]):
            for r in row.get('schedule_matched_evidence',[]):old_lr.append([h.esc(row['id']),h.esc(r['label'].split('_u',1)[-1].split('_minus_',1)[0]),h.fmt(r.get('delta_PDMS'),5),ci_text(r.get('CI95_PDMS'))])
    if old_lr:body+='<details><summary>已经研究过LR：旧10k native fork的完整对照仍保留</summary>'+h.table(['旧方法','全局更新','对旧Safety同预算Δ','95%CI'],old_lr)+'<p>旧低LR12k有正向证据，14k未保持；旧局部cos12／14k区间跨0，均未超旧Safety10k。新P0是fresh IL70／warmup1000下复审并扩展调度保持问题，不是首次LR尝试，也不恢复已停止旧fork。</p></details>'
    directions=[['P0','复审并扩展调度保持问题','同IL70 fresh，20k完成cos衰减后保持floor至批准预算；对照原50k衰减，warmup1000／floor.1／峰值与其他科学配方保持。','检验完整衰减周期包；看全曲线、安全、zero与固定IL70-high保持，不只选最好点。'],['P1','扩大不同场景的优化批次','有效批次8→32个不同场景，保持每场景G8；当前是world8×每rank1场景、4inner。G16只增加同场景候选，不是场景batch扩展。','同时按更新数、fresh场景、候选查询和计算成本比较；单独验证，不能把多曝光当同预算胜出。'],['P2','质量门控的历史RL行为保持','本run TRAIN真实ODE质量通过后入库；同场景当前ODE退化时，把原β预算25%转给历史RL均值保持，fresh PG不改。','区别旧SuccessMemory固定teacher和Protected取消所有负A；2k前不激活，额外ODE／history评分成本单列。']]
    active={r['id']:r for s in studies for r in s['runs']}
    for row,arm in zip(directions,['IMPROVEMENT_EARLY_DECAY_G8','IMPROVEMENT_MULTISCENE32_G8','IMPROVEMENT_VERIFIED_RETENTION_G8']):row.append(_run_status(active[arm],h)if arm in active else '待诊断／未登记训练')
    body+=h.table(['优先级','要解决的问题','可审查的单变量方向','判据与边界','实际执行状态'],directions)
    body+='<p>减少4inner重用、分层顺序、真满分PG0组anchor门控只作后续单独诊断。现有全量顺序已经randperm打乱，不能把共同回摆直接归因于连续难块，也不能据局部非零anchor指认它造成遗忘。现有Inv／Sat长训不会因这些提案而提前停止；当前native不临时改参或重置时钟。</p>'
    body+='<p class="muted"><a href="https://arxiv.org/html/2510.13786v1">ScaleRL</a>区分prompt批次和每prompt生成数，支持按计算预算检验批次；<a href="https://arxiv.org/html/1811.11682v2">CLEAR</a>以历史行为保持配合新经验学习。它们是机制参考，不是驾驶收益复现。<a href="https://arxiv.org/html/2607.04364v3">CPO</a>只用于讨论历史分布保持；其<a href="https://github.com/MaolinLuo/CPO#important-notice-dataset-update-2026-10-12">官方2026-10-12通告</a>披露MedBookVQA答案泄漏并正重评，不据其分数宣传。</p>'
    body+='<p>'+link(standard['phase_potential_path'],'完整潜力／固定队列审计')+' · '+link(standard['phase_lr_path'],'实际LR与设计语义')+'。审计未新评分、未新bootstrap；上述方向仍是可证伪的迁移假说。</p>'
    stale=[r for s in studies for r in s['runs']if r.get('progress_freshness',{}).get('stale')]
    if stale:body+='<details open><summary>运行确认：旧RUNNING不能替代新进度</summary>'+h.table(['arm','最后共同更新','最后日志','当前解释'],[[h.esc(r['id']),h.number(r.get('updates')),h.esc(r['progress_freshness']['last_observed_utc']),'进度过期／当前运行未确认；不据此判数值或算法失败']for r in stale])+'</details>'
    return '<section id="rl-potential"><h2>阶段性收益还有提升能力吗：最新诊断与下一步</h2>'+body+'</section>'


def render_related(report_root,studies,h):
    if not studies:return ''
    from research_report import figure
    bodies=[];current_bodies=[]
    for study in studies:
        reg=study['registration'];root=Path(study['root']);runs=study['runs'];schedule=reg.get('scheduler',{});protocol=study['scheduler_protocol'];exposure=study['exposure'];population=reg.get('data_population_count',reg.get('training_population',study['registry'].get('training_population',exposure.get('population'))))
        label=lambda p,t:h.relative_link(report_root,str(p),t)
        revival=is_revival(study);improvement=is_improvement(study);science_body=''
        body='<div class="notice"><b>新的独立标准化子研究；旧constant九臂作为历史。</b>登记状态 '+h.esc(reg.get('status','入口已发现，正式登记待补'))+'。'+('科学source准备／checkout尚未完成；CPU与启动状态分别以实际回执为准，不能据此称已开训。'if reg.get('source_pending')else '是否已真实学习以worker回执和八rank共同日志为准。')+'不是原constant系列延长后的cos单因素证据；父模型相同也不消除warmup、预算和配方差异。</div>'
        if revival:body='<h3>复审历史判断</h3><p>历史1k／3k观察支持当时固定配方与预算下的结论，不等同于长期无效。本轮保留旧负面配对CI与高分退化证据，在统一配方、更充分曝光下复试；不抹掉历史结果，也不预设长训必然改善。F现有长训并行推进，复试不重复训练其Control。ABCD等用户终止路线不恢复。</p>'+body
        if revival:body+=_historical_reassessment_block(study,h,label)
        if improvement:
            body='<div class="notice"><b>H已独立登记；每臂科学source／Adam时钟分别记录，F Control只读共享。</b>CPU协调启动、child启动、实际八rank更新与完整部署成绩分别判断。登记状态 '+h.esc(reg.get('status','待登记状态'))+'。</div>'
            body+=_improvement_cards(study,h,label)
            if study.get('wm45_coordination')and not any(r['id']=='IMPROVEMENT_RETENTION_EARLY_DECAY_G8'for r in runs):
                wm4_use=Path(study['wm45_coordination_path']).parent/'WM4_ACTUAL_GPU_USE.json'
                wm4_state='wm4实测8张GPU已投入IL／RL并行评测以清理积压'if wm4_use.is_file()else 'wm4优先清理评测积压'
                body+='<p class="notice">新增wm4／wm5已获无到期临时授权；'+wm4_state+'，wm5本次SSH连接被拒。保持＋更早衰减组合仍为准备态，未登记、未派发；现有五臂与实际训练数量分别记录。'+label(study['wm45_coordination_path'],'当前协调状态')+'</p>'
            method_start=len(body)
            body+='<p>P0为同IL70 fresh：warmup1000，<b>20k完成cos衰减，之后保持0.1峰值floor</b>；初预算20k Adam，批准后才续30／50k。50k是原生预算上限，不能误画成F的整50k衰减。2k／5k／10k及后续每2k做完整NAVTEST；不因早期小差自动淘汰。</p>'
            if not any(r.get('improvement_role')=='multiscene32'for r in runs):body+='<p class="muted">Batch32科学CPU检查已准备；尚未登记为本研究运行臂，后续以独立run.source／实际worker为准，不沿用P0的source标记。</p>'
            else:body+='<p>Batch32使用独立Adam时钟：首5k Adam约对应F Control20k的数据量；500／1250／2500及后续每500 Adam评测。1250停于inner2，已采10016场景／80128候选，比名义F5000多16场景／128候选，不称严格等预算。后续7500／12500需独立批准。</p>'
            if any(r.get('improvement_role')=='verified_retention'for r in runs):
                body+='<p><b>P2：质量门控的历史RL行为保持。</b>另用原50k／warm1000配方与1×Adam参考时钟。只保存本run TRAIN中NC／DAC／TTC／Comfort全过且canonical≥.9的真实ODE行为；当前同场景ODE失去该质量或比存量分数低.01时，激活原β50预算的0.75 IL＋0.25历史RL均值保持，非激活组仍为原β50 IL。fresh PG／优势不改，25%预算不等于25%梯度强度。</p>'
                body+='<p>2k前不入库、不做额外评分；之后每collection在64个主PG候选之外最多8次fresh B1 ODE＋8次history B1复评，额外官方查询上限25%，与主候选分账。实际计数按8rank封存state.retention求和，不能将4inner重复日志相加。'+label(study.get('retention_handoff_path'),'P2科学协议与计费边界')+'；该CPU阶段文档的未派发标记不覆盖当前worker回执。</p>'
                waiting=[r for r in runs if r.get('improvement_role')=='verified_retention'and h.finite(r.get('updates'))and r['updates']<2000]
                if waiting:body+='<p class="muted">P2当前仍在2k启用门槛前；未激活是设计，不构成保持方法的效果检验。</p>'
            if any(r.get('improvement_role')=='fp32_output'for r in runs):
                body+='<p><b>真FP32末层：</b>原实现是在BF16 autocast计算action_output后才.float()；新臂明确以FP32进行末层线性投影，且采集、概率重算、冻结reference及部署均使用同一精度路径。借鉴ScaleRL的数值精度研究，但LM head不能直接等同动作投影，当前不是已证明的precision bug。相同IL权重也会改变初始执行函数，额外native0全量NAVTEST单列，不计为RL学习收益。'+label(study.get('fp32_check_path'),'FP32路径CPU证据')+'</p>'
            if any(r.get('improvement_role')=='mixed_noise'for r in runs):
                body+='<p><b>固定G8的时段混合：</b>4late[7,8,9]＋4early[3,6,8]，每候选三innovation slots，实际stoch_idx为[8,3]；仍使用原source clamped-mean proxy而非exact joint density。时段及anchor所见states同时改变，不能解释成只换噪声索引。'+label(study.get('mixed_handoff_path'),'Mixed正式科学协议')+'</p>'
                support=study.get('mixed_support',{});cohorts=support.get('cohorts',{});failure=cohorts.get('historical_failure',{});high=cohorts.get('matched_high_control',{})
                if support.get('status','').startswith('COMPLETE')and failure:
                    a=failure.get('late8',{});b=failure.get('split4late4early',{});ha=high.get('late8',{});hb=high.get('split4late4early',{})
                    body+='<p class="muted">旧θSafety10k的156个选定TRAIN失败场景中，已有G16候选重组的严格见证（审计口径） '+h.number(a.get('found_quality'))+'→'+h.number(b.get('found_quality'))+'，allcanonical0 '+h.number(a.get('canonical_allzero'))+'→'+h.number(b.get('canonical_allzero'))+'，PG0 '+h.number(a.get('PG0'))+'→'+h.number(b.get('PG0'))+'；156个高对照的见证覆盖保持，但PG0 '+h.number(ha.get('PG0'))+'→'+h.number(hb.get('PG0'))+'。这是已有候选子集重组，无新G8 forward、不是NAVTEST提分，也不是新训练过程效果。'+label(study.get('mixed_support_path'),'旧候选支持审计')+'</p>'
            if study.get('new_trials_literature'):body+='<p class="muted">'+label(study['new_trials_literature_path'],'新试验的primary文献、迁移机制及限制')+'；文献／CPU文档中的未部署状态是历史开发阶段，当前启动以worker与全部rank日志为准。</p>'
            science_body=body[method_start:];body=body[:method_start]
        else:body+='<p>同IL70 fresh，首预算 '+h.number(reg.get('initial_budget_updates'))+' 更新；2k／5k仅观测，10k第一次完整阶段评审，之后每2k完整NAVTEST。固定cos总周期 '+h.number(schedule.get('total_updates',protocol.get('horizon_updates')))+'、warmup '+h.number(schedule.get('warmup_updates'))+'、相对floor '+h.fmt(schedule.get('min_ratio'),3)+'；阶段不重置horizon。50k是调度时钟，不是未经审查的自动续满预算。</p>'
        body+='<p class="muted">相关RL实现的调度依任务而异；统一warmup＋cosine是可检验的本轮设计，未被证明为所有任务最优。</p>'
        rows=[]
        for run in runs:
            status=_run_status(run,h)
            raw=run.get('raw_scene_exposures');ratio=h.fmt(raw/population,2,percent=True)+'%'if h.finite(raw)and h.finite(population)and population else '待实际曝光'
            identity=h.esc(run['id'])
            if improvement:identity+='<small>source '+h.esc(Path(run['source']).name)+'／'+h.esc(run.get('source_commit','见本臂source'))+'</small>'
            count=h.number(run.get('updates'))
            if improvement and h.finite(run.get('updates')):count+='<small>名义F数据clock '+h.number(run['updates']*(4 if run.get('improvement_role')=='multiscene32'else 1))+'</small>'
            method=run.get('algorithm',run.get('standard_role','待方法'))
            if improvement:method={'early_decay':'更早衰减并保持floor','multiscene32':'32个不同场景的真实梯度累积','verified_retention':'质量门控的历史RL行为保持','fp32_output':'真FP32末层动作投影','mixed_noise':'固定G8：4late＋4early混合探索','retention_early_decay':'质量门控保持＋更早衰减组合'}.get(run.get('improvement_role'),method)
            rows.append([identity,h.esc(method),status+'<small>'+h.esc(run.get('host','未记录host'))+'</small>',count,h.number(raw)+'<small>'+ratio+' of eligible population</small>',h.number(run.get('new_candidates')),h.number(run.get('target_updates',reg.get('initial_budget_updates')))])
        body+=h.table(['本子研究arm','方法','实际状态','共同更新','实际scene曝光／人口','fresh候选','当前预算'],rows)
        if improvement:
            if study.get('five_arm_deployment'):body+='<p class="muted">'+label(study['five_arm_deployment_path'],'五臂独立8rank实际学习证据')+'；这里只证明真实更新，不能替代完整部署成绩。</p>'
            early=study.get('early_mixed_retention',{});mix=early.get('mixed_actual_training_prefix',{});retain=early.get('retention_actual_training_window',{})
            if mix or retain:
                evidence_rows=[]
                if mix:
                    evidence_rows.append(['Mixed首'+h.number(mix.get('optimizer_prefix_end'))+'更新／'+h.number(mix.get('groups'))+'组','early严格高质量 '+h.number(mix.get('early',{}).get('strict_high_count'))+'／'+h.number(mix.get('per_branch_candidates'))+'；late '+h.number(mix.get('late',{}).get('strict_high_count'))+'／'+h.number(mix.get('per_branch_candidates')),'early-only '+h.number(mix.get('groups_only_early_has_strict_high'))+'组、late-only '+h.number(mix.get('groups_only_late_has_strict_high'))+'组；PG0 '+h.number(mix.get('zero_PG_groups'))+'／allcanonical0 '+h.number(mix.get('allcanonicalzero_groups'))])
                if retain:
                    banks=list(retain.get('latest_observed_bank_size_by_rank',{}).values());window=retain.get('optimizer_window',[])
                    evidence_rows.append(['保持臂更新'+('–'.join(h.number(u)for u in window))+'／'+h.number(retain.get('groups'))+'组','主PG '+h.number(retain.get('mainPG_queries'))+'次；额外fresh ODE '+h.number(retain.get('extra_fresh_ODE_queries'))+'＋history '+h.number(retain.get('extra_history_ODE_queries')),'退化gate '+h.number(retain.get('regression_gate_activations'))+'组；各rank bank '+(h.number(min(banks))+'–'+h.number(max(banks))if banks else '待记录')])
                body+='<details open><summary>机制已实际启动，但尚非NAVTEST收益证明</summary>'+h.table(['限定TRAIN窗口','真实候选／官方查询','互补支持／激活'],evidence_rows)+'<p>每collection只计inner1一次；四inner复用不乘成四次新查询。Mixed是正在变化的训练策略内两分支统计，不是冻结θ因果或heldout改进；保持臂额外查询按主PG分母计，不能宣称总query与Control相等。'+label(study['early_mixed_retention_path'],'正式前缀与激活统计')+'</p></details>'
            completed=[r for r in study['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST']
            initial=[r for r in completed if r.get('updates')==0];trained=[r for r in completed if r.get('updates',0)>0]
            if study.get('fp32_snapshot_check')and any(r.get('improvement_role')=='fp32_output'for r in runs):
                fp32_initial=any(r['arm']=='IMPROVEMENT_FP32_OUTPUT_G8'for r in initial)
                note='native0完整重评已收齐；下述问题保留为评测适配历史。'if fp32_initial else '评测适配已修复／native0完整重评进行中；暂无可发布初始分数。'if study.get('fp32_gpu_parity',{}).get('status')=='PASS'else '评测适配修复／等待native0完整重评；暂无可发布初始分数。'
                body+='<p class="'+('muted'if fp32_initial else 'warning')+'"><b>FP32：'+note+'</b>辅助native_snapshot加载器曾漏读精度flag而触发parity拒绝；训练／导出此前正确，不是科学loss失败或方法无效。旧u0分片整体归档，不与新evaluation_source混拼。'+label(study['fp32_snapshot_check_path'],'独立snapshot恢复CPU回归')+'</p>'
                parity=study.get('fp32_gpu_parity',{})
                if parity.get('status')=='PASS':body+='<p class="muted">修复后实际GPU native／export一致性已PASS，normalized／requested最大误差 '+h.fmt(parity.get('normalized_max_abs'),6)+'／'+h.fmt(parity.get('requested_max_abs'),6)+'；完整重评未齐前不填分。'+label(study['fp32_gpu_parity_path'],'GPU parity原始回执')+'</p>'
                evaluation_sources=list(dict.fromkeys(str(r['evaluation_source'])for r in runs if r.get('improvement_role')=='fp32_output'and r.get('evaluation_source')))
                if evaluation_sources:body+='<p class="muted">FP32独立评测适配source：'+' · '.join(label(p,Path(p).name)for p in evaluation_sources)+'；原训练source与child不变。</p>'
            if initial:body+='<h3>初始执行函数检查：不计为RL学习收益</h3>'+h.table(['arm','Adam更新','场景','初始PDMS','证据'],[[h.esc(r['arm']),'0（未优化）',h.number(r.get('scenes')),h.fmt(r.get('PDMS'),6),label(r.get('evidence'),'SUMMARY')]for r in initial])
            if trained:body+='<h3>H完整NAVTEST：只发布全量结果</h3>'+h.table(['arm','Adam更新','场景','完整PDMS','证据'],[[h.esc(r['arm']),h.number(r.get('updates')),h.number(r.get('scenes')),h.fmt(r.get('PDMS'),6),label(r.get('evidence'),'SUMMARY')]for r in trained])
            else:body+='<div class="notice"><b>H目前没有优化后的完整NAVTEST成绩。</b>原生训练、初始函数检查、checkpoint导出与分片评测均不算提分证据；等待12146场景完整SUMMARY，不发布部分均分。</div>'
            body+='<details><summary>各臂的机制、时钟、初始函数与额外评分成本</summary>'+science_body+'</details>'
            mapping=study.get('algorithm_mapping',{})
            if mapping:body+='<details><summary>算法发展与当前问题如何对应</summary><p>本轮优先验证P0更早收敛、P1真正扩大不同场景批、P2质量门控历史RL保持。GRPO家族文献逐项区分已吸收机制、已有历史测试与后续假说；论文新不等于部署收益，CPU／实现映射不覆盖当前worker与完整评分状态。</p>'+label(study['algorithm_mapping_path'],'正式算法—问题对应及primary来源')+'</details>'
            body+=_resource_block(study,h,label)+_campaign_health_block(studies,h)
            handoff=study.get('consolidation_handoff',{});samples=handoff.get('scheduler_clock',{}).get('samples',{})
            if samples:
                planned_rows=[]
                for u in [0,1000,10000,20000,30000,50000]:
                    p=samples.get(str(u),{})
                    if p:planned_rows.append([h.number(u),h.fmt(p.get('factor'),4),f"{p['action_next_lr']:.7g}",f"{p['memory_next_lr']:.7g}"])
                body+='<details><summary>P0登记计划：20k结束衰减后保持floor（不是实际观测）</summary>'+h.table(['已完成Adam u','下一step倍率','action next LR','memory next LR'],planned_rows)+'<p>计划来自本臂科学协议；真实used LR在下方另列，不借F的50k衰减曲线。</p>'+label(study['consolidation_handoff_path'],'P0正式协议／CPU检查')+'</details>'
            common=study.get('common10k_audit',{})
            if common.get('status','').startswith('COMPLETE'):
                body+='<details><summary>共同10k回落配置审计：未发现reset，尚非根因证明</summary><p>8臂指定窗口共96,000日志行／24,000组，16份checkpoint仅检查mmap元数据；报告错误 '+h.number(common.get('total_errors'))+'。所查LR／Adam step／nexttoken／Active bank与RNG保持连续，未见共同reset。原F已经使用50k cosine，但第10k实际used LR仍约92.7%峰值；这不证明LR造成回落。未发现错误不等于证明没有错误；该时段未记录PG与anchor独立组件梯度，不能用标量loss比例指认原因。</p>'+label(study['common10k_audit_path'],'正式配置／恢复审计')+'</details>'
        body+=_deployment_block(study,h,label)
        if study.get('validator_recoveries'):
            body+='<details><summary>历史验证器故障与恢复证据；当前状态按当前job回执</summary><p>Combined与Inversion的离散统计在GPU极小非零数／CPU零数下曾触发验证拒绝；原失败保留，验证器定向修复后按原生状态恢复。它们是执行验证故障，不能记为算法负收益。</p>'+h.table(['恢复记录','定向检查','验证证据','原生身份迁移'],[[h.esc(r['name']),h.esc(r.get('status')),label(r['evidence'],'等价检查'),label(r['migration'],'迁移回执')]for r in study['validator_recoveries']])+'</details>'
        if study.get('shared_controls'):
            control_rows=[]
            for ref in study['shared_controls']:
                complete=sorted((r for r in ref['results']if r.get('cohort')=='NAVTEST'),key=lambda r:r.get('updates')or 0)
                if complete:
                    for r in complete:control_rows.append([h.esc(ref['arm']),h.esc(ref['study_id']),h.number(r.get('updates')),h.fmt(r.get('PDMS'),5),label(r.get('evidence'),'完整SUMMARY')])
                else:control_rows.append([h.esc(ref['arm']),h.esc(ref['study_id']),'待完整结果','不补点',label(ref['registration_path'],'来源登记')])
            body+='<details><summary>共享Control：引用来源子研究，未重复训练</summary>'+h.table(['Control','来源namespace','更新','完整NAVTEST PDMS','证据'],control_rows)+'<p class="muted">这些是来源子研究已有完整评测；不计为本复试新生成成绩，配对结论仍需同预算完整输入与固定协议。</p></details>'
        body+=_production_prefix_block(study,h,label)
        controllers=[]
        for name,value in study['controllers'].items():controllers.append([h.esc(name),h.esc(value['state'].get('status','待状态')),h.esc(value['owner'].get('host')or value['launch'].get('host')or value['state'].get('host')or'未记录'),h.number(value['owner'].get('pid',value['launch'].get('pid')))])
        if controllers:body+='<details><summary>独立CPU协调记录，不等于GPU学习</summary>'+h.table(['owner','文件状态','host','PID'],controllers)+'</details>'
        history=_lr_history(report_root,study,h);planned=protocol.get('planned_curve_not_actual_observations',[])
        reuse_plan=revival and Path(study.get('scheduler_protocol_path',''))==DEFAULT_ROOT/'SCHEDULER_PROTOCOL.json'
        if reuse_plan:body+='<p class="muted">复试共用上方F登记的全周期计划，不重复绘制相同计划图；下表使用本复试各臂真实日志中的LR，未借用F观测值。</p>'
        if planned and not reuse_plan:
            def draw(plt):
                fig,axes=plt.subplots(2,2,figsize=(12.8,6.2),layout='constrained')
                colors=['#1769aa','#15957d','#8655bd','#d97919']
                observed=[p['completed_before_update']for values in history.values()for p in values]
                zoom_end=max(observed,default=0);zoom_end=max(10,zoom_end+max(5,zoom_end*.05))
                for col,(field,title)in enumerate(zip(['action_lr','memory_lr'],['Action LoRA LR','Scene / predictive memory LR'])):
                    plan_ax,actual_ax=axes[0,col],axes[1,col]
                    planned_seen=set()
                    for i,run in enumerate(runs):
                        peak=run.get('scheduler',{}).get('peak_'+field)
                        if h.finite(peak)and peak not in planned_seen:
                            for ax in [plan_ax,actual_ax]:ax.plot([p['completed_updates']for p in planned],[p['multiplier']*peak for p in planned],ls='--',color='#94a3b8',lw=1,label='Plan peak '+f'{peak:.1e}')
                            planned_seen.add(peak)
                        points=[p for p in history.get(run['id'],[])if h.finite(p.get(field))]
                        if points:actual_ax.scatter([p['completed_before_update']for p in points],[p[field]for p in points],s=9,color=colors[i%len(colors)],label=run['id'])
                    plan_ax.set(title=title+' / registered full cycle',xlabel='Planned completed optimizer updates u',ylabel='Planned LR')
                    actual_ax.set(title=title+' / actual observed range',xlabel='Completed optimizer updates u; LR used by u+1',ylabel='Actual used LR',xlim=(0,zoom_end))
                    visible=[p[field]for values in history.values()for p in values if h.finite(p.get(field))]
                    if visible:actual_ax.set_ylim(0,max(visible)*1.12)
                    else:actual_ax.text(.5,.5,'No actual LR observations yet',transform=actual_ax.transAxes,ha='center',fontsize=9)
                    for ax in [plan_ax,actual_ax]:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=6,frameon=False)
                return fig
            name='standard_lr_'+re.sub('[^A-Za-z0-9_]+','_',study['id'])
            body+=figure(report_root,h,name,'统一调度：全周期计划与当前实际LR分行展示',draw,'上排灰虚线仅登记全周期计划；下排放大当前实际观察范围，彩色点仅来自已提交native日志。action与memory分列，纵轴按各自尺度。u为已完成更新，使用值是下一更新的LR；不把缺失段插值成观测。')
        actual_count=sum(len(v)for v in history.values())
        body+='<p>当前已保存 '+h.number(actual_count)+' 个rank0实际step-used LR观察点；'+('暂无原生日志时不把计划曲线称为实际。'if not actual_count else '若旧日志只给post-step两组值，使用上一行next值并核对当前action使用值；缺失memory留空。')+'</p>'
        latest_lr=[]
        for run in runs:
            points=history.get(run['id'],[])
            if points:
                point=points[-1]
                latest_lr.append([h.esc(run['id']),h.number(point['optimizer_update']),f"{point['action_lr']:.7g}"if h.finite(point.get('action_lr'))else '未观测',f"{point['memory_lr']:.7g}"if h.finite(point.get('memory_lr'))else '未观测'])
        if latest_lr:body+=h.table(['最近真实使用LR','实际更新序号','action LoRA','scene／predictive memory'],latest_lr)
        if improvement:body+='<p class="muted">P0的u是已完成Adam，u1000达峰；Batch32另用warmup250／Adam时钟，注册后分别显示。used LR与post-step next LR相差一步，不能混用。</p>'
        else:body+='<p class="muted">warmup时第1000次step使用u999倍率0.999001，第1001次step才使用u1000峰值；不能把post-step的下一次LR错配给当前更新。</p>'
        milestones=exposure.get('first_pass_counts',[])
        if milestones:
            body+='<details><summary>曝光预算与eligible人口：登记推算，不代表掌握</summary>'+h.table(['optimizer更新','fresh scene曝光','候选','首遍人口比例'],[[h.number(p.get('optimizer_updates')),h.number(p.get('fresh_scene_exposures')),h.number(p.get('candidates')),h.fmt(p.get('first_pass_fraction'),2,percent=True)+'%']for p in milestones])+'</details>'
            if root==DEFAULT_ROOT:body+='<p>人口 '+h.number(population)+'；10k仅约19.36%首遍曝光，50k约96.82%，完整首遍需 '+h.number(exposure.get('full_pass_completed_update'))+' 更新。曝光不是掌握，重复曝光也不等于unique覆盖。u10k next-LR倍率约0.92714，因此10k只检验首段趋势，不能称完整cos周期结论。</p>'
            else:body+='<p>曝光数／eligible人口不自动代表unique覆盖；G16候选量、课程重采和各臂预算需分别计数。曝光不是掌握。</p>'
        complete=[r for r in study['results']if r.get('publishable_complete')]
        if complete and not improvement:body+=h.table(['本子研究完整结果','更新','cohort','场景','PDMS','证据'],[[h.esc(r.get('arm','')),h.number(r.get('updates')),h.esc(r.get('cohort','')),h.number(r.get('scenes')),h.fmt(r.get('PDMS'),5),label(r.get('evidence',''),'SUMMARY')]for r in complete])
        elif not improvement:body+='<p>本子研究尚无完整新成绩，不用旧constant成绩补点。</p>'
        protocol_link=study.get('consolidation_handoff_path')if improvement else study.get('scheduler_protocol_path',root/'SCHEDULER_PROTOCOL.json')
        body+='<p>'+label(study['registration_path'],'实际REGISTRATION')+' · '+label(protocol_link,'科学时钟／计划证据')+' · '+label(study.get('exposure_path',root/'TRAINING_EXPOSURE.json'),'曝光账本')+'</p>'
        cpu_paths=[root/'analysis'/'STANDARDIZED_TRAINING_HANDOFF.json',root/'analysis'/'STANDARDIZED_TRAINING_CPU_CHECK.json']
        existing=[p for p in cpu_paths if p.is_file()]
        if existing:body+='<p class="muted">'+ ' · '.join(label(p,'科学协议／CPU检查'if i==0 else '定向CPU检查')for i,p in enumerate(existing))+'；CPU检查不代表已获得部署收益。</p>'
        title=reg.get('display_name')or('复审历史判断：统一配方下的方法复试'if revival else '当前标准化子研究：统一warmup＋cosine与更充分训练')
        if improvement and any(r.get('improvement_role')in ['fp32_output','mixed_noise']for r in runs):title='H当前研究：调度、场景批、保持、精度与探索'
        document='<section id="'+section_id(study)+'"><h2>'+h.esc(title)+'</h2><p class="deck">独立父身份／输出命名空间；配置、启动、真实LR与完整评测分层记录。</p>'+body+'</section>'
        (current_bodies if improvement else bodies).append(document)
    import importlib,latest_complete_report
    importlib.reload(latest_complete_report)
    latest=latest_complete_report.build_review(studies)
    latest_path=report_root/'analysis'/'REPORT_LATEST_COMPLETE_REVIEW_20261101'/'SUMMARY.json'
    h.atomic_json(latest_path,latest)
    return latest_complete_report.render_review(report_root,latest,latest_path,h)+''.join(current_bodies)+_potential_front(report_root,studies,h)+_benefit_front(report_root,studies,h)+''.join(bodies)
