"""Read-only namespaces for related RL studies; local report artifacts only."""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import re


E_ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031')
DEFAULT_ROOT=E_ROOT/'standard_training_20261101'


def is_revival(study):
    return study.get('registration',{}).get('study_kind')=='historical_revival'or Path(study['root']).name.startswith('revival_')


def section_id(study):
    if Path(study['root'])==DEFAULT_ROOT:return 'standard-training'
    name=re.sub('[^A-Za-z0-9_-]+','-',study.get('id',Path(study['root']).name))
    return 'related-study-'+name+'-'+hashlib.sha256(study['root'].encode()).hexdigest()[:8]


def section_links(studies):
    return [(section_id(s),'历史方法复试'if is_revival(s)else '统一配方新研究'if Path(s['root'])==DEFAULT_ROOT else s.get('registration',{}).get('display_name',s['id']))for s in studies]


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
            folder=Path(path).parent;controllers[folder.name]=dict(state=state,state_path=path,
                owner=h.read_json(folder/'OWNER.json',warnings,inputs)or{},launch=h.read_json(folder/'LAUNCH.json',warnings,inputs)or{})
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
    by_root={s['root']:s for s in studies}
    for study in studies:
        shared=[]
        for ref in h.object_rows(study['registration'].get('shared_controls',[]),'shared_controls'):
            source=by_root.get(ref.get('study_root'));arm=ref.get('arm')
            if not source or not arm:continue
            shared.append(dict(study_id=source['id'],study_root=source['root'],registration_path=source['registration_path'],arm=arm,
                results=[r for r in source['results']if r.get('arm')==arm and r.get('publishable_complete')]))
        study['shared_controls']=shared
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


def render_related(report_root,studies,h):
    if not studies:return ''
    from research_report import figure
    bodies=[]
    for study in studies:
        reg=study['registration'];root=Path(study['root']);runs=study['runs'];schedule=reg.get('scheduler',{});protocol=study['scheduler_protocol'];exposure=study['exposure'];population=reg.get('data_population_count',exposure.get('population'))
        label=lambda p,t:h.relative_link(report_root,str(p),t)
        revival=is_revival(study)
        body='<div class="notice"><b>新的独立标准化子研究；旧constant九臂作为历史。</b>登记状态 '+h.esc(reg.get('status','入口已发现，正式登记待补'))+'。'+('科学source准备／checkout尚未完成；CPU与启动状态分别以实际回执为准，不能据此称已开训。'if reg.get('source_pending')else '是否已真实学习以worker回执和八rank共同日志为准。')+'不是原constant系列延长后的cos单因素证据；父模型相同也不消除warmup、预算和配方差异。</div>'
        if revival:body='<h3>复审历史判断</h3><p>历史1k／3k观察支持当时固定配方与预算下的结论，不等同于长期无效。本轮保留旧负面配对CI与高分退化证据，在统一配方、更充分曝光下复试；不抹掉历史结果，也不预设长训必然改善。F现有长训并行推进，复试不重复训练其Control。ABCD等用户终止路线不恢复。</p>'+body
        if revival:body+=_historical_reassessment_block(study,h,label)
        body+='<p>同IL70 fresh，首预算 '+h.number(reg.get('initial_budget_updates'))+' 更新；2k／5k仅观测，10k第一次完整阶段评审，之后每2k完整NAVTEST。固定cos总周期 '+h.number(schedule.get('total_updates',protocol.get('horizon_updates')))+'、warmup '+h.number(schedule.get('warmup_updates'))+'、相对floor '+h.fmt(schedule.get('min_ratio'),3)+'；阶段不重置horizon。50k是调度时钟，不是未经审查的自动续满预算。</p>'
        body+='<p class="muted">相关RL实现的调度依任务而异；统一warmup＋cosine是可检验的本轮设计，未被证明为所有任务最优。</p>'
        rows=[]
        for run in runs:
            if run.get('worker_returncode')==75:status='资源准入等待（非科学失败）'
            elif run.get('worker_returncode')not in(None,0):status='执行失败待审，原件保留'
            elif run.get('worker_receipt_status')=='RUNNING'and run.get('journal_ranks')==8 and h.finite(run.get('updates'))and run['updates']>0:status='实际八rank优化已见'
            elif run.get('worker_receipt_status')=='RUNNING':status='child启动，待实际更新'
            elif run.get('worker_returncode')==0:status='本原生阶段完成；完整评测另列'
            elif run.get('journal_ranks')==8 and h.finite(run.get('updates'))and run['updates']>0:status='已有八rank更新；当前派发待child确认'
            else:status='已配置，未见实际启动'
            raw=run.get('raw_scene_exposures');ratio=h.fmt(raw/population,2,percent=True)+'%'if h.finite(raw)and h.finite(population)and population else '待实际曝光'
            rows.append([h.esc(run['id']),h.esc(run.get('algorithm',run.get('standard_role','待方法'))),status+'<small>'+h.esc(run.get('host','未记录host'))+'</small>',h.number(run.get('updates')),h.number(raw)+'<small>'+ratio+' of eligible population</small>',h.number(run.get('new_candidates')),h.number(run.get('target_updates',reg.get('initial_budget_updates')))])
        body+=h.table(['本子研究arm','方法','实际状态','共同更新','实际scene曝光／人口','fresh候选','当前预算'],rows)
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
        body+='<p class="muted">warmup时第1000次step使用u999倍率0.999001，第1001次step才使用u1000峰值；不能把post-step的下一次LR错配给当前更新。</p>'
        milestones=exposure.get('first_pass_counts',[])
        if milestones:
            body+='<details><summary>曝光预算与eligible人口：登记推算，不代表掌握</summary>'+h.table(['optimizer更新','fresh scene曝光','候选','首遍人口比例'],[[h.number(p.get('optimizer_updates')),h.number(p.get('fresh_scene_exposures')),h.number(p.get('candidates')),h.fmt(p.get('first_pass_fraction'),2,percent=True)+'%']for p in milestones])+'</details>'
            if root==DEFAULT_ROOT:body+='<p>人口 '+h.number(population)+'；10k仅约19.36%首遍曝光，50k约96.82%，完整首遍需 '+h.number(exposure.get('full_pass_completed_update'))+' 更新。曝光不是掌握，重复曝光也不等于unique覆盖。u10k next-LR倍率约0.92714，因此10k只检验首段趋势，不能称完整cos周期结论。</p>'
            else:body+='<p>曝光数／eligible人口不自动代表unique覆盖；G16候选量、课程重采和各臂预算需分别计数。曝光不是掌握。</p>'
        complete=[r for r in study['results']if r.get('publishable_complete')]
        if complete:body+=h.table(['本子研究完整结果','更新','cohort','场景','PDMS','证据'],[[h.esc(r.get('arm','')),h.number(r.get('updates')),h.esc(r.get('cohort','')),h.number(r.get('scenes')),h.fmt(r.get('PDMS'),5),label(r.get('evidence',''),'SUMMARY')]for r in complete])
        else:body+='<p>本子研究尚无完整新成绩，不用旧constant成绩补点。</p>'
        body+='<p>'+label(study['registration_path'],'实际REGISTRATION')+' · '+label(study.get('scheduler_protocol_path',root/'SCHEDULER_PROTOCOL.json'),'时钟／计划曲线')+' · '+label(study.get('exposure_path',root/'TRAINING_EXPOSURE.json'),'曝光账本')+'</p>'
        cpu_paths=[root/'analysis'/'STANDARDIZED_TRAINING_HANDOFF.json',root/'analysis'/'STANDARDIZED_TRAINING_CPU_CHECK.json']
        existing=[p for p in cpu_paths if p.is_file()]
        if existing:body+='<p class="muted">'+ ' · '.join(label(p,'科学协议／CPU检查'if i==0 else '定向CPU检查')for i,p in enumerate(existing))+'；CPU检查不代表已获得部署收益。</p>'
        title=reg.get('display_name')or('复审历史判断：统一配方下的方法复试'if revival else '当前标准化子研究：统一warmup＋cosine与更充分训练')
        bodies.append('<section id="'+section_id(study)+'"><h2>'+h.esc(title)+'</h2><p class="deck">独立父身份／输出命名空间；配置、启动、真实LR与完整评测分层记录。</p>'+body+'</section>')
    return ''.join(bodies)
