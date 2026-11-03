"""Read-only living performance report. No GPU imports, dispatch or score creation."""
from datetime import datetime, timezone
import argparse
import html
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')
LEGACY_ROOT=Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102')
D46_ROOT=Path('/mnt/project/d46-priority-rl-20261103')
esc=lambda x:html.escape(str(x),quote=True)


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def atomic_text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing-report-'+str(os.getpid()));tmp.write_text(value,encoding='utf-8');os.replace(tmp,path)


def D46_priority_fragment():
    """A status receipt is required; a new directory or config proves no training."""
    path=D46_ROOT/'PRIORITY_STATUS.json';value=read(path,{})
    body='<section id="d46-priority" style="background:#edf7f5;border:2px solid #428b7a;border-radius:14px;padding:24px;margin:20px 0"><small>当前研究优先级</small><h2>D 框架 · IL46k 父策略优先 RL</h2>'
    if not value:
        return body+'<p>等待 D46 正式状态收据；尚未确认启动。既有 IL70 路线保存让位属于研究优先级调整，不判算法无效。</p></section>'
    stamp=value.get('updated_at',value.get('at',value.get('generated_UTC','未记录')))
    if isinstance(stamp,(int,float)):stamp=datetime.fromtimestamp(stamp,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    status=value.get('status','状态待核对')
    label={'PREPARING_EXACT_IL46_POLICY_PARENT_AND_FRESH_RUNS':'准备固定 IL46k 策略与独立 fresh RL，尚未启动',
        'D46_FRESH_IL46_PRIORITY_CAMPAIGN':'D46 六路线已登记，资格与正式训练按各行实际状态区分'}.get(status,status)
    body+='<p><b>最新状态：'+esc(label)+'</b>；收据时间 '+esc(stamp)+'。<a href="'+esc(path)+'">实际优先级与执行状态</a></p>'
    count=value.get('actual_training')
    if isinstance(count,int)and count>=0:body+='<p><b>D46 实际正式训练数量：'+esc(count)+' 条</b>（按该时间的实际核对收据，资格训练与注册数量不计入）。</p>'
    restored_path=D46_ROOT/'operations/RESTORE_SIX_UNBOUNDED_20261103/APPLIED.json'
    restored=read(restored_path,{})
    if restored.get('finite_deadline_applies') is False:
        body+='<p><b>当前资源：六机已对 D46 重新开放，无到期截止。</b>'+esc(' / '.join(restored.get('hosts',[])))+'。<a href="'+esc(restored_path)+'">新授权实际应用</a>。下方原 IL70 的00:00到期和保存退出记录保留为历史，不代表本 D46 新授权已到期；其它任务权限不扩张。</p>'
    parent=value.get('parent',{})
    if parent.get('PDMS') is not None:
        body+='<p>固定 IL46k 父策略完整 NAVTEST：'+esc(f'{parent["PDMS"]:.8f}')+'。这是 IL 起点，不是新 RL 收益。'
        if parent.get('export_sha256'):body+='部署策略 SHA '+esc(parent['export_sha256'])+'。'
        body+='</p>'
    rows=value.get('runs',[])
    if isinstance(rows,dict):rows=[dict(v,id=k)for k,v in rows.items()]
    methods={
        'D3_FPO':('MC4 CFM-energy ratio PPO代理','非精确轨迹似然；clip .01，保留原生KL约束'),
        'D3_AWM':('正优势加权 matching + 参考保持','保守本地适配；非完整 signed AWM，matching不使用PPO clip'),
        'D5':('扩大 D 原生语义/动作适配范围','text LoRA r32 + last8 FiLM低秩 + action I/O/position'),
        'D1':('两阶段语义/终点质量信用','2 exposure epochs后引入Comfort与TRAIN GT endpoint质量；标签不进当前观测'),
        'D2':('真实 B1 对照的 verified teacher 吸收','0.5 exposure epoch后启用；严格安全且较B1高≥.02，clean-x +真实ODE-last'),
        'D0':('D 原生 joint24 Gaussian PPO 基线','clip .1 +自适应 native KL；保存状态条件转移密度，非最终轨迹边缘密度')}
    campaign=read(D46_ROOT/'CAMPAIGN.json',{}).get('runs',{})
    if rows:
        entries=[]
        states={'RUNNING_QUALIFICATION':'八卡资格进行中，非正式训练','WAITING_RESOURCE':'等待资源','RUNNING':'运行中（以实际日志核对）'}
        for r in rows:
            arm=r.get('id','').removeprefix('D46_');registered=campaign.get(arm,{})
            history=registered.get('qualification_jobs',[])if r.get('status')=='RUNNING_QUALIFICATION'else registered.get('jobs',[])
            host=r.get('host')or(history[-1].get('host')if history else None)or'待分配'
            method,gate=methods.get(arm,('以登记配方为准',''))
            entries.append((r.get('id',''),method,gate,host,r.get('updates')if r.get('updates')is not None else'尚无实际日志',states.get(r.get('status'),r.get('status','待确认'))))
        body+=table(['D46 路线','方法','阶段启用与边界','服务器','正式更新（收据）','状态'],entries)
    body+='<p>六条专用路线优先顺序为 D3_FPO / D3_AWM / D5 / D1 / D2 / D0。共同 B64 / G16 / inner2；各自先完成真实八卡10更新资格，再从相同完整 IL46 部署策略与新空 Adam 独立正式起步，绝不继承资格权重或 D44 RL 状态。它们适配 D 原生 Qwen 架构，不能写成把原 IL70 Safety 配方换父权重。<a href="'+esc(D46_ROOT/'CAMPAIGN.json')+'">六路线登记与当前阶段</a> · <a href="'+esc(D46_ROOT/'PARENT_CONTRACT_v2.json')+'">父策略合同v2</a></p>'
    saved_path=D46_ROOT/'operations/USER_PRIORITY_20261103/D44_ACTUAL_NATIVE_SAVED.json'
    saved=read(saved_path,{})
    if saved.get('old_campaign_closed') is True:
        body+='<p>旧 D44 已正常保存并退出：'+esc('；'.join(str(r['arm'])+' / '+str(r['updates'])+' / inner'+str(r['next_inner'])for r in saved.get('saved_runs',[])if r.get('returncode')==0))+'。<a href="'+esc(saved_path)+'">实际原生保存收据</a>；状态未转入 D46，不判算法无效。</p>'
    body+='<p>D4 / D6 依赖的原生 readout 缺失，仍暂缓。保留的部署策略和零差导出核对不等于原完整 IL native 仍存在。模型初始化、资格、已派发与实际正式八 rank 更新分开记录；未完整评测不发布新 RL 分数。既有 IL70 代码、结果与保存状态保留，资源让位不判方法无效。</p></section>'
    return body


def reviewed_I_fragment(root):
    """Consume only the complete sealed population audit and its exact fragment."""
    directory=Path(root)/'analysis/INTEGRATED_I_u002000_FULL_REVIEW'
    handoff=read(directory/'HTML_HANDOFF.json',{});summary=read(directory/'SUMMARY.json',{})
    body=dict(summary);digest=body.pop('sha256',None)
    if (summary.get('status')!='COMPLETE' or summary.get('population',{}).get('scenes')!=12146
            or summary.get('population',{}).get('physical_logs')!=44
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest
            or handoff.get('summary_sha256')!=digest):return ''
    try:raw=(directory/'EMBED_SUMMARY.html').read_bytes()
    except OSError:return ''
    if hashlib.sha256(raw).hexdigest()!=handoff.get('fragment_sha256'):return ''
    text=raw.decode();start=text.find('<figure');end=text.rfind('</section>')
    if start>=0 and end>start:
        text=text[:start]+'<details><summary>查看完整分数分布、逐场景转移、增损及物理 log 三张图</summary>'+text[start:end]+'</details>'+text[end:]
    return '<style>.integrated-i2000-review svg{max-width:100%;height:auto}</style>'+text


def tail(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0,Path(path).stat().st_size-2**20))
            rows=f.read().splitlines()
        for row in reversed(rows):
            try:return json.loads(row)
            except ValueError:pass
    except OSError:pass
    return {}


def journal_rows(path, limit=1024, max_bytes=2**21):
    """Bounded metadata read; discard a possibly partial last JSON line."""
    try:
        with Path(path).open('rb') as f:
            start=max(0,Path(path).stat().st_size-max_bytes);f.seek(start)
            lines=f.read().splitlines()
        if start:lines=lines[1:]
        rows=[]
        for line in lines[-limit:]:
            try:
                row=json.loads(line)
                if isinstance(row,dict):rows.append(row)
            except ValueError:pass
        return rows
    except OSError:return []


def registered_plans(root, registry):
    """Read JSON identity/projection only; never import YAML, Torch or runtime."""
    projection=read(root/'reporting/registered_plans.json',{}).get('runs',{})
    plans={}
    for run in registry.get('runs',[]):
        arm=run['id'];identity_path=Path(run['root'])/'identity.json'
        identity=read(identity_path,{})
        cfg=identity.get('recipe',{})
        if cfg.get('research',{}).get('arm')!=arm:
            item=projection.get(arm,{})
            try:actual=hashlib.sha256(Path(run['config']).read_bytes()).hexdigest()
            except OSError:actual=None
            cfg=item.get('recipe',{}) if actual==run.get('config_sha256')==item.get('config_sha256') else {}
        spec=cfg.get('research',{}).get('performance')
        if spec:plans[arm]=dict(spec=spec,config=run['config'],config_sha256=run.get('config_sha256'),identity=str(identity_path) if identity else None)
    return plans


def verified_quality_evidence(root):
    """Compact sealed qualification metadata; never load native tensors."""
    directory=root/'operations/S6_NATIVE_QUALIFICATION_LANE_V2'
    receipt=read(directory/'COMPLETE_RECEIPT.json',{})
    proof_path=root/'qualification/S6_VERIFIED_QUALITY_NATIVE_V2/COMPLETE.json'
    gradient_path=directory/'GRADIENT_READOUT.json'
    proof=read(proof_path,{});gradient=read(gradient_path,{})
    try:
        proof_sha=hashlib.sha256(proof_path.read_bytes()).hexdigest()
        gradient_sha=hashlib.sha256(gradient_path.read_bytes()).hexdigest()
    except OSError:return {}
    if (receipt.get('status')!='PASS_QUALIFICATION_ONLY' or proof.get('status')!='PASS'
            or proof.get('schema')!='world_verified_quality_actual_world8_native_qualification_v1'
            or (proof.get('world'),proof.get('B'),proof.get('G'),proof.get('M'))!=(8,64,16,4)
            or not proof.get('native_bitexact_at_inner2_and4')
            or receipt.get('complete_sha256')!=proof_sha or receipt.get('gradient_readout_sha256')!=gradient_sha
            or gradient.get('qualification_complete_sha256')!=proof_sha
            or gradient.get('status')!='COMPLETE_ACTUAL_ALL8_RANKS_FOUR_UPDATES_NOT_PERFORMANCE'):
        return {}
    rank0=next((r for r in gradient.get('final_rank_readouts',[]) if r.get('rank')==0),{})
    return dict(proof=str(proof_path),proof_sha256=proof_sha,gradient=str(gradient_path),
        gradient_sha256=gradient_sha,source_commit=receipt.get('source_commit'),
        parameter_count=proof.get('trainable_numel'),teacher_classes=proof.get('actual_first_collection_teacher_classes',{}),
        summary=gradient.get('summary',{}),rank0=rank0)


def train_GT_evidence(root):
    """Only compact aggregate fields from the completed TRAIN-only CPU audit."""
    path=root/'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT/RESEARCH_READOUT.json'
    value=read(path,{})
    if value.get('status')!='COMPLETE_DISCOVERY_ONLY_NOT_TRAINING':return {}
    population=value.get('population',{});budget=value.get('query_budget',{})
    if (population.get('input_count')!=2048 or population.get('valid_whole_trajectory_count')!=2035
            or population.get('NAVTEST_trajectory_or_label_read') is not False
            or budget.get('training_updates')!=0 or budget.get('GPU_model_forwards')!=0):return {}
    cohorts=read(path.with_name('SUMMARY.json'),{}).get('cohort_statistics',{})
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        population={k:population.get(k) for k in ('input_count','valid_whole_trajectory_count','invalid_time_count',
            'physical_log_count','source_TRAIN_count','old_DEV_in_existing_NAVTRAIN_count')},
        budget={k:budget.get(k) for k in ('actual','training_updates','GPU_model_forwards')},
        opportunity=value.get('verified_opportunity',{}),roundtrip=value.get('normalization_roundtrip',{}),
        GT_worse_all=cohorts.get('all',{}).get('GT_score_worse_lt_minus005'),
        GT_worse_strict_high=cohorts.get('B1_strict_high',{}).get('GT_score_worse_lt_minus005'),
        GT_comfort_not_one=population['valid_whole_trajectory_count']-value.get('quality',{}).get('comfort',{}).get('equal_one',0),
        summary=value.get('summary_zh',[]))


def train19_evidence(root):
    path=root/'analysis/IL70_TRAIN19_G16_G32_EXPLORATION/RESEARCH_REVIEW.json'
    value=read(path,{})
    if (value.get('status')!='COMPLETE_CPU_REVIEW_OF_REAL_GPU_PROBE'
            or value.get('protocol',{}).get('Adam_updates')!=0
            or value.get('actual_budget',{}).get('complete_groups')!=38):return {}
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cohorts=value.get('cohorts',{}),budget=value.get('actual_budget',{}))


def S6_live_evidence(root):
    path=root/'analysis/S6_LIVE_ACTIVATION_REVIEW.json';value=read(path,{})
    if (value.get('schema')!='world_S6_live_activation_research_review_v1'
            or not value.get('status','').startswith(('INITIAL_PREFIX_','COMPLETE'))
            or value.get('rank_count')!=8 or value.get('common_updates',0)<1):return {}
    cases=[c for c in value.get('allzero_PG0_cases',[]) if c.get('rawA_PG0')]
    gradients=value.get('gradient_diagnostics',{})
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=value['status'],analysis_at=value.get('analysis_at'),common_updates=value['common_updates'],
        rank_count=value['rank_count'],rank_update_rows=value.get('rank_update_rows'),
        quality=value.get('window_quality_counts',{}),gates=value.get('teacher_gates',{}),
        alpha=value.get('alpha_budget',{}),queries=value.get('query_cost',{}),
        PG0_cases_observed=len(cases),PG0_B1_full_score=sum(c.get('nominal_score')==1. for c in cases),
        local_components=gradients.get('conditional_teacher_slot_component_norms',{}),
        DDP_total=gradients.get('actual_DDP_reduced_total_gradient_norm',{}),
        combined_aux_gradient_norm=gradients.get('combined_aux_gradient_norm'))


def first_window_progress_evidence(root):
    GT_path=root/'analysis/S6_FIRST_WINDOW_GT_FALLBACK_AUDIT/RESEARCH_READOUT.json'
    synthetic_path=root/'analysis/S6_FIRST_WINDOW_SYNTHETIC_PROGRESS_AUDIT/SUMMARY.json'
    GT=read(GT_path,{});synthetic=read(synthetic_path,{})
    complete=read(synthetic_path.with_name('COMPLETE.json'),{})
    if (GT.get('status')!='COMPLETE_CPU_DISCOVERY_ONLY'
            or synthetic.get('status')!='COMPLETE_CPU_SYNTHETIC_DIAGNOSTIC_ONLY'
            or synthetic.get('schema')!='world_S6_TRAIN7_explicit_synthetic_progress_teacher_diagnostic_v1'
            or synthetic.get('no_training_or_native_mutation') is not True
            or complete.get('status')!='COMPLETE'):return {}
    digest=hashlib.sha256(synthetic_path.read_bytes()).hexdigest()
    if complete.get('files_sha256',{}).get('SUMMARY.json')!=digest:return {}
    return dict(GT_path=str(GT_path),synthetic_path=str(synthetic_path),
        GT_file_sha256=hashlib.sha256(GT_path.read_bytes()).hexdigest(),synthetic_file_sha256=digest,
        fallback={k:v for k,v in GT.get('fallback7',{}).items() if k!='cases'},
        GT_budget=GT.get('budget',{}),GT_fallback=GT.get('cohort_statistics',{}).get('fallback7',{}),
        synthetic_cohorts=synthetic.get('cohorts',{}),synthetic_queries=complete.get('actual_CPU_PDM_queries'))


def FAST_review_evidence(root,complete_results):
    directory=root/'analysis/FAST_u000808_FULL_REVIEW';path=directory/'SUMMARY.json'
    value=read(path,{});body=dict(value);seal=body.pop('sha256',None)
    population=value.get('population',{})
    official=next((r for r in complete_results if r['arm']=='S0_FAST_PACK4' and r['updates']==808),None)
    if (not official or value.get('status')!='COMPLETE'
            or (population.get('scenes'),population.get('physical_logs'),population.get('invalid'))!=(12146,44,0)
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal):return {}
    evaluations=value.get('evaluations',{})
    if abs(evaluations.get('FAST808',{}).get('PDMS',-1)-official['PDMS'])>1e-8:return {}
    # Recheck only sealed source metadata, not the12146 scene arrays/ledger.
    for item in evaluations.values():
        source=Path(item['source']);summary=source/'SUMMARY.json'
        if not summary.exists():summary=source.parent/'SUMMARY.json'
        for source_path,key in [(summary,'summary_file_sha256'),(source/'identity.json','identity_file_sha256')]:
            try:raw=source_path.read_bytes();metadata=json.loads(raw)
            except (OSError,ValueError):return {}
            expected=metadata.pop('sha256',None)
            if (hashlib.sha256(raw).hexdigest()!=item['provenance'].get(key)
                    or hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=expected):return {}
    figures=[]
    for name in ('score_bands.svg','band_transitions.svg','gain_loss_and_logs.svg'):
        item=value.get('artifacts',{}).get(name,{})
        try:valid=hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        except (OSError,KeyError):valid=False
        if valid:figures.append(item['path'])
    return dict(path=str(path),seal=seal,score=official['PDMS'],
        scores={k:v['PDMS'] for k,v in evaluations.items()},
        contrasts={k:{field:v[field] for field in ('PDMS_delta','zero','metrics')} for k,v in value['contrasts'].items()},
        fixed_IL70_high=value['fixed_IL70_high_between_arm'],budget=value['native_budget_metadata'],figures=figures)


def S7_evidence(root,registry):
    run=next((r for r in registry.get('runs',[]) if r['id']=='S7_STRUCTURED_PROGRESS'),{})
    registered=run.get('native_qualification',{});path=Path(registered.get('path',''))
    try:raw=path.read_bytes();value=json.loads(raw)
    except (OSError,ValueError):return {}
    if (hashlib.sha256(raw).hexdigest()!=registered.get('file_sha256') or value.get('status')!='PASS'
            or value.get('schema')!='world_structured_progress_actual_world8_native_qualification_v1'
            or (value.get('world'),value.get('B'),value.get('G'),value.get('M'))!=(8,64,16,4)
            or value.get('trainable_numel')!=52707840 or not value.get('native_bitexact_at_inner2_and4')):return {}
    return dict(path=str(path),file_sha256=registered['file_sha256'],source=run.get('source'),source_commit=run.get('source_commit'),
        teachers=value.get('actual_first_collection_teacher_classes',{}),
        structured_queries=value.get('strict_reviews',{}).get('continuous4',{}).get('actual_teacher_only_structured_queries'))


def zero2_evidence(root):
    directory=root/'analysis/TRAIN_ZERO2_STRUCTURED_PROGRESS_AUDIT';path=directory/'RESEARCH_READOUT.json'
    value=read(path,{});evidence=read(directory/'FINAL_EVIDENCE.json',{})
    try:valid=hashlib.sha256(path.read_bytes()).hexdigest()==evidence.get('files_sha256',{}).get('RESEARCH_READOUT.json')
    except OSError:valid=False
    if not valid or value.get('status')!='COMPLETE_DISCOVERY_ONLY_NOT_TRAINING':return {}
    return dict(path=str(path),cohorts=value.get('cohorts',{}),budget=value.get('budget',{}),
        cases=[dict(token=c['token'],best=c['synthetic_all']['best_qualified_score'],
            qualified=c['synthetic_all']['qualifying_teacher'],minimum_delta=c['minimum_successful_delta_path_coordinate_at4_m']) for c in value.get('cases',[])])


def credit_lineage_evidence(root):
    path=root/'analysis/SAFETYEP_CREDIT_LINEAGE_REVIEW.json';value=read(path,{})
    if value.get('status')!='COMPLETE_BOUNDED_CPU_SOURCE_AND_BEHAVIOR_REVIEW':return {}
    return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        groups=sum(v.get('groups',0)for v in value.get('bounded_TRAIN_windows',{}).values()),
        summary=value.get('summary_zh',[]),limits=value.get('limits',[]))


def S5_review_evidence(root,complete_results):
    directory=root/'analysis/S5_u001616_FULL_REVIEW';path=directory/'SUMMARY.json'
    value=read(path,{});body=dict(value);seal=body.pop('sha256',None)
    phase=read(directory/'PHASE_AND_BUDGET.json',{})
    point=next((r for r in complete_results if r['arm']=='S5_PERFORMANCE_MAX' and r['updates']==1616),None)
    if (not point or value.get('status')!='COMPLETE' or phase.get('status')!='COMPLETE'
            or phase.get('actual_Adam_updates')!=1616 or phase.get('actual_recipe_matches_registration') is not True
            or phase.get('checkpoint')!=point.get('checkpoint')
            or abs(value.get('evaluations',{}).get('S5_1616',{}).get('PDMS',-1)-point['PDMS'])>1e-8
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal):return {}
    return dict(path=str(path),phase_path=str(directory/'PHASE_AND_BUDGET.json'),
        zero_IL=value['contrasts']['IL70']['zero'],zero_Control=value['contrasts']['Control10k']['zero'],
        fixed=value['fixed_IL70_high_between_arm']['target_vs_Control10k'],
        phase={k:phase[k]for k in ('phase','actual_exposure_epoch','extra_true_B1_queries','extra_teacher_SDE_queries','extra_history_queries','global_scene_exposures','global_main_PG_queries','matched_compute')})


def TRAIN2048_pair_evidence(root):
    directory=root/'analysis/TRAIN2048_S0u400_S5u800_REVIEW'
    path=directory/'SUMMARY.json';value=read(path,{})
    readout=read(directory/'RESEARCH_READOUT.json',{})
    body=dict(value);seal=body.pop('sha256',None);population=value.get('population',{})
    if (value.get('status')!='COMPLETE' or readout.get('status')!='COMPLETE'
            or population.get('scenes')!=2048 or population.get('invalid')!=0
            or population.get('cohort')!='TRAIN_DIAGNOSTIC'
            or population.get('same_tokens_logs_cache_protocol_numerics') is not True
            or hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=seal
            or hashlib.sha256(path.read_bytes()).hexdigest()!=readout.get('summary_file_sha256')):return {}
    return dict(path=str(path),readout_path=str(directory/'RESEARCH_READOUT.json'),
        population=population,
        evaluations={k:{field:v[field]for field in ('PDMS','bins','strict4_high')}for k,v in value['evaluations'].items()},
        cohorts={k:{field:v[field]for field in ('fixed_IL70_high','fixed_IL70_low')}for k,v in value['strict_high_and_fixed_IL_cohorts'].items()},
        zero=value['cross_arm_zero_sets'],headlines=readout.get('headline_zh',[]),
        budget={k:{field:v[field]for field in ('updates','scene_exposures','actual_PG_queries','extra_queries')}for k,v in value['native_budget_metadata'].items()},
        figures=[str(p)for p in sorted(directory.glob('*.svg'))])


def TRAIN2048_pair_fragment(s):
    value=s['TRAIN2048_pair']
    if not value:return ''
    rows=[]
    for label in ('IL70','S0_u400','S5_u800'):
        point=value['evaluations'][label];contrast=value['cohorts'].get(label.split('_')[0]+'_vs_IL70')
        rows.append((label,f'{point["PDMS"]:.5f}',point['bins']['zero'],point['strict4_high'],
            f'{contrast["fixed_IL70_low"]["net_global_PDMS"]:+.5f}'if contrast else '基准',
            f'{contrast["fixed_IL70_high"]["net_global_PDMS"]:+.5f}'if contrast else '基准'))
    body='<details id="performance-train2048-pair"><summary>TRAIN2048 配对诊断：低分修复与高分退化相抵，不是 NAVTEST 成绩</summary>'
    body+='<p>同 '+esc(value['population']['scenes'])+' 个 TRAIN 场景 / '+esc(value['population']['physical_logs'])+' 个物理 log，invalid0；相同 token、cache、scorer 与真实 B1。末两列是固定 IL 队列对整个 TRAIN2048 均分的贡献，不能与 NAVTEST 排行榜混用。</p>'
    body+=table(['检查点','TRAIN PDMS','零分','strict-high','IL低分432贡献','IL高分1616贡献'],rows)
    for line in value['headlines'][2:]:body+='<p>'+esc(line)+'</p>'
    body+='<p><b>12 个零分修复中 11 个尚未直接曝光，表明本 TRAIN 人口内存在跨场景参数迁移，不能解释为只记住已见 token。</b>这仍不是独立泛化验证；同一检查点只评一次 B1，旧 IL70 采样见证不是当前 RL 同 θ 材料。均值、EP 分解和曝光关联不能证明梯度因果或最终上限；两臂此时都未激活 rescue/history，且 B/Adam 不同。</p>'
    body+='<p><a href="'+esc(value['path'])+'">完整配对审计</a> · <a href="'+esc(value['readout_path'])+'">研究简报与限制</a></p>'
    if value['figures']:
        body+='<details><summary>已有 TRAIN 分布与转移图</summary>'
        for path in value['figures']:body+='<img loading="lazy" style="max-width:100%" src="'+esc(path)+'" alt="TRAIN2048 诊断图">'
        body+='</details>'
    return body+'</details>'


def current_runs(root, registry, owner):
    now=time.time();runs=[]
    for registered in registry.get('runs',[]):
        arm=registered['id'];node=owner.get('runs',{}).get(arm,{})
        target=node.get('stage_target_updates')
        stage=node.get('stages',{}).get(str(target),{})
        job=stage.get('job')
        if not job and node.get('job'):
            # An earlier completed stage is not the current resource-waiting job.
            candidate=Path(node['job'])
            if target is not None and f'formal_to{int(target):06}' in candidate.name:job=str(candidate)
        directory=Path(job) if job else None
        result=read(directory/'RESULT.json') if directory else None
        running=read(directory/'RUNNING.json') if directory else None
        output=Path(registered['root']);latest=[];ages=[];current_invocations=0
        for rank in range(8):
            path=output/f'metrics_rank{rank}.jsonl'
            rows=journal_rows(path,limit=1,max_bytes=2**16)
            if rows:
                latest.append(rows[-1])
                invocation=read(output/'invocations'/str(rows[-1].get('invocation',''))/f'rank{rank}.json',{})
                if running and invocation.get('process_start_unix',0)>=running.get('start_unix',float('inf')):
                    current_invocations+=1
                try:ages.append(max(0,now-path.stat().st_mtime))
                except OSError:pass
        fresh=len(latest)==8 and len(ages)==8 and max(ages)<=600 and current_invocations==8
        if result:status='本阶段已结束' if result.get('returncode')==0 else '执行退出，待复核'
        elif running and fresh:status='实际训练'
        elif running and current_invocations==8:status='进度过期 / 当前运行未确认'
        elif running:status='启动 / 等待首条更新'
        elif 'WAIT' in node.get('status',''):status='等待资源'
        else:status=node.get('status','已登记，尚未启动')
        if not job and 'WAIT' in node.get('status',''):status='等待资源'
        reserved=registered.get('disposition')=='RESERVED_NATIVE_CHECKPOINT_FOR_HIGHER_PRIORITY_QUALITY_STUDY'
        if reserved and status!='实际训练':
            status='保留原生检查点，资源让位高优先级研究；未判无效'
        epochs=[r.get('metrics',{}).get('exposure_epoch') for r in latest]
        epochs=[v for v in epochs if isinstance(v,(int,float))]
        runs.append(dict(id=arm,status=status,resource_note='已登记资源让位，仍以真实退出回执为准' if reserved and status=='实际训练' else None,
            owner_status=node.get('status'),host=(running or result or {}).get('host',node.get('host')),
            job=job,rank_count=len(latest),oldest_rank_age_seconds=max(ages) if ages else None,
            updates=min((r.get('updates',0) for r in latest),default=0),
            exposure_epoch=min(epochs) if epochs else None,target=target,output=str(output)))
    return runs


def snapshot(root=ROOT):
    root=Path(root)
    jobs=[]
    directories=set((root/'jobs').glob('QUAL_*'))|set((root/'jobs').glob('PROBE*'))|set((root/'jobs').glob('S*/formal_*'))
    for directory in sorted(directories):
        result=read(directory/'RESULT.json');running=read(directory/'RUNNING.json')
        value=result or running or read(directory/'DISPATCH.json')
        if value:
            cmd=value.get('command',[]);output=cmd[cmd.index('--output')+1] if '--output' in cmd else ''
            row=tail(Path(output)/'metrics_rank0.jsonl') if output else {}
            jobs.append(dict(id=str(directory.relative_to(root/'jobs')),status=value.get('status','DISPATCHED'),host=value.get('host'),
                qualification='QUAL_' in directory.name or 'PROBE' in directory.name,
                output=output,updates=row.get('updates',0),scene_exposures=row.get('raw_scene_exposures',0),
                exposure_epoch=row.get('metrics',{}).get('exposure_epoch'),returncode=value.get('returncode')))
    # Reuse the stdlib-only sealed SPEC/identity qualification, not bare summaries.
    spec=importlib.util.spec_from_file_location('performance_report_analysis',Path(__file__).with_name('analyze.py'))
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    analysis=analyzer.collect_results(root)
    complete=[r for r in analysis['complete_results'] if r['updates']>0 and r.get('physical_logs')==44]
    registry=read(root/'RUNS.json',{});owner=read(root/'owners/PERFORMANCE_NATIVE/STATE.json',{})
    previous_review=read(Path('/mnt/project/qwen-world-rl-optimization-20261030/analysis/REPORT_LATEST_COMPLETE_REVIEW_20261101/SUMMARY.json'),{})
    previous_peak=previous_review.get('observed_peak',{})
    previous_score=read(previous_peak.get('evidence',''),{})
    previous_body=dict(previous_score);previous_seal=previous_body.pop('sha256',None)
    if not (previous_score.get('status')=='COMPLETE' and previous_score.get('scenes')==12146
            and previous_score.get('invalid_scenes')==0 and previous_score.get('physical_logs')==44
            and previous_score.get('PDMS')==previous_peak.get('PDMS')
            and hashlib.sha256(json.dumps(previous_body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==previous_seal):
        previous_peak={}
    current=current_runs(root,registry,owner)
    legacy_current=current_runs(LEGACY_ROOT,read(LEGACY_ROOT/'RUNS.json',{}),
        read(LEGACY_ROOT/'owners/INTEGRATED_NATIVE/STATE.json',{}))
    legacy_complete=analyzer.collect_results(LEGACY_ROOT)['complete_results']
    legacy_complete=[r for r in legacy_complete if r['updates']>0 and r.get('physical_logs')==44]
    journals={r['id']:journal_rows(Path(r['output'])/'metrics_rank0.jsonl') for r in current}
    return dict(generated_UTC=datetime.now(timezone.utc).isoformat(),jobs=jobs,complete_results=complete,
        active_formal=sum(j['status']=='实际训练' for j in current),current_runs=current,journals=journals,
        active_qualification=sum(j['status']=='RUNNING' and j['qualification'] for j in jobs),
        audit=read(root/'data/AUDIT.json',{}),registry=registry,native_owner=owner,previous_peak=previous_peak,
        legacy_current_runs=legacy_current,legacy_complete_results=legacy_complete,
        plans=registered_plans(root,registry),qualification=read(root/'performance_qualification.json',{}),
        result_review=dict(pending=analysis['pending'],rejected=analysis['rejected']),
        deadline=read(root/'operations/RESOURCES_UNTIL_20261103T000000Z/REQUEST.json',{}),
        resume=read(root/'operations/NATIVE_RESUME_QUALIFICATION.json',{}),
        nominal=read(root/'qualification/REAL_NOMINAL_IL70/COMPLETE.json',{}),
        training_diagnosis=read(root/'analysis/CURRENT_TRAINING_DIAGNOSIS.json',{}),
        execution_qualifications={name:read(root/relative,{}) for name,relative in (
            ('frozen_reference','qualification/REFERENCE_CACHE_FULL_IL70/COMPLETE.json'),
            ('PACK2','benchmarks/COHERENT_PACK2/COMPLETE.json'),
            ('PACK4','benchmarks/COHERENT_PACK4/COMPLETE.json'))},
        permanent_continuation=read(root/'operations/PERMANENT_PRIORITY_BINDING/S1_NOMINAL_RESCUE_NATIVE_RETURN.json',
            read(root/'operations/OPS_V2_HANDOFF/S1_NATIVE_RETURN.json',{})),
        cache_world8=read(root/'qualification/REFERENCE_CACHE_WORLD8_NATIVE/COMPLETE.json',{}),
        packed_world8=read(root/'benchmarks/NATIVE8_PACK4/REVIEWED_COMPLETE_V4.json',{}),
        TRAIN2048_quality=read(root/'diagnostics/IL70_TRAIN2048/QUALITY.json',{}),
        verified_quality=verified_quality_evidence(root),TRAIN_GT=train_GT_evidence(root),TRAIN19=train19_evidence(root),
        S6_fresh=read(root/'operations/S6_FORMAL_LAUNCH/FRESH_IL70_ADAM0_PROOF.json',{}),
        S6_live=S6_live_evidence(root),
        first_window_progress=first_window_progress_evidence(root),
        FAST_review=FAST_review_evidence(root,complete),
        S7=S7_evidence(root,registry),zero2=zero2_evidence(root),
        credit_lineage=credit_lineage_evidence(root),
        S5_review=S5_review_evidence(root,complete),
        TRAIN2048_pair=TRAIN2048_pair_evidence(root),
        resource_window=read(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/WINDOW_CORRECTION.json',{}),
        resource_restore=read(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/RESTORATION_RECEIPT.json',{}),
        S7_permanent_wait=read(root/'operations/S7_PERMANENT_CAPACITY_WAIT/STATE.json',{}),
        full_TRAIN_census=read(root/'diagnostics/FULL_TRAIN_REGISTRATION.json',{}))


def table(headers,rows):
    return '<div style="overflow:auto"><table style="width:100%;border-collapse:collapse"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td style="padding:9px;border-bottom:1px solid #dce3eb">'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def verified_quality_fragment(s):
    proof=s['verified_quality']
    run=next((r for r in s['current_runs'] if r['id']=='S6_VERIFIED_QUALITY'),None)
    status=run['status'] if run else '尚未正式登记启动'
    body='<section id="performance-verified-quality"><h3>S6：质量门控教师与冷启动校准</h3>'
    body+='<p>正式状态：<b>'+esc(status)+'</b>。与当前任务表同源；资格权重不用于正式初始化。S6 从相同 IL70 与 Adam0 独立开始，B64/G16/四 inner；真实 B1 不达严格高分时要求教师增益至少 .02、使用 α=.05，已严格高分时允许至少 .005 的更优教师、使用较弱 α=.02。教师均须 NC/DAC/TTC/Comfort=1、score≥.9；只重新分配原 β50 预算，不叠加新 β。</p>'
    fresh=s['S6_fresh']
    if (fresh.get('status')=='PASS' and fresh.get('optimizer_state_entries')==0
            and fresh.get('from_other_RL_checkpoint') is False and fresh.get('qualification_weights_used') is False):
        body+='<p>正式 native0 已核对同一 IL70、空 Adam 状态与零场景曝光；不是从资格或其它 RL 权重继续。<a href="'+esc(ROOT/'operations/S6_FORMAL_LAUNCH/FRESH_IL70_ADAM0_PROOF.json')+'">fresh IL70 / Adam0 启动证据</a>。实际在训数量仍只由八 rank 新日志决定。</p>'
    live=s['S6_live']
    if live:
        quality=live['quality'];g=live['gates'];a=live['alpha'];q=live['queries']
        body+='<aside style="background:#edf7f4;border-left:4px solid #16816c;padding:12px 16px"><h4>正式 S6 已验证到什么</h4><p>分析截至共同 Adam '+esc(live['common_updates'])+'，覆盖八 rank 的 '+esc(live['rank_update_rows'])+' 条更新日志、'+esc(quality.get('valid_scenes'))+' 个去重新鲜 TRAIN 场景；分析时间 '+esc(live['analysis_at'])+'。</p>'
        body+=table(['实际机制','已观测结果'],[
            ('真实教师门控','修复 '+str(g.get('repair_enabled'))+'/'+str(g.get('repair_scenes'))+'，α='+str(a.get('alpha_repair'))+'；已高分精修 '+str(g.get('already_strict_high_enabled'))+'/'+str(g.get('already_strict_high_scenes'))+'，α='+str(a.get('alpha_already_strict_high'))),
            ('全场景平均原 β 预算',f'IL {100*a["reconstructed_IL_share_all_valid"]:.2f}% / 辅助 {100*a["mean_alpha_all_valid_scenes"]:.2f}%；是系数预算，不是梯度比例'),
            ('实际评分分账','PG '+str(q.get('main_fresh_PG'))+' + 真 B1 '+str(q.get('true_B1'))+' + 条件独立 G16 '+str(q.get('conditional_teacher_only_extra_G16'))+' = '+str(q.get('total_actual_candidate_queries'))+f'；额外/PG {100*q["additional_over_PG_fraction"]:.2f}%（不是 GPU 时间）'),
            ('两种零分别核对','候选官方全零组 '+str(g.get('main_canonical_allzero'))+'；rawA 全零 PG0 组 '+str(g.get('main_rawA_PG0'))+'；PG0 明细中 B1 满分 '+str(live['PG0_B1_full_score'])+'/'+str(live['PG0_cases_observed']))])
        local=live['local_components'];ddp=live['DDP_total']
        cfm=local.get('nominal_scaled_cfm_gradient_norm_last_measured',{}).get('mean')
        ode=local.get('nominal_scaled_ode_gradient_norm_last_measured',{}).get('mean')
        if cfm is not None and ode is not None and ddp.get('mean') is not None:
            body+='<p>已测条件局部分量范数均值：scaled CFM '+esc(f'{cfm:.6g}')+'、ODE '+esc(f'{ode:.6g}')+'；DDP 归约后总梯度范数均值 '+esc(f'{ddp["mean"]:.6g}')+'。两者归约与分母不同，不能直接相除；辅助联合向量范数和梯度夹角尚未记录。</p>'
        body+='<p><b>这里只证明门控与有效辅助路径已激活，未证明更新后的 B1 吸收或 NAVTEST 提升。</b>同一 collection 的四 inner 不重复计新评分；条件额外池是独立 G16，不是 TRAIN19 的真 batch32。无信号的采样组与部署 B1 失败也不能互相等同。<a href="'+esc(live['path'])+'">完整正式激活审核</a></p></aside>'
    if proof:
        classes=proof['teacher_classes'];summary=proof['summary'];r=proof['rank0']
        state=r.get('final_calibration_state',{});norms=state.get('last_norms',{})
        body+='<p><b>v2 真实八卡资格 PASS：</b>52,707,840 个可训练参数，连续四步与保存 inner2 后恢复至四步的参数、Adam、scheduler、校准 PG EMA、八 rank RNG 和数据流逐位一致。64 个真实 TRAIN 场景自然触发高分精修 '+esc(classes.get('.02'))+' 个、低分修复 '+esc(classes.get('.05'))+' 个；无教师 '+esc(classes.get('inactive'))+' 个，没有人工注入门控。<a href="'+esc(proof['proof'])+'">完整原生资格</a> · <a href="'+esc(proof['gradient'])+'">全八 rank 四步梯度读数</a></p>'
        body+='<p>v1 实测发现从 IL0 出发时 IL 锚定梯度很小，按它单独校准会让辅助预算过弱。v2 只将校准目标改为 <code>max(||gIL||, .04×EMA(||gPG||))</code>，PG EMA 衰减 .9、首次正 PG 初始化；当前 PG0 可使用已有 EMA。初始 PG 与 EMA 均为零时保留最小尺度，但不消耗可冻结校准次数。旧 S1 校准、教师门和原 PG/锚定定义保持。</p>'
        if norms:
            body+=table(['rank0 第四步最后有效教师','PG','未加权 IL','原始 CFM','原始 ODE-last'],[
                ('实际局部全参数梯度范数',*[f'{norms[k]:.6g}' for k in ('pg','IL','cfm','ode')])])
            body+='<p>该槽位 PG EMA='+esc(f'{state["PG_gradient_EMA"]:.6g}')+'，目标='+esc(f'{state["last_target_norm"]:.6g}')+'；CFM/ODE scale='+esc(f'{state["scale_cfm"]:.6g}')+' / '+esc(f'{state["scale_ode"]:.6g}')+'。八 rank 末有效槽位的目标/IL 比为 '+esc(f'{summary["last_target_to_IL"]["minimum"]:.0f}')+'–'+esc(f'{summary["last_target_to_IL"]["maximum"]:.0f}')+' 倍，已摆脱近零 IL-only 校准。权重乘以实际分量梯度范数可估计注入量级；它不等于直接测得的联合 BF16/DDP 梯度、参数更新归因或 PDMS 收益。</p>'
    else:
        body+='<p>完整资格收据尚未匹配；不从 CPU 就绪或部分日志推断通过。</p>'
    return body+'</section>'


def train_GT_fragment(s):
    value=s['TRAIN_GT']
    if not value:return ''
    p=value['population'];o=value['opportunity']
    body='<section id="performance-train-GT"><h3>真实 TRAIN GT 审核：有教师机会，也必须拒绝退化标签</h3>'
    body+='<p>既有 NAVTRAIN 诊断 '+esc(p['input_count'])+' 个场景中，'+esc(p['invalid_time_count'])+' 条原始时间有缺口，整条不评分、也不记零分。其余 '+esc(p['valid_whole_trajectory_count'])+' 条用原官方参考与原物理 FP32 GT 各评分一次，共 '+esc(value['budget']['actual'])+' 次新 CPU 查询、0 模型 forward、0 训练更新。<b>这是 TRAIN 诊断，GT 是特权离线标签，不是模型分数或 NAVTEST 收益。</b></p>'
    body+=table(['固定 B1 群体','真实审计结果','含义'],[
        ('原 B1 零分 '+str(o['zero_count']),str(o['zero_strict4'])+' 条 GT 严格四项通过；'+str(o['zero_strict_high'])+' 条 score≥.9','其余安全 GT 也可能未达高分门槛'),
        ('全部合法 '+str(p['valid_whole_trajectory_count']),str(o['all_gt005'])+' 条 strict-high 且 gain>.005；'+str(o['all_ge02'])+' 条 gain≥.02','存在可验证监督机会，不能盲目模仿所有 GT'),
        ('B1 已 strict-high '+str(o['B1_strict_high_count']),str(o['B1_strict_high_GT_strict_high_gain_gt005'])+' 条 GT 改善>.005；'+str(value['GT_worse_strict_high'])+' 条反而下降>.005',
            '全体有'+str(value['GT_worse_all'])+'条 GT 下降>.005，另有'+str(value['GT_comfort_not_one'])+'条 Comfort 未通过')])
    body+='<p>这里 gain=.005 对应 0.5 个 PDMS 点；score≥.9 是严格高分门。仅凭这份 GT 审核不能由 GT 好/B1 差断言 G16 探索失败；下方另列 19 场景的同父模型采样补证。GT 没有 on-policy log-prob，不进入 PPO 优势；本次未把 GT 加入 S6、任何 bank 或正式算法。归一化再解码会改变 FP32 端点位，未来若采用必须重新评分。输入沿既有池口径，含 '+esc(p['old_DEV_in_existing_NAVTRAIN_count'])+' 个原 DEV 场景；NAVTEST token/log 严格隔离。<a href="'+esc(value['path'])+'">完整发现性审核与限制</a></p></section>'
    panel=s['TRAIN19']
    if panel:
        zero=panel['cohorts']['B1_zero'];rows=[]
        for label,key in [('15个固定 B1 零分','B1_zero'),('4个固定高分对照','fixed_strict_high_control')]:
            cohort=panel['cohorts'][key]
            for group in ('16','32'):
                counts=cohort['groups'][group]
                rows.append((label+' / G'+group,str(counts['scenes_with_strict_high'])+'/'+str(counts['scenes']),
                    str(counts['unsafe_candidates'])+'/'+str(counts['actual_PDM_queries']),
                    str(counts['strict_high_negative_rawA'])+'/'+str(counts['strict_high_candidates'])))
        body+='<details><summary>追加真实 TRAIN19 探针：G16/G32 是否已找到好轨迹</summary>'
        body+=table(['定向群体 / 采样组','有 strict-high 见证的场景','strict3 未通过候选','strict-high 中 rawA<0'],rows)
        body+='<p>零分群体两组高见证并集 '+esc(zero['high_found_either'])+'/15；11 个 GT≥.9 场景两组全部采到同门槛候选。两组都无高见证的 2 个场景，其 GT 虽 strict4，却只有 .857625/.890610，因此本 panel 的 GT 独有 strict-high 为 0，不能外推 GT 对全部训练无用。38 组均非官方全零且均有非零 rawA；零分群体的 91 条 strict-high 均 rawA>0。高分对照里的相对负 rawA 不等于 unsafe 获正信用，后者实测为 0。</p>'
        body+='<p>这是 19 个定向 TRAIN 场景、'+esc(panel['budget']['new_PDM_candidate_queries'])+' 次真实候选评分、0 Adam；G16/G32 使用独立随机组，G32 有两倍查询且 BF16 batch shape 不同，不是同算力因果对照。采样好而 B1 差是当前单轨转换机会，不是训练后吸收失败，也不是 NAVTEST 提升。<a href="'+esc(panel['path'])+'">完整 TRAIN19 研究简报</a></p></details>'
    return body


def first_window_progress_fragment(s):
    value=s['first_window_progress']
    if not value:return ''
    fallback=value['fallback'];count=fallback['count'];cohorts=value['synthetic_cohorts']
    found=cohorts['statistics'];old=cohorts['old_extra_G16_statistics'];GT=value['GT_fallback']
    body='<section id="performance-progress-teacher-discovery"><h3>研究发现：安全通过但进度偏低，能否找到更好的教师</h3>'
    body+='<p>正式 S6 首窗口中 '+esc(count)+' 个主 G16 与额外 G16 都未找到高分教师的场景，真实 B1 与 GT 均 '+esc(count)+'/'+esc(count)+' 通过 NC/DAC/TTC/Comfort，<b>是安全但进度较低，不是 '+esc(count)+' 个安全失败。</b>GT score 范围 '+esc(f'{fallback["GT_score_min"]:.6f}')+'–'+esc(f'{fallback["GT_score_max"]:.6f}')+'，达到 .9 的仅 '+esc(fallback['GT_strict_high_count'])+' 个；GT 本身没有增加合格教师槽。</p>'
    rows=[('同场景原独立额外 G16（复用评分）',old['candidates'],old['strict4'],old['qualifying_teacher'],str(old['scenes_with_qualifying_teacher'])+'/'+str(count)),
        ('原 GT roundtrip（既有75次审核的子集）',GT['scored_count'],GT['GT_roundtrip_strict4'],GT['teacher_meets_actual_gate'],'0/'+str(count))]
    for label,key in [('沿 B1 路径的平滑进度候选','B1_path_statistics'),('沿 TRAIN GT 路径的平滑进度候选','GT_path_statistics')]:
        x=cohorts[key];rows.append((label,x['candidates'],x['strict4'],x['qualifying_teacher'],str(x['scenes_with_qualifying_teacher'])+'/'+str(count)))
    body+=table(['同一固定场景组的候选来源','已评分候选','严格四项通过','合格教师','有合格教师场景'],rows)
    body+='<p><b>新增合成候选 '+esc(found['qualifying_teacher'])+'/'+esc(found['candidates'])+' 满足 strict4、score≥.9 且相对真实 B1 gain≥.02，覆盖 '+esc(found['scenes_with_qualifying_teacher'])+'/'+esc(count)+'。</b>GT 审核实际新增 '+esc(value['GT_budget']['actual_new_PDM_candidate_queries'])+' 次 CPU 查询；合成诊断另新增 '+esc(value['synthetic_queries'])+' 次 CPU 查询，0 GPU forward、0 Adam。两类合成分支的场景覆盖有重叠，不能相加。</p>'
    body+='<p>仍有 '+esc(found['candidates']-found['strict4'])+'/'+esc(found['candidates'])+' 个合成候选未通过全部严格条件；平滑构造不保证安全，必须逐条经原官方评分验证。GT 分支使用离线特权 TRAIN 标签，生成机制和成本也不同于原 SDE，不能称公平的策略采样消融、best-of16 部署分数或已学会的能力。没有更新后 B1 重评，因此这是教师发现，不是 B1 吸收或 NAVTEST 提分。</p>'
    S7=next((r for r in s['current_runs'] if r['id'].startswith('S7_')),None)
    body+='<p>独立 S7：'+esc(S7['status'] if S7 else '尚无正式任务')+'；资格状态与真实训练见前部 S7 卡，未有八 rank 新日志不计实际训练。上述诊断不修改 S6 的教师或损失。<a href="'+esc(value['GT_path'])+'">GT 首窗审核</a> · <a href="'+esc(value['synthetic_path'])+'">完整合成进度诊断</a></p></section>'
    return body


def S7_fragment(s):
    proof=s['S7'];run=next((r for r in s['current_runs'] if r['id']=='S7_STRUCTURED_PROGRESS'),None)
    if not proof and not run:return ''
    body='<section id="performance-S7-structured-progress"><h3>S7：轨迹教师发现 → CFM 与真实 ODE-last 吸收</h3>'
    body+='<p>正式状态：<b>'+esc(run['status'] if run else '尚无正式任务')+'</b>，按当前 stage 与八 rank 新日志判定，注册或模型初始化不算已学习。独立从相同 IL70 / Adam0 开始，严禁继承资格权重；主 G16/W3 SafetyEP PG 不变，沿用 PG-EMA 校准和 α=.02/.05 的原 β50 重分配。</p>'
    body+='<p>仅在真实 B1 未达 strict-high、主 G16 和原额外 SDE16 都无 high 时，再构造最多16条 TRAIN GT/B1路径进度候选；逐条经官方评分，通过 strict4、score≥.9、gain≥.02 才作教师。它们仅进入原 .75CFM+.25真实 ODE-last 辅助，不混入 PPO/oldlogp/std 或部署选轨；GT 不进入真实单轨部署输入，额外查询单独分账。</p>'
    if proof:
        t=proof['teachers']
        body+='<p><b>真实八 rank 工程资格 PASS：</b>B64/G16/四 inner、52.7M 参数，连续4步/保存2步/原生恢复4步通过，参数、Adam、RNG/stream及辅助校准恢复一致。自然教师 '+esc(t.get('.02',0)+t.get('.05',0))+' 个：α.02×'+esc(t.get('.02'))+'、α.05×'+esc(t.get('.05'))+'，其中 '+esc(t.get('structured'))+' 个由结构化候选补出，实际新增 '+esc(proof['structured_queries'])+' 次教师候选评分。科学材料比较只排除资格协议已声明的计时/Invocation元数据。<a href="'+esc(proof['path'])+'">完整原生资格</a>。这不是 B1 吸收或 NAVTEST 收益证明。</p>'
    zero=s['zero2']
    if zero:
        counts=zero['cohorts'];total=counts['statistics'];GT=counts['GT_path_statistics'];B1=counts['B1_path_statistics']
        body+='<details><summary>新增独立 CPU 验证：两个真实 B1 零分且 G16/G32 无高见证场景</summary><p>'+esc(zero['budget']['actual_new_CPU_candidate_queries'])+' 次新候选评分，0 GPU / 0 Adam；'+esc(total['qualifying_teacher'])+'/'+esc(total['candidates'])+' 条合格、覆盖 '+esc(total['scenes_with_qualifying_teacher'])+'/2。全部来自 GT 路径族；B1 路径 '+esc(B1['DAC_failed'])+'/'+esc(B1['candidates'])+' 全部 DAC 失败，不能把给任意路径增加进度等同于安全修复。</p>'
        body+=table(['TRAIN 场景','合格教师','最佳教师 score（非模型得分）','最小成功附加4s路径坐标'],[(c['token'],c['qualified'],f'{c["best"]:.6f}',str(c['minimum_delta'])+'m') for c in zero['cases']])
        body+='<p>原 GT 安全但均低于 .9，原 G16/G32 都无 high；GT-path 合成后 '+esc(GT['qualifying_teacher'])+'/'+esc(GT['candidates'])+' 合格。合成轨迹并非原 GT 或模型输出；这是特权 TRAIN 教师发现，<b>不是两个 B1 已修复、部署提分或公平采样消融</b>。<a href="'+esc(zero['path'])+'">32查询完整证据</a></p></details>'
    return body+'</section>'


def FAST_review_fragment(s):
    value=s['FAST_review']
    if not value:return ''
    IL=value['contrasts']['IL70'];control=value['contrasts']['Control10k']
    fixed=value['fixed_IL70_high'];mass=fixed['target_vs_Control10k'];metrics=control['metrics'];budget=value['budget']
    body='<section id="performance-FAST808-review"><h3>FAST 首个全量结果：0.25 epoch / Adam808</h3>'
    body+='<p><b>PDMS '+esc(f'{value["score"]:.8f}')+'</b>，完整 12146 场景 / 44 physical logs / invalid0，真正单轨 B1 ODE K10/shift5/seed2101。相同 IL70 提高 '+esc(f'{IL["PDMS_delta"]:+.6f}')+' 点；低于已观测 Control10k '+esc(f'{control["PDMS_delta"]:.6f}')+' 点。<b>这是首个 0.25 epoch 观察，尚不能据此判定方法无效。</b></p>'
    body+=table(['逐场景对照','对照实际 PDMS','FAST差值','零分修复 / 新增 / 当前'],[
        ('相同 IL70',f'{value["scores"]["IL70"]:.8f}',f'{IL["PDMS_delta"]:+.6f}',str(IL['zero']['repaired'])+' / '+str(IL['zero']['new'])+' / '+str(IL['zero']['target'])),
        ('历史 Control10k',f'{value["scores"]["Control10k"]:.8f}',f'{control["PDMS_delta"]:+.6f}',str(control['zero']['repaired'])+' / '+str(control['zero']['new'])+' / '+str(control['zero']['target']))])
    body+='<p>相 IL70：原零分保留 '+esc(IL['zero']['retained'])+'、修复 '+esc(IL['zero']['repaired'])+'，同时新增 '+esc(IL['zero']['new'])+' 个零分。相 Control10k 的差额中，固定 IL70 高分 '+esc(fixed['scenes'])+' 场景贡献 '+esc(f'{mass["global_PDMS_contribution"]:+.6f}')+' 点，其余场景 '+esc(f'{mass["other_scenes_global_contribution"]:+.6f}')+' 点。EP 分项 '+esc(f'{metrics["ego_progress"]["delta_pp"]:+.6f}')+'pp、DAC '+esc(f'{metrics["drivable_area_compliance"]["delta_pp"]:+.6f}')+'pp，NC/TTC 分别 '+esc(f'{metrics["no_at_fault_collisions"]["delta_pp"]:+.6f}')+' / '+esc(f'{metrics["time_to_collision_within_bound"]["delta_pp"]:+.6f}')+'pp。它们是完整场景的描述性分解，不证明因果或必然安全效率冲突。</p>'
    body+='<p>实际曝光 '+esc(budget['fresh_valid_scenes'])+' 个场景、PG 候选 '+esc(budget['fresh_PG_candidates'])+'；Control10k 为 '+esc(budget['Control10k_fresh_scenes'])+' 场景 / '+esc(budget['Control10k_fresh_PG_candidates'])+' 候选，Adam 时钟也不同，<b>不是等算力或单因子对照</b>。<a href="'+esc(value['path'])+'">完整封存分析</a></p>'
    if value['figures']:
        body+='<details><summary>查看已生成的全量分档、转移与增损/log分布三图</summary>'
        for path in value['figures']:body+='<figure><img loading="lazy" style="width:100%;height:auto" src="'+esc(path)+'" alt="FAST808全量分析"/><figcaption><a href="'+esc(path)+'">原始 SVG</a></figcaption></figure>'
        body+='</details>'
    return body+'</section>'


COLORS=['#087f8c','#cb6d26','#7554a3','#397744','#b94755','#2d66af','#716236']


def svg_text(x,y,value,size=13,color='#253b50',anchor='start'):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'


def svg_document(title,content,width=1240,height=440):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title><rect width="100%" height="100%" fill="#fff"/><g font-family="system-ui,sans-serif">{content}</g></svg>'


def chart_panel(x,y,w,h,title,series,xlabel,ylabel,xmax=None,ymax=None):
    """Finite numeric data only; missing series remain visibly absent."""
    clean=[]
    for item in series:
        points=[(a,b) for a,b in item['points'] if isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)]
        if points:clean.append(dict(item,points=points))
    out=svg_text(x,y,title,16)
    if not clean:return out+svg_text(x+20,y+65,'尚无可绘制的实际观测',14,'#667788')
    left=x+64;top=y+27;pw=w-85;ph=h-113
    xmax=max(xmax or 0,max(a for item in clean for a,b in item['points']),1e-10)
    ymax=max(ymax or 0,max(b for item in clean for a,b in item['points'])*1.08,1e-15)
    for n in range(5):
        yy=top+ph*(1-n/4);value=ymax*n/4
        label=(f'{value:.1e}' if 0<ymax<.01 else f'{value:.3g}')
        out+=f'<path d="M {left} {yy} h {pw}" stroke="#e5ebf0"/>'+svg_text(left-8,yy+4,label,11,anchor='end')
        xx=left+pw*n/4
        out+=svg_text(xx,top+ph+20,f'{xmax*n/4:.3g}',11,anchor='middle')
    out+=svg_text(left,top-7,ylabel,11,'#627487')
    out+=svg_text(left+pw/2,top+ph+42,xlabel,12,anchor='middle')
    for index,item in enumerate(clean):
        points=item['points'];stride=max(1,len(points)//360);points=points[::stride]+([points[-1]] if points[-1]!=points[::stride][-1] else [])
        color=item.get('color',COLORS[index%len(COLORS)])
        coords=' '.join(f'{left+pw*a/xmax:.2f},{top+ph-ph*b/ymax:.2f}' for a,b in points)
        dashed=' stroke-dasharray="5 4"' if item.get('dashed') else ''
        out+=f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dashed}/>'
        if len(points)==1:
            a,b=points[0];out+=f'<circle cx="{left+pw*a/xmax:.2f}" cy="{top+ph-ph*b/ymax:.2f}" r="3" fill="{color}"/>'
        lx=x+(index%3)*(w/3);ly=y+h-18+(index//3)*17
        out+=f'<path d="M {lx} {ly-4} h 17" stroke="{color}" stroke-width="2"{dashed}/>'+svg_text(lx+22,ly,item['label'],10)
    return out


def figure(root,name,svg,caption):
    path=root/'reporting/figures'/name;atomic_text(path,svg)
    return f'<figure style="margin:22px 0"><img style="width:100%;height:auto" src="{esc(path)}" alt="{esc(caption)}"/><figcaption style="color:#53697c;font-size:.9em">{esc(caption)} · <a href="{esc(path)}">SVG</a></figcaption></figure>'


def engineering_figure(root,s):
    rows=s['qualification'].get('configurations',[])
    if not rows:return '<p>吞吐资格结果尚未发布。</p>'
    body=''
    for panel,key,title,unit in [(0,'gpu','实测窗口推算：每曝光 epoch GPUh','GPUh / epoch'),(1,'queries','每曝光 epoch：有效 PG 评分预算','百万次 PG queries')]:
        x=25+panel*610;body+=svg_text(x,28,title,17)
        values=[r['epoch_budget']['estimated_window_only_GPU_hours'] if key=='gpu' else r['epoch_budget']['valid_PG_candidates']/1e6 for r in rows]
        maximum=max(values)*1.16
        for i,(r,v) in enumerate(zip(rows,values)):
            yy=65+i*73;label=f'B{r["B"]} / G{r["G"]}'
            body+=svg_text(x,yy+17,label,13)
            body+=f'<rect x="{x+112}" y="{yy}" width="{360*v/maximum:.2f}" height="24" rx="3" fill="{COLORS[i]}"/>'
            body+=svg_text(x+119+360*v/maximum,yy+17,f'{v:.2f}',12)
            if key=='gpu':
                t=r['timing_seconds']['steady_window'];body+=svg_text(x+112,yy+43,f'{t:.2f} s / window · {r["B"]/t:.2f} scenes/s',11,'#627487')
        body+=svg_text(x+112,388,unit,12)
    return figure(root,'performance_throughput_budget.svg',svg_document('四种 B/G 资格窗口与每 epoch 评分预算',body,height=415),
        '每配置仅一个真实 world8/M4 首窗口：sampling + scoring wait + old-prob + 四次反传；启动约70秒另计。不同主机，非长程稳态或 B/G 因果速度对照；不代表 S2/S3 的 MC/M2 开销。额外 B1、导出和评测未计入。')


def schedule_multiplier(epoch,spec):
    warm=spec['warmup_epochs'];hold=spec['hold_until_epoch'];end=spec['cosine_decay_end_epoch'];floor=spec['minimum_lr_ratio']
    if epoch<warm:return .001+.999*epoch/warm
    if epoch<=hold:return 1.
    if epoch>=end:return floor
    return floor+(1-floor)*(1+math.cos(math.pi*(epoch-hold)/(end-hold)))/2


def planned_figure(root,s):
    pairs=[('S0_LARGE_SAFETYEP','S0 / S1',COLORS[0]),('S5_PERFORMANCE_MAX','S5',COLORS[1])]
    body='';stage_rows=[]
    for panel,key in enumerate(['action_lr','memory_lr']):
        series=[]
        for arm,label,color in pairs:
            plan=s['plans'].get(arm)
            if not plan:continue
            p=plan['spec'];stages=p['stages'];schedule=p['schedule']
            epochs=set(i/100 for i in range(821))|{float(schedule[k]) for k in ['warmup_epochs','hold_until_epoch','cosine_decay_end_epoch']}
            for stage in stages:
                e=stage['start_epoch'];epochs.update([max(0,e-1e-8),float(e)])
            effective=[];peaks=[]
            for e in sorted(epochs):
                stage=[st for st in stages if st['start_epoch']<=e][-1]
                effective.append((e,stage[key]*schedule_multiplier(e,schedule)));peaks.append((e,stage[key]))
            series.extend([dict(label=label+' 有效计划LR',points=effective,color=color),dict(label=label+' 阶段峰值',points=peaks,color=color,dashed=True)])
            if panel==0:
                for st in stages:
                    f=schedule_multiplier(st['start_epoch'],schedule)
                    stage_rows.append((label,st['name'],st['start_epoch'],f'{st["action_lr"]:.1e} / {st["memory_lr"]:.1e}',f'{f:.4g}',f'{st["action_lr"]*f:.2e} / {st["memory_lr"]*f:.2e}'))
        body+=chart_panel(20+panel*610,28,590,365,'Action LoRA' if panel==0 else 'Scene / predictive memory',series,'有效 fresh scene exposure epoch','计划 LR（不是实测）',xmax=8.2)
    return figure(root,'performance_planned_lr.svg',svg_document('登记学习率计划：阶段峰值乘全局曝光调度',body,height=440),
        '计划，不是实测。实线=阶段峰值×连续全局调度，虚线=阶段峰值。warmup .08 epoch，保持至 .75，cosine 至4，之后 floor .05；切阶段不重启。S0/S1 同计划，S5 E4 action 实际计划为5e-7（非1e-5），memory5e-8。')+table(['计划路线','阶段','起始 epoch','Action / memory 阶段峰值','全局倍率','阶段起点有效计划 LR'],stage_rows)


def native_figures(root,s):
    process='';summaries=[];series_by_key={k:[] for k in ['action','memory','gradient','sampled','nominal']}
    for index,run in enumerate(s['current_runs']):
        arm=run['id'];rows=s['journals'].get(arm,[]);color=COLORS[index%len(COLORS)];label=arm.split('_')[0]+(' LOW' if arm=='S0_LOW_LR' else '')
        points={k:[] for k in series_by_key};seen=set();numerator=denominator=b1good=b1count=0
        for row in rows:
            u=row.get('updates');lrs=row.get('optimizer_group_lrs_used',{})
            if isinstance(lrs,dict):
                for key,nativekey in [('action','action_lora'),('memory','scene_predictive_memory')]:
                    if isinstance(lrs.get(nativekey),(int,float)):points[key].append((u,lrs[nativekey]))
            if isinstance(row.get('gradient_norm'),(int,float)):points['gradient'].append((u,row['gradient_norm']))
            # A collection is reused M times. Count support once at inner1.
            if row.get('inner_completed')!=1:continue
            collection=row.get('collection')
            if collection in seen:continue
            seen.add(collection)
            q=row.get('exploration',{}).get('quality_counts_global',{})
            n=q.get('valid_scenes');good=q.get('sampled_main_group_strict_good');b=q.get('nominal_B1_scored');bg=q.get('nominal_B1_strict_good')
            if isinstance(n,(int,float)) and n>0 and isinstance(good,(int,float)):
                numerator+=good;denominator+=n;points['sampled'].append((u,100*good/n))
            if isinstance(b,(int,float)) and b>0 and isinstance(bg,(int,float)):
                b1good+=bg;b1count+=b;points['nominal'].append((u,100*bg/b))
        for key,values in points.items():
            if values:series_by_key[key].append(dict(label=label,points=values,color=color))
        if rows:
            summaries.append((arm,f'{rows[0].get("updates")}–{rows[-1].get("updates")}',len(rows),len(seen),f'{numerator}/{denominator} ({100*numerator/denominator:.1f}%)' if denominator else '未记录',f'{b1good}/{b1count} ({100*b1good/b1count:.1f}%)' if b1count else '尚未评分 / 未启用'))
    if not summaries:return '<p>正式 native 尚无可用实际日志；不绘制零值替代曲线。</p>'
    for panel,(key,title,unit) in enumerate([('action','实际 Action used LR','LR'),('memory','实际 Memory used LR','LR'),('gradient','实际 DDP preclip gradnorm','norm')]):
        process+=chart_panel(12+panel*408,28,397,365,title,series_by_key[key],'实际 Adam update（各路线独立）',unit)
    out=figure(root,'performance_native_process.svg',svg_document('正式训练实际 LR 与 preclip gradient norm',process,height=460),
        '实际有界 rank0 日志：每路线最多末1024行 / 2MiB；LR 取 step 前 optimizer_group_lrs_used，非下一步LR。Action与memory独立坐标。M2/M4 的同 Adam 位置不等曝光或候选成本；资格值不混入。')
    support=chart_panel(20,28,590,365,'Sampled 主组：存在 strict-high 候选的场景',series_by_key['sampled'],'实际 Adam update','场景覆盖率 %',ymax=100)
    support+=chart_panel(630,28,590,365,'真实单轨 B1：strict-high 场景覆盖',series_by_key['nominal'],'实际 Adam update','已评分 B1 中占比 %',ymax=100)
    out+=figure(root,'performance_sampled_vs_B1.svg',svg_document('训练采样支持与实际单轨 B1 覆盖分别展示',support,height=460),
        '只计每 collection 的 inner1一次：sampled 主组分母是有效场景，B1 分母仅实际额外评分场景。Strict-high=NC/DAC/TTC/Comfort全1且canonical≥.9。B1未启用时不绘零线；候选支持不是部署 NAVTEST 成绩。')
    return out+table(['正式路线','bounded update区间','rank0行数','独立 collection','主组 sampled strict-high 场景','实际 B1 strict-high'],summaries)


def render_fragment(root=ROOT):
    root=Path(root);s=snapshot(root)
    retired=[j for j in s['jobs'] if j['status'] not in ('RUNNING','DISPATCHED')]
    checks=s['audit'].get('checks',{})
    best=max(s['complete_results']+s['legacy_complete_results'],key=lambda r:r['PDMS'],default=None)
    previous=s['previous_peak'];prior_pdms=max(91.45359756166451,previous.get('PDMS',0))
    prior_label=('%.8f · '%previous['PDMS']+previous['arm']+' / Adam '+str(previous['updates'])) if previous.get('PDMS',0)>91.45359756166451 else '91.45359756 · SafetyEP Control 10k'
    best_label=('%.8f · '%best['PDMS']+best['arm']+' / Adam '+str(best['updates'])) if best and best['PDMS']>prior_pdms else prior_label
    body=D46_priority_fragment()+'<section id="performance-current" style="background:#fff;border:1px solid #cad9e8;border-radius:14px;padding:24px;margin:20px 0"><small>既有 IL70 系 · PERFORMANCE FIRST · '+esc(s['generated_UTC'])+'</small><h2>最高完整 PDMS：'+esc(best_label)+'</h2>'
    evidence=Path(best['summary']) if best and best['PDMS']>prior_pdms else Path(previous['evidence']) if previous.get('PDMS',0)>91.45359756166451 else Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST/SUMMARY.json')
    if evidence.exists():body+='<p><a href="'+esc(evidence)+'">最高分完整评测证据</a></p>'
    legacy_active=sum(r['status']=='实际训练' for r in s['legacy_current_runs'])
    body+='<p><b>既有 IL70 系当前实际在训 RL 共 '+str(s['active_formal']+legacy_active)+' 条：本轮 '+str(s['active_formal'])+' 条，旧综合 I '+str(legacy_active)+' 条；本轮已登记 '+str(len(s['current_runs']))+' 条；资格测试 '+str(s['active_qualification'])+' 个。</b>此计数不包含上方独立 D46 研究。实际在训要求当前 stage job 有 RUNNING、无 RESULT，且八 rank 均有近10分钟日志；统计时间见上。本轮完整 NAVTEST 结果 '+str(len(s['complete_results']))+' 个；native0、partial、资格测试与候选 oracle 不计为 RL 提分。</p>'
    body+='<p><b>既有研究任务：</b>S1/S6 将已验证好候选吸收到真实 B1；S7 补充结构化进度教师，再经 CFM+真实 ODE-last 吸收；S5 分阶段覆盖、吸收与持续保持。各路线均从相同 IL70k（88.79290991、SHA 7dda5b14…）和 Adam0 独立开始；当前执行状态以任务表为准。</p>'
    body+='<p>实际训练范围为 Action LoRA、Scene Memory 与 Predictive Memory，共 52,707,840 个可训练参数。Qwen 主干与永久 IL 参考冻结；当前可训练 memory/KV 每次 inner 都重新构图。吞吐缓存只复用冻结分支，不把场景表征整体固定成仅训练动作头。</p>'
    body+=table(['正式路线','服务器','实际 Adam（8rank最小）','有效曝光 epoch','当前状态'],[(j['id'],j['host'] or '待分配',j['updates'],f'{j["exposure_epoch"]:.6f}' if j['exposure_epoch'] is not None else '等待首条更新',j['status']+('；'+j['resource_note'] if j.get('resource_note') else '')) for j in s['current_runs']])
    if s['complete_results']:
        results=sorted(s['complete_results'],key=lambda r:Path(r['summary']).stat().st_mtime,reverse=True)
        body+='<h3>本轮完整单轨迹评测</h3>'+table(['评测','Adam','PDMS','有效场景','invalid'],[(r['arm'],r.get('updates'),r.get('PDMS'),r['scenes'],r['invalid_scenes']) for r in results])
        S5=next((r for r in results if r['arm']=='S5_PERFORMANCE_MAX' and r['updates']==1616),None)
        if S5:
            body+='<p><b>S5 首个完整 0.25 epoch：'+esc(f'{S5["PDMS"]:.8f}')+'</b>。该检查点仍为 phase1 基础 SafetyEP 覆盖，rescue 与历史保持均未激活；不能将此分数归功于尚未启用的组件。它与 FAST 的训练批量/Adam时钟不同，尚不能据两个早期点判算法优劣。</p>'
            review=s['S5_review']
            if review:
                z=review['zero_Control'];p=review['phase'];f=review['fixed']
                body+='<p>S5 与 Control10k 的零分总数同为 '+esc(z['target'])+'，但修复 '+esc(z['repaired'])+'、新增 '+esc(z['new'])+'；固定 IL70 高分队列相 Control 的全局差额 '+esc(f'{f["global_PDMS_contribution"]:+.6f}')+' 点，其余 '+esc(f'{f["other_scenes_global_contribution"]:+.6f}')+' 点。八 rank 原生审计确认曝光 '+esc(f'{p["actual_exposure_epoch"]:.6f}')+' epoch，B1/额外教师/history查询均为0，尚未检验 PMax 的后续吸收/保持机制。<a href="'+esc(review['path'])+'">S5全量分析</a> · <a href="'+esc(review['phase_path'])+'">实际阶段与预算</a>；描述性关联，非等算力因果结论。</p>'
    else:body+='<p><b>本轮尚无完整正式 NAVTEST 分数。</b>后续只接受 sealed SPEC + identity、全12146/invalid0/44log、seed2101 的既有结果。</p>'
    body+=FAST_review_fragment(s)
    body+=TRAIN2048_pair_fragment(s)
    body+=S7_fragment(s)
    if s['legacy_current_runs']:
        body+=table(['独立继续的旧实验','服务器','实际 Adam（8rank最小）','状态'],[(j['id'],j['host'],j['updates'],j['status']) for j in s['legacy_current_runs']])
    if s['legacy_complete_results']:
        body+='<p>旧综合 I 已完成的真实单轨迹 NAVTEST（与新 S5 分开）：'+esc('；'.join('Adam '+str(r['updates'])+' / PDMS '+f'{r["PDMS"]:.8f}' for r in s['legacy_complete_results']))+'。未完成的后续评测不发布均分。</p>'
    body+=verified_quality_fragment(s)+first_window_progress_fragment(s)+train_GT_fragment(s)
    body+=reviewed_I_fragment(root)
    body+='<h3>当前结论与接下来的决策</h3><p>主训练仍处于全量均匀覆盖阶段。S1 的 rescue 到 0.5 epoch 才启用；此前它与 S0 的目标相同，独立进程出现相同更新属于预期，不能据此判断 rescue 无效。S5 的历史保持到第 4 epoch 才启用，当前也不能评价其收益。</p>'
    body+='<p>S0（Adam890/inner2）与 S2（Adam404）已原生保存，为 S7 让出资源；S4 与 LOWLR 也保留原生状态。代码、结果与 Adam/RNG 保留，这是研究优先级处置，不判方法无效。正式分数只读完整 NAVTEST；资格、GT 和合成教师诊断不计入排行榜。</p>'
    lineage=s['credit_lineage']
    if lineage:
        body+='<details><summary>优势信用谱系核对：'+esc(lineage['groups'])+' 组有界真实 TRAIN 重放</summary>'
        for line in lineage['summary'][:2]:body+='<p>'+esc(line)+'</p>'
        body+='<p>该窗口不能代表完整训练人口，rawA能量占比不是参数梯度比例；前1024 token集合匹配也不等于G/Adam/算力匹配。保留原至少3/首6epoch长期检验，不能据早期0.25epoch判断上限。<a href="'+esc(lineage['path'])+'">完整谱系与同材料反事实</a></p></details>'
    tq=s['TRAIN2048_quality']
    if tq.get('status')=='COMPLETE':
        body+='<p>固定 TRAIN2048 初始 IL70 真实 B1 诊断完成：零分 '+esc(tq.get('zero_count'))+'，严格高质量 '+esc(tq.get('strict_high_count'))+'。这是训练集诊断，后续按同场景配对；不计入 NAVTEST 成绩。全 NAVTRAIN 初始 B1 普查单独使用 7 张可核验 GPU，查询预算 103288，当前状态 '+esc(s['full_TRAIN_census'].get('status','未启动'))+'，未完成时不发布全量均分。</p>'
    returned=s['permanent_continuation']
    if returned.get('status')=='PASS':
        cp=returned.get('checkpoint',{})
        body+='<p>S1 已从临时主机按完整原生状态迁往长期可用的共享资源：保存于 Adam '+esc(cp.get('updates'))+' / pending inner '+esc(cp.get('next_inner'))+'；保留同一实验的 Adam、RNG、场景游标和待更新轨迹。运行是否已恢复以当前任务表为准。</p>'
    diagnosis=s['training_diagnosis']
    if diagnosis.get('status')=='COMPLETE':
        quality=[]
        for arm,item in diagnosis.get('routes',{}).items():
            q=item.get('quality_journal_census',{})
            if not q:continue
            quality.append((arm,q.get('valid_scenes'),f'{100*q.get("strict_high_support_rate",0):.2f}%',
                f'{100*q.get("PG0_rate",0):.2f}%',f'{100*q.get("canonical_allzero_rate",0):.3f}%',q.get('nominal_B1_scored')))
        if quality:
            body+='<details><summary>完整已观测训练日志：探索支持与无优势组（'+esc(diagnosis.get('completed_at',''))+'）</summary>'
            body+=table(['路线','有效场景','G16 有严格高分候选','PG0 组','G16 官方全零组','真实 B1 查询'],quality)
            body+='<p>按 collection 去重，不重复计算四 inner。PG0 是没有有效组内策略梯度；全零组是所有候选的官方分数均为零，两者不同。有限行为窗口中多数 PG0 来自成功且 EP 饱和。当前尚未查询 B1 的项目显示 0 次查询，不代表 B1 得分为零；G16 覆盖率不能代替部署成绩。不同时间窗口场景不同，比例变化不证明遗忘。</p></details>'
        history=diagnosis.get('historical_SafetyEP_NAVTEST_zero_flow',{}).get('longitudinal',{})
        if history:
            body+='<p><b>历史完整分析：</b>旧独立 SafetyEP 序列在 12146 场景、13 个评测点中，有 '+esc(history.get('persistent_zero'))+' 个场景在所有已观测点均为零，'+esc(history.get('intermittent_zero'))+' 个间歇零分，其中 '+esc(history.get('relapsed_zero'))+' 个出现过修复后再归零。这支持同时研究困难场景学习信号与防退化保持，但尚不能区分内在驾驶冲突和共享参数更新的因果作用。该历史序列与本轮结果分开；NAVTEST 场景身份不进入训练优先级。</p>'
    qualified=s['execution_qualifications'];rows=[]
    cache=qualified['frozen_reference']
    if cache.get('status')=='PASS':
        original=cache.get('results',{}).get('original',{});cached=cache.get('results',{}).get('cached',{})
        rows.append(('冻结 IL 参考计算复用','真实 IL70 / 单 GPU / 4 inner',
            str(original.get('reference_velocity_calls'))+' → '+str(cached.get('reference_velocity_calls'))+' 次参考调用',
            '八卡原生恢复逐位一致，已用于 S1；原目标和 Adam/RNG 保持' if s['cache_world8'].get('status')=='PASS' else '梯度、参数、Adam、scheduler、RNG 逐位一致；待八卡原生恢复验证'))
    for name in ('PACK2','PACK4'):
        proof=qualified[name]
        if proof.get('status')=='PASS':
            first=proof.get('iterations',[{}])[0]
            rows.append((name+' 一致合批采样与更新','真实 IL70 / 单 GPU / '+str(len(proof.get('scenes',[])))+' TRAIN 场景',
                '首更 ratio 偏差 '+str(first.get('fresh_ratio_max_abs')),
                'PACK4 八卡参数/Adam/八rank RNG/stream 逐位恢复通过；独立正式训练从 IL70 开始' if name=='PACK4' and s['packed_world8'].get('status')=='PASS' else '单卡通过；须独立从 IL70 开始，不能替换现有单场景轨迹的更新内核'))
    if rows:body+=table(['吞吐优化','验证范围','实际结果','当前处置'],rows)
    window=s['resource_window'];restore=s['resource_restore'];deadline=window.get('deadline_unix',1793664000.)
    guard=restore.get('new_guard_seconds',900);phase='尚未进入保存窗口' if time.time()<deadline-guard else '已进入保存退出窗口' if time.time()<deadline else '临时授权已到期，退出情况以实际收据为准'
    body+='<details><summary>旧 IL70 资源窗口与归还记录（历史；D46新授权见最前卡片）</summary><p><b>原资源期限：</b>六台 wm4 / rl2 / tsh2 / wmzt / tsh3 / wm2 于 UTC11月3日00:00（北京时间08:00）硬截止；本任务原900秒余量已恢复，UTC23:45（北京时间07:45）开始保存，原窗口：'+esc(phase)+'。UTC23:40对应北京时间07:40。<a href="'+esc(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/WINDOW_CORRECTION.json')+'">窗口纠正</a> · <a href="'+esc(root/'operations/RESTORE_REQUESTED_15MIN_GUARD/CLOSE_WINDOW_LAUNCH.json')+'">close_window实际启动</a>。其它任务授权不变，未知任务不回收。</p>'
    release=root/'operations/RESOURCES_UNTIL_20261103T000000Z/ACTUAL_PRE_EXPIRY_RELEASE.json'
    if release.exists():body+='<p><a href="'+esc(release)+'">23:51实际归还核对：六机自有任务进程为0</a>；永久资源上的训练与foreign任务分开，不把它们误报已退出。</p>'
    body+='</details>'
    body+='<p>S7 长期槽位等待器当前只观察旧 I 自然完成10k；之后仍须满足实际显存、可见 owner 与最多2个已知 native 的准入限制。当前 '+esc(s['S7_permanent_wait'].get('status','尚无收据'))+'；不停止旧 I、第三方 D44 或 B/D，也不降低门槛硬塞。评测未齐不发均分，资源等待不判算法无效。<a href="'+esc(root/'operations/S7_PERMANENT_CAPACITY_WAIT/STATE.json')+'">等待器状态</a></p>'
    body+='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px"><article><h3>已完成：全量数据审计</h3><p>'+esc(checks.get('unique_tokens','待查'))+' 唯一 TRAIN；NAVTEST token/log 交集 0。309864 图像文件、103288 cache 文件逐个筛查无缺失；cache 格式筛查不等同完整语义评分。</p></article><article><h3>尚待回答：是否真正提分</h3><p>重点看固定 TRAIN 中 sampled-good/B1-bad 的后续修复，以及全 NAVTEST 的零分修复和高分退化。不会以扩大采样覆盖率代替单轨迹部署成绩。</p></article></div>'
    body+='<h3>实测工程成本与评分预算</h3><p><a href="'+esc(root/'performance_qualification.json')+'">资格计时与预算原始汇总</a></p>'+engineering_figure(root,s)
    body+='<h3>登记学习率计划：两类参数分别显示</h3>'+planned_figure(root,s)
    body+='<h3>正式训练实测：更新尺度与支持转换</h3>'+native_figures(root,s)
    body+=table(['路线','主要机制','B / G / M','计划与评测'],[
        ('S1 · 第一优先','SafetyEP + verified CFM + 真实 ODE last-state SmoothL1','128 /16 /4','0.25→0.5→1→3→6 epochs'),
        ('S5 · 第一优先','SafetyEP→Rescue→持续历史教师→低 LR 精修','64→128 /16 /4','同一 run 保留 Adam / RNG'),
        ('S6 · 新质量吸收路线','真实 B1 + strict-high gain双门 + PG-EMA校准','64 /16 /4','epoch0启用；首评 .0625 / .125；状态见实时任务表'),
        ('S7 · 结构化进度教师','主/额外G16均无high后新增已验证teacher-only候选；CFM+真实ODE-last','64 /16 /4','八rank资格PASS；注册/启动与实际训练分开'),
        ('S0 / LOWLR','大场景覆盖 + 曝光 warmup/hold/cosine','128 /16 /4','5e-5 与 3e-5 分别 fresh IL'),
        ('S2','保守优势加权 FM，负A二次项有下界','128 /16 /2','MC4，独立 IL70'),
        ('S3','逐 MC energy-ratio FPO + 原物理锚定','128 /16 /2','MC2；非 endpoint density'),
        ('S4','真实时序安全 + nominal rescue','128 /16 /4','原生保存，资源让位；未判无效')])
    body+='<h3>最近工程结论</h3><p>四种 B/G 的真实八卡窗口已通过。B128/G16 的拆分恢复参数、Adam、scheduler 与八 rank RNG/stream 逐位一致；完整 IL70 的真 ODE 末状态监督和全 scope 梯度通过。S1/S2/S3/S4/S5 辅助启用资格均已完成，尚不能称部署提分。</p><p>G32 按完整组归一化；额外条件 G16 仅用于教师挖掘。padding 排除全部 loss/std/有效曝光；当前 memory 每次 inner 重建计算图。首窗口约18小时/epoch，启动、评测、新辅助项和恢复重放另计。</p>'
    body+='<details><summary>资格/调试记录（保留失败，不作为算法负结果）</summary>'+table(['任务','服务器','状态','返回码'],[(j['id'],j['host'],j['status'],j['returncode']) for j in retired])+'<p>首轮真实评分后，恢复哈希未支持 NumPy 数组而在 Adam 前失败；修正后使用新冻结资格版本，不继承资格权重。中间未完成的源码副本派发已停止并保留收据。</p></details>'
    body+='<p><a href="/mnt/project/qwen-world-performance-rl-20261103/report.html">本轮独立报告</a> · <a href="/mnt/project/qwen-world-performance-rl-20261103/data/AUDIT.json">完整审计</a></p></section>'
    compact=dict(s);compact['journals']={arm:dict(path=str(Path(next(r['output'] for r in s['current_runs'] if r['id']==arm))/'metrics_rank0.jsonl'),rows=len(rows),first_update=rows[0].get('updates') if rows else None,last_update=rows[-1].get('updates') if rows else None) for arm,rows in s['journals'].items()}
    compact['figures']=[str(p) for p in sorted((root/'reporting/figures').glob('performance_*.svg'))]
    atomic_text(root/'report_snapshot.json',json.dumps(compact,ensure_ascii=True,indent=2))
    css='body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#203348;max-width:1450px;margin:24px auto;padding:0 20px;line-height:1.6}th{text-align:left;padding:9px;background:#e8eff7}h2{color:#126b57}a{color:#1763a1}'
    atomic_text(root/'report.html','<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World Performance RL</title><style>'+css+'</style><body>'+body+'</body></html>')
    return body


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();render_fragment(args.root)
    report=read(args.root/'report_snapshot.json',{})
    print(json.dumps(dict(report=str(args.root/'report.html'),generated_UTC=report.get('generated_UTC'),actual_training=report.get('active_formal'),complete_NAVTEST=len(report.get('complete_results',[])),figures=len(report.get('figures',[])))))
