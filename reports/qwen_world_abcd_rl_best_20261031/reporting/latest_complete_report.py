"""Report-owned synthesis of existing complete summaries and paired reviews.

No scene records, training selection, bootstrap, dispatch, or process mutations.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


NAMES = {
    'SAFETY_G8_CONTROL': 'Control', 'STANDARD_INVERSION_G8': '纠偏信用',
    'STANDARD_COMBINED_G8': '三信用组合', 'STANDARD_ENDPOINT_LR02_G8': 'Endpoint低LR',
    'REVIVAL_G16': 'G16', 'REVIVAL_DENSE_G8': 'DenseDAC',
    'REVIVAL_SATURATED_G8': '饱和信用', 'REVIVAL_ACTIVE_DATA_G8': 'Active课程',
    'IMPROVEMENT_EARLY_DECAY_G8': 'H·更早衰减',
    'IMPROVEMENT_MULTISCENE32_G8': 'H·不同场景Batch32',
    'IMPROVEMENT_VERIFIED_RETENTION_G8': 'H·质量门控历史RL保持',
    'IMPROVEMENT_FP32_OUTPUT_G8': 'H·真FP32动作投影',
    'IMPROVEMENT_MIXED_NOISE_G8': 'H·固定G8混合探索',
    'IMPROVEMENT_RETENTION_EARLY_DECAY_G8': 'H·质量保持＋更早衰减',
    'INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC': 'I·B64／G16保持＋WKC综合',
}
CONTROL = 'SAFETY_G8_CONTROL'


def render_top_summary(root, snapshot, h):
    """Fresh current-job census, independent of old RUNNING files or registry counts."""
    studies=snapshot.get('editorial_analysis',{}).get('standard_studies',[])
    pool=list(snapshot.get('results',[]));known={r['id']for r in snapshot.get('runs',[])};run_index={}
    for study in studies:
        pool+=study['results'];known.update(r['id']for r in study['runs'])
        for run in study['runs']:run_index[run['id']]=run
    for name in ['exploration_research','abcd_research']:
        value=snapshot.get('editorial_analysis',{}).get(name,{})
        pool+=value.get('results',[]);known.update(r['id']for r in value.get('runs',[])if r.get('id'))
    rl_update=lambda r:r.get('RL_updates')if h.finite(r.get('RL_updates'))else r.get('updates',0)
    valid=[r for r in pool if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'
           and r.get('scenes')==12146 and r.get('invalid_scenes',0)==0 and r.get('arm')in known
           and(rl_update(r)or 0)>0 and 'oracle'not in str(r.get('schema','')).lower()]
    best=max(valid,key=lambda r:r['PDMS'],default={});warnings=[];inputs=[];rows=[]
    counts={k:0 for k in ['training','starting','waiting_resource','resource_reserved','exited_review','paused_saved','ended','unconfirmed']}
    by_study={s['root']:dict.fromkeys(counts,0)for s in studies}
    read=lambda p:h.read_json(Path(p),warnings,inputs)or{}
    for study in studies:
        for original in study['runs']:
            owner=read(original['owner_state'])if original.get('owner_state')else{}
            node=owner.get('runs',{}).get(original['id'],{})
            run={**original,**node};status=str(run.get('status','')).upper()
            target=node.get('stage_target_updates',run.get('stage_target_updates'))
            stage=node.get('stages',{}).get(str(target),{})
            stage_job=stage.get('job');current_job=stage_job
            waiting_without_job=status.startswith('WAITING_')and any(v in status for v in ['RESOURCE','MIGRATION','CAPACITY'])and not stage_job
            if not current_job and not waiting_without_job and node.get('job'):
                try:matches=Path(node['job']).name.startswith(f'formal_to{int(target):06d}_')
                except(TypeError,ValueError):matches=False
                if matches:current_job=node['job']
            job=Path(current_job)if current_job else None
            result=read(job/'RESULT.json')if job else{};running=read(job/'RUNNING.json')if job else{}
            rc=result.get('returncode',result.get('exit_code'))
            reserve=any('RESEARCH_RESERVE'in str(run.get(k,''))for k in ['standard_disposition','revival_disposition','disposition'])
            now=time.time();ranks=[];started=running.get('start_unix')or 0
            if running and not result:
                for rank in range(8):
                    p=Path(run['root'])/f'metrics_rank{rank}.jsonl';tail=h.read_tail(p,limit=1)
                    if tail and p.exists():ranks.append(dict(rank=rank,updates=tail[-1].get('updates'),mtime=p.stat().st_mtime))
            now=time.time()
            fresh=len(ranks)==8 and all(isinstance(r['updates'],(int,float))and r['updates']>0
                and 0<=now-r['mtime']<=180 and r['mtime']>=started-1 for r in ranks)
            if not result and running.get('status')=='RUNNING'and fresh:category='training'
            elif reserve and(result or not running):category='resource_reserved'
            elif waiting_without_job:category='waiting_resource'
            elif result and rc==75 and('WAITING'in status or result.get('child_not_started')is True):category='waiting_resource'
            elif result and rc not in(None,0):category='exited_review'
            elif any(v in status for v in ['PAUSE','MIGRATION','STOPPED_AT_COOPERATIVE_BOUNDARY']):category='paused_saved'
            elif result:category='ended'
            elif 'WAITING'in status or 'RESOURCE'in status:category='waiting_resource'
            elif running and now-started<=180:category='starting'
            else:category='unconfirmed'
            counts[category]+=1
            by_study[study['root']][category]+=1
            rows.append(dict(arm=run['id'],study_root=study['root'],category=category,host=run.get('host'),
                native_status=status,job=str(job)if job else None,returncode=rc,running_receipt_present=bool(running),
                terminal_receipt_present=bool(result),common_updates=min((r['updates']for r in ranks),default=None),
                rank_last_updates={str(r['rank']):r['updates']for r in ranks},
                rank_log_ages_seconds={str(r['rank']):now-r['mtime']for r in ranks}))
    checkpoint=None;arm=best.get('arm');u=rl_update(best)
    if arm in run_index and isinstance(u,(int,float)):
        candidate=Path(run_index[arm]['root'])/'checkpoints'/f'update{int(u):06d}_inner{int(u)%4}'
        if candidate.is_dir():checkpoint=str(candidate)
    observed=h.utc();artifact=root/'reporting'/'FRONT_PAGE_SUMMARY.json'
    summary=dict(schema='report_highest_full_RL_and_current_native_count_v1',observed_utc=observed,
        best={k:best.get(k)for k in ['arm','PDMS','updates','RL_updates','evidence','model_sha256']},
        best_checkpoint=checkpoint,counts=counts,by_study=by_study,runs=rows,training_scope='Registered related-study current native stage jobs; includes I only after real admission',
        freshness_seconds=180,inputs=inputs,warnings=warnings,
        definition='Current stage RUNNING and no RESULT; all8 rank journals newer than worker.start and <=180s. Count tasks, not GPUs.')
    h.atomic_json(artifact,summary);snapshot['front_page_summary']=summary
    link=lambda p,t:h.relative_link(root,p,t)
    method='SafetyEP标准Control'if arm==CONTROL else NAMES.get(arm,arm or'等待完整结果')
    best_html='<article><span>当前最高完整RL成绩 · 12,146场景／invalid0</span><b class="overview-score">'+h.fmt(best.get('PDMS'),8)+'</b><div>'+h.esc(method)+' · RL '+h.number(u)+'更新</div><small>'+link(best.get('evidence'),'完整SUMMARY')+' · '+(link(checkpoint,'checkpoint '+Path(checkpoint).name)if checkpoint else 'checkpoint见结果证据')+'</small></article>'
    scope_label='F／G／H／I'if any(Path(s['root']).name.startswith('integrated_')for s in studies)else 'F／G／H'
    current='<article><span>当前实际正在训练 · '+scope_label+'</span><b class="overview-score">'+h.number(counts['training'])+' 个任务</b><div>当前job RUNNING、无RESULT，且全部8rank日志新鲜</div><small>已退出待复核 '+h.number(counts['exited_review'])+' · 资源预留 '+h.number(counts['resource_reserved'])+' · 启动 '+h.number(counts['starting'])+' · 等资源 '+h.number(counts['waiting_resource'])+'</small><small>保存暂停 '+h.number(counts['paused_saved'])+' · 已结束 '+h.number(counts['ended'])+' · 运行未确认 '+h.number(counts['unconfirmed'])+'</small></article>'
    return '<div class="overview">'+best_html+current+'</div><small>实时统计 '+h.esc(observed)+' · 数的是训练任务，不是独占物理GPU；新鲜阈值180秒。已退出待核不等于科学无效，也不代表已正常迁移。'+link(str(artifact),'逐任务实时证据')+'</small>'


def build_review(studies):
    points = {}; pairs = []; inputs = {}; runs = {}; query_budgets = {}; combination_preparation = {}
    def evidence(path):
        if path and path not in inputs:
            p = Path(path)
            if p.is_file():
                inputs[path] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                    size_bytes=p.stat().st_size)
        return path
    for study in studies:
        root = study['root']
        budget_path=Path(root)/'analysis'/'RETENTION_5K_ACTUAL_QUERY_BUDGET.json'
        if budget_path.is_file():
            value=json.loads(budget_path.read_text())
            if value.get('status')=='COMPLETE_ALL8_SEALED_RANK_STATES':
                query_budgets['IMPROVEMENT_VERIFIED_RETENTION_G8_u005000']={k:v for k,v in value.items()if k!='ranks'}
                query_budgets['IMPROVEMENT_VERIFIED_RETENTION_G8_u005000']['evidence']=evidence(str(budget_path))
        prepared_path=Path(root)/'analysis'/'RETENTION_EARLY_DECAY_PREPARED_20261102.json'
        if prepared_path.is_file():
            value=json.loads(prepared_path.read_text())
            combination_preparation={k:v for k,v in value.items()if k in ['schema','status','arm','source','source_commit','config','CPU_check','registration_draft','handoff']}
            combination_preparation['evidence']=evidence(str(prepared_path))
        for run in study['runs']:
            runs[(root, run['id'])] = {k: run.get(k) for k in
                ['id', 'updates', 'status', 'host', 'journal_ranks', 'new_candidates',
                 'raw_scene_exposures', 'worker_returncode', 'worker_receipt_status',
                 'progress_freshness', 'job', 'improvement_role', 'standard_disposition',
                 'revival_disposition', 'disposition', 'resource_priority_pause_receipt']}
        for result in study['results']:
            if not result.get('publishable_complete') or result.get('cohort') != 'NAVTEST':
                continue
            key = (root, result['arm'], result['updates'])
            points[key] = dict(arm=result['arm'], updates=result['updates'], PDMS=result['PDMS'],
                study_root=root, evidence=evidence(result.get('evidence')), scenes=12146)
        for pair in study.get('paired_reviews', []):
            evidence(pair['evidence'])
            if (root, pair['arm'], pair['updates']) in points:
                pairs.append(dict(pair, study_root=root))
    def historic(root, arm, update):
        return next((p for p in pairs if p['study_root'] == root and p['arm'] == arm
                     and p['updates'] == update and p['kind'] == 'historical'
                     and p['reference'] == 'IL70'), None)
    control_root = next((s['root'] for s in studies if Path(s['root']).name == 'standard_training_20261101'), '')
    for (root, arm, u), point in points.items():
        il = historic(root, arm, u)
        if il:
            r = il['result']
            point['zeros'] = r.get('baseline_zero_denominator', 0) - r.get('repaired_zero', 0) + r.get('new_zero', 0)
            point['fixed_IL70_high_mean_delta_PDMS'] = r.get('fixed_control_high', {}).get('mean_delta_PDMS')
            point['fixed_IL70_high_count'] = r.get('fixed_control_high', {}).get('count')
    comparisons = []
    for pair in pairs:
        if pair['kind'] != 'between_arms' or pair['reference'] != CONTROL or pair.get('reference_root') != control_root:
            continue
        target = points[(pair['study_root'], pair['arm'], pair['updates'])]
        base = points.get((control_root, CONTROL, pair.get('reference_updates')))
        if not base:
            continue
        r = pair['result']; factor = 16 if pair['arm']=='INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'else 2 if pair['arm'] == 'REVIVAL_G16' else 4 if pair['arm'] == 'IMPROVEMENT_MULTISCENE32_G8' else 1
        a = target.get('fixed_IL70_high_mean_delta_PDMS'); b = base.get('fixed_IL70_high_mean_delta_PDMS')
        fixed_delta = a-b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else pair.get('fixed_IL70_high_advantage_PDMS')
        comparisons.append(dict(arm=pair['arm'], updates=pair['updates'], reference_updates=base['updates'],
            study_root=pair['study_root'], target_PDMS=target['PDMS'], baseline_PDMS=base['PDMS'],
            delta_PDMS=r.get('PDMS_delta'), CI95_PDMS=r.get('PDMS_CI95'),
            repaired_zero=r.get('repaired_zero'), new_zero=r.get('new_zero'),
            baseline_zeros=r.get('baseline_zero_denominator'), target_zeros=target.get('zeros'),
            fixed_IL70_high_mean_delta_PDMS=fixed_delta,
            fixed_IL70_high_population_mass_delta_PDMS=fixed_delta*8724/12146 if fixed_delta is not None else None,
            fixed_IL70_high_count=8724, fixed_IL70_high_CI95_PDMS=None,
            fixed_Control_high_mean_delta_PDMS=r.get('fixed_control_high', {}).get('mean_delta_PDMS'),
            nominal_source_candidate_matched=base['updates'] == factor*pair['updates'],
            source_update_clock_multiplier=factor, budget_contract=pair.get('budget_contract'),
            source_scene_clock_multiplier=8 if pair['arm']=='INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'else 1 if pair['arm']=='REVIVAL_G16'else factor,
            source_candidate_clock_multiplier=factor,
            comparison_axis=pair.get('comparison_axis'),
            actual_collection_budget=pair.get('actual_collection_budget'), evidence=pair['evidence']))
    arms = []
    for root, arm in sorted({(root, arm) for root, arm, u in points} | set(runs)):
        curve = sorted([p for (rt, a, u), p in points.items() if rt == root and a == arm], key=lambda p: p['updates'])
        matched = sorted([p for p in comparisons if p['study_root'] == root and p['arm'] == arm
                          and p['nominal_source_candidate_matched']], key=lambda p:p['updates'])
        arms.append(dict(arm=arm, name=NAMES.get(arm, arm), study_root=root, curve=curve,
            latest_complete=curve[-1] if curve else None, observed_peak=max((p for p in curve if p['updates']>0), key=lambda p:p['PDMS'], default=None),
            latest_candidate_matched=matched[-1] if matched else None,
            candidate_matched_points=matched, current_native=runs.get((root, arm), {})))
    review=dict(schema='report_existing_complete_long_run_comparison_v1',
        status='COMPLETE_SNAPSHOT_OF_AVAILABLE_FULL_RESULTS', created_utc=datetime.now(timezone.utc).isoformat(),
        complete_points=len(points), complete_comparisons=len(comparisons), arms=arms,
        observed_peak=max((p for p in points.values() if p['updates']>0), key=lambda p:p['PDMS'], default=None),
        comparisons=comparisons, actual_training_query_budgets=query_budgets, combination_preparation=combination_preparation,
        integrated_studies=[dict(root=s['root'],registration=s['registration_path'],status=s['registration'].get('status'))for s in studies if Path(s['root']).name.startswith('integrated_')], input_files=inputs,
        scope='F/G/H only; existing complete12146/invalid0 summaries and sealed paired artifacts; no new score/bootstrap.',
        limitations=['All reported CIs reused: one training seed,44 physical logs,2000 cluster-bootstrap draws,seed42.',
            'Repeated checkpoint selection and multiple comparisons are not corrected; observed peaks are descriptive.',
            'Fixed IL70 high cohort8724 is invariant; its mean deltas have no CI here and are not significance claims.',
            'Candidate budgets do not equate scenes, Adam steps, scheduler phase, or model compute for G16/Batch32.',
            'Retention extra official ODE queries are separate; source-PG query matching is not total-query matching.',
            'NAVTEST labels are diagnostic only and never become a training bank.'])
    review['stage_gain_vs_observed_peak']=_stage_gain_context(review)
    return review


def _stage_gain_context(review):
    peak=review.get('observed_peak')or{};rows=[]
    selections=[('IMPROVEMENT_VERIFIED_RETENTION_G8',5000,5000),('IMPROVEMENT_EARLY_DECAY_G8',12000,12000),
                ('IMPROVEMENT_EARLY_DECAY_G8',14000,14000),('REVIVAL_G16',10000,20000)]
    for arm,u,base_u in selections:
        p=next((p for p in review['comparisons']if p['arm']==arm and p['updates']==u and p['reference_updates']==base_u),None)
        if p:
            rows.append({k:p.get(k)for k in ['arm','updates','reference_updates','baseline_PDMS','target_PDMS','delta_PDMS','CI95_PDMS','evidence']}|
                        dict(delta_to_observed_peak_PDMS=p['target_PDMS']-peak['PDMS'],
                             candidate_multiplier_vs_same_update_control=2 if arm=='REVIVAL_G16'else 1))
    return dict(reference_peak=peak,rows=rows,
        caveat='Matched deltas use each stated Control checkpoint. Distance to the selected peak is arithmetic across budgets, not a new matched CI or a causal method ranking.')


def _render_stage_gain_context(review,h,link,ci):
    context=review.get('stage_gain_vs_observed_peak')or _stage_gain_context(review)
    peak=context['reference_peak'];rows=[]
    budget=review.get('actual_training_query_budgets',{}).get('IMPROVEMENT_VERIFIED_RETENTION_G8_u005000',{})
    for p in context['rows']:
        arm=p['arm'];note='同Adam更新；官方验证查询另计'if arm=='IMPROVEMENT_VERIFIED_RETENTION_G8'else '同Adam／主PG候选量'if arm!='REVIVAL_G16'else '10k对Control20k同主候选量；同10k候选为2倍'
        rows.append([h.esc(NAMES.get(arm,arm))+'<small>'+h.number(p['updates'])+'更新</small>',
                     'Control '+h.number(p['reference_updates'])+'<small>'+h.fmt(p['baseline_PDMS'],8)+'</small>',
                     h.fmt(p['target_PDMS'],8),h.fmt(p['delta_PDMS'],6)+'<small>'+ci(p['CI95_PDMS'])+'</small>',
                     h.fmt(p['delta_to_observed_peak_PDMS'],6),note+'<small>'+link(p['evidence'],'原始完整配对')+'</small>'])
    if not rows:return ''
    body='<h3>同阶段增益≠超过最高分</h3><p>下面的增益分别相对明确列出的Control检查点，<b>没有加在最高 '+h.fmt(peak.get('PDMS'),8)+' 上</b>。全部从同一原始IL70独立fresh初始化；最高点与各方法的预算不同，最后一列前的峰值差只是实际分数的算术距离，不是新的配对显著性结论。</p>'
    body+=h.table(['方法／实际更新','增益所用基线／实际分','方法实际分','对该基线Δ／95%CI','距本页最高分Δ','预算与证据'],rows)
    body+='<p>目前这些正向对照支持部分阶段的优化或保持改善，尚未证明提高最优部署分数上限。保持5k的更新数少于最高10k；Early的12k／14k则多于10k；不能只按距最高的差值判方法长期优劣。</p>'
    if budget:
        source=budget.get('sourcePG',{});extra=budget.get('extra_verification',{})
        body+='<p class="muted">保持臂5k的已封存实际训练评分：主PG '+h.number(source.get('global_candidates'))+'＋fresh B1 '+h.number(extra.get('fresh_TRAIN_B1'))+'＋history B1 '+h.number(extra.get('history_TRAIN_B1'))+'＝'+h.number(budget.get('total_official_training_candidate_scores'))+'，较主PG额外 '+h.fmt(budget.get('extra_ratio'),3,percent=True)+'%；不含周期NAV评分。同5k并非总训练查询严格相等。'+link(budget['evidence'],'8rank实际查询预算')+'</p>'
    body+='<details open><summary>91.4536的最高配方已经有哪些组件，变体具体改了什么</summary><p><b>SafetyEP标准Control10k：</b>SafetyEP信用、G8／world8、4inner；warmup1000后固定50k cosine／floor0.1，峰值action LR 5e−5、memory LR 5e−6；IL70均值anchor β50。scene／predictive memory与actor LoRA共52,707,840参数实际可训练，Qwen冻结。数据为full103,288；不是仅训动作头，也不是未加warmup或cosine的弱基线。</p>'
    body+=h.table(['相同强基线上的变体','单独改变的核心机制','保持的主体配置'],[
        ['更早衰减','cos衰减终点50k→20k，之后保持0.1floor；warmup和峰值LR不变','SafetyEP／G8／4inner／IL70 β50 anchor／可训练memory不变'],
        ['质量门控的历史RL保持','2k后收集本run实际ODE验证的高质量行为；抽历史场景用当前ODE重评，退化时在原β50内分配0.75 IL＋0.25历史RL均值保持','fresh PG／优势与原50k cosine不变；额外官方验证查询单列'],
        ['G16','每场景候选8→16，同更新数主候选为2倍；与Control20k的候选预算对齐并不对齐Adam／场景／scheduler时钟','同原IL70 fresh、SafetyEP与基础scope；不是32个不同场景的batch扩展'],
    ])+'</details>'
    if review.get('integrated_studies'):
        body+='<p><b>新计划已改为独立I综合试验：</b>旧H两因素prepared方案不启动；I从同原IL70 fresh验证B64不同场景、G16、历史保持与WKC组合，实际状态见前方独立I卡。多组件结果不能按上表涨幅相加，也不能单独归因。原H已测单因素数据继续保留作比较依据。</p>'
        return body
    combo=next((a for a in review['arms']if a['arm']=='IMPROVEMENT_RETENTION_EARLY_DECAY_G8'),None)
    if combo:
        n=combo['current_native'];state='已登记，等待实际更新'
        if n.get('worker_receipt_status')=='RUNNING':state='child启动，待实际8rank更新'if n.get('journal_ranks')!=8 or not(n.get('updates')or 0)>0 else '已观察到8rank优化 '+h.number(n['updates'])+'更新'
        if n.get('worker_returncode')==75:state='当前等待资源准入，不是科学失败'
        if combo.get('latest_complete'):state+='；完整成绩见当前结果表'
    else:state='准备态；尚未登记或派发，无组合分数，不计入当前训练数量'
    body+='<p><b>整合首选：保持＋更早衰减。</b>'+state+'。组合从原IL70、fresh Adam／RNG42／update0另起；它与从最高Control10k继续训练是不同设计。两机制可能相互影响，不能相加各自涨幅，更不能把保持5k与Early14k的差值拼成组合成绩。2k前保持尚未启用，不是组合效果检验；完整同阶段输入齐后，分别对Control、Early、Retention核对总分、零分修复／新增、固定IL高分保持与官方查询成本。</p>'
    prepared=review.get('combination_preparation',{})
    if prepared.get('evidence'):body+='<p class="muted">'+link(prepared['evidence'],'组合代码／CPU检查／可审配置与准备阶段收据')+'：'+h.esc(prepared.get('status','状态见原始文件'))+'。'+('此为历史准备阶段；当前注册／worker与实际更新优先，不能用旧准备标签覆盖当前状态。'if combo else 'CPU检查不等于部署或收益，原H登记与owner不因准备文件而扩容。')+'</p>'
    return body


def render_review(root, review, review_path, h):
    ci = lambda v: '['+', '.join(h.fmt(x, 4) for x in v)+']' if isinstance(v, list) and len(v) == 2 else '未有配对区间'
    link = lambda p, text: h.relative_link(root, p, text)
    peak = review.get('observed_peak') or {}
    body = '<div class="notice"><b>已出现多个阶段的配对正点，但尚未形成持续的新方法领先。</b>当前F/G/H完整观测峰仍为 '+h.esc(NAMES.get(peak.get('arm'), peak.get('arm')))+' '+h.number(peak.get('updates'))+'更新 '+h.fmt(peak.get('PDMS'), 5)+'；'+('尚未达到92。' if peak.get('PDMS', 0)<92 else '已达92，保持性另检。')+'本次汇总 '+h.number(review['complete_points'])+' 个完整NAVTEST点／'+h.number(review['complete_comparisons'])+' 组既有Control配对；不把部分评测补成成绩。</div>'
    body+=_render_stage_gain_context(review,h,link,ci)
    rows = []
    is_current=lambda a:a.startswith(('IMPROVEMENT_','INTEGRATED_'))
    ordered = sorted(review['arms'], key=lambda r:(not is_current(r['arm']), r['arm'] != CONTROL, r['arm']))
    for arm in ordered:
        latest = arm['latest_complete']; p = arm['latest_candidate_matched']
        if latest is None:
            rows.append([h.esc(arm['name']), '等待完整NAVTEST', '—', '—', '—', '—'])
            continue
        label = h.number(latest['updates'])+'／'+h.fmt(latest['PDMS'], 5)+'<small>'+('native0初始函数，非RL学习收益；' if latest['updates']==0 else '')+'零分 '+h.number(latest.get('zeros'))+'；'+link(latest['evidence'], '全量SUMMARY')+'</small>'
        if p:
            budget=h.number(p['updates'])+' vs Control '+h.number(p['reference_updates'])
            stale='；较新点尚无匹配Control' if latest['updates'] != p['updates'] else ''
            rows.append([h.esc(arm['name']),label, budget+'<small>'+h.fmt(p['delta_PDMS'],5)+'／'+ci(p['CI95_PDMS'])+stale+'</small>',
                h.number(p['repaired_zero'])+'／'+h.number(p['new_zero']),h.fmt(p['fixed_IL70_high_mean_delta_PDMS'],4),link(p['evidence'],'完整配对')])
        else:
            rows.append([h.esc(arm['name']),label,'只作对照，不与自身配对'if arm['arm']==CONTROL else '尚无完整同预算配对','—',h.fmt(latest.get('fixed_IL70_high_mean_delta_PDMS'),4)+'<small>此行为对IL70自身基准</small>','—'])
    columns=['路线','最新完整Adam／PDMS','最近主PG候选同量比较：更新／Δ及95%CI','零修复／新增','固定IL70≥.9均值Δ','证据']
    current_rows=[row for arm,row in zip(ordered,rows)if is_current(arm['arm'])or arm['arm']==CONTROL]
    historical_rows=[row for arm,row in zip(ordered,rows)if not is_current(arm['arm'])and arm['arm']!=CONTROL]
    body+='<h3>当前独立研究与共享Control：完整部署收益单列</h3>'+h.table(columns,current_rows)
    if historical_rows:body+='<details><summary>F／G长训全部路线的最新完整成绩与公平配对</summary>'+h.table(columns,historical_rows)+'</details>'
    body+='<p class="muted">方法行的高群体Δ是同一固定IL70高分8724场景相对当点Control的均值差，未给CI，不据此宣称显著。Control行则是相对IL70。G16用2倍、Batch32用4倍Control更新对齐source-PG查询；场景曝光、Adam和scheduler时钟仍不同。当前H保持臂未完整时留空；其2k后额外官方ODE评分仍单独分账。</p>'
    recent=[]
    for arm,u in [('IMPROVEMENT_VERIFIED_RETENTION_G8',5000),('IMPROVEMENT_EARLY_DECAY_G8',12000),('IMPROVEMENT_EARLY_DECAY_G8',14000),('IMPROVEMENT_MIXED_NOISE_G8',5000)]:
        p=next((p for p in review['comparisons']if p['arm']==arm and p['updates']==u and p['reference_updates']==u),None)
        if p:recent.append([h.esc(NAMES[arm]),h.number(u),h.fmt(p['target_PDMS'],5),h.fmt(p['delta_PDMS'],5)+'<small>'+ci(p['CI95_PDMS'])+'</small>',h.number(p['repaired_zero'])+'／'+h.number(p['new_zero']),h.fmt(p['fixed_IL70_high_mean_delta_PDMS'],4)])
    if recent:
        body+='<h3>新增收益：有正向证据，保持性仍要单独验证</h3>'+h.table(['H方法','同预算Adam','完整PDMS','对Control Δ／95%CI','零修复／新增','固定IL高均值Δ'],recent)
        early=next((a for a in review['arms']if a['arm']=='IMPROVEMENT_EARLY_DECAY_G8'),{})
        ep={p['updates']:p['PDMS']for p in early.get('curve',[])}
        if 10000 in ep and 12000 in ep:body+='<p>更早衰减在12k／14k对当点Control连续有正证据，但自身10k '+h.fmt(ep[10000],5)+'→12k '+h.fmt(ep[12000],5)+'仍回落，不能把相对优势说成已解决后期下降。保持臂5k是新的正向结果，需后续预算验证持续性；Mixed5k为负向证据。Batch32／FP32当前完整同预算结果尚未确认收益，见最新配对表；FP32初始函数差异另列。</p>'
    g16 = next((a for a in review['arms'] if a['arm']=='REVIVAL_G16'), {})
    gp = g16.get('candidate_matched_points', [])
    if gp:
        body+='<p><b>G16不能再只按早期负点判断：</b>'+ '；'.join(h.number(p['updates'])+' vs Control '+h.number(p['reference_updates'])+'：Δ'+h.fmt(p['delta_PDMS'],5)+'，CI '+ci(p['CI95_PDMS']) for p in gp) +'。后期出现同候选量正向证据，值得保留一个后续公平复核点；这不是纯探索量单因素，也未超过当前绝对峰值。</p>'
    decisions = [
        ['P0：优先完成H最小检验','更早衰减／Batch32／质量保持','首个Early2k与Batch500对F2k区间跨0；不能据早期结果停止。后续按最新完整预算点复核，保持臂必须跨2k激活后再看作用。'],
        ['新单变量，等待完整评测','真FP32动作投影／固定G8混合探索','按本臂worker和全rank日志判断启动；无完整NAV则留空。FP32先用native0区分初始执行函数变化；Mixed的旧候选重组仅是探索支持假说，不是已获部署收益。'],
        ['P1：有条件保留','G16、饱和信用、纠偏信用','G16已有后期同候选量正点；Sat／Inv有阶段正向但20–22k未持续。保留预定后续复核，不默认一直加到50k，也不把单个负点判永久无效。'],
        ['P2：不优先额外扩预算','DenseDAC、三信用组合、原Active课程','长训已有多点，最新完整对照未形成持续优势；可把超出已批准复核点的资源让给新问题。Active的22k反弹需与20k负点同时看，禁止把低分修复当总分增益。'],
        ['P3：建议优先让出当前配方资源','旧Endpoint低LR','已有多个完整长期点显著低于Control，12k仍未恢复；有限过程的候选支持改善未转成部署收益。是当前配方的配置分配建议，不否定endpoint算法族；是否停训由root按实际收据处理。'],
    ]
    body+='<details open><summary>资源优先级建议：依据完整长曲线，不执行自动淘汰</summary>'+h.table(['优先级','路线','建议与证据边界'], decisions)+'</details>'
    reserved=[a['name'] for a in review['arms'] if any('RESEARCH_RESERVE' in str(a['current_native'].get(k,'')) for k in ['standard_disposition','revival_disposition','disposition'])]
    if reserved:body+='<p><b>已执行的资源安排：</b>'+h.esc('、'.join(reserved))+'已纳入资源优先级预留，当前实际停止／边界见下方运行表；现有代码、checkpoint与评测保留。未完成的新阶段分数不据此判差，方法族未被否定。</p>'
    body+='<p>本报告只汇总研究分配建议与实际回执，不发停止或迁移信号。历史负面CI、高分退化和早期正点全部保留；NAVTEST名单、未来标签和互补oracle均不进入TRAINbank。</p>'
    body+='<p class="muted">'+link(str(review_path),'完整曲线／全部配对及输入SHA清单')+' · 生成 '+h.esc(review['created_utc'])+'。单seed／44日志配对区间，未校正多重比较；观测峰不是无偏最优值。完整曲线图见下方。</p>'
    return '<section id="rl-latest-evidence"><h2>最新完整证据与下一轮资源优先级</h2>'+body+'</section>'
