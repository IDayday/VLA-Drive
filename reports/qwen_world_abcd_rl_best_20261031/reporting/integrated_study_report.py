"""Presentation only for the independently registered B64/G16 integrated trial."""
from pathlib import Path
import math

ARM='INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'


def budget(updates):
    windows=math.ceil(updates/4)
    return dict(windows=windows,scenes=windows*64,candidates=windows*1024,
                scene_clock=updates*8,candidate_clock=updates*16,
                extra_scenes=windows*64-updates*16,extra_candidates=windows*1024-updates*256,
                next_inner=updates%4)


def first_result_evidence(study,h,link):
    """Read complete paired evidence only; the first gate precedes history use."""
    point=next((r for r in study['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'and r.get('updates')==250),None)
    if not point:return ''
    pairs=[p for p in study.get('paired_reviews',[])if p.get('updates')==250]
    il=next((p for p in pairs if p.get('reference')=='IL70'),None)
    control=next((p for p in pairs if p.get('kind')=='between_arms'and p.get('reference')=='SAFETY_G8_CONTROL'and p.get('reference_updates')==2000),None)
    peak=next((p for p in pairs if p.get('reference')=='FIXED_GLOBAL_BEST'),None)
    rows=[]
    for p,label in [(il,'原始IL70'),(control,'Control 2k：名义场景量匹配'),(peak,'历史最高Control 10k：非预算匹配')]:
        if not p:continue
        r=p['result'];delta=r.get('PDMS_delta');ci=r.get('PDMS_CI95');n=r.get('baseline_zero_denominator');repair=r.get('repaired_zero');new=r.get('new_zero')
        baseline=point['PDMS']-delta if isinstance(delta,(int,float))else None
        targetzero=n-repair+new if all(isinstance(v,(int,float))for v in [n,repair,new])else None
        rows.append([label,h.fmt(baseline,8)+' → '+h.fmt(point['PDMS'],8),h.fmt(delta,6)+'<small>'+('['+', '.join(h.fmt(v,6)for v in ci)+']'if isinstance(ci,list)else 'CI未发布')+'</small>',h.number(n)+' → '+h.number(targetzero)+'<small>修复 '+h.number(repair)+'／新增 '+h.number(new)+'</small>',link(p['evidence'],'完整配对')])
    body='<div class="notice"><b>I首个完整250点：'+h.fmt(point['PDMS'],8)+'，12,146场景／invalid0。</b>较原始IL70改善，但早期落后于场景曝光量接近的Control2k，尚未超过历史最高配方。250尚未进入历史保持，不能据此评价完整综合方案的最终效果。</div>'
    if rows:body+=h.table(['比较对象','基线实际分 → I250','Δ PDMS／44log配对95%CI','零分变化','证据'],rows)
    if control:
        high=control['result'].get('fixed_control_high',{})
        if high:body+='<p><b>当前短板是好场景保持：</b>固定Control2k≥0.9的 '+h.number(high.get('count'))+' 场景，I250队列均分差 '+h.fmt(high.get('mean_delta_PDMS'),6)+'，对全人口总分贡献 '+h.fmt(high.get('global_contribution_PDMS'),6)+'。这是历史检查点条件队列的描述性分解，不单独证明遗忘或某个组件致因。</p>'
    components=study.get('integrated_first_review',{}).get('metric_mean_deltas_percentage_points',{})
    if components:
        body+='<p>相对Control2k，EP '+h.fmt(components.get('ego_progress'),5)+'pp，DAC '+h.fmt(components.get('drivable_area_compliance'),5)+'pp、NC '+h.fmt(components.get('no_at_fault_collisions'),5)+'pp、TTC '+h.fmt(components.get('time_to_collision_within_bound'),5)+'pp。该点出现效率提高与安全下降并存，单点不证明两者不可调和。'+link(study['integrated_first_review_path'],'完整分量与恢复说明')+'</p>'
    body+='<p>I250实际4032次scene曝光／64,512主PG候选；Control2k为4000／32,000，Adam更新为250／2000。候选量约两倍且窗口有pending差额，<b>不是同算力对照</b>；没有完整Control4k匹配时不补造候选预算结论。'+link(point['evidence'],'I250全量SUMMARY')+'</p>'
    return body


def execution_evidence(study,live,h,link):
    check=study.get('integrated_reporting_check',{})
    if check.get('status')!='PASS':return ''
    row=next((r for r in live if r.get('arm')==ARM),{})
    category=row.get('category');updates=row.get('common_updates')
    evidence=link(study['integrated_reporting_check_path'],'真实collection65统计序列化复现／CPU窄修')
    recovery=study.get('integrated_recovery',{})
    recovered=recovery.get('status')=='ACTUAL_ALL8_RANKS_RECOVERED_PAST_FAILED_HISTORY_WINDOW'
    if recovered and category=='training':
        gates=sorted({r.get('updates')for r in study['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'and isinstance(r.get('updates'),(int,float))})
        full_note='完整NAVTEST仍只有250点。'if gates==[250]else '完整NAVTEST见下方结果表。'
        return '<div class="notice"><b>I日志序列化故障已修复，当前持续训练。</b>完整Adam／RNG／bank／pending从250 inner2恢复，8rank已越过原257故障点；重放251–256的loss、gradient norm、LR最大差均为0，chain identity相同。仅统计转Python int，科学配方与既有250分数保持。<b>当前训练更新不是新评测成绩；'+full_note+'</b>'+link(study['integrated_recovery_path'],'实际8rank恢复证据')+'</div><details><summary>历史日志故障与窄修</summary><p>θ256后的step257曾因history_gates为NumPy int64而无法写JSON；记录loss／gradient有限，不能称数值发散或算法负结果。'+evidence+'</p></details>'
    if category=='exited_review':
        job=row.get('job');result=Path(job)/'RESULT.json'if job else None
        return '<div class="notice"><b>I当前执行已退出，等待恢复证据。</b>8rank共同日志停在 '+h.number(updates)+'；θ256后的step257在写统计JSON时遇到NumPy int64，完整checkpoint仍为250 inner2。已记录loss／gradient有限，此次是日志序列化故障，不能当作数值发散或算法负结果。CPU窄修已过，尚不能据此写成恢复训练。'+evidence+(' · '+link(result,'实际失败RESULT')if result and result.is_file()else '')+'</div>'
    title='I已观察到故障后的真实8rank更新。'if category=='training'and isinstance(updates,(int,float))and updates>256 else 'I恢复状态以本次实际stage与8rank日志为准。'
    return '<details><summary>'+title+'</summary><p>前次θ256后的step257曾因history_gates的NumPy int64不能写JSON而退出；修复仅将统计转换Python int，不改门控或训练数学。CPU检查不代替恢复证明，实际训练分类见本节实时卡。'+evidence+'</p></details>'


def render_integrated(report_root,study,h,section_id,native_summary=None):
    root=Path(study['root']);reg=study['registration'];runs=study['runs']
    link=lambda p,t:h.relative_link(report_root,str(p),t)
    live=(native_summary or{}).get('runs',[]);live=[r for r in live if r.get('study_root')==study['root']]
    counts={k:sum(r.get('category')==k for r in live)for k in ['training','starting','waiting_resource','exited_review','resource_reserved','unconfirmed']}
    confirmed=native_summary is not None
    body='<div class="notice"><b>独立I综合试验，沿用最高SafetyEP配方和同一原始IL70 fresh初始化。</b>同时加入64个不同场景、G16、质量门控历史RL保持与warmup→keep→cosine。它检验整套配方，不能把单因素阶段涨幅相加，也不能单独归因给某个组件。旧H两因素prepared计划被此方案取代，不启动旧H第六臂。</div>'
    body+=first_result_evidence(study,h,link)
    body+=execution_evidence(study,live,h,link)
    body+='<p class="muted">本I科学source '+h.esc(Path(reg.get('source','待冻结')).name)+'／'+h.esc(reg.get('source_commit','待实际source'))+'；已有F／G／H科学source分别保留，不因新CPU协调namespace而改写。</p>'
    body+='<div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr))"><article><span>I实际正在训练</span><strong>'+(h.number(counts['training'])if confirmed else '待实时证据')+'</strong><small>独立namespace；当前stage无RESULT、全部8rank新日志</small></article><article><span>I启动／等资源／退出待核</span><strong>'+('/'.join(h.number(counts[k])for k in ['starting','waiting_resource','exited_review'])if confirmed else '待本次统计')+'</strong><small>本I登记 '+h.number(len(runs))+' 臂；未确认 '+h.number(counts['unconfirmed'])+'。与首屏同时间：'+h.esc((native_summary or{}).get('observed_utc','尚未统计'))+'</small></article></div>'
    method_rows=[['基准主体','原IL70／fresh Adam0／RNG42；SafetyEP信用、IL70 mean-anchor β50、52,707,840可训练memory＋LoRA；Qwen冻结'],
        ['不同场景批次','world8×每rank8个不同scene＝global64；每scene G16，fresh PG每窗口1024候选，四inner复用；每inner真实8个micro backward后一次Adam'],
        ['历史RL保持','actual250为启用阈值；真实TRAIN B1 ODE严格质量验证，gate触发时原β50内0.75 IL＋0.25历史RL均值保持，fresh PG不混历史'],
        ['WKC调度','固定12500 actual-Adam时钟：从0.001峰值warmup至125→峰值keep至625→cosine至2500→0.1floor保持；峰值action5e−5／memory5e−6，同倍率且阶段不重置'],
        ['部署评分','原canonical、single B1 ODE K10 shift5 seed2101／NAVTEST12146；不是best-of-G，也不改变评分定义']]
    body+=h.table(['综合组件','已登记设计／计划，不等同收益证明'],method_rows)
    development=[p for p in [root/'analysis'/'INTEGRATED64_HANDOFF.json',root/'analysis'/'INTEGRATED64_CPU_CHECK.json']if p.is_file()]
    if development:body+='<p class="muted">'+' · '.join(link(p,'科学协议／CPU检查')for p in development)+'；实际8rank tinyScope／DDP／恢复检查不冒充本I的完整52.7M生产训练或部署提分证据。</p>'
    proof=root/'operations'/'DEPLOYMENT_20261102'/'ACTUAL_TRAINING_PROOF.json'
    if proof.is_file():body+='<p>历史生产首批证据已确认8rank真实优化、64个不同fresh场景／1024归一化、52,707,840实际trainable；当前运行状态由本次stage与8rank日志确认。'+link(proof,'独立生产启动回执')+'</p>'
    holds=root/'operations'/'DEPLOYMENT_20261102'/'ROUTE_HOLDS_RESTORED.json'
    if holds.is_file():body+='<p class="muted">旧低优先路线保存后为新I预留固定资源；临时future-route holds已恢复，既有F／G／H CPU未接管，BD IL与wm4／wm5评测继续。'+link(holds,'原路线hold恢复与协调回执')+'</p>'
    progress=[]
    for run in runs:
        row=next((r for r in live if r['arm']==run['id']),{})
        labels={'training':'实际训练','starting':'启动中，待8rank更新','waiting_resource':'等待资源','exited_review':'执行退出，待复核／恢复','resource_reserved':'资源储备','unconfirmed':'当前运行未确认','ended':'当前阶段已结束','paused_saved':'边界保存后暂停'}
        progress.append([h.esc(run['id']),h.esc(labels.get(row.get('category'),run.get('status','待当前阶段证据'))),
            h.number(row.get('common_updates',run.get('updates'))),h.esc(row.get('host',run.get('host','待派发'))),
            h.number(run.get('raw_scene_exposures')),h.number(run.get('new_candidates')),h.number(run.get('target_updates',6250))])
    body+=h.table(['I运行臂','当前证据分类','actual Adam共同更新','host','原生scene曝光','原生主PG候选','当前批准Adam预算'],progress)
    body+='<h3>三个时钟分开：没有单一“等价步数”</h3><p>主source场景时钟为actual Adam×8，主PG候选时钟为×16；预算与Adam成本不同。窗口数据按ceil(u/4)采集，pending inner的整窗口已经生成／评分，不能只按已完成inner比例计成本。</p>'
    plan=[]
    for u in [250,625,1250,2500,6250,12500]:
        b=budget(u)
        plan.append([h.number(u),h.number(b['windows']),h.number(b['scenes']),h.number(b['candidates']),
            h.number(b['scene_clock'])+'／'+h.number(b['candidate_clock']),
            h.number(b['extra_scenes'])+'／'+h.number(b['extra_candidates'])+'<small>next_inner '+h.number(b['next_inner'])+'</small>'])
    body+=h.table(['actual Adam u','窗口计划ceil(u/4)','scene曝光计划','主PG候选计划','名义Control scene／PG时钟','pending surplus：scene／PG'],plan)
    body+='<p><b>初批准6250 actual Adam：</b>按整窗口计算的计划为100,032 scene／1,600,512 PG候选、pending2；未到该点前均非实际观测。有益后另审至12500；这是固定调度上限，不是自动续满承诺。raw曝光不等于unique掌握，fullTRAIN eligible103,288需与跨epoch重复分开。u为已完成Adam，计划倍率供下一次step使用。</p>'
    body+='<p>第一个250评测尚未进入retention；受整窗口与pending影响，首次history admission的behavior为θ252，不能据250评测判断保持组件效应。启用后每窗口在1024个fresh PG评分之外，最多64个fresh B1＋64个history B1官方验证，即额外上限<b>12.5%</b>，不能沿用H单scene/G8的25%查询比例。25%仍指gate激活时的β50历史anchor预算份额；实际验证次数等待封存计数，不能把四inner复用乘成四次新query。</p>'
    body+='<p>全量评测gate：250／625／1250，之后每250 actual Adam；这是source scene时钟10k后的每2k节奏。两种预算的完整Control缺哪种就待匹配；例如I625对Control5k是scene预算、对Control10k是PG预算，且pending多48scene／768PG，不能称严格等成本。</p>'
    complete=sorted((r for r in study['results']if r.get('publishable_complete')and r.get('cohort')=='NAVTEST'),key=lambda r:r.get('updates',0))
    if complete:body+='<h3>I完整NAVTEST</h3>'+h.table(['方法','actual Adam','场景','完整PDMS','证据'],[[h.esc(r['arm']),h.number(r.get('updates')),h.number(r.get('scenes')),h.fmt(r.get('PDMS'),8),link(r['evidence'],'全量SUMMARY')]for r in complete])
    else:body+='<div class="notice"><b>I尚无完整NAVTEST成绩。</b>配置／CPU检查／child启动／训练候选改善都不代替12146场景完整部署分数，不发布partial均值。</div>'
    pairs=[]
    for p in study.get('paired_reviews',[]):
        r=p['result'];u=p.get('updates');v=p.get('reference_updates');axis='待核预算'
        if p.get('kind')=='historical'and p.get('reference')=='FIXED_GLOBAL_BEST':
            axis='固定历史最高（非预算匹配）';v=10000
        elif p.get('kind')!='between_arms':continue
        elif isinstance(u,(int,float)):
            if v==u*8:axis='名义scene同量'
            elif v==u*16:axis='名义主PG同量'
        ci=r.get('PDMS_CI95');ci='['+', '.join(h.fmt(x,5)for x in ci)+']'if isinstance(ci,list)else '无完整配对区间'
        pairs.append([h.number(u),axis,h.esc(p.get('reference'))+' '+h.number(v),h.fmt(r.get('PDMS_delta'),6)+'<small>'+ci+'</small>',h.number(r.get('repaired_zero'))+'／'+h.number(r.get('new_zero')),link(p['evidence'],'完整配对')])
    if pairs:body+=h.table(['I actual Adam','比较预算','完整基线','Δ PDMS／95%CI','零修复／新增','证据'],pairs)
    else:body+='<p class="muted">尚无完整双时钟配对；不会用别的Control预算点插值或替补。</p>'
    body+='<p>'+link(study['registration_path'],'I正式REGISTRATION')+' · '+link(root/'RUNS.json','I实际运行登记')+'；数据与观测人口沿用原IL70研究合同，NAVTEST标签不进入TRAINbank。实际source／checkpoint／结果分别归档，既有训练和评测是否调整仅看root真实操作收据。</p>'
    return '<section id="'+section_id+'"><h2>I综合试验：SafetyEP＋B64／G16＋历史保持＋WKC</h2><p class="deck">独立fresh训练与命名空间；设计、真实更新、窗口查询和完整部署结果分开。</p>'+body+'</section>'
