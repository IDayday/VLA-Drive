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


def render_integrated(report_root,study,h,section_id,native_summary=None):
    root=Path(study['root']);reg=study['registration'];runs=study['runs']
    link=lambda p,t:h.relative_link(report_root,str(p),t)
    live=(native_summary or{}).get('runs',[]);live=[r for r in live if r.get('study_root')==study['root']]
    counts={k:sum(r.get('category')==k for r in live)for k in ['training','starting','waiting_resource','exited_review','resource_reserved','unconfirmed']}
    confirmed=native_summary is not None
    body='<div class="notice"><b>独立I综合试验，沿用最高SafetyEP配方和同一原始IL70 fresh初始化。</b>同时加入64个不同场景、G16、质量门控历史RL保持与warmup→keep→cosine。它检验整套配方，不能把单因素阶段涨幅相加，也不能单独归因给某个组件。旧H两因素prepared计划被此方案取代，不启动旧H第六臂。</div>'
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
    if proof.is_file():body+='<p>已另有本I正式生产首批证据：8rank真实优化、64个不同fresh场景／1024归一化、52,707,840实际trainable；仅说明生产机制已运行，不等于稳定性或NAVTEST提分。'+link(proof,'独立生产启动回执')+'</p>'
    holds=root/'operations'/'DEPLOYMENT_20261102'/'ROUTE_HOLDS_RESTORED.json'
    if holds.is_file():body+='<p class="muted">旧低优先路线保存后为新I预留固定资源；临时future-route holds已恢复，既有F／G／H CPU未接管，BD IL与wm4／wm5评测继续。'+link(holds,'原路线hold恢复与协调回执')+'</p>'
    progress=[]
    for run in runs:
        row=next((r for r in live if r['arm']==run['id']),{})
        progress.append([h.esc(run['id']),h.esc(row.get('category',run.get('status','待当前阶段证据'))),
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
