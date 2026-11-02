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


def trick_diagnostics(study,h,link):
    audit=study.get('integrated_trick_diagnostics',{})
    if audit.get('all8_ranks')is not True:return ''
    summary=audit.get('full_summary',{});counts=summary.get('counts',{});rates=summary.get('group_percent',{})
    if not counts.get('groups'):return ''
    optimizer=summary.get('mean_optimizer_record_metrics',{})
    blocked=optimizer.get('quality_high_positive_PPO_blocked')
    blocked=100*blocked if isinstance(blocked,(int,float))else None
    rows=[
        ['历史保持','选中 '+h.number(counts.get('history_slots_selected'))+' 次，gate激活 '+h.number(counts.get('history_gates'))+' 次（'+h.fmt(summary.get('history_gate_percent_of_selected'),4)+'%）','25%是gate条件下原β50的历史份额；按全部scene折算的名义历史anchor预算仅 '+h.fmt(summary.get('nominal_history_anchor_share_percent_of_all_scene_slots'),4)+'%，不是参数梯度占比'],
        ['额外官方B1验证',h.number(summary.get('extra_B1_queries'))+' 次，相对主PG查询 '+h.fmt(summary.get('extra_B1_queries_percent_of_fresh_PG'),4)+'%','已按rank／collection去重；查询次数不等于GPU耗时，四inner复用不重复计新query'],
        ['真实PG零信用',h.number(counts.get('zero_PG'))+'／'+h.number(counts.get('groups'))+'（'+h.fmt(rates.get('zero_PG'),4)+'%）','canonical全零仅 '+h.number(counts.get('all_zero'))+' 组：'+h.number(counts.get('all_zero_with_PG'))+' 组仍有SafetyEP方向，'+h.number(counts.get('all_zero_without_PG'))+' 组无PG；不能把PG0都当失败'],
        ['优质候选支持','含strict-good候选的组 '+h.fmt(summary.get('groups_with_strict_high_percent'),4)+'%','TRAIN候选可发现不等于单B1 ODE部署会选中；需配套部署差距诊断'],
        ['好候选的相对信用','strict-good中负A占 '+h.fmt(summary.get('strict_high_negative_advantage_percent'),3)+'%','可能只是相对更好候选的正常下调；负号不等于参数负梯度或实现错误'],
        ['PPO保护／阻断','好正A候选平均真正阻断 '+h.fmt(blocked,3)+'%','该范围不支持“clip普遍封死学习”；标量阻断也不代表梯度竞争结论']]
    body='<h3>综合trick实际生效多少，下一步怎样验证</h3><p>截至实际 '+h.number(audit.get('through_optimizer_update'))+' Adam，完整8rank训练记录覆盖 '+h.number(counts.get('groups'))+' 个去重TRAIN组／'+h.number(summary.get('unique_fresh_tokens'))+' 个unique场景、'+h.number(counts.get('fresh_PG_candidates'))+' 主PG候选；这是已运行前缀的全量审计，不是完整TRAIN人口或NAVTEST收益。strict-good指NC／DAC／TTC／Comfort均1且canonical≥0.9。</p>'
    discovery=audit.get('within_G16_first8_counterfactual',{});dc=discovery.get('counts',{});dr=discovery.get('percent_of_all_groups',{})
    if dc:
        body+='<p><b>G16确实增加候选发现：</b>同一G16抽样内，前8条找到strict-good的组 '+h.number(dc.get('strict_high_first8'))+'／'+h.number(dc.get('groups'))+'（'+h.fmt(dr.get('strict_high_first8'),4)+'%），全部16条为 '+h.number(dc.get('strict_high_full16'))+'（'+h.fmt(dr.get('strict_high_full16'),4)+'%）；后8条新增 '+h.number(dc.get('strict_high_rescued_by_second8'))+' 组（+'+h.fmt(dr.get('strict_high_rescued_by_second8'),5)+'pp）。这是同组边际发现，不是独立G8策略对照，更不是PDMS增益。</p>'
    nominal=study.get('integrated_nominal_sampled',{});nc=nominal.get('counts',{})
    if nominal.get('TRAIN_only')is True and nominal.get('NAVTEST_labels_used')is False and nc:
        body+='<p><b>好候选尚有转化为B1的空间：</b>逐组同一behavior版本的 '+h.number(nc.get('groups_with_actual_B1'))+' 个fresh TRAIN场景，G16有strict-good '+h.number(nc.get('sampled_strict_high'))+'，实际B1为 '+h.number(nc.get('nominal_strict_high'))+'；'+h.number(nc.get('sampled_high_but_nominal_not_high'))+' 个场景采样有好解但B1不优质（含 '+h.number(nc.get('sampled_high_but_nominal_unsafe'))+' 个B1不安全／'+h.number(nc.get('sampled_high_but_nominal_zero'))+' 个B1零分，两项不相加），反向 '+h.number(nc.get('nominal_high_but_no_sampled_high'))+' 个B1优质而G16未找到。'+h.number(nc.get('rescue_pair_with_margin_0p02'))+' 对满足候选比B1至少高0.02的TRAIN救援条件。<b>这些均为更新前fresh测量，是转化机会；不是训练后仍未学会的证明。</b>需以后续固定TRAIN复访验证持续性，不使用NAVTEST标签。'+link(study['integrated_nominal_sampled_path'],'同版本B1／采样全量诊断')+'</p>'
    body+='<p>历史保持 '+h.number(counts.get('history_gates'))+'／'+h.number(counts.get('history_slots_selected'))+'（'+h.fmt(summary.get('history_gate_percent_of_selected'),4)+'%）触发，全局名义历史anchor份额 '+h.fmt(summary.get('nominal_history_anchor_share_percent_of_all_scene_slots'),4)+'%；额外B1查询 '+h.fmt(summary.get('extra_B1_queries_percent_of_fresh_PG'),4)+'%而非GPU时间。好正A候选阻断约 '+h.fmt(blocked,3)+'%，不支持直接把clip视为主瓶颈。'+link(study['integrated_trick_diagnostics_path'],'全8rank机制审计')+'</p>'
    body+='<details><summary>展开完整机制统计与分母</summary>'+h.table(['组件／机制','实际观测','解释边界'],rows)+'</details>'
    proposals=study.get('integrated_refinement_proposals',{})
    if proposals.get('status')=='PROPOSALS_NOT_REGISTERED_NOT_DISPATCHED':
        ideas=[['1','严格候选救援／自模仿','将满足质量与≥0.02差距的本场景G16好解，作为独立flow-matching辅助目标；在固定β50内预留5%条件预算，不混入fresh PPO。','检验能否转化B1与保持高分；5%是目标系数，不是梯度份额'],
               ['2','小历史保持底座＋退化门控','有效历史teacher存在时，历史份额改为5%＋20%×gate，其余给IL；只改该权重规则。','可能保住次优旧策略；早期gate稀疏不证明原10k保持无效'],
               ['3','只延后cosine终点','cos结束actual2500→5000；warm125／keep625／floor0.1／horizon12500及两组峰值保持。','独立fresh对照是否过早收敛；不热改当前run，不把调度疑虑当已证原因']]
        body+='<h4>按证据排序的三个后续消融：尚未注册、尚未训练</h4>'+h.table(['优先级','待验方向','仅拟改动','判据／风险'],ideas)
        body+='<p><b>当前10k参考试验保持不变，等实际2k／5k／10k完整NAVTEST。</b>'+link(study['integrated_refinement_proposals_path'],'未部署方案与独立预算设计')+'。候选好行为再学习借鉴 <a href="https://arxiv.org/abs/1806.05635">Self-Imitation Learning</a>，本地flow辅助并非原算法复现；<a href="https://arxiv.org/html/2510.13786v1">ScaleRL</a>的采样规模证据来自LLM，只作迁移假说。</p>'
    return body


def render_integrated(report_root,study,h,section_id,native_summary=None):
    root=Path(study['root']);reg=study['registration'];runs=study['runs']
    registry_runs=study.get('registry',{}).get('runs',[])
    registered=next((r for r in registry_runs if r.get('id')==ARM),{})if isinstance(registry_runs,list)else registry_runs.get(ARM,{})
    target=registered.get('target_updates',6250)
    eval_gates=registered.get('evaluation_gates',reg.get('evaluation_gates'))
    if isinstance(eval_gates,list):eval_gates=sorted({u for u in eval_gates if isinstance(u,int)and u>0})
    else:eval_gates=None
    link=lambda p,t:h.relative_link(report_root,str(p),t)
    live=(native_summary or{}).get('runs',[]);live=[r for r in live if r.get('study_root')==study['root']]
    counts={k:sum(r.get('category')==k for r in live)for k in ['training','starting','waiting_resource','exited_review','resource_reserved','unconfirmed']}
    confirmed=native_summary is not None
    body='<div class="notice"><b>独立I综合试验，沿用最高SafetyEP配方和同一原始IL70 fresh初始化。</b>同时加入64个不同场景、G16、质量门控历史RL保持与warmup→keep→cosine。它检验整套配方，不能把单因素阶段涨幅相加，也不能单独归因给某个组件。旧H两因素prepared计划被此方案取代，不启动旧H第六臂。</div>'
    body+=first_result_evidence(study,h,link)
    body+=execution_evidence(study,live,h,link)
    body+=trick_diagnostics(study,h,link)
    if target>=10000 and eval_gates:
        body+='<div class="notice"><b>现行判断门槛：完整实际 '+h.number(target)+' 次Adam更新后，再评审是否值得长训。</b>未来完整NAVTEST仅按实际 '+ '／'.join(h.number(u)for u in eval_gates)+' 更新安排；已完成250保留为历史，不重复评测。场景／候选时钟不替代实际Adam步数，现有科学调度与native状态保持。</div>'
        applied=root/'operations'/'USER_10K_2K5K10K_20261102'/'APPLIED.json'
        if applied.is_file():body+='<p>新预算与评测节奏已实际应用；10k前不因阶段分数提前淘汰。仅升级CPU运营控制，科学配置、warm125／keep625／cos2500／floor0.1与12500调度总时钟保持，Adam／RNG／历史bank连续，现有GPU任务未重启。'+link(applied,'预算与评测计划APPLIED收据')+'</p>'
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
            h.number(run.get('raw_scene_exposures')),h.number(run.get('new_candidates')),h.number(target)])
    body+=h.table(['I运行臂','当前证据分类','actual Adam共同更新','host','原生scene曝光','原生主PG候选','当前批准Adam预算'],progress)
    body+='<h3>三个时钟分开：没有单一“等价步数”</h3><p>主source场景时钟为actual Adam×8，主PG候选时钟为×16；预算与Adam成本不同。窗口数据按ceil(u/4)采集，pending inner的整窗口已经生成／评分，不能只按已完成inner比例计成本。</p>'
    plan=[]
    milestones=sorted(set([250,625,target,12500]+(eval_gates if eval_gates is not None else[1250,2500])))
    for u in milestones:
        b=budget(u)
        purpose='历史完整评测'if u==250 else '当前批准预算终点'if u==target else '当前评测点'if eval_gates and u in eval_gates else 'native交接节点'if u==625 and eval_gates is not None else '科学调度上限；未自动批准'if u==12500 else '历史计划节点'
        plan.append([h.number(u)+'<small>'+purpose+'</small>',h.number(b['windows']),h.number(b['scenes']),h.number(b['candidates']),
            h.number(b['scene_clock'])+'／'+h.number(b['candidate_clock']),
            h.number(b['extra_scenes'])+'／'+h.number(b['extra_candidates'])+'<small>next_inner '+h.number(b['next_inner'])+'</small>'])
    body+=h.table(['actual Adam u','窗口计划ceil(u/4)','scene曝光计划','主PG候选计划','名义Control scene／PG时钟','pending surplus：scene／PG'],plan)
    approved=budget(target)
    body+='<p><b>当前批准 '+h.number(target)+' actual Adam：</b>按整窗口计算的主训练计划为 '+h.number(approved['scenes'])+' scene／'+h.number(approved['candidates'])+' PG候选，pending '+h.number(approved['next_inner'])+'；真实B1额外验证另计，未到该点前均非实际观测。固定科学调度上限12500不等于自动批准续满。raw曝光不等于unique掌握，fullTRAIN eligible103,288需与跨epoch重复分开。u为已完成Adam，计划倍率供下一次step使用。</p>'
    body+='<p>第一个250评测尚未进入retention；受整窗口与pending影响，首次history admission的behavior为θ252，不能据250评测判断保持组件效应。启用后每窗口在1024个fresh PG评分之外，最多64个fresh B1＋64个history B1官方验证，即额外上限<b>12.5%</b>，不能沿用H单scene/G8的25%查询比例。25%仍指gate激活时的β50历史anchor预算份额；实际验证次数等待封存计数，不能把四inner复用乘成四次新query。</p>'
    if eval_gates is not None:
        body+='<p>未来全量评测gate：'+ '／'.join(h.number(u)for u in eval_gates)+' actual Adam。250只保留历史；native保存／交接节点与评测节点分开，不按旧625／1250等密集计划新增NAVTEST。两种预算的完整Control缺哪种就待匹配，不插值或用未齐点代替。</p>'
    else:
        body+='<p>登记的全量评测gate见现行RUNS；两种预算的完整Control缺哪种就待匹配。例如I625对Control5k是scene预算、对Control10k是PG预算，且pending多48scene／768PG，不能称严格等成本。</p>'
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
