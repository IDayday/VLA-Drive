"""Current-first RL presentation with expandable mechanism and history."""
from __future__ import annotations

import json
from pathlib import Path

from research_report import figure, update

SAFETY='PPO_SAFETY_EP'
PAIR_ARMS=('PPO_SAFETY_SUCCESS_MEMORY','PPO_SAFETY_FPO')
FULLTIME='PPO_FPO_FULLTIME'
QUALITY='PPO_SAFETY_FEASIBLE_SCORE'
FRESH_ARMS=(*PAIR_ARMS,FULLTIME,QUALITY)
LR_ARMS={'PPO_SAFETY_LRLOW','PPO_SAFETY_LRCOS','PPO_SAFETY_DATA','PPO_SAFETY_LRDATA'}
METRICS={'no_at_fault_collisions':'NC','drivable_area_compliance':'DAC','time_to_collision_within_bound':'TTC','ego_progress':'EP'}


def render_safety_focus_report(root,snapshot,history,h):
    esc=h.esc;fmt=h.fmt;number=h.number;table=h.table
    link=lambda path,label='证据':h.relative_link(root,str(path)if path is not None else None,label)
    def detail(title,body):return '<details><summary>'+title+'</summary>'+body+'</details>'
    def signed(value,digits=4):return f'{value:+.{digits}f}'if h.finite(value)else '待记录'
    def ci(value,digits=4):return '['+', '.join(signed(x,digits)for x in value)+']'if isinstance(value,(list,tuple))and len(value)==2 else '区间待记录'
    editorial=snapshot.get('editorial_analysis',{})
    focus=editorial.get('safety_focus',{});focus_stopped={row['arm']:row for row in focus.get('stop_summary',{}).get('stopped',[])if row.get('arm')and row.get('returncode')==0 and row.get('native_transport_exited')is True}
    diagnosis=editorial.get('safety_process',{});directions=editorial.get('safety_directions',{})
    mechanism=editorial.get('safety_mechanism',{});gradient_probe=editorial.get('safety_gradient_probe',{})
    fresh_pair=editorial.get('safety_fresh_pair',{})
    credit_gradient=editorial.get('safety_credit_gradient',{});policy_cases=editorial.get('safety_policy_cases',{})
    decisions=editorial.get('decisions',{});stopped=editorial.get('stopped_receipts',{})
    runs={row['id']:row for row in snapshot['runs']};run=runs.get(SAFETY,{})
    all_test=[row for row in snapshot['results']if row.get('publishable_complete')and row.get('cohort')=='NAVTEST'and row.get('scenes')==12146]
    safety=sorted((row for row in all_test if row.get('arm')==SAFETY),key=update)
    by_update={update(row):row for row in safety}
    best=max(safety,key=lambda row:row['PDMS'],default={});latest=safety[-1]if safety else {}
    base=snapshot.get('ppo_full_first_stage_paired_analysis',{}).get('reference_PDMS',88.79290990699972)
    peak10=by_update.get(10000,{})
    research_best=max((row for row in all_test if row.get('arm')in(SAFETY,*FRESH_ARMS)),key=lambda row:row['PDMS'],default={})
    chapters=[]
    def section(key,title,deck,body):chapters.append('<section id="'+key+'"><h2>'+title+'</h2><p class="deck">'+deck+'</p>'+body+'</section>')
    safety_stop=stopped.get(SAFETY,{})
    stop_result=safety_stop.get('result',{})
    safety_stopped=(safety_stop.get('native_job_exited')is True and stop_result.get('returncode',stop_result.get('exit_code'))==0)
    stop_requested=decisions.get(SAFETY,{}).get('disposition')=='STOP_FUTILITY'
    primary_state=('已协作停止本配方长训，原生状态保留'if safety_stopped else '已请求停止，等待实际原生退出收据'if stop_requested else '原生训练中'if run.get('worker_receipt_status')=='RUNNING'else '本次原生阶段已完成'if run.get('worker_returncode')==0 else str(run.get('status','待状态证据')))
    cards=[('共同IL70基线',fmt(base,5),'未来方案统一从该IL身份fresh启动'),
           ('Safety已测最高',fmt(best.get('PDMS'),5),number(update(best))+' updates · 完整NAVTEST'),
           ('最近完整Safety',fmt(latest.get('PDMS'),5),number(update(latest))+' updates；不是当前训练候选分'),
           ('Safety停止位置'if safety_stopped else '当前原生更新',number(safety_stop.get('checkpoint',{}).get('updates')if safety_stopped else run.get('updates')),primary_state+' · '+number(run.get('journal_ranks'))+' rank日志')]
    body='<div class="notice"><b>目标：完整NAVTEST达到92及以上。</b>'+('原Safety恒定LR长训已依据完整20k／30k结果协作停止，最佳10k及原生状态保留，资源优先四个fresh IL70研究。'if safety_stopped else '原Safety已进入证据止损流程，等待实际退出收据；新fresh研究继续。'if stop_requested else '跟踪SafetyEP控制路线，同时依据过程证据验证少量fresh IL70新算法。')+'不限定必须沿用SafetyEP代码结构；旧路线不复活，IL独立研究继续。</div>'
    body+='<div class="cards">'+''.join('<article><span>'+esc(a)+'</span><strong>'+b+'</strong><small>'+esc(c)+'</small></article>'for a,b,c in cards)+'</div>'
    if research_best:body+='<p><b>距92目标：</b>当前研究路线已完整最高 '+fmt(research_best.get('PDMS'),5)+'，'+('尚差 '+fmt(92-research_best['PDMS'],5)+' 点'if research_best['PDMS']<92 else '达到92目标；继续按完整协议复核收益与风险')+'；训练候选分、best-of8或反事实重评分均不计为达标。</p>'
    new_latest=[max((row for row in all_test if row.get('arm')==arm),key=update,default={})for arm in FRESH_ARMS]
    if any(new_latest):body+='<p><b>新方法最近完整阶段：</b>'+esc('；'.join(row['arm']+' '+number(update(row))+'更新 '+fmt(row['PDMS'],5)+(('，对同预算Safety '+signed(row['PDMS']-by_update[update(row)]['PDMS'],5))if update(row)in by_update else '')for row in new_latest if row))+'。配对区间与安全／效率分量见新研究节。</p>'
    if stop_requested:
        decision=decisions[SAFETY]
        body+='<p class="notice"><b>本次算力分配决定：</b>root根据20k与30k对10k的完整配对区间上界均低于零、30k没有明确NC／DAC／TTC提升，手动结束这一恒定LR续训配方；这是事后研究算力分配，<b>不是35k预注册规则已满足</b>，35k未观察不补结果。'+('实际停止于 '+number(safety_stop.get('checkpoint',{}).get('updates'))+' / inner'+number(safety_stop.get('checkpoint',{}).get('next_inner'))+'，exit0与完整Adam／RNG原生状态已保留。'if safety_stopped else '当前只记录停止请求，退出与checkpoint收据齐全后再改为已停。')+'不否定算法家族，新四个fresh试验与已有28k恢复评测不取消。'+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'PPO_SAFETY_EP.json','分配决定')+(' · '+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'PPO_SAFETY_EP.STOPPED.json','实际退出与原生seal')if safety_stopped else '')+'</p>'
    if latest and peak10:
        body+='<p><b>当前核心事实：</b>10k为 '+fmt(peak10['PDMS'],5)+'；最近完整 '+number(update(latest))+' 为 '+fmt(latest['PDMS'],5)+'，相对10k '+signed(latest['PDMS']-peak10['PDMS'],5)+' 点。'+('后续完整点尚未刷新10k。'if best and update(best)==10000 else '当前最高点以本页实际完整结果为准。')+'训练更久并不保证持续提升；曲线回摆本身不能证明学习率、数据或表征中的任何一个是根因。</p>'
    progress30=editorial.get('safety_30k_review',{})
    if progress30.get('status')=='COMPLETE':
        comparison=progress30.get('comparison',{});metrics30=comparison.get('metrics',{});score30=metrics30.get('score',{});ep30=metrics30.get('ego_progress',{})
        body+='<p class="notice"><b>最新30k完整配对：</b>相对10k ΔPDMS '+signed(score30.get('delta_pp'),5)+'，95%区间 '+ci(score30.get('CI95_pp'),5)+'；EP '+signed(ep30.get('delta_pp'),5)+' pp，区间 '+ci(ep30.get('CI95_pp'),5)+'。NC／DAC／TTC点估计虽略高，区间均未给出明确改善；仅为固定训练seed下44物理log的不确定性，不是多seed或已完成35k的结论。'+link(root/'analysis'/'SAFETY_EP_30K_PROGRESS_REVIEW_20261031.json','30k原始组件与配对依据')+'</p>'
    elif diagnosis.get('status')=='COMPLETE'and diagnosis.get('component_attribution'):
        paired_latest=max(diagnosis['component_attribution'],key=lambda row:row['updates']);paired_ci=paired_latest.get('vs10k_CI95',[])
        paired_verdict='区间跨零，不能称总体显著退化'if len(paired_ci)==2 and paired_ci[0]<=0<=paired_ci[1]else '区间低于零'if len(paired_ci)==2 and paired_ci[1]<0 else '区间高于零'if len(paired_ci)==2 and paired_ci[0]>0 else '配对区间待核'
        body+='<p class="notice"><b>完整配对的关键区分：</b>'+number(paired_latest['updates'])+'对10k总分 '+signed(paired_latest.get('vs10k_PDMS_delta'),5)+'，95%区间 '+ci(paired_ci,5)+'，'+paired_verdict+'。10k为已观测峰值，这属于阶段诊断，不是预注册的独立显著性胜负检验。固定10k高分子集与总体分开判断；zero净减少也可能伴随新的高分失败。</p>'
    body+='<div class="two"><article class="callout"><h3>已确认与仍未知</h3><p>SafetyEP的条件信用修正有阶段总分收益，但并未自动解决部署安全风险、零分场景与高分能力保持。需要把部署ODE、训练G8候选、相对优势与真实optimizer更新分开。</p></article><article class="callout"><h3>下一轮比较方式</h3><p>已授权成功memory与Flow endpoint-CFM ratio两条fresh研究：1k／5k／10k复核，上限10k；另追加FPO时间覆盖及可行内质量信用两项独立对照：各1k／5k复核，上限5k。统一同IL70、fresh AdamW／RNG42／update0，实际启动与收益分别按证据更新。</p></article></div>'
    closed_lr=[arm for arm in LR_ARMS if arm in focus_stopped]
    if len(closed_lr)==4:body+='<p class="notice">四条LR／数据分叉已按用户调整研究焦点协作退出，保留checkpoint与实验资产；这是研究范围调整，不是据新成绩判定四种算法失败。历史方法及各自已测结果见折叠档案。</p>'
    else:body+='<p class="muted">四条LR／数据分叉按用户要求退出；已取得实际退出收据 '+number(len(closed_lr))+' / 4。缺少收据者仍标为待退出，不将停止请求当作已经退出。</p>'
    health=editorial.get('readonly_controller_status',{})
    if health:
        live=sum(node.get('exact_process_alive')is True for node in health.values())
        active_rows=[]
        for arm in(SAFETY,*FRESH_ARMS):
            node=runs.get(arm,{});seal=node.get('progress_receipt',{})
            active_rows.append([esc(arm),number(node.get('updates'))+' / '+number(node.get('journal_ranks'))+'rank',number(seal.get('updates')),esc(node.get('worker_receipt_status','待worker收据')),esc(node.get('host','待记录'))])
        body+=detail('原生路线与CPU观察状态（只读）',table(['路线','8rank共同日志更新／已见rank','最近已发布native进度','worker收据','实际host'],active_rows)+table(['CPU职责','当前OWNER PID','精确PID＋start仍存活','STATE状态'],[[esc(scope),number(node.get('pid')),'是'if node.get('exact_process_alive')else '已完成／关闭（见STATE）'if any(word in str(node.get('state_status',''))for word in('COMPLETE','SUPERSEDED','STOPPED'))else '否／未知',esc(node.get('state_status'))]for scope,node in health.items()])+'<p>本次 '+number(live)+' / '+number(len(health))+' 个CPU owner／guard的PID与start身份当前存活；已完成或被supersede的观察器不应反复恢复。日志更新与最近发布checkpoint分开显示；没有新native进度文件不等于没有训练更新。进程存活不构成模型健康或提分证明；报告不发送信号、不修改owner。</p><p>'+link(root/'reporting'/'READONLY_RL_STATUS.json','本次只读状态证据')+'</p>')
    export28=editorial.get('navtest_cadence',{}).get('evaluations',{}).get('PPO_SAFETY_EP_u028000_NAVTEST',{})
    if export28 and 28000 not in by_update:
        if export28.get('superseded_historical_failure'):body+=detail('28k已独立恢复评测：旧失败不覆盖新部署', '<p>首次CPU导出写入失败与残留.building均已保留归档；root已quarantine未发布残留并发布正式export与'+number(export28.get('world'))+'rank评测spec。当前独立恢复状态 '+esc(export28.get('status'))+'，worker收据 '+esc(export28.get('worker_counts'))+'。同CP／同v3 scorer／同推理协议，旧cadence的FAILED历史不覆盖新恢复；完整SUMMARY出来前不补分数。</p><p>'+link(root/'owners'/'SAFETY28_EXPORT_RECOVERY'/'STATE.json','独立恢复状态')+' · '+link(root/'evaluation_specs'/'PPO_SAFETY_EP_u028000_NAVTEST.json','实际评测spec')+' · '+link(root/'analysis'/'SAFETY28K_EXPORT_FAILURE_READONLY_20261031.json','历史失败只读诊断')+'</p>')
        else:body+=detail('28k评测缺口：CPU导出故障与科学成绩分开', '<p>28k导出先遇到PyTorch输出写入失败，随后被残留未发布.building目录拒绝；当前评测状态 '+esc(export28.get('status'))+'。未有完整分数不补点，不把导出故障解释为算法失败；30k完整结果独立展示。修复／重试由root负责，报告仅只读记录。</p><p>'+link(root/'analysis'/'SAFETY28K_EXPORT_FAILURE_READONLY_20261031.json','输出写入失败、原生seal与最小恢复接口')+'</p>')
    section('focus','SafetyEP历史对照：收益与长训停止依据','以92+为目标；阶段收益、机制证据与新算法验证分别呈现。',body)

    def score_plot(plt):
        fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
        if safety:
            ax.plot([update(row)/1000 for row in safety],[row['PDMS']for row in safety],'-o',color='#059669',lw=2,ms=4)
            ax.scatter([update(best)/1000],[best['PDMS']],color='#0f766e',s=60,zorder=3)
        for arm,color,label in zip(FRESH_ARMS,['#2563eb','#7c3aed','#d97706','#dc2626'],['Success memory','FPO sigma .2-.8','FPO sigma .2-1.0','Feasible quality q']):
            complete=sorted((row for row in all_test if row.get('arm')==arm),key=update)
            if complete:ax.plot([update(row)/1000 for row in complete],[row['PDMS']for row in complete],'--o',color=color,lw=1.6,ms=4,label=label)
        if safety:ax.lines[0].set_label('SafetyEP control (closed)'if safety_stopped else 'SafetyEP control')
        if len(ax.lines)>1:ax.legend(frameon=False,fontsize=8)
        ax.set(xlabel='Global optimizer updates (thousands)',ylabel='Complete NAVTEST PDMS')
        ax.grid(axis='y',alpha=.18)
        return fig
    score_figure=figure(root,h,'safety_focus_score','同IL70 fresh完整NAVTEST：主线与新对照',score_plot,'只连接已完整12146场景的单候选ODE结果。同预算比较；新对照未完整不补点，budget上限各自独立。纵轴放大看阶段变化，IL70基线见首卡。')
    def component_plot(plt):
        fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
        if peak10:
            for metric,color in zip(METRICS,['#dc2626','#7c3aed','#d97706','#2563eb']):
                points=[row for row in safety if update(row)>=10000 and h.finite(row.get('means',{}).get(metric))and h.finite(peak10.get('means',{}).get(metric))]
                if points:ax.plot([update(row)/1000 for row in points],[100*(row['means'][metric]-peak10['means'][metric])for row in points],'-o',color=color,ms=3,label=METRICS[metric])
        ax.axhline(0,color='#64748b',lw=1,ls='--');ax.grid(axis='y',alpha=.18)
        ax.set(xlabel='Global optimizer updates (thousands)',ylabel='Component change vs 10k (percentage points)')
        if peak10:ax.legend(frameon=False,ncol=4)
        return fig
    components_figure=figure(root,h,'safety_focus_components','安全与效率分量：相对10k的观察点差',component_plot,'NC／DAC／TTC／EP分别看，不能用总分替代风险变化。这是均值轨迹，未将观察点差画成统计置信区间。')
    rows=[[number(update(row)),fmt(row['PDMS'],6),signed(row['PDMS']-base,5),signed(row['PDMS']-peak10['PDMS'],5)if peak10 else '—',link(row.get('evidence'),'完整SUMMARY')]for row in safety]
    body='<div class="two">'+score_figure+components_figure+'</div>'+table(['全局更新','完整PDMS','对IL70：观察点差','对Safety10k：观察点差','证据'],rows)
    body+='<p class="protocol">统一协议：NAVTEST12,146场景／44物理log；单候选ODE K10／shift5／seed2101。训练seed42，不增加多种子；TRAIN诊断与G8候选不混入此曲线。</p>'
    if mechanism.get('status')=='COMPLETE':body+=_matched_mechanism_tables(mechanism,h,link,editorial.get('safety_normalization',{}))
    section('trajectory','完整轨迹：分数与安全—效率分量','先把真实部署变化看清，再检验关于优化、探索和信用分配的解释。',body)

    body='<p>本节只采用已完成的Safety过程诊断；持续zero、曾经zero与最新zero是不同集合。高分保持也要固定10k场景子集，不能随checkpoint改分母。</p>'
    diagnosed_updates=max((row.get('updates',0)for row in diagnosis.get('evaluation_points',[])),default=0)
    if latest and update(latest)>diagnosed_updates:body+='<p class="notice">完整部署曲线已更新到 '+number(update(latest))+'，本次zero／高分／贡献诊断仍截至 '+number(diagnosed_updates)+'。下文分量结论只对应标注的诊断截点，不推广到后来checkpoint。</p>'
    if diagnosis.get('status')=='COMPLETE':body+=_diagnosis_tables(root,diagnosis,h,link)
    else:body+='<p class="muted">完整纵向zero／10k高分保持／分量配对诊断正在生成，未完成前不沿用旧四观测的persistent数量冒充最新全序列。'+link(root/'analysis'/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','Safety过程诊断')+'</p>'
    body+=_train_zero_support_tables(editorial.get('train_zero_support',{}),editorial.get('dac_tie_counterfactual',{}),h,link)
    section('failures','哪些场景修复了，哪些能力丢失了','部署zero不等于训练全失败，也不等于没有正优势；先确定集合与可用学习信号。',body)

    body='<p><b>当前机制：</b>安全／可行EP分量优势按候选指标分配信用，保留原W3随机探索与IL70参考；这不是具备路权、遮挡或交通交互推理的证明。低EP可能是必要等待，只有同场景确有安全可行替代时才谈错过通行机会。</p>'
    body+=table(['候选信用机制','当前实际含义'],[
        ['有可行候选的mixed组','f为NC＝DAC＝TTC＝1，m为8条中的可行数；安全项f−m/8，EP仅在可行子集内中心化。unsafe不会因EP获得正A。'],
        ['整个组无可行候选','按mean(c)−c保留较小NC／DAC／TTC deficit的相对方向；deficit不是事故严重程度，正A也不等于该候选安全。'],
        ['组间尺度与保护项','64条fresh候选的raw A统一population std，之后cap5；四inner复用同A。beta50物理均值anchor指向原IL70，不是旧策略KL，不能仅从scalar loss认定拉回过强。']])
    if mechanism.get('status')=='COMPLETE':body+=_credit_mechanism_tables(mechanism,h,link)
    if credit_gradient.get('status')=='COMPLETE':body+=_credit_gradient_tables(credit_gradient,h,link)
    if gradient_probe.get('status')=='COMPLETE':body+=_gradient_tables(gradient_probe,h,link)
    if policy_cases.get('status')=='COMPLETE_EXISTING_RECORDS_ONLY':body+=_policy_case_tables(root,policy_cases,h,link)
    body+='<p>Qwen特征冻结／缓存；scene与predictive memory以及动作attention LoRA实际参与优化，约52.7M参数，不是只训练动作头。可训练不等于已学到充分风险信息；当前缺少的驾驶知识应反馈IL，RL继续负责在该知识基础上强化轨迹。</p>'
    if diagnosis.get('status')=='COMPLETE':body+=_rollout_tables(diagnosis,h,link)
    body+='<p class="notice">旧W2、低噪声与Rescue组合的完整负结果说明：候选分更高、teacher确实给全零组提供方向，都不保证部署更好。它们不证明所有探索或数据策略无效，也不应成为继续叠加组件的理由。</p>'
    section('learning','学习过程中还缺哪一环','把信息、候选探索、信用、优化与部署五个环节分开，避免单一总分解释全部现象。',body)

    body='<div class="notice"><b>统一未来比较：</b>固定同一IL70 checkpoint及action／scene／predictive初始化，fresh AdamW、fresh RNG42、update0，参考与数据人口先固定；每次只验证一个主要假设。所有分支用相同预算与完整canonical评测，比较安全／效率、zero修复／新生和固定高分保持。这不是旧RL10k续训分叉；新方案实际状态见下表，不把授权、注册、child启动与optimizer更新混为一谈。</div>'
    body+='<p>未来高分主队列预先固定为IL70 score≥0.9；历史Safety10k高分队列只作次级保持诊断。TRAIN的zero／正低分队列同样从IL起点预定义，不从NAVTEST名单选训练场景。既有fresh-Safety同预算结果仅在身份／数据／推理协议一致时复用；旧RL10k分叉不能充当fresh起点。候选先提议1k／5k复核，是否延长另按证据决定；optimizer步数与额外forward／GPU成本分别记录。</p>'
    body+=_fresh_pair_tables(fresh_pair,runs,h,link)
    body+=_feasible_score_tables(editorial.get('feasible_score',{}),runs,h,link)
    body+=_fpo_fixed_cohort_tables(editorial.get('fpo_fulltime',{}).get('time_coverage',{}),h,link)
    body+=_fulltime_tables(editorial.get('fpo_fulltime',{}),runs,h,link)
    if directions.get('status'):body+=_direction_tables(directions,h,link,gradient_measured=gradient_probe.get('status')=='COMPLETE',fresh_pair=fresh_pair,runs=runs)
    else:body+='<p class="muted">基于Safety过程与primary文献的3–4个方向正在整理；未将候选方案写成已注册或已运行。</p>'
    proximal=editorial.get('flow_direct_proximal',{})
    if proximal.get('status')=='COMPLETE_CPU_RESEARCH_ONLY':
        body+=detail('保留为后续研究：Flow直接proximal／条件KL，目前不追加GPU', '<p>两个有界8rank窗口里，按有效候选步加权的PPO阻断率为2.82%／2.34%；有正负不对称，但尚未证明是主要瓶颈。条件Gaussian KL可以用于测量与候选门控；官方实现的mean logp／mean KL也不能当全文全密度重要性比。当前保留CPU原型与单位方案，优先等待已运行P1／FPO的完整结果。</p><p>'+link(root/'analysis'/'FLOW_DIRECT_PROXIMAL_RESEARCH_20261031.json','primary文献、实际窗口与暂缓决定')+'</p>')
    if editorial.get('llm_trick_transfer',{}).get('status')=='COMPLETE_LITERATURE_TO_IMPLEMENTATION_REVIEW':body+=detail('LLM RL技巧迁移：保留机制边界', '<p>恒定LR配warmup有论文先例；LLM不使用token KL不能推出删除这里的物理轨迹anchor。JustRL未单独证明batch-level std，因此不把它当三篇共享结论直接套用，而以本任务已有归一化、梯度与同预算评测作迁移依据。'+link(root/'analysis'/'LLM_RL_TRICK_TRANSFER_CHECK_20261031.json','primary文献与实际实现复核')+'</p>')
    section('next','实验设计、文献与完整配对分析','当前四组实验的详细实现、研究依据与阶段统计。',body)

    historical=[]
    for arm,other in sorted(runs.items()):
        if arm==SAFETY or arm in FRESH_ARMS:continue
        scores=sorted((row for row in all_test if row.get('arm')==arm),key=update)
        if not scores and not other.get('job'):continue
        decision=decisions.get(arm,{})
        if arm in LR_ARMS:state='用户调整研究焦点停止；已取得退出收据'if arm in focus_stopped else '用户调整研究焦点停止；待实际退出收据'
        elif decision.get('stop_mode','').startswith('completed '):state='已完成，决定不续训当前配方'
        elif arm in stopped:state='原研究已结束；保留其原始停止依据'
        else:state='已归档；实际执行状态 '+str(other.get('status','待记录'))
        record=max(scores,key=lambda row:row['PDMS'],default={});last=scores[-1]if scores else {}
        if arm in focus_stopped:state+='；停止于'+number(focus_stopped[arm].get('native_checkpoint',{}).get('updates'))+' / inner'+number(focus_stopped[arm].get('native_checkpoint',{}).get('next_inner'))
        history_evidence=link(root/'owners'/'SAFETY_EP_FOCUS'/'STOPPED_SUMMARY.json','用户调整焦点退出收据')if arm in focus_stopped else link(root/'owners'/'CAMPAIGN'/'arm_decisions'/f'{arm}.STOPPED.json','退出收据')if arm in stopped else link(root/'owners'/'CAMPAIGN'/'arm_decisions'/f'{arm}.json','研究决定')
        historical.append([esc(arm),esc(state),fmt(record.get('PDMS'),5)+'<small>'+number(update(record))+'更新</small>'if record else '无完整成绩',fmt(last.get('PDMS'),5)+'<small>'+number(update(last))+'更新</small>'if last else '无完整成绩',history_evidence])
    body=detail('历史RL方法与四条LR分叉：成绩保留，当前不再作为并行主线',table(['历史路线','结束／归档原因','本路线已测最高','最近完整成绩','证据'],historical)+'<p>不同预算／初始化不可按单个分数排算法。四条LR因用户收缩研究范围结束，不能称新算法失败；既往futility／no-extension决定保持各自原始科学原因。</p>')
    lr_review=editorial.get('frozen_lr_review',{})
    if lr_review.get('status')=='COMPLETE_FINITE_EXISTING_POINTS_ONLY':
        lr_rows=[]
        for row in lr_review.get('observed_points',[]):
            contrast=row.get('vs_samebudget_Safety',{})
            lr_rows.append([esc(row.get('arm')),number(row.get('updates')),fmt(row.get('PDMS'),5),signed(contrast.get('delta_PDMS'))+'<small>'+ci(contrast.get('CI95_PDMS'))+'</small>',' / '.join(signed(contrast.get(key,{}).get('delta_pp'))for key in ('NC','TTC','EP'))])
        body+=detail('旧LR／数据分叉的有限历史复核：不恢复训练或缺失评测',table(['旧分叉','更新','完整PDMS','对同预算Safety：Δ［区间］','ΔNC／TTC／EP pp'],lr_rows)+'<p>7个已完整12k／14k点均未超过Safety10k峰；本集合最高为DATA14k 91.08105。LOW12k的同预算增益到14k未持续；COS无优势证据。DATA的IL70高分队列有改善，但整体NC／TTC负向区间与EP提升并存，不能只说无效或安全提升。LRDATA14k缺失保留未知，不补点、不复活旧臂；这些是Safety10k原生分叉，不能当fresh IL70调度的完整因果检验。</p><p>'+link(root/'analysis'/'FROZEN_LR_EXISTING_EVIDENCE_REVIEW_20261031.json','既有7点、固定IL高分队列与分量区间')+'</p>')
    body+=detail('人口、身份与解释边界','<p>共同原始IL70父模型；NAVTRAIN103,288包括旧DEV1,696，旧DEV属于训练内诊断。NAVTEST与TRAIN隔离。NAVTEST已用于研究选择，相关名单不回流训练。只有COMPLETE、完整人口、invalid0且SUMMARY seal正确才发布分数；partial不发布均分。配对区间按物理log，固定seed结果不代表训练seed不确定性。</p><p>'+link(root/'IL_PARENT','固定IL父模型')+' · '+link(root/'analysis'/'REPRESENTATION_TRAINABILITY_AUDIT_20261030.json','表征可训练性审计')+' · '+link(root/'analysis'/'RL_LR_NATIVE_AUDIT_20261031.json','原生LR审计')+' · '+link(root/'analysis'/'RL_LR_PRIMARY_RESEARCH_20261031.json','primary LR文献')+'</p>')
    body+=detail('完整分析与历史研究归档','<p>'+link(root/'analysis'/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','Safety过程诊断')+' · '+link(root/'analysis'/'SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json','少量freshIL改进方向')+' · '+link(root/'analysis'/'COMPLETED_5K_PILOT_REVIEWS_20261031.json','六配方完成复核')+' · '+link(root/'owners'/'CAMPAIGN'/'STATE.json','原owner执行历史')+'</p>')
    body+='<p class="muted">'+link(root/'SAFETY_EP_FOCUSED_RESEARCH_20261031.json','旧Safety-only范围（后续fresh算法授权已扩展）')+' · '+link(root/'owners'/'SAFETY_EP_FOCUS'/'STOPPED_SUMMARY.json','四分叉实际退出，pending0')+'</p>'
    watcher=snapshot.get('report_watcher',{})
    if watcher.get('status')=='RUNNING'and watcher.get('last_render',{}).get('status')=='RENDERED':body+='<p class="muted">自动报告owner '+number(watcher.get('pid'))+' 正常；上次真实watcher成功渲染 '+esc(watcher['last_render'].get('generated_utc'))+'，常规1800秒，下次 '+esc(watcher.get('next_regular_utc'))+'。'+('OWNER中旧error文本属于已恢复的历史证据，当前健康以RUNNING和最新成功记录判断。'if watcher.get('error')else '')+link(root/'reporting'/'WATCHER_HEALTH.json','自动更新健康与历史错误分类')+'</p>'
    if snapshot.get('warnings'):body+='<p class="warning">'+esc('；'.join(snapshot['warnings']))+'</p>'
    section('archive','历史方法与原始证据','保留已有工作与结果，首页不再被历史路线淹没。',body)
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172b43;font:15px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}header{background:#123248;color:#fff;padding:36px max(22px,calc((100vw - 1300px)/2));border-bottom:4px solid #10b981}h1{font-size:30px;margin:8px 0}header p{color:#c3d8e5;max-width:1000px}nav{position:sticky;top:0;background:#fffffff5;z-index:2;border-bottom:1px solid #dce6ef;display:flex;gap:25px;padding:12px max(22px,calc((100vw - 1300px)/2));overflow:auto;white-space:nowrap}nav a{font-size:13px;color:#36596d}main{max-width:1344px;margin:auto;padding:24px 22px 40px}section{background:#fff;border:1px solid #dce6ef;border-radius:13px;padding:28px 30px;margin-bottom:25px;scroll-margin-top:64px}h2{font-size:23px;margin:0}h3{font-size:17px;margin:18px 0 8px}.deck,.muted{color:#64768a;font-size:13px}.deck{margin:4px 0 20px}p{margin:11px 0}.notice,.callout{background:#eef8f4;border-left:3px solid #10b981;padding:14px 18px;border-radius:8px}.warning{background:#fff1e9;padding:14px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}.cards article{background:#f3f7fb;border:1px solid #e1e9f0;border-radius:10px;padding:17px}.cards span,small{display:block;color:#65788d;font-size:12px}.cards strong{display:block;font-size:28px;margin:5px 0;color:#123e51}.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;margin:20px 0;align-items:start}.callout{background:#f6f8fb;border-left:0}.callout h3{margin-top:0}figure{margin:0;background:#fbfcfe;border:1px solid #e2e9f0;border-radius:10px;padding:15px;min-width:0}figcaption{font-size:14px;font-weight:600}figure svg{width:100%;height:auto}.figure-note,.downloads{font-size:12px;color:#64768a}.table-wrap{overflow:auto;border:1px solid #e2e9f0;border-radius:10px;margin:17px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #e7edf3;overflow-wrap:anywhere}th{font-size:12px;background:#f0f5f9;color:#426075;white-space:nowrap}tr:last-child td{border-bottom:0}details{border:1px solid #dce6ef;border-radius:10px;padding:15px 18px;margin:13px 0}summary{cursor:pointer;font-size:14px;font-weight:600;color:#35566c}details[open] summary{margin-bottom:17px}a{color:#1979aa;text-decoration:none}a:hover{text-decoration:underline}.protocol,footer{color:#64768a;font-size:12px}footer{text-align:center;padding:10px 20px 30px}pre{font-size:11px;white-space:pre-wrap;background:#f5f7fa;padding:14px;max-height:330px;overflow:auto}@media(max-width:950px){.two{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}section{padding:22px}h1{font-size:26px}}@media(max-width:550px){main{padding:15px 9px}section{padding:18px 14px}.cards{gap:9px}.cards article{padding:12px}.cards strong{font-size:23px}}'''
    import importlib
    import current_research_report
    importlib.reload(current_research_report)
    front=current_research_report.build_current_sections(root,snapshot,h)
    css+=current_research_report.CSS
    titles=['SafetyEP历史收益与停止依据','历史对照曲线与分量归因','零分场景与高分保持诊断','信用分配、真实梯度与策略案例','四组实验的完整设计、文献与配对分析','已结束路线与原始证据']
    background='<section id="background"><h2>机制、场景诊断与历史资料</h2><p class="deck">当前工作与最新结果已在前方汇总；需要追溯方法、证据或历史路线时再展开。</p>'+''.join('<details class="background-chapter"><summary>'+title+'</summary>'+chapter+'</details>'for title,chapter in zip(titles,chapters))+'</section>'
    nav=[('current','当前实验与进度'),('current-results','最新评测结果'),('current-analysis','结论与下一步'),('background','机制与历史资料')]
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World RL：当前实验、评测与研究进展</title><style>'+css+'</style></head><body><header><small>QWEN-WORLD · RL RESEARCH</small><h1>RL研究进展：当前实验、评测与结论</h1><p>四组同IL70实验并行推进，目标完整NAVTEST达到92。先看当前在做什么、已评测到哪里，以及结果支持什么判断。</p><small>报告生成 '+esc(snapshot['generated_utc'])+' · 常规30分钟＋完整阶段事件更新</small></header><nav>'+''.join('<a href="#'+key+'">'+title+'</a>'for key,title in nav)+'</nav><main>'+''.join(front)+background+'</main><footer>固定路径 '+esc(root/'report.html')+' · 本报告只读汇总，不控制训练、评测或资源</footer></body></html>'


def _matched_mechanism_tables(mechanism,h,link,normalization=None):
    signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
    interval=lambda value:'['+', '.join(signed(x)for x in value)+']'if isinstance(value,(list,tuple))and len(value)==2 else '区间待记录'
    complete=[row for row in mechanism.get('NAVTEST_matched',[])if row.get('status')=='COMPLETE']
    body='<h3>SafetyEP相对原PPO：同预算收益来自哪里</h3><p>同IL70 fresh初始化、full103,288、W3探索、学习率、PPO保护与IL70物理anchor保持一致；实际科学配置只改变候选优势公式。canonical评分没有增加安全奖励。优势归一化同时改变PG相对anchor的尺度，因此不能把收益只归因于安全门控。</p>'
    body+=h.table(['更新','原PPO FULL','SafetyEP','ΔPDMS［44log 95%区间］','ΔNC／DAC／TTC／EP pp','zero修复／新增（对PPO）'],[[h.number(row['updates']),h.fmt(row.get('baseline_PDMS'),5),h.fmt(row.get('target_PDMS'),5),signed(row.get('delta_PDMS'))+'<small>'+interval(row.get('CI95_PDMS'))+'</small>',' / '.join(signed(row.get('metrics',{}).get(key,{}).get('delta_pp'))for key in METRICS),h.number(row.get('zeros',{}).get('repaired'))+' / '+h.number(row.get('zeros',{}).get('new_zero'))]for row in complete])
    body+='<p class="notice"><b>收益有明确阶段性与权衡：</b>1k／5k／10k总分配对区间为正，20k区间跨零；10k主要伴随EP提升约1.00pp，NC与TTC点估计下降。不能把“条件安全信用”称为所有安全维度改善，也不能把10k领先外推到长训。</p>'
    extras=[]
    for row in complete:
        cohort=row.get('fixed_IL70_high_ge09_primary',{})
        extras.append([h.number(row['updates']),h.number(cohort.get('scenes')),signed(cohort.get('delta_PDMS'))+'<small>'+interval(cohort.get('CI95_PDMS'))+'</small>'])
    train=[row for row in mechanism.get('TRAIN1024_matched',[])if row.get('status')=='COMPLETE']
    body+='<details><summary>预定义IL70高分保持与TRAIN1024：固定分母、不同用途</summary>'+h.table(['更新','固定IL70≥.9人口','Safety−PPO子集ΔPDMS［区间］'],extras)+h.table(['更新','TRAIN1024 ΔPDMS［162log区间］'],[[h.number(row['updates']),signed(row.get('delta_PDMS'))+'<small>'+interval(row.get('CI95_PDMS'))+'</small>']for row in train])+'<p>IL70高分队列为比较前固定的primary；Safety10k高分仅secondary。TRAIN为分层训练内面板，既不是完整TRAIN均值，也不是泛化代理。完整缓存没有raw progress米数，因此不把EP换名当原始位移。所有区间为固定训练seed下探索性物理log bootstrap，未作多重比较校正。</p></details>'
    ablations=(normalization or {}).get('contrasts',{});norm=ablations.get('PPO_BATCHSTD_minus_PPO_FULL_u010000',{});gate=ablations.get('PPO_SAFETY_EP_minus_PPO_BATCHSTD_u010000',{})
    if norm and gate:body+='<details><summary>已有归一化拆解：收益不能全部归给安全门控</summary><p>10k仅改centered-advantage std的BATCH相对FULL '+signed(norm.get('delta_PDMS'))+'，95%区间 '+interval(norm.get('CI95_PDMS'))+'；Safety相对BATCH '+signed(gate.get('delta_PDMS'))+'，区间 '+interval(gate.get('CI95_PDMS'))+' 跨零。归一化已有独立收益证据，但目前不能确证Safety胜过BATCH，也不能用两项分数的比例作因果分摊；现有设计并非完整2×2，不外推到FPO。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_NORMALIZATION_EXISTING_ABLATION_20261031.json','已有归一化实验与信用拆解')+'</p></details>'
    body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_EP_VS_PPO_MECHANISM_20261031.json','同预算机制分析、完整分量区间与身份核对')+'</p>'
    return body


def _credit_mechanism_tables(mechanism,h,link):
    aggregate=mechanism.get('same_Safety_rollout_credit_counterfactual',{}).get('aggregate',{})
    original=aggregate.get('original_PPO_counterfactual',{});actual=aggregate.get('Safety_actual',{})
    if not aggregate:return ''
    body='<h3>同一批真实候选重算：到底改变了什么信用</h3><p>原PPO：组内减canonical reward均值，再除全8rank的raw reward std。SafetyEP：安全相对项＋可行子集EP项，再除全8rank raw A std。下表只重用Safety已保存的 '+h.number(aggregate.get('groups'))+' 个TRAIN组／'+h.number(aggregate.get('candidates'))+' 条候选，隔离公式直接作用；它不是原PPO在同θ的新rollout，也不是参数梯度测量。</p>'
    metrics=[('有可行替代时仍获正A的unsafe候选','positive_unsafe_with_feasible_alternative'),('mixed组中的负A可行候选','feasible_negative_candidates_in_mixed'),('PG为零的组','zero_PG_groups')]
    rows=[[label,h.number(original.get(key)),h.number(actual.get(key))]for label,key in metrics]
    rows.append(['可行子集EP与信用负投影的组',h.number(original.get('within_feasible_EP_credit_projection',{}).get('negative')),h.number(actual.get('within_feasible_EP_credit_projection',{}).get('negative'))])
    body+=h.table(['同候选标量信用统计','原PPO公式：CPU反事实','Safety实际保存A'],rows)
    negative=aggregate.get('original_negative_EP_projection_classification',{})
    body+='<p class="notice">原PPO的 '+h.number(negative.get('groups'))+' 个负EP投影组，全部存在可行候选之间的Comfort差异；它可以反映原评分的舒适性权衡，不能一概称为错误。Safety更直接鼓励可行内进度，同时丢掉这条Comfort信用。标量信用投影不是对模型参数的梯度方向。</p>'
    ddc_old=original.get('DDC_independent_quality',{});ddc_new=actual.get('DDC_independent_quality',{})
    body+='<details><summary>驾驶方向质量审计：DDC与当前评分的边界</summary><p>当前canonical逐候选满足 NC×DAC×(5EP＋5TTC＋2Comfort)/12，DDC没有乘入评分。通过NC／DAC／TTC但DDC&lt;1且正A：原PPO '+h.number(ddc_old.get('strict3feasible_DDCbad_positive_A'))+' 条，Safety '+h.number(ddc_new.get('strict3feasible_DDCbad_positive_A'))+' 条。它是独立驾驶质量读数；不把额外乘DDC后的假想分数叫作现PDMS损失或可达提升，不修改scorer制造92。</p></details>'
    return body


def _gradient_tables(probe,h,link):
    rows=[]
    for case in probe.get('cases',[]):
        scope=case.get('scopes',{}).get('all',{});norms=scope.get('global_mean_gradient_norm',{});pg=norms.get('pg');anchor=norms.get('anchor')
        rows.append([h.number(case.get('case')),h.fmt(anchor/pg if h.finite(pg)and pg else None,4),h.fmt(scope.get('cosine',{}).get('anchor__pg'),4),h.fmt(scope.get('summed_gradient_norm')/pg if h.finite(pg)and pg and h.finite(scope.get('summed_gradient_norm'))else None,4)])
    body='<h3>28k原生PG／anchor真实梯度：局部冲突，未见anchor压倒PG</h3><p>实际原生28k checkpoint，'+h.number(probe.get('groups'))+' 个TRAIN组／'+h.number(probe.get('candidates'))+' 条候选，8rank真实梯度归约；零optimizer更新。不是CPU标量loss推断，也不代表总体或完整训练期。</p>'
    body+=h.table(['独立collection case','‖anchor‖／‖PG‖','cos(anchor, PG)','‖PG＋anchor‖／‖PG‖'],rows)
    body+='<p class="notice">三组anchor范数约为PG的15%–26%，其中两组方向冲突；合成范数仍约为PG的90%–106%。当前小窗口不支持“anchor压倒学习”的判断，因此不优先盲删anchor或刷新reference。Adam实际参数步长和跨阶段分布仍未由这次零更新探针回答。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_EP_NATIVE28K_GRADIENT_PROBE_20261031/SUMMARY.json','原生梯度范数、action／memory分解与协议')+'</p>'
    return body


def _credit_gradient_tables(diagnosis,h,link):
    cases=diagnosis.get('gradient_results',{}).get('case_metrics',[])
    if not cases:return ''
    rows=[]
    for case in cases:
        scopes=case.get('scope_metrics',{});whole=scopes.get('all',{});memory=scopes.get('memory',{})
        residual=whole.get('BF16_separate_attribution_sum_residual_relative_to_Safety')
        rows.append([h.number(case.get('case')),h.fmt(whole.get('Safety_to_PPO_PG_norm_ratio'),3),h.fmt(whole.get('PPO_Safety_PG_cosine'),3),h.fmt(memory.get('PPO_Safety_PG_cosine'),3),h.fmt(memory.get('EP_to_safe_norm_ratio'),3),h.fmt(memory.get('safe_EP_cosine'),3),h.fmt(100*residual if h.finite(residual)else None,2)+'%'])
    body='<h3>同θ真实梯度回放：信用变化确实进入memory</h3><p>固定原生Safety28k和同24个已评分TRAIN组，重用相同候选／logp，新增生成、评分和optimizer更新均为0。三个case各为真实8rank归约，不将范数或cos的均值冒充24组pool梯度。</p>'
    body+=h.table(['case','全模型‖SafetyPG‖／‖PPO PG‖','全模型cos(PPO, Safety)','Memory cos(PPO, Safety)','Memory ‖EP‖／‖safe‖','Memory cos(safe, EP)','全模型BF16归账残差'],rows)
    body+='<p class="notice"><b>不只是动作头，也不是EP全面主导：</b>信用公式改变了全模型和memory的真实梯度方向；Memory的EP分量仅约为safe范数的14%–22%，且三次夹角cos轻负。它证明优化信号发生变化，不证明语义知识已经掌握，也不证明梯度更大导致最终提分。</p>'
    comfort=diagnosis.get('gradient_results',{}).get('Comfort_variable_Safety_PG0_groups',[])
    body+='<p>另有 '+h.number(len(comfort))+' 组NC／DAC／TTC／EP全1、Comfort却有差异：Safety PG为0，原PPO实测局部全模型PG范数约0.03253／0.03102。<b>三安全＋EP1饱和不等于canonical PDMS满分</b>；这是当前信用忽略Comfort区分的直接反例，不能把所有原PPO反向EP信用都判错。</p>'
    body+='<p class="muted">独立BF16反传分量并非逐bit可加：全模型归账残差0.35%–0.74%。该数不同于另一次P1早期探针的0.34%，不能混用；这里是first-inner、cap／clip均无阻塞的局部回放，尚未包含Adam预条件或真实参数位移，不能做终局因果归因。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_CREDIT_GRADIENT_REPLAY_INTERPRETATION_20261031.json','同θ梯度、Comfort反例与数值边界')+'</p>'
    return body


def _policy_case_tables(root,diagnosis,h,link):
    cards=[]
    for case in diagnosis.get('cases',[]):
        svg=Path(case.get('artifacts',{}).get('svg',{}).get('path',''))
        if not svg.is_file():continue
        document=svg.read_text(encoding='utf-8');start=document.find('<svg')
        if start<0:continue
        full=case.get('metrics',{}).get('PPO_FULL_u010000_NAVTEST',{})
        safety=case.get('metrics',{}).get('PPO_SAFETY_EP_u010000_NAVTEST',{})
        caption='<figcaption>'+h.esc(case.get('title_zh','策略案例'))+'</figcaption><p class="figure-note">PPO10k '+h.fmt(full.get('PDMS'),2)+' → Safety10k '+h.fmt(safety.get('PDMS'),2)+'；规则合格 '+h.number(case.get('eligible_scene_count'))+' 场景中选出的案例，token '+h.esc(case.get('token'))+'。</p>'
        explanation='<p class="figure-note">'+h.esc(case.get('interpretation_zh',''))+'</p>'
        downloads='<p class="downloads">'+link(svg,'SVG')+' · '+link(case.get('artifacts',{}).get('png',{}).get('path'),'PNG')+'</p>'
        cards.append('<figure>'+caption+document[start:]+explanation+downloads+'</figure>')
    if not cards:return ''
    body='<details><summary>四个策略案例：进度改善、zero修复、高分变零与七观测持续零</summary><p>这些NAVTEST案例只用于报告，不进入训练；按规则挑选的极端／反例不代表总体发生率或提升因果。图为真实8pose请求轨迹，GT为记录的人类未来，不是PDM参考规划器或最优行动证书。</p><div class="two">'+''.join(cards)+'</div>'
    body+='<p class="muted">无地图／障碍物证据，不推断撞谁、遮挡对象或模型是否理解红灯；低EP也不能单独判定无故停车。这里持续零仅为IL70及两策略1k／5k／10k共7个观测，与上文Safety全序列10点persistent133属于不同范围，不能混用。</p><p>'+link(root/'analysis'/'SAFETY_EP_POLICY_CASES_20261031.json','挑选规则、7次观测、原始轨迹与相机证据')+'</p></details>'
    return body


def _fresh_runtime_state(run,fresh):
    guard=fresh.get('futility_guard',{}).get('state',{}).get('arms',{}).get(run.get('id'),{})
    statuses={guard.get('status'),run.get('registry_status'),run.get('status')}
    if 'STOPPED_FUTILITY'in statuses:return '已按证据规则停止；退出证据见guard状态'
    if 'STOP_REQUESTED_FUTILITY'in statuses:return '已发证据止损请求；尚待原生退出收据'
    if 'COMPLETED_AT_REGISTERED_10K'in statuses:return '注册10k训练已完成；评测独立更新'
    if 'COMPLETED_AT_REGISTERED_CAP'in statuses:return '注册训练上限已完成；评测独立更新'
    if run.get('worker_returncode')not in (None,0):return '当前启动／执行退出，待实际复核'
    if run.get('worker_receipt_status')=='RUNNING'and isinstance(run.get('updates'),(int,float))and run['updates']>0:return '实际原生训练；已见optimizer更新'
    if run.get('worker_receipt_status')=='RUNNING':return 'child已开始；尚未见optimizer更新证据'
    if fresh.get('registration'):return '已注册；等待实际启动证据'
    return '科学实现／冻结中；尚未见启动证据'


def _fresh_pair_tables(fresh,runs,h,link):
    labels={'PPO_SAFETY_SUCCESS_MEMORY':'P1：当前run验证成功memory','PPO_SAFETY_FPO':'P2：Flow endpoint-CFM ratio替代PG'}
    rows=[]
    for arm in PAIR_ARMS:
        run=runs.get(arm,{})
        state=_fresh_runtime_state(run,fresh)
        rows.append([labels[arm],state+'<small>'+h.esc(run.get('host','等待实际host'))+'</small>',h.number(run.get('updates'))if run else '—','1k／5k／10k；总研究上限10k'])
    body='<h3>92+新研究：统一fresh起点，单独记录实际启动</h3>'+h.table(['新路线','当前可核状态','新run更新','评测与预算'],rows)
    startup=fresh.get('startup',{});confirmed=[]
    for arm,node in startup.get('arms',{}).items():
        updates=[row.get('actual_update')for row in node.get('rank_updates',[])if row.get('finite')is True and isinstance(row.get('actual_update'),(int,float))]
        if node.get('all8_real_updates')and len(updates)==8:confirmed.append(labels.get(arm,arm)+' '+h.number(min(updates))+'更新／8rank finite')
    if confirmed:body+='<p class="notice"><b>已核实启动：</b>'+h.esc('；'.join(confirmed))+'。这是启动收据的时间截面，最新更新数读上表，部署结果独立列在完整配对表。'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_FRESH_PAIR_NATIVE_STARTUP_20261031.json','实际原生启动证据')+'</p>'
    body+='<p>P1通过独立有界aux replay保留当前run已经验证成功的轨迹方向；历史teacher不混入on-policy PG。P2尝试普通PPO的逐MC CFM surrogate ratio替代transition PG，不宣称精确endpoint似然。运行状态不用于推导分数，部署成绩只采用下方完整结果；5k阶段评审，10k明确完结，配置中的50k不构成自动续训授权。</p>'
    body+=_fresh_early_process_tables(fresh.get('process_early',{}),h,link)
    if fresh.get('registration',{}).get('resource_decision'):body+='<details><summary>本轮资源选择与隔离</summary><p>本轮首次实际部署：P1在training-vlawm-zt，FPO在training-vlawm-tsinghua-zt5；当时Tsh6已有独立IL_A、Tsh3已有原Safety，未为本次研究迁移或打断它们。最新host以上表为准。资源状态只读中央复核与实际准入证据，不把历史空闲样本当当前可用保证。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/resources/RESOURCE_REVIEW_LATEST.json','中央资源复核')+'</p></details>'
    review=fresh.get('matched_review',{});pairs=review.get('contrasts',{})
    complete=[(key,row)for key,row in pairs.items()if isinstance(row,dict)and row.get('status')=='COMPLETE']if isinstance(pairs,dict)else []
    if complete:
        signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
        body+='<h3>新fresh研究已完整的配对：PARTIAL不隐藏完整子结果</h3>'+h.table(['对照','完整目标PDMS','ΔPDMS','95%区间','ΔNC／DAC／TTC／EP pp'],[[h.esc(key),h.fmt(row.get('target_PDMS'),5),signed(row.get('delta_PDMS')),'['+', '.join(signed(value)for value in row.get('CI95_PDMS',[]))+']',' / '.join(signed(row.get('components',{}).get(metric,{}).get('delta_pp'))for metric in METRICS)]for key,row in complete])
    elif review:body+='<p class="muted">完整配对分析器状态 '+h.esc(review.get('status','待完整输入'))+'；没有完整pair的门槛不补分数。</p>'
    for key,label in [('policy','92+研究授权与身份'),('registration','新两路线注册')]:
        if fresh.get(key):body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/'+('SAFETY_RL_92_TARGET_POLICY_20261031.json'if key=='policy'else 'EXPERIMENT_SAFETY_FRESH_PAIR_REGISTRATION_20261031.json'),label)+'</p>'
    guard=fresh.get('futility_guard',{});guard_state=guard.get('state',{})
    if guard.get('owner',{}).get('pid')and guard_state:
        status_labels={'OBSERVING':'观察中；尚未达到双点止损条件','STOP_REQUESTED_FUTILITY':'已请求停止；待原生退出与checkpoint收据','STOPPED_FUTILITY':'已按证据规则停止'}
        guard_rows=[[h.esc(arm),h.esc(status_labels.get(node.get('status'),node.get('status','待状态')))]for arm,node in guard_state.get('arms',{}).items()]
        body+='<details><summary>已启动的证据止损guard：请求与真正退出分开</summary><p>同一arm在1k和5k两点均须同时满足：相对同预算Safety ΔPDMS&lt;−1、总分配对CI上界&lt;0、固定IL70高分队列CI上界&lt;0。单一1k或partial不会自动停；自然10k只进入review。停止请求之后，只有原生exit0、精确transport退出、完整checkpoint／8rank RNG／success-bank与owner确认齐全才记已停；配方退化不等于算法家族无效。</p>'+h.table(['新路线','guard最新状态'],guard_rows)+'<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/owners/SAFETY_FRESH_PAIR_FUTILITY_GUARD/STATE.json','实际guard状态与停止证据')+'</p></details>'
    return body


def _feasible_score_tables(trial,runs,h,link):
    if not trial.get('registration'):return ''
    run=runs.get(QUALITY,{})
    body='<article class="callout"><h3>新质量信用对照：q=(5EP＋2Comfort)/7</h3><p><b>实际状态：</b>'+h.esc(_fresh_runtime_state(run,trial))+'；8rank共同更新 '+h.number(run.get('updates'))+'，已见 '+h.number(run.get('journal_ranks'))+' rank，host '+h.esc(run.get('host','待记录'))+'。同IL70 fresh，接替已退出原Safety释放的research槽；1k／5k完整NAVTEST与TRAIN1024异步，上限5k，不自动延长。</p>'
    if not h.finite(run.get('updates'))or run['updates']<=0:body+='<p>当前首批采集中；已建立但为空的metrics文件不作为失败，等待真实8rank optimizer更新证据。</p>'
    body+='<p>保留Safety可行性／allfail成本fallback及其他PPO设置，将可行内EP信用替换为质量q，重新区分EP1但Comfort不同的候选。它不是旧SafeComfort把C置1，也不改canonical scorer；主对照原Safety同预算，次级BATCHSTD。</p><p class="muted">q同时把EP系数降为5/7，并非只额外补入Comfort；SDE候选的Comfort差异可能含采样噪声，且本变体不修allfail costtie。CPU同候选信用变化不等于真实梯度或部署提分。</p>'
    review=trial.get('matched_review',{});pairs=review.get('contrasts',{})
    complete=[(key,row)for key,row in pairs.items()if isinstance(row,dict)and row.get('status')=='COMPLETE']if isinstance(pairs,dict)else []
    signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
    interval=lambda values:'['+', '.join(signed(value)for value in values)+']'if isinstance(values,(list,tuple))and len(values)==2 else '区间待记录'
    if complete:
        body+=h.table(['已完整对照','目标PDMS','ΔPDMS［95%区间］','ΔNC／DAC／TTC／EP pp','zero修复／新增'],[[h.esc(key),h.fmt(row.get('target_PDMS'),5),signed(row.get('delta_PDMS'))+'<small>'+interval(row.get('CI95_PDMS'))+'</small>',' / '.join(signed(row.get('components',{}).get(metric,{}).get('delta_pp'))for metric in METRICS),h.number(row.get('zeros',{}).get('repaired'))+' / '+h.number(row.get('zeros',{}).get('new_zero'))]for key,row in complete])
        body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_FEASIBLE_SCORE_MATCHED_REVIEW_20261031.json','完整分量区间、固定IL高／低分与TRAIN诊断')+'</p>'
    else:body+='<p class="notice">当前尚无此质量q变体的新完整NAVTEST成绩；实际启动、过程统计与最终收益分别判断，曲线不补占位分数。</p>'
    body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/EXPERIMENT_SAFETY_FEASIBLE_SCORE_REGISTRATION_20261031.json','封存质量信用注册')+' · '+link('/mnt/project/qwen-world-rl-optimization-20261030/owners/SAFETY_FEASIBLE_SCORE/STATE.json','独立有限owner与评测状态')+'</p></article>'
    return body


def _fpo_fixed_cohort_tables(design,h,link):
    evidence=design.get('complete_1k_cohort_evidence',{})
    signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
    interval=lambda values:'['+', '.join(signed(value)for value in values)+']'if isinstance(values,(list,tuple))and len(values)==2 else '区间待记录'
    rows=[]
    for cohort in('NAVTEST','TRAIN1024'):
        comparisons=evidence.get(cohort,{}).get('comparisons',{}).get('samebudget_SafetyEP1k',{})
        for key,label in [('fixed_IL70_high_ge09','IL70原高分≥.9'),('fixed_IL70_low_lt09','IL70原低分<.9')]:
            pair=comparisons.get(key,{})
            if not pair:continue
            ttc=pair.get('components',{}).get('time_to_collision_within_bound',{})
            rows.append([cohort+' · '+label,h.number(pair.get('scenes')),signed(pair.get('delta_PDMS'))+'<small>'+interval(pair.get('CI95_PDMS'))+'</small>',signed(ttc.get('delta_pp'))+'<small>'+interval(ttc.get('CI95_pp'))+'</small>',h.number(pair.get('zeros',{}).get('repaired'))+' / '+h.number(pair.get('zeros',{}).get('new_zero'))])
    if not rows:return ''
    body='<h3>FPO 1k为什么还没有总体胜出：固定高／低分队列分开看</h3><p class="notice"><b>同预算Safety是比较基线，IL70只负责预先固定队列：</b>NAVTEST原IL高分组改善约+0.321点，但低分组下降约−1.244点，抵消后整体为−0.120点、区间跨零，点估计未超过Safety。不能把高分保持改善单独当作全局提升，也不能把FPO相对IL70的绝对收益混成此处的同预算差值。</p>'
    body+=h.table(['人口／固定队列','场景数','FPO−Safety1k ΔPDMS［95%区间］','ΔTTC pp［区间］','zero修复／新增'],rows)
    body+='<p>TRAIN1024中高／低分队列均小幅提高，TTC差为0，主要体现EP变化；它是分层训练内诊断，不能替代NAVTEST泛化结论。下一步看既定5k完整结果，并单独检验时间覆盖上界；当前还没有时间缺口导致低分退化的因果证据。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/FPO_TIME_COVERAGE_DESIGN_20261031.json','完整队列区间、同预算与绝对IL基线分离')+'</p>'
    return body


def _fulltime_tables(trial,runs,h,link):
    if not trial.get('registration'):return ''
    run=runs.get(FULLTIME,{})
    state=_fresh_runtime_state(run,trial)
    body='<article class="callout"><h3>P2的时间覆盖对照：FPO sigma上界0.8 → 1.0</h3>'
    body+='<p><b>实际状态：</b>'+h.esc(state)+'；已观察更新 '+h.number(run.get('updates'))+'，'+h.number(run.get('journal_ranks'))+' rank；host '+h.esc(run.get('host','待记录'))+'。研究上限5k，1k／5k各完整NAVTEST＋TRAIN1024；5k明确结束，不自动延长。</p>'
    body+='<p>同IL70 fresh，保留原FPO entry、M=2、sigma_min=0.2、原W3、Safety信用、beta50、数据及LR；只把sigma_max从0.8改为1.0。主比较为旧FPO同预算，Safety为次级对照，均检查固定IL70高／低分队列与zero修复／新增。</p>'
    coverage=trial.get('time_coverage',{}).get('time_coverage',{})
    original=coverage.get('original_FPO_sigma',{});proposed=coverage.get('proposed_FPO_sigma',{})
    body+='<p class="muted">时间支持由10个部署ODE节点中的 '+h.number(coverage.get('original_support_nodes'))+' 个，扩至支撑闭包 '+h.number(coverage.get('new_support_closure_nodes'))+' 个；sigma=1只是边界，连续MC不保证抽中任何指定节点，仍不是完整论文Uniform[0,1]复现。共同区间密度下降25%，E[sigma⁻²] '+h.fmt(original.get('expected_inverse_sigma_squared'),2)+' → '+h.fmt(proposed.get('expected_inverse_sigma_squared'),2)+'，这是上界变动同时带来的权重／抽样分布后果，尚未证明退化因果。固定seed也不意味着变换后的sigma或后续训练参数逐值相同。</p>'
    review=trial.get('matched_review',{});contrasts=review.get('contrasts',{})
    complete=[(key,row)for key,row in contrasts.items()if isinstance(row,dict)and row.get('status')=='COMPLETE']if isinstance(contrasts,dict)else []
    signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
    interval=lambda values:'['+', '.join(signed(value)for value in values)+']'if isinstance(values,(list,tuple))and len(values)==2 else '区间待记录'
    def cohort(row,key):
        entry=row.get(key,{})
        return signed(entry.get('delta_PDMS'))+'<small>'+interval(entry.get('CI95_PDMS'))+'</small>'
    if complete:
        body+='<h3>时间覆盖对照已完整的NAVTEST配对</h3>'+h.table(['对照','完整目标PDMS','ΔPDMS［95%区间］','固定IL高分Δ［区间］','固定IL低分Δ［区间］','zero修复／新增'],[[h.esc(key),h.fmt(row.get('target_PDMS'),5),signed(row.get('delta_PDMS'))+'<small>'+interval(row.get('CI95_PDMS'))+'</small>',cohort(row,'fixed_IL70_high_ge09'),cohort(row,'fixed_IL70_low_lt09'),h.number(row.get('zeros',{}).get('repaired'))+' / '+h.number(row.get('zeros',{}).get('new_zero'))]for key,row in complete])
        body+='<details><summary>时间覆盖对照的安全／效率分量区间</summary>'+h.table(['对照','NC pp','DAC pp','TTC pp','EP pp'],[[h.esc(key),*[signed(row.get('components',{}).get(metric,{}).get('delta_pp'))+'<small>'+interval(row.get('components',{}).get(metric,{}).get('CI95_pp'))+'</small>'for metric in METRICS]]for key,row in complete])+'</details>'
    else:body+='<p class="notice">当前尚无此新变体的完整NAVTEST配对；child启动或optimizer推进均不构成提分证据，曲线不会补占位分数。</p>'
    body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/EXPERIMENT_FPO_FULLTIME_REGISTRATION_20261031.json','封存单因素注册')+' · '+link('/mnt/project/qwen-world-rl-optimization-20261030/owners/FPO_FULLTIME/STATE.json','独立有限owner状态')+' · '+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/FPO_TIME_COVERAGE_DESIGN_20261031.json','时间支持、密度与权重边界')+'</p>'
    if review:body+='<p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/FPO_FULLTIME_MATCHED_REVIEW_20261031.json','实际配对与TRAIN诊断')+'</p>'
    body+='</article>'
    return body


def _fresh_early_process_tables(process,h,link):
    if not str(process.get('status','')).startswith('COMPLETE'):return ''
    rows=[]
    for arm,node in process.get('arms',{}).items():
        probes=node.get('probes',{})
        positive=[(int(key),probe)for key,probe in probes.items()if str(key).isdigit()and int(key)>0 and isinstance(probe,dict)]
        if not positive:continue
        update,probe=max(positive,key=lambda item:item[0])
        scope=probe.get('scopes',{}).get('all',{})
        norms=scope.get('global_mean_gradient_norm',{});pg=norms.get('pg')
        if not h.finite(pg)or not pg:continue
        for component,norm in norms.items():
            if component=='pg'or not h.finite(norm):continue
            cosine=scope.get('cosine',{}).get(component+'__pg',scope.get('cosine',{}).get('pg__'+component))
            rows.append([h.esc(arm),h.number(update),h.esc(component),h.fmt(norm/pg,4),h.fmt(cosine,4)])
    body='<details><summary>新两路线的有界早期过程：辅助项与PG／anchor的实际尺度</summary>'
    if rows:body+=h.table(['路线','原生更新截点','梯度分量（已乘系数）','‖分量‖／‖PG‖','与PG的cos'],rows)
    notes=process.get('conclusions_zh',[])
    if isinstance(notes,list):body+=''.join('<p>'+h.esc(note)+'</p>'for note in notes[:6]if isinstance(note,str))
    body+='<p>这里只读真实早期窗口和零optimizer更新梯度探针；单点比例不说明系数最优，不代替完整NAVTEST提分证据。PG、辅助项与anchor可能冲突，不能用scalar loss比例替代梯度比例。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_FRESH_PAIR_PROCESS_EARLY_20261031.json','早期窗口、首collection材料与梯度原证据')+'</p></details>'
    return body


def _count(value,h):
    if h.finite(value):return value
    if isinstance(value,list):return len(value)
    if isinstance(value,dict):
        for key in ('count','tokens','scenes','n','groups'):
            node=value.get(key)
            if h.finite(node):return node
            if isinstance(node,list):return len(node)
    return None


def _train_zero_support_tables(audit,counterfactual,h,link):
    if audit.get('status')!='COMPLETE_CPU_EXISTING_DATA':return ''
    longitudinal=audit.get('longitudinal',{});exposure=audit.get('native_stream_exposure',{});alignment=audit.get('exact_alignment',{})
    body='<h3>TRAIN诊断的四类zero瓶颈：分别决定下一步</h3><p>固定TRAIN1024、162物理log的离散观测：曾zero '+h.number(longitudinal.get('ever_zero'))+'，全观测持续zero '+h.number(longitudinal.get('persistent'))+'，最新zero '+h.number(longitudinal.get('latest_zero'))+'。这是训练内诊断，不能与NAVTEST的133个持续zero混成一个人口；下列类别也不是互斥分桶。</p>'
    body+=h.table(['瓶颈／状态','已有实际证据','下一步与解释边界'],[
        ['尚未直接曝光',h.number(exposure.get('everzero_tokens_not_yet_exposed'))+'个everzero截至原生'+h.number(exposure.get('audited_updates'))+'更新仍未直接进入训练','先看曝光机会；缺候选是未知，不能填成allfail或A0。'],
        ['历史已有可行候选／正A',h.number(exposure.get('historical_candidate_groups_with_strict_feasible'))+'个已曝光场景有历史可行／正A见证','检查成功到部署的转移与保持；这些跨theta历史不能当该次部署zero的同时候选。'],
        ['实际allfail且成本同分',h.number(exposure.get('historical_allfail_cost_tie_PG0'))+'组229a历史候选全失败／costtie而PG0','条件性研究更细信用；不能据一个组称全局dense信号不足。'],
        ['修复后又复发',h.number(longitudinal.get('recurrent_nonpersistent'))+'个非持续重复zero；最新新增'+h.number(longitudinal.get('latest_new'))+'个为复发','跟踪恢复／复发转换，避免只把最新zero静态加入困难池。']])
    body+='<p class="notice"><b>同token与同theta是两道不同门槛：</b>严格对齐的native G8与门槛zero观测交集为 '+h.number(alignment.get('exact_native_training_G8_join_count'))+' / '+h.number(alignment.get('native_training_zero_evaluation_instances'))+'，不能把附近训练窗口冒充同时证据。独立10k固定theta诊断覆盖当时全部6个zero：SDE.1在3个场景发现可行候选、.03在2个、联合4个；这是诊断候选，不是已使用的训练梯度或部署修复。</p>'
    body+='<p>这12个everzero最早在5316更新才被直接训练；其中4个原IL zero已经在1k通过其他场景的学习被修复。因此未直接曝光不等于必然学不会，发现成功也不等于部署ODE已能稳定复现。'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_TRAIN_ZERO_SUPPORT_AUDIT_20261031.json','固定人口、实际曝光、严格对齐与逐scene证据')+'</p>'
    if counterfactual.get('status')=='COMPLETE_CPU_COUNTERFACTUAL_NOT_REGISTERED':
        coverage=counterfactual.get('actual_window_coverage',{})
        window=counterfactual.get('actual_window_counterfactual',{}).get('global64_normalization_effect',{})
        body+='<details><summary>DAC-only同分反事实：仅CPU、未注册／未部署，不是提分结果</summary><p>1280个既有窗口组中仅 '+h.number(coverage.get('explicit_DAC_only_tie_groups'))+' 个满足严格条件：8条均NC＝TTC＝1、DAC＝0且allfail costtie；另有单独229a定向组。后40时刻真实越界占比可生成A_tie＝组内均值−占比，但候选canonical PDMS仍全部为0，不是安全teacher，也没有新forward／评分／梯度／optimizer更新。</p><p>229a的越界占比为17.5%–87.5%；raw progress与未提取的边界距离均未进入信用。即使其他raw A不动，重算global64 std仍会共同缩放其他normalized A：窗口约×'+h.fmt(window.get('unchanged_raw_normalized_scale'),6)+'，229a约×0.965256；不能声称对其他组完全无影响。这是低覆盖、有条件的候选提案，不与正在训练的质量q混同。</p><p>'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_DAC_TIE_COUNTERFACTUAL_20261031.json','实际字段、覆盖分母与缩放反事实')+'</p></details>'
    return body


def _diagnosis_tables(root,diagnosis,h,link):
    esc=h.esc;fmt=h.fmt;number=h.number;table=h.table
    signed=lambda value:f'{value:+.4f}'if h.finite(value)else '待记录'
    interval=lambda value:'['+', '.join(signed(x)for x in value)+']'if isinstance(value,(list,tuple))and len(value)==2 else '区间待记录'
    longitudinal=diagnosis.get('zero_longitudinal',{});points=diagnosis.get('evaluation_points',[])
    body='<h3>zero的固定核心与换入换出</h3>'+table(['统计口径','场景数'],[
        ['全部已观测Safety checkpoint均为zero',number(_count(longitudinal.get('all_observations_persistent'),h))],
        ['至少一次zero（并集）',number(_count(longitudinal.get('ever_zero'),h))],
        ['间歇zero',number(_count(longitudinal.get('intermittent'),h))],
        ['修复后又出现zero的场景',number(_count(longitudinal.get('relapsed_scenes'),h))],
        ['最近完整checkpoint的zero',number(_count(longitudinal.get('latest_zero'),h))]])
    post10=longitudinal.get('post10k',{})
    body+='<p class="muted">观测更新点：'+esc('、'.join(str(x)for x in longitudinal.get('observed_updates',[])))+'；持续仅指这一观测序列，不补推未评测阶段或整个TRAIN人口。诊断生成 '+esc(diagnosis.get('created_at','见原证据时间'))+'。</p>'
    if post10:body+='<p class="notice"><b>对10k的净变化不能替代转换：</b>最近完整点修复 '+number(post10.get('latest_repaired_relative10k'))+' 个原zero，同时新增 '+number(post10.get('latest_new_zero_relative10k'))+' 个zero。需要同时看修复与能力丢失，而不是只把持续zero重新排成困难名单；NAVTEST场景名单不回流训练。</p>'
    transitions=longitudinal.get('transitions',[])
    body+=table(['相邻完整点','当前zero','与前一点共同zero','新zero','修复zero'],[[number(row.get('previous_updates'))+' → '+number(row.get('updates')),number(row.get('zero')),number(row.get('persistent_from_previous')),number(row.get('new_zero')),number(row.get('repaired_zero'))]for row in transitions])
    if points:
        def draw(plt):
            fig,axes=plt.subplots(1,2,figsize=(13.5,3.8),layout='constrained')
            valid=[row for row in points if h.finite(row.get('zeros',{}).get('zeros'))]
            if valid:axes[0].plot([row['updates']/1000 for row in valid],[row['zeros']['zeros']for row in valid],'-o',color='#dc2626',lw=2,ms=4)
            axes[0].set(xlabel='Global optimizer updates (thousands)',ylabel='Canonical zero scenes / 12,146')
            if transitions:
                positions=list(range(len(transitions)));width=.36
                axes[1].bar([p-width/2 for p in positions],[row.get('new_zero',0)for row in transitions],width,color='#dc2626',label='New zero')
                axes[1].bar([p+width/2 for p in positions],[row.get('repaired_zero',0)for row in transitions],width,color='#059669',label='Repaired')
                axes[1].set_xticks(positions,[str(row['updates']//1000)+'k'for row in transitions]);axes[1].legend(frameon=False)
            axes[1].set(xlabel='Target checkpoint vs previous complete checkpoint',ylabel='Scenes')
            for ax in axes:ax.grid(axis='y',alpha=.15)
            return fig
        body+=figure(root,h,'safety_focus_zeros','zero总量与相邻修复／新增：分开看',draw,'左图为各完整checkpoint的部署zero场景数；右图是相邻完整点转换，不是全序列持续zero。新增与修复可以同时发生，不能只看净数量。')
    high=diagnosis.get('high10k_retention',[])
    body+='<h3>固定Safety10k高分子集：是否保住已有能力</h3>'+table(['目标更新','固定子集人口','子集目标PDMS','相对10k：Δ与95%区间','高分变零','下降≥10pp／提升≥10pp'],[[number(row.get('updates')),number(row.get('cohort_size')),fmt(row.get('target_PDMS'),5),signed(row.get('delta_PDMS'))+'<small>'+interval(row.get('CI95_PDMS'))+'</small>',number(row.get('high_to_zero')),number(row.get('drops_ge10pp'))+' / '+number(row.get('gains_ge10pp'))]for row in high])
    if high:
        high_latest=max(high,key=lambda row:row['updates'])
        body+='<p class="notice">固定队列定义：'+esc(high_latest.get('cohort_definition','见证据中的固定10k阈值'))+'，'+number(high_latest.get('cohort_size'))+' 场景；'+number(high_latest['updates'])+'相对子集自身10k '+signed(high_latest.get('delta_PDMS'))+'，区间 '+interval(high_latest.get('CI95_PDMS'))+'，其中 '+number(high_latest.get('high_to_zero'))+' 个变为zero。它是条件选出的保持诊断，向上空间受限，不是整体性能的无偏估计；不能把该子集区间替代总体区间。</p>'
        high_components=high_latest.get('components',{})
        if all(h.finite(high_components.get(key,{}).get('delta_pp'))and high_components[key]['delta_pp']<0 for key in METRICS):body+='<p>该固定高分子集的NC／DAC／TTC／EP点估计均下降；总体TTC改善与该子集安全及效率同时受损可以并存。</p>'
    cohort_accounting=diagnosis.get('cohort_gain_loss_accounting',[])
    if cohort_accounting:
        last_accounting=max(cohort_accounting,key=lambda row:row['updates']);cohorts=last_accounting.get('cohorts',{})
        labels={'fixed10k_zero':'10k原zero','fixed10k_positive_below09':'10k正低分(<0.9)','fixed10k_high_ge09':'10k高分(≥0.9)'}
        body+='<h3>为什么修复了旧zero，总分却仍未更高</h3>'+table(['固定10k分组','场景数','组内PDMS变化','加权到全局的PDMS贡献'],[[labels.get(key,key),number(row.get('scenes')),signed(row.get('delta_PDMS_within_cohort')),signed(row.get('contribution_to_global_PDMS_delta'))]for key,row in cohorts.items()])
        body+='<p>'+number(last_accounting['updates'])+'对10k：原zero修复贡献 '+signed(cohorts.get('fixed10k_zero',{}).get('contribution_to_global_PDMS_delta'))+' 点，却被原高分群体 '+signed(cohorts.get('fixed10k_high_ge09',{}).get('contribution_to_global_PDMS_delta'))+' 点抵消；全局合计 '+signed(last_accounting.get('total_PDMS_delta'))+'。这是历史peak条件分组的描述性归账，不直接证明参数遗忘、anchor致因或新算法胜负。</p>'
    attribution=diagnosis.get('component_attribution',[])
    def metric_delta(row,key):
        value=row.get('component_metric_deltas',{}).get(key)
        return signed(value.get('delta_pp'))+'<small>'+interval(value.get('CI95_pp'))+'</small>'if isinstance(value,dict)else signed(value)
    body+='<h3>对10k的完整配对：总分与安全／效率分量</h3>'+table(['目标更新','ΔPDMS［物理log 95%区间］','ΔNC pp［区间］','ΔDAC pp［区间］','ΔTTC pp［区间］','ΔEP pp［区间］'],[[number(row.get('updates')),signed(row.get('vs10k_PDMS_delta'))+'<small>'+interval(row.get('vs10k_CI95'))+'</small>',*[metric_delta(row,key)for key in METRICS]]for row in attribution])
    body+='<p class="muted">区间按同44物理log配对bootstrap；固定训练seed，不包含跨训练seed不确定性。多阶段／多分量属于探索性比较，未作多重比较校正。</p>'
    if attribution:
        last_attribution=max(attribution,key=lambda row:row['updates']);last_metrics=last_attribution.get('component_metric_deltas',{})
        body+='<p class="notice">'+number(last_attribution['updates'])+'的总体TTC变化 '+signed(last_metrics.get('time_to_collision_within_bound',{}).get('delta_pp'))+' pp，而EP变化 '+signed(last_metrics.get('ego_progress',{}).get('delta_pp'))+' pp。总体变化与固定高分群体受损可以并存；不能概括为“只追EP”或“所有安全维度都在下降”。</p>'
    if attribution:
        accounting=table(['目标更新','NC','DAC','TTC','EP','Comfort','总变化：PDMS点'],[[number(row.get('updates')),*[signed(row.get('score_accounting',{}).get('contributions',{}).get(key))for key in ('NC','DAC','TTC','EP','Comfort')],signed(row.get('score_accounting',{}).get('total_delta'))]for row in attribution])
        body+='<details><summary>评分公式贡献拆解：描述性归账，不作因果归因</summary>'+accounting+'</details>'
    body+='<p>'+link(root/'analysis'/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','完整集合、配对区间与固定分母证据')+'</p>'
    return body


def _rollout_tables(diagnosis,h,link):
    esc=h.esc;table=h.table;number=h.number;windows=diagnosis.get('rollout_windows',[])
    aggregate=diagnosis.get('rollout_aggregate',{});categories=aggregate.get('categories',{})
    body='<h3>已保存TRAIN窗口：有没有产生可学习的候选差异</h3>'
    if aggregate:body+='<p class="notice"><b>PG=0主要是三安全／EP1信用饱和：</b>'+number(aggregate.get('groups'))+' 个有界TRAIN组中PG=0为 '+number(aggregate.get('PG_zero_groups'))+' 组，其中成功EP1饱和 '+number(aggregate.get('successful_credit_saturation_groups'))+' 组、全fail成本tie '+number(categories.get('all_fail_cost_tie',{}).get('groups'))+' 组；EP1饱和不保证Comfort通过或canonical PDMS满分。mixed组中unsafe正A '+number(aggregate.get('positive_unsafe',{}).get('mixed_group_candidates'))+' 条；allfail相对成本排序有 '+number(aggregate.get('positive_unsafe',{}).get('all_fail_group_candidates'))+' 条正A，不应混为同一问题。</p>'
    body+=table(['全局更新窗口','真实G8 groups','候选数','PG=0组','正A unsafe候选／mixed中','mixed中可行候选有负A：组数'],[[number(row.get('updates')),number(row.get('summary',{}).get('groups')),number(row.get('summary',{}).get('candidates')),number(row.get('summary',{}).get('PG_zero_groups')),number(row.get('summary',{}).get('positive_unsafe',{}).get('candidates'))+' / '+number(row.get('summary',{}).get('positive_unsafe',{}).get('mixed_group_candidates')),number(row.get('summary',{}).get('negative_feasible',{}).get('mixed_groups'))]for row in windows])
    body+='<p>每个窗口仅16collections×8rank＝128个TRAIN组，四inner复用不重复计为新候选；不是全程或全TRAIN分布。PG=0可能是指标饱和成功，也可能是失败同分。全失败时较小deficit获正信用属于成本fallback，不应误称“有安全替代仍奖励冒险”。</p>'
    if aggregate:
        mixed=categories.get('mixed_feasible_failed',{}).get('groups');negative_mixed=aggregate.get('negative_feasible',{}).get('mixed_groups')
        body+='<p><b>更值得检验的信用压力：</b>'+number(mixed)+' 个mixed组中，'+number(negative_mixed)+' 组有通过三安全指标的候选因EP相对项获得负A，发生在可行候选数为6／7的组；少数不可行替代带来的安全margin较小，效率差可覆盖它。通过三个指标不等于客观最安全；这支持检验mixed组的EP压力与保持能力，不支持把所有负A或必要等待都判为错误。</p>'
    sampler=diagnosis.get('same_theta_sampler_evidence',{})
    if sampler.get('status')=='REUSED_COMPLETE_SELECTED_TRAIN15':body+='<p class="notice"><b>探索仍有两面：</b>Safety10k同θ、选定15个TRAIN场景中，'+number(sampler.get('canonical_B1_zero'))+' 个部署B1 zero里有 '+number(sampler.get('canonical_B1_zero_SDE8_has_feasible'))+' 个在SDE8出现可行候选；同120根中，SDE损坏 '+number(sampler.get('matched_roots_ODE_feasible_SDE_failed'))+' 个ODE可行候选，同时修复 '+number(sampler.get('matched_roots_SDE_feasible_ODE_failed'))+' 个。局部探索中奖与大量噪声损害并存，不能全关探索，也不能把best-of8当部署。'+link(sampler.get('source'),'同θ受控证据')+'</p>'
    body+='<p class="muted">这些native窗口没有同θ、同场景的ODE配对，sampler与部署差异保留UNKNOWN；上面的同θ结论只来自独立selected15探针。日志只有联合梯度与scalar loss，不能据anchor loss较大推断anchor梯度主导。</p>'
    rows=[]
    for window in windows:
        for category,node in window.get('summary',{}).get('categories',{}).items():
            rows.append([number(window.get('updates')),esc(category),number(node.get('groups'))if isinstance(node,dict)else number(node),h.fmt(node.get('fraction'),2,percent=True)+'%'if isinstance(node,dict)and h.finite(node.get('fraction'))else '待记录'])
    if rows:body+='<details><summary>窗口分型与分母：成功饱和、全失败、差异候选分别看</summary>'+table(['全局更新','分型（原定义）','groups','该窗口比例'],rows)+'</details>'
    return body


def _direction_tables(directions,h,link,gradient_measured=False,fresh_pair=None,runs=None):
    esc=h.esc;body='';diagnostic=directions.get('P0_gradient_and_credit_diagnosis',{})
    if diagnostic and not gradient_measured:
        body+='<article class="callout"><h3>P0：先区分PG、物理anchor与真实参数更新</h3><p>'+esc(diagnostic.get('problem_zh',''))+'</p><p>'+esc(diagnostic.get('unknown_zh',''))+'</p><p>要测真实8rank归约后的PG／anchor梯度范数、夹角及Adam实际步长，分别看action与memory；scalar loss和Adam二阶矩不能代替这一步。当前状态 '+esc(diagnostic.get('status','尚未实测'))+'，不是已发现anchor主导故障。</p></article>'
        fact=diagnostic.get('CPU_fact',{})
        if fact:body+='<p class="muted">独立近10k窗口：'+h.number(fact.get('groups'))+' 组中 '+h.number(fact.get('zeroPG_groups'))+' 组local PG=0，'+h.number(fact.get('zeroPG_saturated'))+' 组为成功饱和；'+h.number(fact.get('global8rank_collections'))+' 个完整8rank collection里全rank A0为 '+h.number(fact.get('global_collections_with_all_rank_A0'))+'。这与跨checkpoint的1280组是不同窗口，不混分母，也不把local PG0称为全局optimizer停摆。</p>'
    cards=[]
    for row in sorted(directions.get('directions',[]),key=lambda item:item.get('priority',99)):
        risks=row.get('risks_zh',[]);risk='；'.join(risks)if isinstance(risks,list)else str(risks)
        source_links=' · '.join(link(url,'primary '+str(i+1))for i,url in enumerate(row.get('sourceURLs',[])[:3]))
        text='<article class="callout"><h3>'+esc(str(row.get('priority','')))+' · '+esc(row.get('title_zh',row.get('title',row.get('name','方向'))))+'</h3>'
        text+='<p><b>问题：</b>'+esc(row.get('problem_zh',row.get('problem','')))+'</p><p><b>假设：</b>'+esc(row.get('hypothesis_zh',row.get('hypothesis','')))+'</p>'
        text+='<p><b>只改什么：</b>'+esc(row.get('single_core_change',row.get('core_change','')))+'</p>'
        if row.get('trigger_zh'):text+='<p><b>触发条件：</b>'+esc(row['trigger_zh'])+'</p>'
        arm={'VERIFIED_SUCCESS_MEMORY':'PPO_SAFETY_SUCCESS_MEMORY','FLOW_ENDPOINT_RATIO':'PPO_SAFETY_FPO'}.get(row.get('id'))
        status=_fresh_runtime_state((runs or {}).get(arm,{}),fresh_pair or {})if arm else '条件研究提案；尚未注册或部署'
        text+='<p class="muted"><b>风险与限制：</b>'+esc(risk)+'</p><small>'+esc(status)+'</small><p>'+source_links+'</p></article>'
        cards.append(text)
    body+='<div class="two">'+''.join(cards)+'</div>'
    deferred=directions.get('deferred_single_factor_controls')
    if deferred:body+='<details><summary>暂缓的独立变量：不把所有组件同时组合</summary><pre>'+esc(json.dumps(deferred,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<p>原提案文档保留历史版本；P1／P2当前状态以上述新注册和实际运行证据为准，1k／5k／10k复核、10k研究上限覆盖旧提案预算。P3／P4仍为条件提案；所有路线均不代表已证明涨分。方法受primary文献启发，不能默认迁移其trick、概率解释或安全保证。'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json','问题驱动的freshIL设计与完整来源')+'</p>'
    return body
