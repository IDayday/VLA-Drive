"""SafetyEP-focused read-only presentation; historical trials stay in an appendix."""
from __future__ import annotations

import json

from research_report import figure, update

SAFETY='PPO_SAFETY_EP'
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
    decisions=editorial.get('decisions',{});stopped=editorial.get('stopped_receipts',{})
    runs={row['id']:row for row in snapshot['runs']};run=runs.get(SAFETY,{})
    all_test=[row for row in snapshot['results']if row.get('publishable_complete')and row.get('cohort')=='NAVTEST'and row.get('scenes')==12146]
    safety=sorted((row for row in all_test if row.get('arm')==SAFETY),key=update)
    by_update={update(row):row for row in safety}
    best=max(safety,key=lambda row:row['PDMS'],default={});latest=safety[-1]if safety else {}
    base=snapshot.get('ppo_full_first_stage_paired_analysis',{}).get('reference_PDMS',88.79290990699972)
    peak10=by_update.get(10000,{})
    chapters=[]
    def section(key,title,deck,body):chapters.append('<section id="'+key+'"><h2>'+title+'</h2><p class="deck">'+deck+'</p>'+body+'</section>')
    primary_state=('原生训练中'if run.get('worker_receipt_status')=='RUNNING'else '本次原生阶段已完成'if run.get('worker_returncode')==0 else str(run.get('status','待状态证据')))
    cards=[('共同IL70基线',fmt(base,5),'未来方案统一从该IL身份fresh启动'),
           ('Safety已测最高',fmt(best.get('PDMS'),5),number(update(best))+' updates · 完整NAVTEST'),
           ('最近完整Safety',fmt(latest.get('PDMS'),5),number(update(latest))+' updates；不是当前训练候选分'),
           ('当前原生更新',number(run.get('updates')),primary_state+' · '+number(run.get('journal_ranks'))+' rank日志')]
    body='<div class="notice"><b>当前只保留SafetyEP为RL主线。</b>把重点收回到它的学习过程、失败机制与少量可检验改进；其他路线保留历史成绩与停止原因。IL独立研究继续，不受本次RL收缩影响。</div>'
    body+='<div class="cards">'+''.join('<article><span>'+esc(a)+'</span><strong>'+b+'</strong><small>'+esc(c)+'</small></article>'for a,b,c in cards)+'</div>'
    if latest and peak10:
        body+='<p><b>当前核心事实：</b>10k为 '+fmt(peak10['PDMS'],5)+'；最近完整 '+number(update(latest))+' 为 '+fmt(latest['PDMS'],5)+'，相对10k '+signed(latest['PDMS']-peak10['PDMS'],5)+' 点。'+('后续完整点尚未刷新10k。'if best and update(best)==10000 else '当前最高点以本页实际完整结果为准。')+'训练更久并不保证持续提升；曲线回摆本身不能证明学习率、数据或表征中的任何一个是根因。</p>'
    if diagnosis.get('status')=='COMPLETE'and diagnosis.get('component_attribution'):
        paired_latest=max(diagnosis['component_attribution'],key=lambda row:row['updates']);paired_ci=paired_latest.get('vs10k_CI95',[])
        paired_verdict='区间跨零，不能称总体显著退化'if len(paired_ci)==2 and paired_ci[0]<=0<=paired_ci[1]else '区间低于零'if len(paired_ci)==2 and paired_ci[1]<0 else '区间高于零'if len(paired_ci)==2 and paired_ci[0]>0 else '配对区间待核'
        body+='<p class="notice"><b>完整配对的关键区分：</b>'+number(paired_latest['updates'])+'对10k总分 '+signed(paired_latest.get('vs10k_PDMS_delta'),5)+'，95%区间 '+ci(paired_ci,5)+'，'+paired_verdict+'。10k为已观测峰值，这属于阶段诊断，不是预注册的独立显著性胜负检验。固定10k高分子集与总体分开判断；zero净减少也可能伴随新的高分失败。</p>'
    body+='<div class="two"><article class="callout"><h3>已确认与仍未知</h3><p>SafetyEP的条件信用修正有阶段总分收益，但并未自动解决部署安全风险、零分场景与高分能力保持。需要把部署ODE、训练G8候选、相对优势与真实optimizer更新分开。</p></article><article class="callout"><h3>下一轮比较方式</h3><p>提出少量方向后再决定实验；未来统一同IL基线、fresh AdamW／RNG42／update0。旧Safety10k原生分叉只作历史，不能冒充这一统一fresh设计。</p></article></div>'
    closed_lr=[arm for arm in LR_ARMS if arm in focus_stopped]
    if len(closed_lr)==4:body+='<p class="notice">四条LR／数据分叉已按用户调整研究焦点协作退出，保留checkpoint与实验资产；这是研究范围调整，不是据新成绩判定四种算法失败。历史方法及各自已测结果见折叠档案。</p>'
    else:body+='<p class="muted">四条LR／数据分叉按用户要求退出；已取得实际退出收据 '+number(len(closed_lr))+' / 4。缺少收据者仍标为待退出，不将停止请求当作已经退出。</p>'
    section('focus','SafetyEP：现在知道什么，下一步研究什么','本页只聚焦保留主线；阶段收益、过程解释与未来假设分别呈现。',body)

    def score_plot(plt):
        fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
        if safety:
            ax.plot([update(row)/1000 for row in safety],[row['PDMS']for row in safety],'-o',color='#059669',lw=2,ms=4)
            ax.scatter([update(best)/1000],[best['PDMS']],color='#0f766e',s=60,zorder=3)
        ax.set(xlabel='Global optimizer updates (thousands)',ylabel='Complete NAVTEST PDMS')
        ax.grid(axis='y',alpha=.18)
        return fig
    score_figure=figure(root,h,'safety_focus_score','Safety完整NAVTEST：峰值与后期回摆',score_plot,'只连接已完整12146场景的单候选ODE结果。纵轴放大看阶段变化；IL70基线见首卡，未完成阶段不补点。')
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
    section('trajectory','完整轨迹：分数与安全—效率分量','先把真实部署变化看清，再检验关于优化、探索和信用分配的解释。',body)

    body='<p>本节只采用已完成的Safety过程诊断；持续zero、曾经zero与最新zero是不同集合。高分保持也要固定10k场景子集，不能随checkpoint改分母。</p>'
    if diagnosis.get('status')=='COMPLETE':body+=_diagnosis_tables(root,diagnosis,h,link)
    else:body+='<p class="muted">完整纵向zero／10k高分保持／分量配对诊断正在生成，未完成前不沿用旧四观测的persistent数量冒充最新全序列。'+link(root/'analysis'/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','Safety过程诊断')+'</p>'
    section('failures','哪些场景修复了，哪些能力丢失了','部署zero不等于训练全失败，也不等于没有正优势；先确定集合与可用学习信号。',body)

    body='<p><b>当前机制：</b>安全／可行EP分量优势按候选指标分配信用，保留原W3随机探索与IL70参考；这不是具备路权、遮挡或交通交互推理的证明。低EP可能是必要等待，只有同场景确有安全可行替代时才谈错过通行机会。</p>'
    body+=table(['候选信用机制','当前实际含义'],[
        ['有可行候选的mixed组','f为NC＝DAC＝TTC＝1，m为8条中的可行数；安全项f−m/8，EP仅在可行子集内中心化。unsafe不会因EP获得正A。'],
        ['整个组无可行候选','按mean(c)−c保留较小NC／DAC／TTC deficit的相对方向；deficit不是事故严重程度，正A也不等于该候选安全。'],
        ['组间尺度与保护项','64条fresh候选的raw A统一population std，之后cap5；四inner复用同A。beta50物理均值anchor指向原IL70，不是旧策略KL，不能仅从scalar loss认定拉回过强。']])
    body+='<p>Qwen特征冻结／缓存；scene与predictive memory以及动作attention LoRA实际参与优化，约52.7M参数，不是只训练动作头。可训练不等于已学到充分风险信息；当前缺少的驾驶知识应反馈IL，RL继续负责在该知识基础上强化轨迹。</p>'
    if diagnosis.get('status')=='COMPLETE':body+=_rollout_tables(diagnosis,h,link)
    body+='<p class="notice">旧W2、低噪声与Rescue组合的完整负结果说明：候选分更高、teacher确实给全零组提供方向，都不保证部署更好。它们不证明所有探索或数据策略无效，也不应成为继续叠加组件的理由。</p>'
    section('learning','学习过程中还缺哪一环','把信息、候选探索、信用、优化与部署五个环节分开，避免单一总分解释全部现象。',body)

    body='<div class="notice"><b>统一未来比较：</b>固定同一IL70 checkpoint及action／scene／predictive初始化，fresh AdamW、fresh RNG42、update0，参考与数据人口先固定；每次只验证一个主要假设。所有分支用相同预算与完整canonical评测，比较安全／效率、zero修复／新生和固定高分保持。当前没有因此自动启动新实验。</div>'
    body+='<p>未来高分主队列预先固定为IL70 score≥0.9；历史Safety10k高分队列只作次级保持诊断。TRAIN的zero／正低分队列同样从IL起点预定义，不从NAVTEST名单选训练场景。既有fresh-Safety同预算结果仅在身份／数据／推理协议一致时复用；旧RL10k分叉不能充当fresh起点。候选先提议1k／5k复核，是否延长另按证据决定；optimizer步数与额外forward／GPU成本分别记录。</p>'
    if directions.get('status'):body+=_direction_tables(directions,h,link)
    else:body+='<p class="muted">基于Safety过程与primary文献的3–4个方向正在整理；未将候选方案写成已注册或已运行。</p>'
    section('next','只提出少量下一步：统一IL基线fresh验证','先讲清问题、机制、单变量和失败判据，再分配新的训练预算。',body)

    historical=[]
    for arm,other in sorted(runs.items()):
        if arm==SAFETY:continue
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
    body+=detail('人口、身份与解释边界','<p>共同原始IL70父模型；NAVTRAIN103,288包括旧DEV1,696，旧DEV属于训练内诊断。NAVTEST与TRAIN隔离。NAVTEST已用于研究选择，相关名单不回流训练。只有COMPLETE、完整人口、invalid0且SUMMARY seal正确才发布分数；partial不发布均分。配对区间按物理log，固定seed结果不代表训练seed不确定性。</p><p>'+link(root/'IL_PARENT','固定IL父模型')+' · '+link(root/'analysis'/'REPRESENTATION_TRAINABILITY_AUDIT_20261030.json','表征可训练性审计')+' · '+link(root/'analysis'/'RL_LR_NATIVE_AUDIT_20261031.json','原生LR审计')+' · '+link(root/'analysis'/'RL_LR_PRIMARY_RESEARCH_20261031.json','primary LR文献')+'</p>')
    body+=detail('完整分析与历史研究归档','<p>'+link(root/'analysis'/'SAFETY_EP_PROCESS_DIAGNOSIS_20261031.json','Safety过程诊断')+' · '+link(root/'analysis'/'SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json','少量freshIL改进方向')+' · '+link(root/'analysis'/'COMPLETED_5K_PILOT_REVIEWS_20261031.json','六配方完成复核')+' · '+link(root/'owners'/'CAMPAIGN'/'STATE.json','原owner执行历史')+'</p>')
    body+='<p class="muted">'+link(root/'SAFETY_EP_FOCUSED_RESEARCH_20261031.json','当前Safety-only用户研究范围')+' · '+link(root/'owners'/'SAFETY_EP_FOCUS'/'STOPPED_SUMMARY.json','四分叉实际退出，pending0')+'</p>'
    watcher=snapshot.get('report_watcher',{})
    if watcher.get('status')=='RUNNING'and watcher.get('last_render',{}).get('status')=='RENDERED':body+='<p class="muted">自动报告owner '+number(watcher.get('pid'))+' 正常；上次真实watcher成功渲染 '+esc(watcher['last_render'].get('generated_utc'))+'，常规1800秒，下次 '+esc(watcher.get('next_regular_utc'))+'。'+('OWNER中旧error文本属于已恢复的历史证据，当前健康以RUNNING和最新成功记录判断。'if watcher.get('error')else '')+link(root/'reporting'/'WATCHER_HEALTH.json','自动更新健康与历史错误分类')+'</p>'
    if snapshot.get('warnings'):body+='<p class="warning">'+esc('；'.join(snapshot['warnings']))+'</p>'
    section('archive','历史方法与原始证据','保留已有工作与结果，首页不再被历史路线淹没。',body)
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172b43;font:15px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}header{background:#123248;color:#fff;padding:36px max(22px,calc((100vw - 1300px)/2));border-bottom:4px solid #10b981}h1{font-size:30px;margin:8px 0}header p{color:#c3d8e5;max-width:1000px}nav{position:sticky;top:0;background:#fffffff5;z-index:2;border-bottom:1px solid #dce6ef;display:flex;gap:25px;padding:12px max(22px,calc((100vw - 1300px)/2));overflow:auto;white-space:nowrap}nav a{font-size:13px;color:#36596d}main{max-width:1344px;margin:auto;padding:24px 22px 40px}section{background:#fff;border:1px solid #dce6ef;border-radius:13px;padding:28px 30px;margin-bottom:25px;scroll-margin-top:64px}h2{font-size:23px;margin:0}h3{font-size:17px;margin:18px 0 8px}.deck,.muted{color:#64768a;font-size:13px}.deck{margin:4px 0 20px}p{margin:11px 0}.notice,.callout{background:#eef8f4;border-left:3px solid #10b981;padding:14px 18px;border-radius:8px}.warning{background:#fff1e9;padding:14px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}.cards article{background:#f3f7fb;border:1px solid #e1e9f0;border-radius:10px;padding:17px}.cards span,small{display:block;color:#65788d;font-size:12px}.cards strong{display:block;font-size:28px;margin:5px 0;color:#123e51}.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;margin:20px 0;align-items:start}.callout{background:#f6f8fb;border-left:0}.callout h3{margin-top:0}figure{margin:0;background:#fbfcfe;border:1px solid #e2e9f0;border-radius:10px;padding:15px;min-width:0}figcaption{font-size:14px;font-weight:600}figure svg{width:100%;height:auto}.figure-note,.downloads{font-size:12px;color:#64768a}.table-wrap{overflow:auto;border:1px solid #e2e9f0;border-radius:10px;margin:17px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #e7edf3;overflow-wrap:anywhere}th{font-size:12px;background:#f0f5f9;color:#426075;white-space:nowrap}tr:last-child td{border-bottom:0}details{border:1px solid #dce6ef;border-radius:10px;padding:15px 18px;margin:13px 0}summary{cursor:pointer;font-size:14px;font-weight:600;color:#35566c}details[open] summary{margin-bottom:17px}a{color:#1979aa;text-decoration:none}a:hover{text-decoration:underline}.protocol,footer{color:#64768a;font-size:12px}footer{text-align:center;padding:10px 20px 30px}pre{font-size:11px;white-space:pre-wrap;background:#f5f7fa;padding:14px;max-height:330px;overflow:auto}@media(max-width:950px){.two{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}section{padding:22px}h1{font-size:26px}}@media(max-width:550px){main{padding:15px 9px}section{padding:18px 14px}.cards{gap:9px}.cards article{padding:12px}.cards strong{font-size:23px}}'''
    nav=[('focus','当前结论'),('trajectory','完整曲线'),('failures','zero与保持'),('learning','学习过程'),('next','少量下一步'),('archive','历史与证据')]
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>SafetyEP RL研究报告</title><style>'+css+'</style></head><body><header><small>QWEN-WORLD · SAFETY-EP</small><h1>聚焦SafetyEP：理解学习过程，再做少量改进</h1><p>保留一条主线，检查安全与驾驶效率、零分修复和高分保持；未来候选统一同IL基线fresh启动。</p><small>报告生成 '+esc(snapshot['generated_utc'])+' · 常规30分钟＋完整阶段事件更新</small></header><nav>'+''.join('<a href="#'+key+'">'+title+'</a>'for key,title in nav)+'</nav><main>'+''.join(chapters)+'</main><footer>固定路径 '+esc(root/'report.html')+' · 本报告只读汇总，不控制训练、评测或资源</footer></body></html>'


def _count(value,h):
    if h.finite(value):return value
    if isinstance(value,list):return len(value)
    if isinstance(value,dict):
        for key in ('count','tokens','scenes','n','groups'):
            node=value.get(key)
            if h.finite(node):return node
            if isinstance(node,list):return len(node)
    return None


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
    if aggregate:body+='<p class="notice"><b>PG=0主要是成功饱和：</b>'+number(aggregate.get('groups'))+' 个有界TRAIN组中PG=0为 '+number(aggregate.get('PG_zero_groups'))+' 组，其中成功EP1饱和 '+number(aggregate.get('successful_credit_saturation_groups'))+' 组、全fail成本tie '+number(categories.get('all_fail_cost_tie',{}).get('groups'))+' 组。mixed组中unsafe正A '+number(aggregate.get('positive_unsafe',{}).get('mixed_group_candidates'))+' 条；allfail相对成本排序有 '+number(aggregate.get('positive_unsafe',{}).get('all_fail_group_candidates'))+' 条正A，不应混为同一问题。</p>'
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


def _direction_tables(directions,h,link):
    esc=h.esc;body='';diagnostic=directions.get('P0_gradient_and_credit_diagnosis',{})
    if diagnostic:
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
        text+='<p class="muted"><b>风险与限制：</b>'+esc(risk)+'</p><small>'+esc(row.get('status','提案'))+'</small><p>'+source_links+'</p></article>'
        cards.append(text)
    body+='<div class="two">'+''.join(cards)+'</div>'
    deferred=directions.get('deferred_single_factor_controls')
    if deferred:body+='<details><summary>暂缓的独立变量：不把所有组件同时组合</summary><pre>'+esc(json.dumps(deferred,ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<p>方案状态 '+esc(directions.get('status','待完整研究建议'))+'：提案不是已注册实验，也不是已证明能涨分。方法受primary文献启发，不能默认迁移其trick、概率解释或安全保证。'+link('/mnt/project/qwen-world-rl-optimization-20261030/analysis/SAFETY_EP_FOCUSED_DIRECTIONS_20261031.json','问题驱动的freshIL设计与完整来源')+'</p>'
    return body
