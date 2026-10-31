"""Editorial research report; evidence collection stays in render_report.py."""
from __future__ import annotations

from collections import defaultdict
import io
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime

ARMS=['PPO_FULL','PPO_BATCHSTD','PPO_SAFE_COMFORT','CISPO_FULL','PPO_HARD','PPO_ADAPTIVE','PPO_ANCHOR40','NFT_FULL']
PPO_ARMS=ARMS[:5]
ALGO_ARMS=ARMS[:4]
COLORS=dict(zip(ARMS,['#2563eb','#059669','#7c3aed','#d97706','#64748b','#0891b2','#be185d','#dc2626']))
SHORT={'PPO_FULL':'Full','PPO_BATCHSTD':'BatchStd','PPO_SAFE_COMFORT':'SafeComfort','CISPO_FULL':'CISPO','PPO_HARD':'Hard','PPO_ADAPTIVE':'Adaptive','PPO_ANCHOR40':'Anchor40','NFT_FULL':'NFT'}
IDEAS={'PPO_FULL':'全量103,288；原始PPO对照','PPO_BATCHSTD':'仅改中心化优势的全局std分母','PPO_SAFE_COMFORT':'安全候选Comfort置1，局部突出EP','CISPO_FULL':'全量数据；仅改策略目标','PPO_HARD':'困难15,171；数据人口对照','PPO_ADAPTIVE':'训练内低分／退化／覆盖再采样','PPO_ANCHOR40':'仅将均值anchor权重β50改为40','NFT_FULL':'纯ODE采样＋NFT目标／EMA'}
IDEAS.update(PPO_MEMORYLR20='仅memory学习率5e−6→2e−5；既有IL知识适配',PPO_SAFETY_EP='仅安全／可行EP分量优势；候选指标条件信用')
IDEAS.update(PPO_PREF_REPLAY='80%全量＋20%有效偏好场景；采样对照',PPO_PREF_CFM='与Replay同采样＋离线CFM偏好辅助')
IDEAS.update(PPO_NOISE03='rigorous eta0.1→0.03，mean与diffusion联动；正式5k对照')
IDEAS.update(PPO_LASTDET='末步确定性；PG与anchor同时改W2[7,8]，RNG消耗亦变',PPO_RESCUE_REPLAY='Safety10k权重新phase；80全量／10低分／10高分',PPO_RESCUE_CFM='与Rescue Replay同设置＋verified positive CFM')
COLORS.update(PPO_MEMORYLR20='#ea580c',PPO_SAFETY_EP='#059669')
SHORT.update(PPO_MEMORYLR20='MemoryLR20',PPO_SAFETY_EP='SafetyEP')


def update(row):
    value=row.get('RL_updates',row.get('updates'))
    return value if isinstance(value,(int,float)) else 0


def figure(root,h,name,title,draw,caption=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'path','svg.hashsalt':'qwen-world-editorial',
        'axes.spines.top':False,'axes.spines.right':False,'font.size':10,
        'axes.labelcolor':'#475569','xtick.color':'#64748b','ytick.color':'#64748b'})
    fig=draw(plt)
    directory=root/'reporting'/'figures';directory.mkdir(exist_ok=True,parents=True)
    buffer=io.StringIO();fig.savefig(buffer,format='svg',metadata={'Date':None},bbox_inches='tight')
    svg=buffer.getvalue();svg=svg[svg.index('<svg'):]
    h.atomic_text(directory/(name+'.svg'),svg)
    for extension in ['png','pdf']:
        buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=180,bbox_inches='tight',
            metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
        fd,temporary=tempfile.mkstemp(prefix='.'+name+'.',dir=directory)
        with os.fdopen(fd,'wb') as handle:
            handle.write(buffer.getvalue());handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,directory/(name+'.'+extension))
    plt.close(fig)
    return '<figure><figcaption>'+h.esc(title)+'</figcaption>'+svg+'<p class="figure-note">'+h.esc(caption)+'</p><div class="downloads">'+h.relative_link(root,str(directory/(name+'.png')),'PNG')+' · '+h.relative_link(root,str(directory/(name+'.pdf')),'PDF')+'</div></figure>'


def curve(root,h,name,title,series,*,baseline=None,ylabel='Canonical PDMS',caption=''):
    def draw(plt):
        fig,ax=plt.subplots(figsize=(7.0,3.9),layout='constrained')
        for arm,points in series.items():
            points=sorted(points)
            if points:ax.plot([x/1000 for x,y in points],[y for x,y in points],'-o',
                lw=2,ms=4,color=COLORS.get(arm,'#334155'),label=SHORT.get(arm,arm))
        if baseline is not None:ax.axhline(baseline,color='#0f172a',ls='--',lw=1.2,label='IL70 reference')
        ax.set(xlabel='RL optimizer updates (thousands)',ylabel=ylabel)
        ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,ncol=3,fontsize=8,loc='best')
        return fig
    return figure(root,h,name,title,draw,caption)


def contrast(item,target,baseline):
    rows=item.get('contrasts',{})
    direct=rows.get(target+'_minus_'+baseline)
    if direct:return direct
    inverse=rows.get(baseline+'_minus_'+target)
    if inverse:
        result=dict(inverse,delta_PDMS=-inverse['delta_PDMS'])
        if len(inverse.get('CI95_PDMS',[]))==2:result['CI95_PDMS']=[-inverse['CI95_PDMS'][1],-inverse['CI95_PDMS'][0]]
        return result
    return {}


def matched_noise003_complete(baseline,lower):
    """Publish the third mode only after the full matched collection completes."""
    if not isinstance(baseline,dict) or not isinstance(lower,dict):return False
    if baseline.get('status')!='COMPLETE' or lower.get('status')!='COMPLETE' or lower.get('errors'):
        return False
    old=baseline.get('results',{});new=lower.get('results',{})
    if not isinstance(old,dict) or not isinstance(new,dict) or not old or set(old)!=set(new):return False
    for arm,a in old.items():
        b=new[arm]
        if not isinstance(a,dict) or not isinstance(b,dict):return False
        if a.get('status')!='COMPLETE' or b.get('status')!='COMPLETE' or b.get('diagnostic_noise_level')!=.03:return False
        if a.get('checkpoint')!=b.get('checkpoint') or a.get('updates')!=b.get('updates') or a.get('count')!=b.get('count'):return False
        ta=[r.get('token')for r in a.get('scenes',[])];tb=[r.get('token')for r in b.get('scenes',[])]
        if len(ta)!=a.get('count') or len(set(ta))!=len(ta) or set(ta)!=set(tb) or len(tb)!=len(ta):return False
        for key in ('candidate_count','feasible','canonical_zero'):
            if a.get('ODE8',{}).get(key)!=b.get('ODE8',{}).get(key):return False
        expected=8*a['count']
        for mode in ('ODE8','SDE8'):
            row=b.get(mode,{})
            if row.get('candidate_count')!=expected:return False
            if any(not isinstance(row.get(k),(int,float)) or not 0<=row[k]<=expected for k in ('feasible','canonical_zero')):return False
    return True


def render_research_report(root,snapshot,history,h):
    esc=h.esc;fmt=h.fmt;number=h.number;table=h.table
    link=lambda path,label='证据':h.relative_link(root,str(path) if path is not None else None,label)
    complete=[r for r in snapshot['results'] if r.get('publishable_complete')]
    test=[r for r in complete if r.get('cohort')=='NAVTEST' and r.get('scenes')==12146]
    panel=[r for r in complete if r.get('cohort')=='TRAIN_DIAGNOSTIC' and r.get('scenes')==1024]
    points={(r.get('arm'),update(r)):r for r in test}
    panel_points={(r.get('arm'),update(r)):r for r in panel}
    runs={r['id']:r for r in snapshot['runs']}
    display_arms=ARMS+[arm for arm,run in runs.items() if arm not in ARMS and run.get('job')]
    base=snapshot.get('ppo_full_first_stage_paired_analysis',{}).get('reference_PDMS',88.79290990699972)
    panel_base=next((r['PDMS'] for r in panel if r.get('evaluation_kind')=='IL_BASELINE'),93.64668412698353)
    editorial=snapshot.get('editorial_analysis',{})
    decision=editorial.get('nft_decision',{})
    stopped=editorial.get('nft_stopped',{})
    decisions=editorial.get('decisions',{});stop_receipts=editorial.get('stopped_receipts',{})
    completed_no_extension={arm:value for arm,value in decisions.items() if value.get('disposition')=='STOP_FUTILITY' and value.get('stop_mode','').startswith('completed ')}
    safety_hold=editorial.get('safety_ep_review',{})
    safety_continue=editorial.get('safety_ep_continuation',{}) or editorial.get('safety_ep_continuation_pending',{})
    nft_state='已协作停止' if stopped else '停止决策已下，等待退出收据' if decision else '待停止决策／执行收据'
    analysis10=editorial.get('matched_10k',{})
    analysis20=editorial.get('matched_20k',{})
    scene_research=editorial.get('scene_safety_efficiency',{})
    pilot_startups=editorial.get('pilot_startups',{})
    pilot_registrations=editorial.get('pilot_registrations',{})
    active_pilots={arm for arm,value in pilot_startups.items() if value.get('status')=='ACTUAL_NATIVE_UPDATES_ALL8_FINITE' and len(value.get('rows',[]))==8 and all(row.get('finite') is True for row in value['rows'])}
    adaptive_amendment=editorial.get('adaptive_coverage_amendment',{})
    pilot_review=editorial.get('new_pilot_matched_review',{})
    preference_study=editorial.get('preference_mixture_study',{})
    preference_receipts=editorial.get('preference_training_receipts',{})
    pref_children_started={arm for arm,receipts in preference_receipts.items() if receipts.get('RUNNING.json',{}).get('child_pid') and receipts.get('RUNNING.json',{}).get('status')=='RUNNING'}
    fixed_probe_complete=editorial.get('fixed_theta_probe',{}).get('status')=='COMPLETE'
    noise003_result=editorial.get('noise003_result',{})
    noise003_matched=matched_noise003_complete(editorial.get('fixed_theta_probe',{}),noise003_result)
    rescue=editorial.get('rescue_study',{});rescue_registration=rescue.get('registration',{})
    sampler_reviews=editorial.get('sampler_matched_reviews',{})
    sampler_pairs=[(arm,key,row)for arm,review in sampler_reviews.items()for key,row in review.get('contrasts',{}).items()if row.get('status')=='COMPLETE']
    def pilot_pair(arm,u=10000):
        value=pilot_review.get('contrasts',{}).get(f'{arm}_u{u:06d}_NAVTEST_minus_PPO_FULL_u{u:06d}_NAVTEST',{})
        return value if value.get('status')=='COMPLETE' else {}
    safety_pair=pilot_pair('PPO_SAFETY_EP');memory_pair=pilot_pair('PPO_MEMORYLR20')
    pilot_focus=bool(safety_pair and ('PPO_SAFETY_EP',10000) in points)
    adaptive_analysis=editorial.get('adaptive_process',{})
    budgets=sorted({update(r) for r in test if r.get('arm') in PPO_ARMS})
    budget=max((u for u in budgets if ('PPO_FULL',u) in points and sum((a,u) in points for a in PPO_ARMS)>=3),default=1000)
    matched=[points[a,budget] for a in PPO_ARMS if (a,budget) in points]
    latest={a:max((r for r in test if r.get('arm')==a),key=update,default=None) for a in display_arms}
    best=max(test,key=lambda r:r['PDMS'],default=None)
    nft=latest.get('NFT_FULL')
    nft_best=max((r['PDMS'] for r in test if r.get('arm')=='NFT_FULL'),default=None)
    source_time=max((r.get('evidence_mtime',0) or 0 for r in complete),default=0)
    chapters=[]
    def chapter(index,id,title,deck,body):
        chapters.append('<section class="chapter" id="'+id+'"><div class="chapter-head"><span class="number">'+index+'</span><div><h2>'+title+'</h2><p class="deck">'+deck+'</p></div></div>'+body+'</section>')
    def evidence(level,text):return '<span class="badge '+level+'">'+text+'</span>'
    def detail(title,body):return '<details><summary>'+title+'</summary>'+body+'</details>'
    def signed(value,digits=4):return f'{value:+.{digits}f}' if h.finite(value) else '待记录'
    def ci_text(row):
        interval=row.get('CI95_PDMS',[])
        return '['+', '.join(signed(v) for v in interval)+']' if len(interval)==2 else '配对分析待完成'
    def verdict(row):
        interval=row.get('CI95_PDMS',[])
        if len(interval)!=2:return '只有点差，未判断优劣'
        return '区间高于零' if interval[0]>0 else '区间低于零' if interval[1]<0 else '区间跨零'

    # 01: conclusions first, with explicit strength and action.
    best_name=best.get('arm') if best else '待完整结果'
    cards=[('本轮已测最佳checkpoint',fmt(best['PDMS'],4) if best else '待评测',esc(best_name)+' · RL '+number(update(best)) if best else ''),
        ('该checkpoint对IL70增益',signed(best['PDMS']-base) if best else '—','历史最佳不等于最新阶段；点差不等于算法因果'),
        ('新pilot匹配预算' if pilot_focus else '匹配对照预算',number(10000 if pilot_focus else budget),'optimizer updates；不同预算不直接排优劣'),
        ('当前关键风险','NC下降' if pilot_focus else '待诊断','SafetyEP10k新增碰撞23／修复11' if pilot_focus else '安全分量与总分分别判断')]
    body='<div class="note"><b>核心研究目标：根据场景信息，平衡安全与驾驶效率。</b>识别何时必须让行、何时存在安全通行窗口，以及何时只是观测信息不足；不是把安全与舒适性作为主要对立轴。SafeComfort是突出EP的局部奖励去干扰，尚未实现真正的场景条件风险权衡。</div><div class="metric-grid">'+''.join('<div class="metric"><span>'+a+'</span><strong>'+b+'</strong><small>'+c+'</small></div>' for a,b,c in cards)+'</div>'
    body+='<p><b>IL与RL两线并行：</b>IL提供并补齐离线驾驶知识基础；RL在固定IL知识基础上强化轨迹行为，不全面重做IL。知识缺口反馈IL；新IL达标后，用相同RL配方另起对照，不热切换正在训练的旧父模型。</p>'
    priority_review=editorial.get('route_priority_review',{})
    if priority_review:
        retired_rows=[]
        for action in priority_review.get('actions',[]):
            arm=action.get('arm');confirmed=arm in stop_receipts
            peak=action.get('best_completed',{})
            last=max(action.get('completed_scores',[]),key=lambda r:r.get('updates',0),default={})
            if latest.get(arm) and update(latest[arm])>last.get('updates',0):last=latest[arm]
            retired_rows.append([esc(arm),'已实际结束' if confirmed else '停止已决定，待退出证据',number(action.get('sealed_updates')) if confirmed else '待seal',fmt(peak.get('PDMS'),5)+' / '+number(peak.get('updates'))+'更新<small>最近 '+fmt(last.get('PDMS'),5)+' / '+number(last.get('updates'))+'更新</small>','已sealed27k仅做一次完整NAVTEST，再决定是否重启' if arm=='PPO_ADAPTIVE' else '保留已测checkpoint，不追加长训预算'])
        body+='<div class="note"><b>当前资源优先级：</b>SafetyEP推进20k自然复核；偏好与采样pilot先到1k／5k；'+('Rescue两臂从最高分Safety10k启动额外5k，验证低分修复与高分保持；'if rescue_registration else '')+'新问题用有限诊断验证。50k是上限，当前没有默认跑满的承诺。</div><h3>本轮平台期路线：停止证据与下一复核点</h3>'+table(['路线','实际决策状态','最终原生更新','本路线最佳／最近完整分','下一步'],retired_rows)
        if priority_review.get('status')=='FOUR_NATIVE_TRAINING_CONTINUATIONS_STOPPED':body+='<p>四条旧线均已return0并保留optimizer／RNG，'+number(priority_review.get('freed_training_slots'))+' 个8rank训练槽位结束，分布在 '+number(priority_review.get('distinct_physical_hosts'))+' 个物理host；不等于32张独占GPU已空闲。Full／Safe直到48.9k／49.96k才停，复核确实来晚了，已耗算力不能回收；不把它写成10k提前止损的节省。</p>'
        body+='<p class="muted">资源去向：Safety最高分主线、偏好采样／辅助损失对照、有限低噪声探针；Adaptive只完成一次已封存27k正式Test，不恢复优化，等新覆盖阶段的结果再决定。'+link(root/'analysis'/'ROUTE_PRIORITY_REVIEW_20261031.json','本轮完整路线与资源决策')+'</p>'
        adaptive27=points.get(('PPO_ADAPTIVE',27000));adaptive_review=editorial.get('adaptive27k_review',{})
        if adaptive_review.get('status')=='COMPLETE' and adaptive27:
            adaptive20=points.get(('PPO_ADAPTIVE',20000));delta27=adaptive27['PDMS']-adaptive20['PDMS']if adaptive20 else None
            body+='<p class="note">Adaptive27k有限复核已COMPLETE：完整NAVTEST '+fmt(adaptive27['PDMS'],6)+' / 12,146场景。'+('较自身20k '+signed(delta27,6)+' 点；'if adaptive20 else '')+'此分数与训练panel分开；本次复核未提供恢复优化的正向证据，保留已停止状态，重启需新独立决定。'+link(adaptive27.get('evidence'),'27k完整SUMMARY')+'</p>'
        else:body+='<p class="muted">Adaptive27k有限复核尚未COMPLETE，不报部分分数；不因评测在跑而恢复训练。'+link(root/'analysis'/'ADAPTIVE27K_FINITE_REVIEW.json','有限复核实际状态')+'</p>'
    if pilot_focus:
        sm=safety_pair.get('safety_metrics',{});nc=sm.get('no_at_fault_collisions',{});ep=sm.get('ego_progress',{})
        body+='<div class="conclusion-grid"><article class="finding success">'+evidence('strong','已确认 · 完整10k同预算配对')+'<h3>SafetyEP达到本轮已测最佳，总分收益有配对支持</h3><p>完整NAVTEST '+fmt(safety_pair.get('target_PDMS'),6)+'，相对Full10k '+signed(safety_pair.get('delta_PDMS'),6)+'，95%区间 '+ci_text(safety_pair)+'。EP提高 '+signed(ep.get('delta_percentage_points'),5)+' pp；这验证了本轮候选指标条件信用分解的阶段收益，尚不是场景风险推理能力的证明。</p></article>'
        body+='<article class="finding caution">'+evidence('limited','必须同时面对 · 安全交换')+'<h3>总分提升没有解决碰撞偏好冲突</h3><p>SafetyEP10k的NC '+signed(nc.get('delta_percentage_points'),5)+' pp，配对区间低于零；新增失败 '+number(nc.get('new_failure'))+'、修复 '+number(nc.get('repaired_failure'))+'。下一步按同场景通过／冒险候选构建偏好诊断，检查必要等待和可实现效率，不以单一总分覆盖风险。</p></article>'
        body+='<article class="finding caution">'+evidence('limited','阴性结果 · memory适配强度')+'<h3>单纯提高memory学习率尚无提分证据</h3><p>MemoryLR20完整10k '+fmt(memory_pair.get('target_PDMS'),6)+'，对Full '+signed(memory_pair.get('delta_PDMS'),6)+'，区间 '+ci_text(memory_pair)+'；总分跨零，NC/TTC却有负向配对证据。这不能推出IL知识已充分，也不支持仅靠更大学习率修复风险信息。</p></article></div>'
        body+='<p class="muted">历史对照仍保留：Full1k相对IL70提高+1.639627；Batch10k早期优势没有持续到20k；NFT与固定Hard已止损。训练预算增加不等于性能单调提高，当前重点转向安全—效率冲突信用和纵向失败集合。</p>'
    else:
        body+='<div class="conclusion-grid"><article class="finding success">'+evidence('strong','已确认 · 完整评测与父模型配对')+'<h3>PPO带来收益，增加预算并非单调提升</h3><p>IL70基线 '+fmt(base,6)+'。PPO_FULL在1k提高 +1.639627，配对95%区间 [＋1.214409, ＋2.039287]；保留历史最佳checkpoint，不能默认训练更久就更好。</p></article>'
    batch_pair=contrast(analysis10,'PPO_BATCHSTD','PPO_FULL')
    if not pilot_focus and budget>=20000 and ('PPO_FULL',20000) in points and ('PPO_BATCHSTD',20000) in points:
        full20=points['PPO_FULL',20000];batch20=points['PPO_BATCHSTD',20000];safe20=points.get(('PPO_SAFE_COMFORT',20000))
        safe_pair20=contrast(analysis20,'PPO_SAFE_COMFORT','PPO_FULL')
        safe_summary=('SAFE20k−FULL '+signed(safe_pair20.get('delta_PDMS'))+'，区间 '+ci_text(safe_pair20)+'，支持该阶段局部改动收益；' if safe_pair20 else 'SAFE20k观察分 '+fmt(safe20['PDMS'],5)+'；' if safe20 else '')
        body+='<article class="finding caution">'+evidence('limited','阶段更新 · 10k不能外推')+'<h3>10k的BatchStd领先未普遍持续</h3><p>10k的+0.134370有配对支持；12k点分低于Full，20k Batch−Full仅 '+signed(batch20['PDMS']-full20['PDMS'])+' 点。'+safe_summary+'这仍不能证明策略已学会按场景平衡安全与效率。</p></article>'
    elif not pilot_focus and batch_pair.get('CI_excludes_zero') and batch_pair.get('delta_PDMS',0)>0:
        body+='<article class="finding caution">'+evidence('strong','支持 · 单seed物理log配对')+'<h3>BatchStd的10k增益有区间支持</h3><p>相对PPO_FULL '+signed(batch_pair['delta_PDMS'])+' 点，95%区间 '+ci_text(batch_pair)+'。这是本轮固定seed与10k阶段的证据；SAFE−FULL、BATCH−SAFE区间仍跨零，不把数值排名当作普遍算法结论。</p></article>'
    elif not pilot_focus:body+='<article class="finding caution">'+evidence('limited','待判断 · 小差需要匹配比较')+'<h3>新增trick的优劣尚需区间支持</h3><p>仅比较同预算、同人口的完整结果；配对区间未完成或跨零时，不宣布算法胜出。</p></article>'
    if not pilot_focus:body+='<article class="finding stop">'+evidence('action','决策 · 低潜力路线释放预算')+'<h3>NFT缺少持续部署收益</h3><p>'+(number(update(nft))+'更新完整分 '+fmt(nft['PDMS'],6)+'，相对IL70 '+signed(nft['PDMS']-base)+'；已测最高仅 '+fmt(nft_best,6)+'。' if nft else '等待完整NFT结果。')+' '+esc(nft_state)+'。</p></article></div>'
    body+='<div class="protocol">统一读数：NAVTEST 12,146场景 · 单候选ODE K10 / shift5 · 训练seed42 / 推理seed2101。固定种子研究，不安排多种子。TRAIN panel与随机训练候选分数另列。</div>'
    if rescue_registration:
        rescue_running=[a for a in ('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM')if runs.get(a,{}).get('worker_receipt_status')=='RUNNING'and h.finite(runs.get(a,{}).get('updates'))]
        body+='<p class="note"><b>当前新增主问题：采到的好轨迹如何真正被学会，同时保住已有能力？</b>Rescue '+('两臂已有真实原生优化更新'if len(rescue_running)==2 else '已登记，实际启动／更新看路线表')+'。从SafetyEP10k当前权重开始额外phase，以低分verified teacher补学习方向，以高分retention池防遗忘；尚未建立新最佳。'+link(root/'analysis'/'RESCUE_PAIR_NATIVE_STARTUP_20261031.json','8rank启动与warmstart身份')+'</p>'
    lastdet_first=next((row for arm,key,row in sampler_pairs if arm=='PPO_LASTDET'and '_u001000_'in key),{})
    if lastdet_first:
        body+='<p class="alert"><b>LastDet1k的实际负结果：</b>完整Test '+fmt(lastdet_first.get('target_PDMS'),6)+'，相对Full1k '+signed(lastdet_first.get('delta_PDMS'),6)+'，95%区间 '+ci_text(lastdet_first)+'。局部去末噪声反事实改善未转成该训练配方的部署收益；继续既定5k再筛，不把W2视作已证明有效。'+link(root/'analysis'/'LASTDET_MATCHED_REVIEW.json','完整同预算配对与过程诊断')+'</p>'
    if preference_study:
        body+='<h3>结论、假设与实验状态</h3>'+table(['证据等级','现在的判断','下一步怎样验证'],[
            ['明确结论','零分有固定核心，也有换入换出；SafetyEP新高仍伴随NC退化。'+('选定固定theta探针中SDE净损害较大，同时有少量独有修复。' if fixed_probe_complete else ''),'持续集合、相邻修复／新零和安全组件继续分开观察，不能据局部均值全关探索。'],
            ['待验证假设','候选偏好监督能否改善部署安全—效率，以及局部噪声效应能否推广，均待验证。','Replay隔离采样作用，CFM对Replay隔离辅助损失；固定theta结果只解释选定cohort。'],
            ['进行中研究','两项5k偏好pilot已有native child启动回执；优化进度见路线表。' if len(pref_children_started)==2 else '两项5k偏好pilot已登记；实际派发／更新以路线与独立收据为准。','保持canonical单候选评测，不预填训练结果，也不把best-of8当部署。']])
    if snapshot.get('warnings'):body+='<div class="alert">'+esc('；'.join(snapshot['warnings']))+'</div>'
    chapter('01','conclusions','现在可以得出的结论','先区分已观察到的提升、尚未证明的算法优势，以及已经决定调整的投入。',body)

    # 02: a compact ledger of doing / done / stopped / next.
    stopped_actions={
        'PPO_FULL':'平台期：保留10k峰值及20k／35k控制checkpoint，停止只为跑满50k的优化。',
        'PPO_SAFE_COMFORT':'10k后持续回落；本次接近50k才结束，不夸大提前节省。保留10k／20k已有收益证据。',
        'PPO_ANCHOR40':'beta40未见持续优势；保留10k峰值，不自动继续长训。',
        'PPO_ADAPTIVE':'先停止优化；仅对已sealed27k做一次完整NAVTEST，复核17k起coverage新阶段后再决定是否重启。',
        'PPO_MEMORYLR20':'已完成10k，无明确总分增益且NC／TTC下降；不延长。',
        'PPO_BATCHSTD':'保留10k历史好点；20k优势未持续、NC／TTC退化，停止后续预算。',
        'CISPO_FULL':'20k总分近Full而NC／TTC下降，保留已测checkpoint。',
        'PPO_HARD':'保留5k好点与困难数据诊断；固定hard-only10k无优势，不推论困难场景不值得训练。',
        'NFT_FULL':'已测多个预算点缺少持续部署收益，保留原生状态与结果。'}
    route_rows=[]
    for arm in display_arms:
        run=runs.get(arm,{})
        result=latest.get(arm)
        status=run.get('owner_receipt',{}).get('status') or run.get('status','未登记')
        if arm in stop_receipts:state='已协作停止';next_action=stopped_actions.get(arm,'保留checkpoint、optimizer/RNG与结果；预算转向更高优先研究')
        elif arm in completed_no_extension:state='pilot完成，取消延长（STOP_FUTILITY）';next_action='保留10k结果与原生状态；本轮已完成，无需再发子进程停止信号'
        elif arm=='PPO_SAFETY_EP' and safety_continue:
            new_native=run.get('stage_budget_updates',0)>10000 and run.get('worker_receipt_status')=='RUNNING'
            state='续训native child已启动' if new_native else '继续训练已决定，等待新启动收据'
            next_action='高优先推进20k复核；35k／50k再按证据决策，50k仅上限；偏好实验并行'
        elif arm=='PPO_SAFETY_EP' and safety_hold.get('disposition')=='RETAIN_CHECKPOINT_HOLD_AT_REGISTERED_10K_REVIEW':state='保留10k，等待安全取舍复核';next_action='NC退化与总分收益联合判断；下一预算单独决策'
        elif arm=='PPO_NOISE03' and run.get('worker_terminal_receipt') and h.finite(run.get('worker_returncode')) and run['worker_returncode']!=0:
            state='等待容量：child前拒绝'if run['worker_returncode']==75 else '启动失败待修复：未形成科学训练结果'
            next_action='RESULT优先于旧RUNNING；保留失败execution／receipts，修复后按新的真实启动回执更新'
        elif arm in decisions:state='停止决策已下，等待退出收据';next_action='协作保存原生checkpoint，等待退出确认'
        elif status=='PILOT_COMPLETE_REVIEW_REQUIRED':state='首轮pilot已完成，等待预算复核';next_action='完整同预算总分与安全组件联合判断；新预算以独立决策为准'
        elif status=='WAITING_OWN_TRAIN_FEEDBACK':state='等待自己的TRAIN反馈';next_action='bank就绪后继续；不等待其他臂或Test'
        elif status.startswith('WAITING'):state='等待容量／阶段条件';next_action='按退出事件与容量heartbeat重排'
        elif run.get('job'):
            state=('已观察全rank原生日志' if run.get('journal_ranks')==run.get('expected_ranks',8) and h.finite(run.get('updates')) else
                   'worker已启动，原生更新待日志' if run.get('job_receipts',{}).get('RUNNING.json') else '已登记派发，启动／更新待收据')
            next_action='当前阶段 '+number(run.get('stage_budget_updates'))+'；'+('首轮研究cap '+number(run.get('research_budget_updates'))+'，到预算再评审' if run.get('research_budget_updates') else '周期完整评测异步')
        else:state='未派发';next_action='等待研究决策与资源'
        native_cell=('停止于 '+number(stop_receipts[arm].get('checkpoint',{}).get('updates')) if arm in stop_receipts else
                     '结束于 '+number(completed_no_extension[arm].get('matched_budget_updates'))+'，不再延长' if arm in completed_no_extension else
                     number(run.get('updates'))+' → 20,000自然节点' if arm=='PPO_SAFETY_EP' and safety_continue else '10,000：hold review' if arm=='PPO_SAFETY_EP' and safety_hold else number(run.get('updates'))+' / '+number(run.get('stage_budget_updates')))
        if arm in ('PPO_PREF_REPLAY','PPO_PREF_CFM') and arm not in decisions:next_action='1k／5k完整配对：Replay−Full隔离采样，CFM−Replay隔离辅助；首轮cap5k。'
        if arm in ('PPO_NOISE03','PPO_LASTDET') and arm not in decisions and run.get('worker_receipt_status')=='RUNNING':next_action='真实恢复训练；首轮1k／5k同预算Full配对，完整证据后选择性延长。'
        if arm in ('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM'):
            native_cell='新phase '+number(run.get('updates'))+' / '+number(run.get('stage_budget_updates'))
            next_action='额外phase1k／5k：对固定Safety10k查修复／新零与≥0.9保持；CFM−Replay隔离辅助。'
        state_cell=esc(state)+('<small>'+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/f'{arm}.STOPPED.json','实际结束收据')+'</small>' if arm in stop_receipts else '')
        route_rows.append(['<b>'+esc(arm)+'</b><small>'+esc(IDEAS.get(arm,run.get('hypothesis','新登记独立pilot')))+'</small>',
            fmt(result['PDMS'],4)+'<small>'+('额外phase 'if arm.startswith('PPO_RESCUE_')else '')+number(update(result))+'更新 · 完整Test</small>' if result else '暂无完整Test',
            native_cell,
            state_cell,esc(next_action)])
    body=table(['路线与单因素','最近完整结果','原生日志 / 当前目标','执行决策','依据与下次复核'],route_rows)
    body+='<p class="note">资源优先流向SafetyEP的20k复核、两臂5k偏好对照与针对新问题的有限诊断；平台期路线结束后保留历史最佳checkpoint与控制结果。实际退出按逐臂收据，接近自然预算末尾的SAFE不计作提前节省大量训练；库存UUID和slot结束也不等于整张物理GPU空闲。</p>'
    ongoing_count=sum(a in runs and a not in decisions for a in display_arms)
    anchor_result=latest.get('PPO_ANCHOR40')
    anchor_result_text=('ANCHOR40最近完整Test为'+fmt(anchor_result['PDMS'],6)+' / '+number(update(anchor_result))+'更新，单独保留其预算。' if anchor_result else 'ANCHOR40是独立单因素，尚无正式完整Test成绩。')
    stopped_names='、'.join(sorted(stop_receipts)) or '暂无'
    pending_stop_names='、'.join(sorted(set(decisions)-set(stop_receipts)-set(completed_no_extension))) or '暂无'
    body+='<p class="note">已协作停止：'+esc(stopped_names)+'。完成pilot后取消延长：'+esc('、'.join(sorted(completed_no_extension)) or '暂无')+'。已决定停止、待退出收据：'+esc(pending_stop_names)+'。当前 '+number(ongoing_count)+' 条未止损研究路线，包含等待复核者；已测checkpoint继续保留，包含Batch10k历史好点。'+anchor_result_text+'</p>'
    if safety_continue:body+='<p class="note">SafetyEP最高分路线已恢复高优先续训，先到20k自然复核节点，偏好实验并行；后续35k／50k按新证据决策，50k仅配置上限，不默认跑满。NC下降持续跟踪，不单独作为停训条件；原10k hold仅保留为历史。实际新child与更新状态见上表。'+link(editorial.get('safety_ep_continuation_path') or root/'reporting'/'SAFETY_EP_CONTINUATION_PENDING_20261031.json','最新续训决策／执行证据')+'</p>'
    elif safety_hold:body+='<p class="muted">SafetyEP保留10k新高与NC交换证据，当前hold review，优先检验偏好干预。'+link(root/'analysis'/'SAFETY_EP_10K_REVIEW_20261031.json','SafetyEP10k复核决定')+'</p>'
    if editorial.get('safety_evidence_review'):body+='<p class="muted">有限CPU证据评审已启动：20k提出建议；35k若两个节点均未刷新10k、总分差95%区间上界低于0且NC／DAC／TTC无明确改善，才按登记条件协作停止。不因单个安全指标下降或预算上限强停。'+link(root/'analysis'/'SAFETY_EVIDENCE_REVIEW_20261031.json','Safety证据评审状态')+'</p>'
    resources=snapshot.get('active_resources',[]);uuids={r.get('uuid') for r in resources if r.get('uuid')}
    body+='<p class="muted">最新可核查资源身份：'+number(len(snapshot.get('current_authorized_hosts') or []))+'许可主机 / '+number(len(uuids))+'已登记GPU UUID，来源含库存快照与native child启动回执；授权、准入合格和实际部署分开记录，UUID不等于当前空闲。运营细节见折叠附录。</p>'
    if snapshot.get('resource_inventory_unknown_hosts'):body+='<p class="muted">当前重新授权但物理库存／就绪度未核实的主机：'+esc('、'.join(snapshot['resource_inventory_unknown_hosts']))+'；不填8张可用卡，不将连接失败写成部署完成。</p>'
    temporary=snapshot.get('resource_temporary_window',{})
    reauthorization=snapshot.get('resource_reauthorization_note',{})
    if snapshot.get('resource_finite_window_superseded'):
        request=snapshot.get('resource_authorization_request',{})
        body+='<p class="muted">最新用户REQUEST已将 '+esc('、'.join(request.get('hosts',[])))+' 改为临时不限时许可，覆盖旧8小时窗口；00:22:59保护线／00:42:59到期仅保留为历史记录。许可不证明连接、库存、空闲或部署就绪。'+link(snapshot.get('resource_authorization_request_path'),'实际用户授权收据')+'</p>'
    elif reauthorization:
        granted=set(reauthorization.get('hosts',[]))<=set(snapshot.get('current_authorized_hosts') or [])
        try:window_ended=snapshot.get('generated_unix',0)>=datetime.fromisoformat(reauthorization['expires_utc'].replace('Z','+00:00')).timestamp()
        except (KeyError,ValueError):window_ended=False
        window_status=('该8小时窗口已结束，当前有效许可看最新grant。' if window_ended else '已进入当前有效grant。' if granted else 'grant／库存记录正在更新，旧撤回记录不能当作永久禁用。')
        body+='<p class="muted">用户新增8小时再授权：vla-zt3 / tsinghua-zt2，'+esc(reauthorization.get('starts_utc'))+' → '+esc(reauthorization.get('expires_utc'))+'。'+window_status+'这不表示新部署已发生；当前状态以时间窗与独立startup收据为准。</p>'
    chapter('02','routes','路线状态：已完成、进行中、停止与下一步','研究预算依据证据调整；控制进程的状态不替代模型更新或评测完成证据。',body)

    # 03: matched-budget curves and point deltas sit side by side.
    plot_budget=10000 if pilot_focus else budget
    plot_arms=['PPO_FULL','PPO_MEMORYLR20','PPO_SAFETY_EP'] if pilot_focus else ALGO_ARMS
    series={a:[(update(r),r['PDMS']) for r in test if r.get('arm')==a and update(r)<=plot_budget] for a in plot_arms}
    line=curve(root,h,'research_matched_ppo','新pilot完整NAVTEST：同预算曲线' if pilot_focus else 'PPO算法完整NAVTEST：同预算曲线',series,
        caption='纵轴放大用于看路线差异；IL70='+fmt(base,4)+'见首卡总提升。数据路线与NFT分别另图。')
    paired_source={'contrasts':{arm+'_minus_PPO_FULL':pilot_pair(arm,plot_budget) for arm in plot_arms if arm!='PPO_FULL'}} if pilot_focus else analysis20 if budget==20000 and analysis20.get('status')=='COMPLETE' else analysis10 if budget==10000 else {}
    base_row=points.get(('PPO_FULL',plot_budget))
    def delta_draw(plt):
        fig,ax=plt.subplots(figsize=(7,3.9),layout='constrained')
        arms=[a for a in plot_arms if a!='PPO_FULL' and (a,plot_budget) in points]
        for i,arm in enumerate(arms):
            delta=points[arm,plot_budget]['PDMS']-base_row['PDMS'] if base_row else 0
            pair=contrast(paired_source,arm,'PPO_FULL');interval=pair.get('CI95_PDMS',[])
            ax.barh(i,delta,color=COLORS[arm],height=.58,alpha=.85)
            if len(interval)==2:ax.errorbar(delta,i,xerr=[[max(0,delta-interval[0])],[max(0,interval[1]-delta)]],fmt='o',color='#0f172a',capsize=4,ms=3)
            else:ax.text(delta+.012 if delta>=0 else delta-.012,i,f'{delta:+.3f}',va='center',ha='left' if delta>=0 else 'right',fontsize=9)
        ax.set_yticks(range(len(arms)),[SHORT[a] for a in arms]);ax.axvline(0,color='#475569',lw=1)
        ax.set(xlabel='PDMS points vs PPO_FULL',title=f'Matched {plot_budget//1000}k checkpoint');ax.grid(axis='x',alpha=.18);ax.margins(x=.25)
        return fig
    delta=figure(root,h,'research_matched_deltas','相对PPO_FULL的点差与配对区间',delta_draw,
        '有区间才画误差线；没有误差线表示配对分析尚未完成，不能据条形长度宣布优劣。')
    rows=[]
    table_budgets=sorted({update(r) for r in test if r.get('arm') in plot_arms and update(r)<=plot_budget})
    for arm in plot_arms:
        cells=[]
        for u in table_budgets:
            row=points.get((arm,u));cells.append(fmt(row['PDMS'],6) if row else '—（无完整点）')
        rows.append(['<b>'+esc(arm)+'</b>']+cells)
    body='<div class="plots-two">'+line+delta+'</div>'+table(['完整NAVTEST']+[str(u//1000)+'k' for u in table_budgets],rows)
    if pilot_focus:
        old_budgets=sorted({update(r)for r in test if r.get('arm')in ALGO_ARMS})
        old_rows=[[esc(arm)]+[fmt(points[arm,u]['PDMS'],6)if (arm,u)in points else '—' for u in old_budgets]for arm in ALGO_ARMS]
        body+=detail('保留原算法路线的完整阶段历史（含35k）',table(['原算法路线']+[str(u//1000)+'k'for u in old_budgets],old_rows)+'<p>历史高点与后续回落均保留，不能把早期名次外推到新预算。</p>')
    if budget>=20000:
        change_rows=[]
        for arm in ALGO_ARMS:
            earlier=points.get((arm,10000));later=points.get((arm,20000))
            if earlier and later:change_rows.append([esc(arm),fmt(earlier['PDMS'],6),fmt(later['PDMS'],6),signed(later['PDMS']-earlier['PDMS'],6)])
        historical_change=table(['同一路线','10k完整分','20k完整分','20k−10k：观察点差'],change_rows)+'<p class="note">20k结果不支持把10k的最佳checkpoint优势外推为普遍持续收益。阶段点回落与算法间差异分开解释；20k是否存在可重复的安全—效率改进，需看同预算配对及场景分组，不能只追一个总分。</p>'
        body+=detail('旧路线10k→20k：保留预算非单调证据',historical_change) if pilot_focus else historical_change
    if analysis20.get('status')=='COMPLETE':
        rows20=[[esc(key.replace('_minus_',' − ')),signed(row.get('delta_PDMS')),ci_text(row),verdict(row)]for key,row in analysis20.get('contrasts',{}).items()]
        historical_pairs=table(['20k同预算配对','点差','95%区间：PDMS点','判断'],rows20)+link(editorial.get('matched_20k_path'),'20k完整配对与组件拆解')
        body+=detail('旧路线20k同预算配对',historical_pairs) if pilot_focus else historical_pairs
    elif budget>=20000:body+='<p class="muted">20k配对分析尚未发布；右图此阶段只画真实点差，不画10k或跨预算区间。</p>'
    if analysis10.get('status')=='COMPLETE':
        paired_rows=[]
        for key,row in analysis10.get('contrasts',{}).items():
            if 'PPO_HARD' in key:continue
            paired_rows.append([esc(key.replace('_minus_',' − ')),signed(row.get('delta_PDMS')),ci_text(row),verdict(row)])
        body+=detail('保留10k历史配对：不能自动沿用到20k',table(['10k配对','点差','95%区间：PDMS点','该阶段判断'],paired_rows)+link(root/'analysis'/'PORTFOLIO_10K_PAIRED.json','10k完整配对与匹配证据'))
    else:body+='<p class="note">10k配对分析正在复用既有scene记录完成。当前表格证明这些阶段点的完整分数，不单独证明BatchStd或SafeComfort胜过PPO_FULL。此前1k的CISPO−FULL、FULL−HARD区间都跨零。</p>'
    historical=snapshot.get('historical',{}).get('points',[])
    old=max((r for r in historical if h.finite(r.get('PDMS'))),key=lambda r:r['PDMS'],default=None)
    if old:body+='<p class="muted">历史IL40k起点的已测最高 '+fmt(old['PDMS'],6)+'（'+esc(old.get('route'))+' / '+number(old.get('updates'))+'更新）。本轮父模型为IL70k；跨轮预算和父模型不同，仅作历史参照。</p>'
    chapter('03','matched','完整结果：优先比较匹配预算','分数由封存SUMMARY读取；TRAIN与partial不会进入这些图。',body)

    # 04: explain the PPO result rather than repeat a leaderboard.
    metrics=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort']
    def components_draw(plt):
        import numpy as np
        fig,ax=plt.subplots(figsize=(7,3.9),layout='constrained')
        arms=[a for a in plot_arms if (a,plot_budget) in points]
        values=np.array([[100*(points[a,plot_budget].get('means',{}).get(m,0)-base_row.get('means',{}).get(m,0)) for m in metrics]for a in arms]) if base_row else np.zeros((len(arms),5))
        limit=max(.05,float(abs(values).max())) if values.size else .05
        image=ax.imshow(values,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(5),['NC','DAC','TTC','EP','Comfort']);ax.set_yticks(range(len(arms)),[SHORT[a] for a in arms])
        for i in range(len(arms)):
            for j in range(5):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',fontsize=9)
        fig.colorbar(image,ax=ax,label='Percentage points vs Full',shrink=.8)
        return fig
    heat=figure(root,h,'research_ppo_components',str(plot_budget//1000)+'k同预算安全、效率与其他评价分量',components_draw,'每格是相对同预算PPO_FULL的均值百分点。Comfort保留为官方评价分量，核心研究问题是场景条件安全—效率权衡。')
    full_pair=snapshot.get('ppo_full_first_stage_paired_analysis',{});paired=full_pair.get('paired',{})
    def repair_draw(plt):
        fig,axes=plt.subplots(1,2,figsize=(7,3.9),layout='constrained')
        changed=full_pair.get('changed_scores',{})
        axes[0].bar(['Improved','Degraded','Unchanged'],[changed.get('improved',0),changed.get('degraded',0),changed.get('exactly_unchanged',0)],color=['#059669','#d97706','#94a3b8'])
        axes[0].set(ylabel='Scenes',title='Full 1k vs IL70');axes[0].tick_params(axis='x',rotation=25)
        axes[1].bar(['G+','G−'],[full_pair.get('G_plus_PDMS',0),full_pair.get('G_minus_PDMS',0)],color=['#059669','#d97706']);axes[1].axhline(0,color='#475569',lw=1);axes[1].set(ylabel='Contribution to mean PDMS',title='Paired score decomposition')
        for ax in axes:ax.grid(axis='y',alpha=.15)
        return fig
    repairs=figure(root,h,'research_full_repairs','早期收益的组成：少数严重失败修复很重要',repair_draw,'这是已完成1k父模型配对的机制线索，不直接作为10k改动的因果归因。')
    body='<div class="plots-two">'+heat+repairs+'</div>'
    if pilot_focus:
        rows=[]
        for arm in ['PPO_SAFETY_EP','PPO_MEMORYLR20']:
            row=pilot_pair(arm);cells=[]
            for metric in ['no_at_fault_collisions','time_to_collision_within_bound','ego_progress']:
                m=row.get('safety_metrics',{}).get(metric,{});ci=m.get('CI95_delta_percentage_points',[])
                cells.append(signed(m.get('delta_percentage_points'),5)+'<small>['+', '.join(signed(x,5)for x in ci)+']</small>')
            rows.append([esc(arm),signed(row.get('delta_PDMS'),6)+'<small>'+ci_text(row)+'</small>']+cells)
        body+='<h3>10k pilot：效率提升与安全交换要同时解释</h3>'+table(['同10k对Full','ΔPDMS［95%区间］','ΔNC pp［区间］','ΔTTC pp［区间］','ΔEP pp［区间］'],rows)+'<p class="note">SafetyEP提高了可实现进度的信用，但NC新增失败23、修复11；MemoryLR20没有总分优势，NC/TTC有下降证据。不能把“拆了安全分量”当成已经学到安全偏好，也不能把提高memory学习率等同补齐信息。下一步用同场景反例和纵向zero集合定位信用、候选覆盖与表征问题。</p>'
    if analysis20.get('status')=='COMPLETE':
        def metric20(row,metric):
            item=row.get('safety_metrics',{}).get(metric,{});interval=item.get('CI95_delta_percentage_points',[])
            return signed(item.get('delta_percentage_points'),5)+('<small>['+', '.join(signed(v,5)for v in interval)+']</small>' if len(interval)==2 else '')
        rows20=[]
        for arm in ['PPO_SAFE_COMFORT','PPO_BATCHSTD','CISPO_FULL']:
            row=contrast(analysis20,arm,'PPO_FULL')
            if row:rows20.append([esc(arm)+' − FULL',signed(row.get('delta_PDMS'),5)+'<small>'+ci_text(row)+'</small>',metric20(row,'ego_progress'),metric20(row,'no_at_fault_collisions'),metric20(row,'time_to_collision_within_bound')])
        body+='<h3>20k：效率收益与风险分量分别判断</h3>'+table(['同20k对照','ΔPDMS［95%区间］','ΔEP pp［区间］','ΔNC pp［区间］','ΔTTC pp［区间］'],rows20)+'<p class="note">SAFE−FULL的20k总分区间高于零，主要伴随EP提高；NC/TTC区间跨零，不代表所有安全分量改善。BATCH和CISPO的总分差区间跨零，NC/TTC下降区间却低于零。这提示需要按场景检查“何时前进、何时等待”，不能把全局奖励重加权视为已解决风险权衡。</p>'
    if batch_pair:
        components=batch_pair.get('safety_metrics',{});transitions=batch_pair.get('paired',{}).get('safety_transitions',{});zeros=batch_pair.get('paired',{})
        loss=batch_pair.get('loss_decomposition',{});unsafe_loss=loss.get('new_safety_violation_Gminus_PDMS');total_loss=batch_pair.get('G_minus_PDMS')
        fraction=unsafe_loss/total_loss if h.finite(unsafe_loss) and h.finite(total_loss) and total_loss<0 else None
        nc=transitions.get('no_at_fault_collisions',{})
        body+='<div class="note"><b>10k历史机制线索：</b>BatchStd增益伴随EP '+signed(components.get('ego_progress',{}).get('delta_percentage_points'),5)+' pp；NC '+signed(components.get('no_at_fault_collisions',{}).get('delta_percentage_points'),5)+' pp（新增失败'+number(nc.get('new_failure'))+'、修复'+number(nc.get('repaired_failure'))+'），DAC '+signed(components.get('drivable_area_compliance',{}).get('delta_percentage_points'),5)+' pp、TTC '+signed(components.get('time_to_collision_within_bound',{}).get('delta_percentage_points'),5)+' pp。零分修复'+number(zeros.get('repaired_zero'))+'、新增'+number(zeros.get('new_zero'))+'。'+number(loss.get('new_safety_violation_degraded_scenes'))+'个新增安全违规的下降场景占对照损失约'+fmt(fraction,1,percent=True)+'%。这是10k的拆解，不能直接当作20k效果或场景权衡能力。</div>'
    probes=snapshot.get('additional_navtest_probes',{})
    if probes.get('evaluations'):
        probe_rows=[]
        for probe in probes['evaluations']:
            result=points.get((probe.get('arm'),probe.get('updates')))
            progress=next((r for r in snapshot['results']if r.get('cohort')=='NAVTEST' and r.get('arm')==probe.get('arm') and update(r)==probe.get('updates')),None)
            progress_text=('评测进行中：'+number(progress.get('scenes'))+' / 12,146' if progress and progress.get('scenes') else '准备export／等待异步派发')
            probe_rows.append([esc(probe.get('arm')),number(probe.get('updates')),
                fmt(result['PDMS'],6) if result else progress_text+'；未发布均分',
                fmt(100*result['means']['no_at_fault_collisions'],6) if result else '—',
                link(result.get('evidence'),'完整SUMMARY') if result else link(probe.get('checkpoint'),'已封存12k checkpoint')])
        probes_done=all((p.get('arm'),p.get('updates'))in points for p in probes['evaluations'])
        body+='<h3>'+('12k补充复查：两个模型均已完整' if probes_done else '12k补充复查执行进度')+'</h3><p>该复查在20k之前登记，用来提前检查10k收益与NC风险是否持续；不改变learner目标、不新增seed。现在按原预算保留历史结果。</p>'+table(['路线','精确预算','完整PDMS','NC：0–100','证据'],probe_rows)+'<p>'+link(root/'ADDITIONAL_NAVTEST_PROBES.json','两份12,000更新完整Navtest注册')+'</p>'
    matched12=snapshot.get('matched_12k_analysis',{})
    if matched12.get('status')=='COMPLETE' and matched12.get('matching_evidence',{}).get('scenes')==12146:
        def component_ci(values,metric):
            item=values.get('safety_metrics',{}).get(metric,{})
            interval=item.get('CI95_delta_percentage_points',[])
            return signed(item.get('delta_percentage_points'),5)+('<small>['+', '.join(signed(v,5)for v in interval)+']</small>' if len(interval)==2 else '<small>区间待记录</small>')
        paired12_rows=[]
        for key,value in matched12.get('contrasts',{}).items():
            label=key.replace('_minus_',' − ').replace('_u012000_NAVTEST',' 12k').replace('_u010000_NAVTEST',' 10k')
            paired12_rows.append([esc(label),signed(value.get('delta_PDMS'),5)+'<small>'+ci_text(value)+'</small>',
                component_ci(value,'no_at_fault_collisions'),component_ci(value,'time_to_collision_within_bound'),component_ci(value,'ego_progress')])
        conclusions=matched12.get('conclusions',{})
        body+=detail('12k历史配对：检查10k优势是否持续','<p>同12k BATCH−FULL及各自12k−10k；固定seed、44物理log簇bootstrap2,000次，未作多重比较校正。分量单位为百分点。</p>'+table(['精确对照','ΔPDMS［区间］','ΔNC pp［区间］','ΔTTC pp［区间］','ΔEP pp［区间］'],paired12_rows)+'<p class="note">'+esc(conclusions.get('persistence_zh',''))+' '+esc(conclusions.get('NC_risk_zh',''))+'</p><p>'+link(root/'analysis'/'MATCHED_12K_PPO_CONTRASTS.json','完整12k配对、安全分量区间与来源')+'</p>')
    elif matched12:
        body+='<p class="muted">12k有限后处理状态：'+esc(matched12.get('status','待完整结果'))+'；未生成的差值与区间不填入结论，主线训练独立继续。</p>'
    body+='<p>PPO_FULL 1k修复原零分 '+number(paired.get('repaired_zero'))+' 个，新增零分 '+number(paired.get('new_zero'))+' 个。改善场景数量少于小幅下降场景，均值仍明显上升；只数“涨了多少场景”会漏掉严重失败修复的权重。</p>'
    scale=snapshot.get('advantage_normalization',{}).get('arms',{}).get('PPO_FULL',{}).get('raw_to_centered_scale',{}).get('mean')
    batch_disposition=('已决定止损，退出状态见路线表' if 'PPO_BATCHSTD' in decisions else '阶段优势需要重新检验')
    body+='<div class="review-grid"><article><h3>BatchStd：'+batch_disposition+'</h3><p>10k正收益是真实历史好点，予以保留；12k/20k没有持续成同预算总分优势，20k NC/TTC又有负向配对证据。首1k尺度倍率 '+fmt(scale,3)+' 解释更新强度变化，不能替代场景安全—效率信用判断。</p></article><article><h3>SafeComfort：局部突出EP，尚非场景风险权衡</h3><p>对NC=DAC=TTC全过的候选，将训练Comfort置1，以削弱局部奖励干扰、突出EP；其他候选仍用canonical。它没有判断路口等待是否合理、低风险是否应前进或观测是否不确定，不能等同真正的场景条件策略。</p></article><article><h3>CISPO／固定hard-only：按证据止损</h3><p>CISPO20k总分近Full而NC/TTC劣化；固定Hard10k也落后Full且安全分量较弱。停止预算不删除历史checkpoint，也不否定困难场景在新场景条件诊断中的价值；实际退出以收据为准。</p></article></div>'
    body+='<p class="muted">两个新臂更新前的首个fresh collection在全部8rank与PPO_FULL逐值一致；后续rollout会随策略改变。近10k SAFE塑形后的PG有效组约90.5%，FULL为98.9%，奖励重排也改变了对比信号。现有日志仅含总DDP梯度范数，不据scalar loss声称anchor梯度占比。'+link(root/'analysis'/'PROCESS_DIAGNOSTIC_10K.json','近10k过程诊断')+' · '+link(root/'analysis'/'ALGORITHM_ARMS_FIRST_COLLECTION_MATCH.json','起点匹配')+' · '+link(root/'analysis'/'safe_comfort_counterfactual_u001000.json','奖励反事实与风险')+'</p>'
    if 'PPO_ANCHOR40' in runs:
        body+='<div class="note"><b>已新增PPO_ANCHOR40：</b>使用NFT释放的8卡研究槽，从同一IL70、seed42与全量数据开始，仅β50→40，保留RAWstd，不叠加SAFE或中心化std。目的是检验PG与anchor压力的近似再平衡；A cap5与Adam使它不具严格等价变换。近10k PPO阻塞仅约2.4%–2.6%，梯度clip1未触发，因此当前不优先Clip-Higher。'+link(root/'EXPERIMENT_PPO_ANCHOR40_REGISTRATION.json','正式注册与科学边界')+'</div>'
    chapter('04','ppo-diagnosis','PPO路线诊断：安全—效率变化从何而来','先识别场景与可用信息，再解释局部奖励、更新尺度和总分的关系。',body)

    # 05: NFT gets its own scale and decision trail.
    nft_series={'NFT_FULL':[(update(r),r['PDMS']) for r in test if r.get('arm')=='NFT_FULL']}
    nft_curve=curve(root,h,'research_nft_test','NFT完整NAVTEST：没有持续增益',nft_series,baseline=base,
        caption='单独的纵轴展示小幅波动；不与PPO曲线混用纵轴，也不将训练候选分数当部署成绩。')
    nft_metrics={m:[(update(r),100*r.get('means',{}).get(m,0)) for r in test if r.get('arm')=='NFT_FULL']for m in metrics[:4]}
    def nft_components_draw(plt):
        fig,ax=plt.subplots(figsize=(7,3.9),layout='constrained')
        start=points.get(('NFT_FULL',5000))
        for m,label,color in zip(metrics[:4],['NC','DAC','TTC','EP'],['#2563eb','#059669','#dc2626','#d97706']):
            if start:
                ps=sorted((u,y-100*start['means'][m])for u,y in nft_metrics[m]if u>=5000)
                ax.plot([u/1000 for u,y in ps],[y for u,y in ps],'-o',label=label,color=color,ms=4,lw=2)
        ax.axhline(0,color='#64748b',lw=1);ax.set(xlabel='RL updates (thousands)',ylabel='Percentage points vs NFT 5k');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,ncol=4)
        return fig
    nft_components=figure(root,h,'research_nft_components','NFT后续变化：安全走弱，EP略升',nft_components_draw,'不同分量的此消彼长没有转化为完整canonical PDMS的持续提高。')
    body='<div class="plots-two">'+nft_curve+nft_components+'</div>'
    body+='<div class="finding stop"><h3>'+esc(nft_state)+'</h3><p>停止依据是四个完整预算点及相对PPO的机会成本，不是把1k的−0.0301点误判为显著下降。1k父模型配对区间跨零；扩到20k仍无持续收益，足以作本轮资源分配决策。该判断针对当前NFT实现／基座／数据组合，不否定整个算法族。</p></div>'
    nft_pairs=[]
    for key,value in editorial.get('nft_comparisons',{}).items():
        nft_pairs.append([esc(value.get('target','')+' − '+value.get('baseline','')),signed(value.get('delta_PDMS')),ci_text(value),verdict(value)])
    if nft_pairs:body+=table(['NFT完整配对','PDMS点差','95%区间','证据'],nft_pairs)
    body+='<p>首1k奖励同分组：NFT 38.7%，PPO_FULL 0.9%。这说明组内奖励对比信号较少；NFT同分时仍可能有EMA回拉梯度，不称总梯度为零。目标、EMA、采样器同时不同，尚不能归因于一个因素。</p>'
    body+='<p>'+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'NFT_FULL.json','停止决策')+' · '+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'NFT_FULL.STOPPED.json','已停执行收据')+' · '+link(root/'analysis'/'NFT_FUTILITY_REVIEW_20261030.json','完整停止依据')+' · '+link(root/'analysis'/'first1000_reward_signal.json','组内信号诊断')+'</p>'
    if stopped:body+='<p class="note">已完成协作停止：最终原生seal '+number(stopped.get('checkpoint',{}).get('updates'))+' / inner'+number(stopped.get('checkpoint',{}).get('next_inner'))+'，RESULT0与精确transport退出已记录；保留EMA、optimizer与RNG，释放8卡研究slot。停止不是模型失败。</p>'
    chapter('05','nft','NFT：停止依据与保留的研究结论','四个完整预算点、安全分量和机会成本共同支持停止当前实现。',body)

    # 06: the learning process and next experiments.
    panel_series={a:[(update(r),r['PDMS']) for r in panel if r.get('arm')==a]for a in ['PPO_ADAPTIVE','PPO_FULL','PPO_HARD']}
    panel_chart=curve(root,h,'research_train_panel','固定TRAIN1024：反馈信号，不是泛化分数',panel_series,baseline=panel_base,
        caption='103,288全量臂中这些场景属于训练人口；panel固定物理log分层，不等于全TRAIN加权均值。')
    data_series={a:[(update(r),r['PDMS'])for r in test if r.get('arm')==a]for a in ['PPO_FULL','PPO_HARD','PPO_ADAPTIVE']}
    data_chart=curve(root,h,'research_data_navtest','数据路线的完整NAVTEST',data_series,caption='只连接已完整阶段，未完成阶段不补点；纵轴放大显示数据路线差异。')
    adaptive_tenk=points.get(('PPO_ADAPTIVE',10000))
    adaptive_sentence=('Adaptive的10k完整Test为'+fmt(adaptive_tenk['PDMS'],6)+'；' if adaptive_tenk else 'Adaptive的5k Test较Full高0.157064点，但10k尚未完整；')
    body='<h3>新的研究主线：场景条件安全—驾驶效率</h3><p>先审计当前观测能支持哪些风险与交互判断，再定义场景分组、表征诊断与策略判据。不能仅凭低EP把等待判为低效，也不能把所有停滞都解释为安全；已确认风险与信息不足须分开。</p>'
    body+='<h3>当前要回答的三个问题</h3>'+table(['问题','需要的证据','对应方法与判断边界'],[
        ['高进度但冒险的轨迹，为何仍得到正向信用？','同一TRAIN场景：通过指标的替代轨迹、冒险候选、EP差、实际优势与anchor贡献','构建安全—效率冲突偏好候选；必要等待与无故停滞分开，不能把高EP自动当优选。'],
        ['零分是否总是同一批场景？','同一路线跨checkpoint的zero交集／并集、持续率、修复与新生；同时比较不同路线','区分persistent与rotating失败；先对齐完整人口和推理协议，再决定知识补齐、覆盖或稳定性改进。'],
        ['rollout已经采到好轨迹，却没学进去，还是根本没采到？','每组8候选的通过率、最好候选、优势正负、全失败／同分组和更新后部署表现','拆开候选覆盖与信用分配；全同分PG信号不足不等于总梯度为零，IL anchor仍可更新。']])
    preference=editorial.get('preference_research',{});longitudinal=editorial.get('zero_longitudinal',{})
    if preference:
        summary=preference.get('summary',{});counts=summary.get('pair_counts_by_reason',{})
        body+='<h3>TRAIN离线偏好数据与CPU代理检查</h3><p>产物证据时间 '+esc(preference.get('utc','待记录'))+'；保存时状态 '+esc(preference.get('status','已发布离线研究证据'))+'，当前训练执行看下方独立登记。既有 '+number(summary.get('unique_scenes'))+' 个共同TRAIN场景、'+number(summary.get('groups'))+'组／'+number(summary.get('candidates'))+'候选，构造 '+number(summary.get('pairs'))+' 对；构建过程不新增rollout或评分，不把历史pair混入on-policy GRPO优势。'+link(preference.get('manifest'),'离线偏好manifest')+'</p>'
        body+=table(['偏好类型','已选pairs','方向与限制'],[
            ['通过三项指标优于未通过',number(counts.get('pass_over_fail')),'只表示NC=DAC=TTC指标条件优先；不是已知路权／必要等待标签。'],
            ['通过候选内部比较效率',number(counts.get('feasible_efficiency')),'EP差≥0.05，raw progress更高且Comfort不差；不把所有低EP或停车判错。'],
            ['整个pool无通过见证时较小deficit',number(counts.get('allfail_deficit')),'混合pool的fail-fail不放此类；要求安全分量逐项不更差，交叉优劣排除。指标缺失成本不是事故严重程度，也不是安全解。']])
        body+='<p>标记的安全—效率取舍 '+number(summary.get('selected_safety_efficiency_tradeoffs'))+' 对；'+number(summary.get('labels_conflicting_with_canonical_score'))+' 对标签方向与canonical总分相反。其余pair也可显式表达安全与条件效率，不代表大多数总分标签错误。TRAIN身份已核，NAVTEST重叠为0。</p>'
        rollout_rows=[]
        for arm,node in summary.get('counts_by_arm',{}).items():
            rollout_rows.append([esc(arm),number(node.get('groups')),number(node.get('scalar_canonical_tie_groups')),number(node.get('actual_zero_PG_groups')),number(node.get('unsafe_positive_with_feasible_alternative')),number(node.get('zero_PG_groups_with_cross_behavior_preference_witness'))])
        if rollout_rows:
            body+=table(['近10k既有TRAIN窗口','组数','canonical同分组','算法实际PG=0组','unsafe正A且有通过替代：候选数','PG=0组有历史跨behavior偏好证据'],rollout_rows)+'<p class="note">SafetyEP在该窗口消除了“unsafe正优势且有通过替代”的候选，但正式10k NC仍下降，说明训练信用修正不等于部署场景偏好已解决。算法PG=0组、canonical同分组与部署zero场景是三种口径；跨behavior证据只支持合法离线样本，不把旧轨迹当新on-policy样本。</p>'
        tie=preference.get('zero_tie_diagnosis',{})
        if tie.get('by_arm'):
            tie_rows=[]
            for arm,node in tie['by_arm'].items():
                tie_rows.append([esc(arm),number(node.get('zero_PG_groups')),number(node.get('zero_PG_all3pass_and_all_EP1')),number(node.get('zero_PG_allfail')),number(node.get('zero_PG_allfail_has_own_deficit_direction')),number(node.get('zero_PG_with_other_policy_pass_witness'))])
            body+='<h3>PG=0的关键拆分：指标饱和成功与全失败不是一回事</h3>'+table(['相同近10k窗口','PG=0组','全8候选三安全通过且EP=1','全8候选均未通过','全失败但本组deficit有方向','别的behavior有通过见证'],tie_rows)+'<p>SafetyEP的100个PG=0组全部属于三指标与EP饱和成功，不是100个失败／停滞场景；没有更高安全EP替代时，不应为抹平zero比例强造梯度。Full的3个全失败PG=0组中，2组仍有指标deficit方向，可研究结构化信用；所有候选同败同cost时则没有数据内相对方向。这里只是窗口诊断，不是persistent总体结论。'+link(tie.get('path'),'逐组可辨识性诊断')+'</p>'
        recommendations=preference.get('algorithm_recommendations',[])
        if recommendations:
            body+=table(['准备研究的方法','优先级','实现／证据边界'],[[esc(node.get('name')),number(node.get('priority')),esc(node.get('scope',''))]for node in recommendations])
        body+='<p>CPU prototype已检查梯度方向与clean-x／CFM恒等关系；chain logscore或CFM负误差仍是代理分数，不是精确端点DPO likelihood。'+esc(preference.get('limits_zh',''))+' '+link(editorial.get('preference_research_path'),'偏好研究、CPU检查与实施边界')+'</p>'+detail('偏好样本、CPU原型与完整证据','<pre>'+esc(json.dumps(preference,ensure_ascii=False,indent=2))+'</pre>')
    else:body+='<p class="muted">TRAIN安全—效率冲突偏好manifest与CPU候选方案正在准备；尚不声明偏好训练已注册、已上线或已提分。NAVTEST场景不直接回流为训练名单。</p>'
    if preference_study:
        eligible=preference_study.get('preference_scenes');source_scenes=preference.get('summary',{}).get('unique_scenes')
        no_pair=source_scenes-eligible if h.finite(source_scenes) and h.finite(eligible) else None
        body+='<h3>两臂5k偏好研究：分离采样效应与辅助损失</h3><p>原 '+number(source_scenes)+' 个源TRAIN场景中，'+number(eligible)+' 个有合法偏好pair，另 '+number(no_pair)+' 个无pair，不计入优先池。两臂共同采用静态80%全量103,288＋20%有效偏好场景采样；无pair场景仍可从全量分量采到。固定IL70、seed42、sampler和canonical部署，不把历史pair混进on-policy PG优势。</p>'
        study_rows=[]
        for spec in preference_study.get('configs',[]):
            arm=spec.get('arm');run=runs.get(arm,{})
            status=run.get('owner_receipt',{}).get('status') or run.get('status','REGISTERED')
            if arm in stop_receipts:state='已有停止收据，见路线表'
            elif status=='PILOT_COMPLETE_REVIEW_REQUIRED':state='首轮预算完成，等待复核'
            elif run.get('journal_ranks')==run.get('expected_ranks',8) and h.finite(run.get('updates')) and run['updates']>0:state='全rank原生记录 '+number(run['updates'])+' 更新'
            elif arm in pref_children_started and not preference_receipts[arm].get('RESULT.json'):state='native child已启动（RUNNING）；优化更新待原生日志'
            elif run.get('job_receipts',{}).get('RUNNING.json'):state='worker已启动，原生更新待日志'
            elif run.get('job'):state='已派发，启动／更新待回执'
            else:state='已登记，尚无派发／更新证据'
            treatment='仅改变训练场景分布；PG与anchor保留Full' if arm=='PPO_PREF_REPLAY' else '相同采样＋离线CFM pair代理辅助；coef0.001／beta0.1'
            contrast_design='对Full隔离采样作用' if arm=='PPO_PREF_REPLAY' else '对Replay隔离辅助损失；对Full看完整方法'
            latest_result=latest.get(arm)
            result_cell=fmt(latest_result['PDMS'],6)+' / '+number(update(latest_result))+'更新完整Test' if latest_result else '暂无完整成绩'
            study_rows.append([esc(arm),treatment,contrast_design,number(spec.get('research_budget_updates'))+' cap<small>1k／5k复核；scheduler '+number(spec.get('native_scheduler_max_updates'))+'不是当前研究预算</small>',esc(state),result_cell,link(spec.get('path'),'配置')])
        body+=table(['研究臂','主要变化','可归因比较','首轮预算','实际执行状态','成绩','来源'],study_rows)+'<p>这是候选指标条件偏好监督，不是精确端点DPO，也没有新增红灯／让行真值。成功饱和组跳过aux；PG mask为0时若pref_aux_enabled且weighted_loss有效，仍可能有辅助梯度，legacy的anchor_only_updates_local不能单独解释成纯anchor。'+link(root/'PREFERENCE_MIXTURE_STUDY_REGISTRATION.json','两臂共同采样与预注册比较')+'</p>'
        deployment=editorial.get('preference_deployment',{})
        if deployment:
            deploy_rows=[]
            for arm,folder in deployment.get('training',{}).items():
                receipt=preference_receipts.get(arm,{}).get('RUNNING.json',{})
                deploy_rows.append([esc(arm),esc(receipt.get('host','待回执')),number(len(receipt.get('gpus',[]))),number(receipt.get('child_pid')),link(str(Path(folder)/'RUNNING.json'),'真实child启动')])
            body+=detail('已启动的两臂：来源与原生child收据',table(['研究臂','实际主机','GPU数','native child PID','来源'],deploy_rows)+'<p>科学snapshot '+esc(deployment.get('source','待记录'))+' 独立封存；旧GPU child不因此重启。'+link(root/'analysis'/'PREFERENCE_STUDY_DEPLOYMENT_20261031.json','实际派发与probe owner收据')+'</p>')
        mixture_review=editorial.get('preference_mixture_review',{})
        completed=[(key,row)for key,row in mixture_review.get('contrasts',{}).items()if row.get('status')=='COMPLETE']
        if completed:
            review_rows=[]
            for key,row in completed:
                deltas=[]
                for metric in ['no_at_fault_collisions','time_to_collision_within_bound','ego_progress']:
                    m=row.get('safety_metrics',{}).get(metric,{});ci=m.get('CI95_delta_percentage_points',[])
                    deltas.append(signed(m.get('delta_percentage_points'),5)+'<small>['+', '.join(signed(v,5)for v in ci)+']</small>')
                zeros=row.get('zero_transitions',row.get('paired',{}))
                review_rows.append([esc(key.replace('_NAVTEST','').replace('_minus_',' − '))+'<small>'+esc(row.get('comparison_question_zh',''))+'</small>',signed(row.get('delta_PDMS'),6)+'<small>'+ci_text(row)+'</small>']+deltas+[number(zeros.get('repaired_zero'))+' / '+number(zeros.get('new_zero'))])
            body+='<h3>偏好研究：已完整配对 '+number(len(completed))+' / '+number(mixture_review.get('target_pairs',6))+'</h3>'+table(['匹配对照与归因问题','ΔPDMS［95%区间］','ΔNC pp［区间］','ΔTTC pp［区间］','ΔEP pp［区间］','zero修复／新增'],review_rows)+'<p>'+link(root/'analysis'/'PREFERENCE_MIXTURE_MATCHED_REVIEW.json','完整人口配对及组件区间')+'</p>'
        else:body+='<p class="muted">偏好研究尚无已完整匹配对照，不填分数／区间。未来1k与5k各三组比较逐对发布，任一组双方完整即可展示，不必等待全部六组。</p>'
        auxiliary=mixture_review.get('native_auxiliary_diagnostics',{})
        if auxiliary:
            aux_rows=[]
            for key,node in auxiliary.items():
                active=node.get('activation',{});logit=node.get('active_only_logit',{});loss=node.get('active_only_weighted_loss',{})
                aux_rows.append([esc(key),esc(node.get('status','UNKNOWN')),number(node.get('selected_rank_update_rows'))+' / '+number(node.get('expected_rank_update_rows')),number(active.get('known_flag_rows')),number(active.get('active_rows')),fmt(active.get('active_fraction'),2,percent=True)+'%',fmt(logit.get('mean'),6),fmt(loss.get('mean'),6),'UNKNOWN'])
            body+=detail('CFM原生aux激活诊断：只读一次有界窗口',table(['正式checkpoint','诊断状态','捕获rank-update行','有激活flag行','active行','active／已知行','active logit均值','active weighted loss均值','独立aux梯度幅度'],aux_rows)+'<p>每rank最多2MiB、checkpoint末64更新，非训练全程覆盖；缺失字段保留UNKNOWN。零logit仍可有非零梯度，loss标量不能代替参数梯度幅度；Replay为N/A。</p>')
        else:body+='<p class="muted">CFM aux原生激活／量级诊断等待正式checkpoint；未知不按0或未激活处理。Replay不适用。若无收益，只能评价固定beta0.1／coef0.001配方，不能据弱注入或缺字段否定整条偏好路线。</p>'
        if mixture_review.get('auxiliary_interpretation_guard_zh'):body+='<p class="muted">'+esc(mixture_review['auxiliary_interpretation_guard_zh'])+'</p>'
        primary=editorial.get('primary_research_links',{})
        if primary.get('papers'):
            body+='<h3>三篇primary文献：关联与迁移边界</h3>'+table(['原论文','可借鉴的机制','与本项目的关系／研究判断'],[[link(node.get('url'),node.get('name','论文'))+(' · '+link(node['pdf_url'],'官方PDF') if node.get('pdf_url') else ''),esc(node.get('finding_zh','')),esc(node.get('relevance_zh',''))]for node in primary['papers']])+'<p class="muted">文献用于提出和约束实验假设，不宣称当前实现复现了这些方法，也不直接比较跨论文分数。</p>'
    if longitudinal:
        body+='<h3>纵向zero：同一难点与轮换失败分开统计</h3><p>状态 '+esc(longitudinal.get('status','已发布诊断证据'))+'；证据时间 '+esc(longitudinal.get('at',longitudinal.get('utc','待记录')))+'。persistent只表示在表中列出的观测点全部为零，不代表未观测更新也始终失败；不同路线的观测点数与预算不同，不能直接按持续zero数排算法。</p>'
        nav_zero=longitudinal.get('NAVTEST_diagnostic_only',{});nav_series=nav_zero.get('series',{})
        zero_rows=[];transition_rows=[];attribution_rows=[]
        for arm,node in nav_series.items():
            sets=node.get('sets',{});observed=node.get('observed_updates',[])
            zero_rows.append([esc(arm),esc('／'.join(str(u//1000)+'k'for u in observed)),number(sets.get('latest_zero')),number(sets.get('persistent_zero_all_observed')),number(node.get('persistent_RL_only_zero_count')),number(sets.get('intermittent_zero')),number(sets.get('relapsed_after_observed_recovery'))])
            if node.get('adjacent_transitions'):
                t=node['adjacent_transitions'][-1]
                transition_rows.append([esc(arm),number(t.get('from_updates'))+'→'+number(t.get('to_updates')),number(t.get('stayed_zero')),number(t.get('new_zero')),number(t.get('recovered')),fmt(t.get('zero_jaccard'),2,percent=True)+'%'])
            if node.get('checkpoints'):
                cp=node['checkpoints'][-1];a=cp.get('zero_attribution',{})
                attribution_rows.append([esc(arm),number(cp.get('updates')),number(a.get('NC0_only')),number(a.get('DAC0_only')),number(a.get('NC0_and_DAC0')),number(a.get('EP0_among_zero'))])
        if zero_rows:
            body+=table(['完整NAVTEST路线','观测更新（0=IL）','最新zero','全部观测点均zero：含IL','全部RL点均zero：不含IL','曾zero但非全部点','观察到修复后再zero'],zero_rows)
            body+='<p class="note">部署zero是固定seed单候选ODE的canonical零分，不等于训练G8全部失败，也不等于算法PG=0。NAVTEST只作诊断，不导出token名单到训练；持续失败仍需结合TRAIN合法候选与当前可见信息判断原因。</p>'
            full_sets=nav_series.get('PPO_FULL',{}).get('sets',{})
            if full_sets:
                body+='<p><b>观测结论：固定难点与轮换失败同时存在。</b>Full的全部已观测点中，曾zero '+number(full_sets.get('ever_zero'))+' 场景，其中 '+number(full_sets.get('persistent_zero_all_observed'))+' 在全部点都zero、'+number(full_sets.get('intermittent_zero'))+' 为间歇zero；最新仍zero '+number(full_sets.get('latest_zero'))+'，并非始终只有同一批。此分类不能直接归因于知识缺口、采样不足或训练不稳定，需与合法TRAIN rollout的通过见证和实际优势关联。</p>'
        cross=nav_zero.get('cross_arm10k',{})
        if cross:body+='<p>同10k跨 '+number(len(cross.get('arms',[])))+' 条路线，全部路线共同zero '+number(cross.get('common_zero_all_arms'))+' 场景，至少一路zero的并集 '+number(cross.get('zero_union_any_arm'))+'。这是同预算场景交集，不能据此断言不可学习或IL知识必然缺失。</p>'
        if transition_rows:body+=detail('最近相邻观测：失败修复与新生是否抵消',table(['路线','相邻更新','仍zero','新zero','已修复','zero集合Jaccard'],transition_rows))
        if attribution_rows:body+=detail('最新部署zero的指标归因',table(['路线','更新','仅NC0','仅DAC0','NC0且DAC0','zero中EP0：可重叠'],attribution_rows)+'<p>EP0可能与安全失败重叠，不能相加；canonical零分不自动等于低效停车。</p>')
        train_series=longitudinal.get('TRAIN1024',{}).get('series',{})
        if train_series:
            train_rows=[]
            for arm,node in train_series.items():
                sets=node.get('sets',{})
                train_rows.append([esc(arm),number(len(node.get('observed_updates',[]))),number(sets.get('latest_zero')),number(sets.get('persistent_zero_all_observed')),number(sets.get('intermittent_zero')),number(sets.get('relapsed_after_observed_recovery'))])
            body+=detail('训练内固定1024 panel的纵向zero（与Test隔开）',table(['TRAIN1024路线','观测点数：含IL','最新zero','全部点zero','间歇zero','修复后复发'],train_rows)+'<p>TRAIN可反馈，旧DEV对全量训练臂不是heldout；不把其持续率当完整NAVTEST结论。</p>')
        body+='<p>rollout／优势关联状态：'+esc(longitudinal.get('rollout_analysis_status','见原始诊断'))+'。'+link(editorial.get('zero_longitudinal_path'),'完整纵向集合、分层与rollout诊断')+'</p>'
        rollout=longitudinal.get('TRAIN_rollout_advantage_diagnosis',{})
        if rollout.get('status')=='COMPLETE':
            history_rows=[];post_rows=[]
            for arm,node in rollout.get('arms',{}).items():
                cohort=node.get('cohorts',{}).get('persistent_ODE_zero',{});c=cohort.get('counts',{})
                history_rows.append([esc(arm),number(cohort.get('tokens')),number(cohort.get('tokens_ever_resampled')),number(cohort.get('tokens_no_rollout_observed_unknown')),number(cohort.get('groups')),number(c.get('at_least_one_feasible')),number(c.get('has_positive_actual_A')),number(c.get('canonical_all_zero')),number(c.get('zero_PG'))])
                post=node.get('post_ODE_zero_windows',{})
                post_rows.append([esc(arm),number(post.get('windows')),number(post.get('observed_G8')),number(post.get('unknown_not_resampled'))])
            body+='<h3>持续ODE零分不等于没有训练机会或优势</h3>'+table(['TRAIN固定panel路线','持续ODE0 token','实际再采到token','未见rollout：unknown','历史G8组','至少1可行候选组','有实际正A组','canonical全0组','PG=0组'],history_rows)
            ad=rollout.get('arms',{}).get('PPO_ADAPTIVE',{}).get('cohorts',{}).get('persistent_ODE_zero',{});ac=ad.get('counts',{})
            if ad:
                body+='<p class="note">ADAPT的 '+number(ad.get('tokens'))+' 个持续ODE零分TRAIN场景，在 '+number(ad.get('groups'))+' 组历史G8中，'+number(ac.get('at_least_one_feasible'))+' 组至少一条可行，'+number(ac.get('has_positive_actual_A'))+' 组有正优势，仅 '+number(ac.get('canonical_all_zero'))+' 组canonical全0。它们跨theta、跨历史训练时间，历史数据本身不能证明sampler原因；固定checkpoint结论见下方实际配对探针。</p>'
            body+='<p>未再采到必须保留unknown，不能记为G8持续全失败。此表是TRAIN选定cohort的历史组统计，与前面近10k固定1000组的成功饱和诊断不是同一总体；也不把best-of8可行性当单候选部署分数。</p>'
            ad_low=rollout.get('arms',{}).get('PPO_ADAPTIVE',{}).get('cohorts',{}).get('ever_ODE_low50',{}).get('counts',{})
            if ad_low:body+='<p>ADAPT选定零／低分cohort中的 '+number(ad_low.get('canonical_all_zero'))+' 个全零G8组，有 '+number(ad_low.get('all_zero_cost_varied'))+' 组指标deficit不同、'+number(ad_low.get('all_zero_first_infraction_time_varied'))+' 组首次违规时间不同；标量乘零会遮蔽这些方向。实际SafetyEP已采用kappa0的cost-centered全失败fallback，首次违规时刻尚未进入现有优势，不热改封存训练。</p>'
            initial=rollout.get('initial_state_zero_group_counts',{});checked=sum(n.get('all_canonical0_groups',0) for n in initial.values() if isinstance(n,dict))
            if initial:body+='<p>所选 '+number(checked)+' 组全零G8的初始状态审计，未见全候选t=0碰撞／不可行驶区问题；这是该cohort证据，不能外推为全量训练不存在初始状态问题。'+link(root/'analysis'/'ZERO_LOW_SCORE_ROLLOUT_DIAGNOSIS.json','历史rollout、实际优势与初始状态审计')+'</p>'
            body+=detail('ODE零分之后的训练窗口：未再采样单列unknown',table(['路线','ODE零分窗口','之后有G8窗口','未再采到：unknown'],post_rows))
    failure=editorial.get('failure_guidance',{})
    if failure:
        fs=failure.get('evidence',{}).get('summary',{})
        body+='<h3>失败密集信号：仅条件提案，尚未部署</h3><p>这 '+number(fs.get('groups'))+' 组只来自 '+number(fs.get('unique_TRAIN_tokens'))+' 个TRAIN场景，其中 '+number(fs.get('dominant_token_groups'))+' 组来自同一场景。'+number(fs.get('cost_tied'))+' 个成本tie组中 '+number(fs.get('tied_cost_ND_occupancy_varied'))+' 组保留越界时间占比差异，是历史局部可辨识性，不是独立场景发生率，也不证明当前最佳SafetyEP的主瓶颈是dense信号不足。设计状态 '+esc(failure.get('status','提案待验证'))+'；当前优先偏好CFM与固定theta探针，不新增dense训练成绩。'+link(root/'analysis'/'FAILURE_GUIDANCE_DESIGN_20261031.json','条件设计与启动判据')+' · '+link(root/'analysis'/'FAILURE_GUIDANCE_FIELD_AUDIT_20261031.json','独立字段审计')+'</p>'
    probe=editorial.get('fixed_theta_probe',{});probe_registration=editorial.get('fixed_theta_probe_registration',{})
    if probe:
        body+='<h3>固定theta ODE8／SDE8探针：选定场景的受控结果</h3><p>状态 '+esc(probe.get('status','见结果收据'))+'；同 '+number(probe_registration.get('TRAIN_unique_scenes'))+' 个选定TRAIN场景，各模型固定checkpoint、8个相同初始root、前7步bit-identical，仅后段SDE改变。G8均值／通过数不是best-of8部署成绩。'+link(root/'analysis'/'FIXED_THETA_ODE_SDE_ZERO_PROBE.json','固定theta完整诊断JSON')+'</p>'
        probe_rows=[];probe_complete=[]
        for arm,node in probe.get('results',{}).items():
            if node.get('status')!='COMPLETE' or node.get('count')!=probe_registration.get('TRAIN_unique_scenes') or len(node.get('scenes',[]))!=node.get('count'):continue
            ode=node.get('ODE8',{});sde=node.get('SDE8',{});probe_complete.append((arm,node))
            mode_rows=[ode,sde]+([noise003_result['results'][arm]['SDE8']]if noise003_matched else [])
            probe_rows.append([esc(arm)+'<small>固定 '+number(node.get('updates'))+' 更新</small>',number(ode.get('candidate_count')),' → '.join(number(row.get('feasible'))for row in mode_rows),' → '.join(number(row.get('canonical_zero'))for row in mode_rows),' → '.join(fmt(row.get('mean_PDMS'),4)for row in mode_rows)])
        if probe_rows:
            def probe_draw(plt):
                fig,axes=plt.subplots(1,2,figsize=(10.3,3.7),layout='constrained')
                positions=list(range(len(probe_complete)));labels=[SHORT.get(arm,arm)+'\n'+str(node['updates']//1000)+'k' for arm,node in probe_complete]
                for ax,key,title in zip(axes,['feasible','canonical_zero'],['Feasible candidates','Canonical zero candidates']):
                    modes=[('ODE8',-.26,'#2563eb'),('SDE0.1',0,'#d97706'),('SDE0.03',.26,'#059669')]if noise003_matched else [('ODE8',-.18,'#2563eb'),('SDE0.1',.18,'#d97706')]
                    for mode,shift,color in modes:
                        vals=[(node['ODE8'] if mode=='ODE8' else node['SDE8'] if mode=='SDE0.1' else noise003_result['results'][arm]['SDE8'])[key]for arm,node in probe_complete];bars=ax.bar([i+shift for i in positions],vals,width=.24 if noise003_matched else .34,color=color,label=mode)
                        ax.bar_label(bars,padding=3,fontsize=9)
                    ax.set_xticks(positions,labels);ax.set(ylabel='Candidate count (each arm: 120)',title=title,ylim=(0,max(node['ODE8']['candidate_count']for arm,node in probe_complete)*1.12));ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,fontsize=9)
                return fig
            body+=figure(root,h,'research_fixed_theta_probe','同模型固定theta：可行候选与零分候选并排看',probe_draw,'仅15个selected TRAIN场景、每模型120个候选。'+('完整匹配后并列ODE／SDE0.1／SDE0.03。'if noise003_matched else '当前只画已完整的ODE／SDE0.1；0.03未完整前不补值。')+'三个模型checkpoint不同，不排算法名次；这是候选诊断，不是heldout总分或部署成绩。')
            mode_label='ODE→0.1→0.03'if noise003_matched else'ODE→0.1'
            body+=table(['固定模型','每模式候选数','可行候选：'+mode_label,'零分候选：'+mode_label,'候选均分：'+mode_label+'（诊断）'],probe_rows)+'<p class="note"><b>已确认的0.1局部损害与探索价值分开看：</b>在此selected cohort上，SDE0.1净降低可行数、增加零分，同时有独有修复；不据此全关SDE。0.03只有完整匹配后才入图，局部候选比较不能外推成全TRAIN或heldout收益。</p>'
        diagnosis=editorial.get('fixed_probe_diagnosis',{})
        if diagnosis.get('status')=='COMPLETE':
            lost_repaired=[[esc(arm),number(node.get('counts',{}).get('ODE_feasible_SDE_not')),number(node.get('counts',{}).get('SDE_feasible_ODE_not'))]for arm,node in diagnosis.get('arms',{}).items()]
            witness=diagnosis.get('persistent_shared_TRAIN_zero',{});witness_rows=[]
            for arm,node in witness.get('by_arm',{}).items():
                ode=node.get('ODE8',{});sde=node.get('SDE8',{})
                witness_rows.append([esc(arm),fmt(node.get('canonical_single_ODE_PDMS'),4),number(ode.get('positive_canonical'))+' → '+number(sde.get('positive_canonical')),number(ode.get('strictly_feasible'))+' → '+number(sde.get('strictly_feasible'))])
            body+='<h3>探索的两面：大量损坏已有好候选，也能救出ODE找不到的候选</h3>'+table(['同模型matched roots','ODE可行→SDE不可行：损失','ODE不可行→SDE可行：修复'],lost_repaired)
            body+='<p><b>持续zero的受控反例：</b>共同TRAIN场景 '+esc(witness.get('token','见证据'))+' 在三个固定checkpoint中，单候选ODE为0、ODE8也全部0，SDE8却有正分及严格可行轨迹。它直接否定“部署zero就没有rollout信号”，也说明不能只按局部均值取消所有随机探索。</p>'+table(['同一TRAIN场景，各自固定checkpoint','原单候选ODE','正分候选：ODE8→SDE8','严格可行：ODE8→SDE8'],witness_rows)+'<p>冻结公式的优势非零这里只是诊断计算，没有新optimizer更新；采到好候选能否转成部署收益，仍是偏好／endpoint／信用分配的待检问题。0.03有限对照只增加360条SDE候选并复用360条ODE与45条canonical基线，执行／结果状态单列，不扩写成新长训练或提分结论。'+link(root/'analysis'/'FIXED_THETA_ZERO_PROBE_DIAGNOSIS_20261031.json','逐scene分类、45对root及前7步一致证据')+'</p>'
    if editorial.get('noise003_deployment'):
        noise_result=editorial.get('noise003_result',{})
        noise_status=str(noise_result.get('status',''));noise_failed='FAIL' in noise_status;fix=editorial.get('noise003_scorer_fix',{})
        attempts=editorial.get('noise003_attempts',{});attempt_states=[str(node.get('status',''))for node in attempts.values()]
        if noise003_matched:noise_title='0.03有限诊断已完整：同图三模式比较'
        elif any('FAIL' in state for state in attempt_states):noise_title='0.03有限诊断：修复后新尝试存在运行失败，待复核'
        elif any(state=='RUNNING'for state in attempt_states) or noise_status.startswith('RUNNING') and not attempt_states:noise_title='0.03有限诊断：修复后重试运行中，完整前不补值'
        elif noise_failed and fix.get('status')=='PASS':noise_title='0.03首轮失败已通过CPU修复，重试运行待最新回执'
        elif noise_failed:noise_title='0.03有限诊断：首轮运行失败待修复'
        elif noise_status=='COMPLETE':noise_title='0.03结果完整匹配待核查，暂不发布三模式统计'
        else:noise_title='0.03有限诊断已派发，完整结果待回执'
        body+='<h3>'+noise_title+'</h3><p>仅新生成／评分360条SDE候选；已有ODE360与canonical45复用，不把配对观察总数720算作新增GPU候选。'+('路径规范化修复CPU复核PASS，实际scorer import／protocol与0.1一致；首次失败是初始化问题，不是算法结果。'+link(root/'analysis'/'NOISE003_SCORER_PATH_CPU_FIX_REVIEW_20261031.json','CPU路径修复复核') if fix.get('status')=='PASS' else '')+' 当前收据状态 '+esc(noise_status or '待结果')+'，'+link(root/'analysis'/'FIXED_THETA_NOISE003_ZERO_PROBE.json','0.03独立结果JSON')+'。只有三模型均完整且checkpoint／scene集合／复用ODE计数匹配后，才把0.03加入原并排图；不是heldout总分，也不跨不同checkpoint排算法。正式PPO_NOISE03的5k recipe与本诊断分开，实际执行看独立登记。'+link(root/'analysis'/'NOISE003_PROBE_DEPLOYMENT_20261031.json','实际派发收据')+'</p>'
        failure_archive=next((root/'analysis').glob('FIXED_THETA_NOISE003_ZERO_PROBE*ATTEMPT001*FAILURE*.json'),None)
        if failure_archive:body+='<p class="muted">'+link(failure_archive,'首次失败永久归档')+'；原0.1完整结果保持独立。</p>'
        if attempts:body+=detail('最新尝试的真实worker状态：RESULT优先于旧RUNNING',table(['模型','最新attempt','实际状态','来源'],[[esc(arm),esc(node.get('attempt')),esc(node.get('status')),link(node.get('job'),'attempt收据')]for arm,node in attempts.items()]))
        formal=editorial.get('noise003_registration',{})
        if formal:
            formal_run=runs.get('PPO_NOISE03',{});formal_started=formal_run.get('worker_receipt_status')=='RUNNING'
            formal_failed=formal_run.get('worker_terminal_receipt') and h.finite(formal_run.get('worker_returncode')) and formal_run['worker_returncode']!=0
            formal_state='启动阶段失败待修复，未形成科学训练结果'if formal_failed else 'native child已启动，实际更新见路线表'if formal_started else '已登记／派发，尚无当前native RUNNING证明'
            body+='<p><b>正式PPO_NOISE03：</b>'+formal_state+'；首轮cap '+number(formal.get('research_budget_updates'))+'，1k／5k完整复核。此为canonical PPO仅改eta0.1→0.03，mean与diffusion联动；有限15场景诊断不是这条训练路线的泛化成绩。'+link(root/'EXPERIMENT_PPO_NOISE03_REGISTRATION.json','正式5k独立登记')+'</p>'
    elif editorial.get('preference_deployment',{}).get('probes'):
        jobs=editorial['preference_deployment']['probes'];owner=editorial.get('probe_owner_state',{})
        body+='<h3>固定theta ODE8／SDE8探针：三个任务已派发</h3><p>同 '+number(probe_registration.get('TRAIN_unique_scenes'))+' 个TRAIN场景，'+number(probe_registration.get('total_diagnostic_candidates'))+' 个诊断候选；部署在vlawm-zt2 GPU0／1／2，GPU6仍排除。CPU分析owner状态 '+esc(owner.get('status','见收据'))+'；暂无实际结果JSON，不据历史跨theta差异宣布sampler原因。</p>'+table(['固定模型','实际派发主机','GPU','worker证据'],[[esc(job.get('arm')),esc(job.get('host')),number(job.get('gpu')),link(job.get('job'),'实际任务')]for job in jobs])
    elif probe_registration:
        body+='<p class="muted">固定theta ODE8／SDE8探针方案已登记：'+number(probe_registration.get('TRAIN_unique_scenes'))+' 个共同TRAIN场景、'+number(probe_registration.get('total_diagnostic_candidates'))+' 个候选；当前尚无实际结果JSON，不推断采样器优劣。'+link(root/'data'/'probes'/'ZERO_ODE8_SDE8_FIXED_THETA_20261031'/'REGISTRATION.json','有限探针登记')+'</p>'
    if not longitudinal:body+='<p class="muted">纵向zero集合与rollout信号分析正在复用已保存证据；persistent／rotating比例尚待对齐诊断，不先把单阶段零分率当作持续失败。</p>'
    counterfactual=editorial.get('last_step_counterfactual',{})
    if counterfactual.get('status')=='COMPLETE':
        cf_rows=[]
        for arm,node in counterfactual.get('arms',{}).items():
            modes=node.get('modes',{});selected=[modes.get(key,{})for key in ['SDE01','mu9_reconstructed','det9_reconstructed']]
            cf_rows.append([esc(arm),' → '.join(number(row.get('strictly_feasible'))for row in selected),' → '.join(number(row.get('canonical_zero'))for row in selected),' → '.join(fmt(row.get('PDMS'),4)for row in selected)])
        body+='<h3>另一个证据层次：末步CPU重构反事实，非新policy rollout</h3><p>'+number(counterfactual.get('new_CPU_official_scores'))+' 次官方CPU重评分、'+number(counterfactual.get('new_model_forwards'))+' 次新模型前向。mu9重构去末innovation但保留drift；det9重构同时去末drift与innovation，保留先前探索前缀；FP32重构误差界保存在来源中。它们不是实际0.03 rollout、不是NAVTEST提升。</p>'+detail('真实SDE0.1 → mu9重构 → det9重构（仅选定TRAIN候选）',table(['各自固定模型','严格可行数','canonical零分数','候选均分：非部署'],cf_rows))
        mechanism=counterfactual.get('terminal_mechanism',{});std=mechanism.get('actual_last_noise_per_coordinate_std',[])
        if len(std)==3:body+='<p>末步实际物理noise std约x '+fmt(std[0],2)+' m、y '+fmt(std[1],2)+' m、heading '+fmt(mechanism.get('std_heading_degrees'),2)+'°，按future pose独立，之后没有model去噪。requested轨迹差分粗糙度不是实际车辆加速度。</p>'
        body+='<p class="note">重构后的局部均值／可行性改善不等于可以取消探索：持续zero场景7af在mu9／det9下三模型又全部8条为0，原SDE的2／2／1条可行逃逸解全部丢失。正式LastDet已有实际训练，其部署结果单列；不能把CPU反事实冒充新训练收益。'+link(root/'analysis'/'FINAL_SDE_STEP_COUNTERFACTUAL_DIAGNOSIS_20261031.json','官方重构反事实评分与误差界')+' · '+link(root/'analysis'/'FINAL_SDE_STEP_DIAGNOSIS_20261031.json','末步物理噪声机制')+'</p>'
    if sampler_pairs:
        body+='<h3>采样机制落到训练后：实际完整结果优先</h3>'+table(['完整同预算对照','目标／对照PDMS','ΔPDMS与95%区间','零分修复／新增／高分变零'],[[esc(key),fmt(row.get('target_PDMS'),6)+' / '+fmt(row.get('baseline_PDMS'),6),signed(row.get('delta_PDMS'),6)+'<small>'+ci_text(row)+'</small>',number(row.get('paired',{}).get('repaired_zero'))+' / '+number(row.get('paired',{}).get('new_zero'))+' / '+number(row.get('paired',{}).get('high_to_zero'))]for arm,key,row in sampler_pairs])
        lastdet_diagnostic=sampler_reviews.get('PPO_LASTDET',{}).get('native_process_diagnostics',{}).get('1000',{})
        window=lastdet_diagnostic.get('exploration_once_per_collection',{})
        if window:
            body+='<p class="note">LastDet1k末64更新的 '+number(lastdet_diagnostic.get('first_inner_collection_rank_rows'))+' 个去重collection-rank组：训练canonical候选均分 '+fmt(window.get('canonical_random_mean',{}).get('mean'),4)+'；zero-PG '+fmt(100*window.get('zero_PG',{}).get('mean',0),2)+'%，安全指标组内有差异 '+fmt(100*window.get('safety_differing',{}).get('mean',0),2)+'%，仅进度有差异 '+fmt(100*window.get('progress_only',{}).get('mean',0),2)+'%。高候选分与缺少安全区分可能掩盖学习信号不足，但这只是局部窗口。相对Full1k的新增零与能力损失需正视；继续既定5k复核。W3→W2同时改变PG／anchor平均权重与后续RNG流，不能单因归咎于去掉末噪声。'+link(root/'analysis'/'LASTDET_MATCHED_REVIEW.json','LastDet配对、原生日志窗口与归因边界')+'</p>'
    if rescue_registration:
        bank=rescue.get('bank',{});stats=bank.get('statistics',{});census=rescue.get('census',{});teacher=rescue.get('persistent_zero_teacher',{})
        body+='<h3>当前重点：Rescue补学习方向，同时保住高分能力</h3><p><b>研究假设：</b>部署零分不等于训练没有正优势；偶尔采到好轨迹也不等于能稳定学会。将真实评分通过的好轨迹作为positive CFM目标，检验能否在现有强策略上修复低分，而不是用更高候选均分替代部署验证。</p>'
        body+=table(['设计','两臂共享设置／隔离变量'],[
            ['起点与预算','只加载SafetyEP10k的当前LoRA＋scene／predictive权重，父完整NAVTEST '+fmt(rescue_registration.get('starting_complete_NAVTEST_PDMS'),6)+'；fresh AdamW、fresh RNG42、phase update0，冻结reference仍IL70。首轮额外phase1k／5k，不继承父optimizer／RNG；不是原生不中断续训。'],
            ['相同训练配方','SafetyEP信用＋finaldet W2[7,8]；80%全量、10%低分、10%高分。Replay隔离数据与新phase；CFM−Replay同额外预算隔离teacher辅助；它不单独隔离W2。'],
            ['CFM单独增加','verified positive velocity MSE，低池λ1／高池λ0.2。fresh候选严格NC＝DAC＝TTC＝Comfort＝1且score比static高超过1e−6时才升级teacher；每collection选定后四inner固定。teacher不进入on-policy PG；全8候选五项均满分则跳过辅助。'],
            ['判断是否值得延长','每臂对固定Safety10k看总分与安全／EP配对区间、零分修复／新增、固定父score≥0.9场景保持；CFM对Replay看同phase辅助收益。额外phase不能冒充fresh IL70同预算对照。']])
        body+='<p>TRAIN-only bank有 '+number(stats.get('low_verified_teachers'))+' 个低分teacher（'+number(stats.get('canonical_SAFETY10k_zero'))+' 个原零分、'+number(stats.get('canonical_positive_low'))+' 个正低分），以及 '+number(stats.get('high_selected'))+' 个高分teacher，覆盖 '+number(stats.get('high_unique_physical_logs'))+' 个eligible物理log／'+number(stats.get('high_selected_joint_strata'))+' 个分层。它们是metric-verified，不是人类最优标签；旧DEV仍属于训练内诊断。</p>'
        if teacher:body+='<p>持续零分反例7af已有严格四项通过的真实teacher，canonical '+fmt(teacher.get('teacher_score'),6)+'、EP '+fmt(teacher.get('teacher_EP'),6)+'；只证明有可用正向目标，尚未证明部署能学会。'+link(rescue_registration.get('bank'),'teacher manifest与源轨迹身份')+'</p>'
        if census:body+=detail('曝光与覆盖：候选人口不等于已遍历', '<p>计划 '+number(sum(census.get('exposure_counts',{}).values()))+' 次场景曝光＝8000 full＋1000 low＋1000 high；真实unique场景 '+number(census.get('distinct_scene_tokens_total'))+'。full可采人口103,288不表示已遍历。按token lookup，low／high理论匹配teacher '+number(census.get('aux_roles_for_all_slots',{}).get('low'))+'／'+number(census.get('aux_roles_for_all_slots',{}).get('high'))+' 次，因full中也可遇到bank场景；实际激活还要扣除成功饱和跳过。'+link(root/'analysis'/'RESCUE_RETENTION_CENSUS_AUDIT_20261031.json','固定census')+'</p>')
        rescue_decisions=rescue.get('next_decisions',{});three_scales=rescue_decisions.get('coverage',{}).get('three_scales',{})
        if rescue_decisions.get('status')=='COMPLETE'and three_scales:
            known_low=three_scales.get('known_Safety10k_low_below095',{})
            body+='<p><b>覆盖仍有明显边界：</b>521teacher已覆盖TRAIN1024五路线全部已观测ever-zero，旧零分恢复场景也可在high池；但Safety10k在该1024面板有 '+number(known_low.get('scenes'))+' 个score&lt;0.95场景，只有 '+number(known_low.get('low_priority_pool'))+' 个进入low优先池。覆盖已知零分不等于已经修复，也不代表103,288全人口的失败已识别。ADAPT的27是纵向观测次数，持续zero为1个scene。'+link(root/'analysis'/'ZERO_RESCUE_NEXT_DECISIONS_20261031.json','覆盖三层口径与下一研究判断')+'</p>'
        early=next((row for row in rescue.get('early_signal',{}).get('runs',[])if row.get('arm')=='PPO_RESCUE_CFM'),{})
        if early:
            collections=early.get('collections',0);rescue_early_metrics=early.get('metrics',{})
            mode_count=lambda key:round(rescue_early_metrics[key]['mean']*collections)if collections and isinstance(rescue_early_metrics.get(key),dict)and h.finite(rescue_early_metrics[key].get('mean'))else None
            zero_semantics=rescue.get('early_signal',{}).get('zeroPG_reward_semantics_followup',{});zero_semantics_text=''
            if zero_semantics.get('status')=='EXACT_ORIGINAL_UPDATE_WINDOWS'and zero_semantics.get('original_collections_recovered')==collections:
                allzero=zero_semantics.get('zeroPG_active_canonical_all_zero_collections');zero_active=zero_semantics.get('teacher_active_zero_PG_collections')
                if h.finite(allzero)and h.finite(zero_active):zero_semantics_text='恢复同一原窗口后确认：其中 '+number(allzero)+' 组确为8候选canonical全零，另 '+number(zero_active-allzero)+' 组是组内信用并列；这 '+number(zero_active)+' 组来自low／high '+number(zero_semantics.get('zeroPG_active_low_pool'))+'／'+number(zero_semantics.get('zeroPG_active_high_pool'))+'。teacher提供了全零组原PG没有的学习方向，尚不证明部署零分已修复。'
            body+='<p class="note"><b>当前真正验证到的过程：</b>末64更新局部窗口、8rank共 '+number(collections)+' 个collection中，'+number(early.get('teacher_active_collections'))+' 组aux active，其中 '+number(early.get('teacher_active_zero_PG_collections'))+' 组PG为0仍得到teacher监督；'+number(mode_count('rescue_aux_online_upgrade'))+' 组在线升级，low／high '+number(mode_count('rescue_aux_pool_low'))+'／'+number(mode_count('rescue_aux_pool_high'))+'。'+zero_semantics_text+'Replay按设计无aux。此证据确认辅助实际触达了零PG组，尚不代表部署提分；现有gradnorm为联合梯度，不能用scalar loss比例推断CFM／PG梯度占比。'+link(root/'analysis'/'RESCUE_PAIR_EARLY_SIGNAL_20261031.json','原生有界窗口')+'</p>'
        rescue_review=rescue.get('matched_review',{});rescue_pairs=[(key,row)for key,row in rescue_review.get('contrasts',{}).items()if row.get('scenes')==12146 and h.finite(row.get('delta_PDMS'))and len(row.get('CI95_PDMS',[]))==2]
        if rescue_pairs:
            body+=table(['完整NAVTEST对照（额外phase）','目标PDMS／Δ与95%区间','修复／新零／高分变零','固定Safety10k≥0.9子集：Δ与95%区间'],[[esc(key),fmt(row.get('target_PDMS'),6)+' / '+signed(row.get('delta_PDMS'),6)+'<small>'+ci_text(row)+'</small>',number(row.get('zeros',{}).get('repaired'))+' / '+number(row.get('zeros',{}).get('new_zero'))+' / '+number(row.get('zeros',{}).get('high_to_zero')),signed(row.get('fixed_SAFETY10k_high_ge09',{}).get('delta_PDMS'),6)+'<small>'+ci_text(row.get('fixed_SAFETY10k_high_ge09',{}))+'</small>']for key,row in rescue_pairs])
            body+=detail('完整配对的NC／DAC／TTC／EP分量',table(['对照','分量','Δpp与95%区间'],[[esc(key),esc(metric),signed(value.get('delta_pp'),5)+'<small>'+esc(str(value.get('CI95_pp','UNKNOWN')))+'</small>']for key,row in rescue_pairs for metric,value in row.get('components',{}).items()if metric in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress')]))
        else:body+='<p class="muted">两臂原生优化进度看路线表；尚无完整Rescue NAVTEST配对，不预填新分数或低分修复率。独立owner异步完成1k／5k完整评测后自动更新。'+link(root/'owners'/'RESCUE_PAIR'/'STATE.json','独立评测owner')+'</p>'
        body+='<p class="muted">方法关联：<a href="https://arxiv.org/abs/1806.05635">SIL</a>从过去成功经验补学习信号；<a href="https://arxiv.org/abs/1910.00177">AWR</a>通过价值估计加权行为回归。本实验只借鉴“保留并学习已验证好行为”的思想，使用无critic的positive CFM；没有AWR的优势加权似然，也不是精确DPO或完整重做IL。'+link(root/'EXPERIMENT_RESCUE_PAIR_REGISTRATION_20261031.json','正式独立登记')+' · '+link(root/'analysis'/'RESCUE_PAIR_MATCHED_REVIEW.json','完整结果产生后自动读取的配对入口')+'</p>'
        if rescue_decisions.get('status')=='COMPLETE':body+=detail('后续候选及启动条件：当前均未上线',table(['优先级／方向','当前动作','必须先观察到什么'],[[esc(str(row.get('priority')))+' · '+esc(row.get('method')),esc(row.get('action_now')),esc(row.get('trigger'))]for row in rescue_decisions.get('decisions',[])if row.get('priority',0)>0]))
    examples=scene_research.get('scene_examples',[])
    example_rows=[[esc(row.get('scene','')),esc(row.get('evidence_needed','')),esc(row.get('desired',''))]for row in examples] if examples else [
        ['路口让行／冲突参与者通过','路权、冲突间隙、参与者可见性、可行通行窗口','必要等待不作低效惩罚；区分尚无安全窗口与错过安全窗口。'],
        ['低风险通畅道路','安全可行空间、导航意图、自车状态与进度机会','应稳定利用安全通行空间；不能用停滞换取表面安全高分。'],
        ['遮挡或交互信息不足','观测可见性、信息缺失与置信，不把未知直接当已知危险','不确定性与确认风险单列；检查信息是否足以支持走／等决定。']]
    body+=table(['场景例子（研究判据）','需提取／核对的信息','合理行为与评价边界'],example_rows)
    observed=scene_research.get('observed_information',{})
    if observed:
        body+=table(['信息层次','当前事实／边界'],[
            ['actor当前输入',esc('；'.join(observed.get('actor_current_inputs',[])))],
            ['当前未提供',esc('；'.join(observed.get('not_currently_provided',[])))],
            ['特权反馈仅作训练标签／奖励',esc('；'.join(observed.get('privileged_training_only',[])))]])
        body+='<p class="note">已确认的风险与观测不确定性是两个维度。未来真值不直接作为actor输入；predictive memory的存在也不等于已校准的交互风险预测。场景分组须审计当前观测是否足以支持该判断。</p>'
    audit=editorial.get('representation_audit',{})
    body+='<div class="note"><b>表征并非全冻结，也不是只训动作头。</b>Qwen及其中间特征冻结／缓存；scene memory的144个spatial＋16个global token provider、32个predictive token provider与actor attention LoRA可训练，当前总可训练参数52,707,840。每个inner重新计算current memory，缓存只覆盖Qwen冻结特征与frozen IL reference。</div>'
    if audit.get('status')=='COMPLETE':
        module_labels={'scene_provider':'scene provider','predictive_provider':'predictive provider','action_expert':'action attention LoRA'}
        module_rows=[[module_labels.get(name,name),number(v.get('numel')),number(v.get('tensors'))]for name,v in audit.get('modules',{}).items()]
        body+=table(['原生20k审计：可训练部分','参数数','tensor数'],module_rows)
        groups=audit.get('optimizer_groups',[])
        body+='<p><b>真实优化证据：</b>'+ '；'.join(esc(v.get('name',''))+' '+number(v.get('nonzero_second_moment_tensors'))+' / '+number(v.get('state_tensors'))+'个Adam二阶矩非零，step '+number(v.get('step_min'))+'–'+number(v.get('step_max')) for v in groups)+'。这是实际优化到memory的证据，不代表风险语义已足够、预测已校准，也不能归因于PG专属梯度。'+link(root/'analysis'/'REPRESENTATION_TRAINABILITY_AUDIT_20261030.json','CPU原生optimizer审计，无新GPU计算')+'</p>'
    body+='<p>因此要区分知识基础与RL适配：缺少短历史或结构化运动输入时，调reward不能自动补齐不可观测信息；单一总分与IL均值anchor也不等于持续的object／future监督。但RL不需全面重做IL。表征／知识缺口反馈IL补齐，RL侧先检验memory适配强度与安全—效率信用分配；辅助监督的归属与作用单独登记。</p>'
    information_audit=editorial.get('scene_information_audit',{})
    if information_audit:
        scope=information_audit.get('RL_trainable_scope_and_losses',{})
        body+='<p>当前RL没有继续调用IL的DINO、current-object、future、post-action等readout损失；这不证明已经发生语义漂移，却说明均值anchor没有直接约束这些语义任务的准确率。场景标签、奖励特权信息与actor当前可见信息已分层审计。'+link(root/'analysis'/'SCENARIO_SAFETY_EFFICIENCY_AUDIT_20261030.json','可见信息、特权标签与辅助监督边界')+'</p>'
    if scene_research:
        scene_summary=scene_research.get('summary_zh') or scene_research.get('objective_zh') or scene_research.get('objective')
        if isinstance(scene_summary,str):body+='<p>'+esc(scene_summary)+'</p>'
        objective=scene_research.get('objective',{})
        if isinstance(objective,dict):
            body+='<p><b>研究判据：</b>'+esc(objective.get('safety','按场景检查安全约束与尾部'))+' '+esc(objective.get('efficiency','在场景制约内评价可实现效率'))+'</p><p>'+esc(objective.get('all_fail_rule','8个候选全未通过不等于场景无安全解。'))+'</p><p class="muted">实施状态：'+('受控pilot已实际训练；场景条件策略与部署收益仍待检验。' if active_pilots else esc(objective.get('status','研究方案准备中，未声明新策略已获收益')))+'</p>'
        preliminary=scene_research.get('preliminary_training_evidence',{})
        if preliminary:
            body+='<p>初步信用分配证据（'+esc(preliminary.get('scope','既有TRAIN候选，离线诊断'))+'）：unsafe正优势候选 '+number(preliminary.get('unsafe_positive_advantage'))+'，其中 '+number(preliminary.get('unsafe_positive_with_feasible_alternative'))+' 个有通过候选可替代，'+number(preliminary.get('unsafe_positive_EP_above_all_feasible'))+' 个EP高于所有通过候选。另有 '+number(preliminary.get('groups_with_at_least2_feasible'))+' 组至少两个通过候选，'+number(preliminary.get('groups_feasible_EP_span_ge_005'))+' 组的通过候选EP跨度≥0.05。这里“通过”仅指离线NC=DAC=TTC=1，不等于掌握全部场景风险；这些是待验证的信用分配线索。</p>'
        body+='<p>'+link(root/'analysis'/'SCENE_SAFETY_EFFICIENCY_RESEARCH.json','场景安全—效率研究目标、证据与实施状态')+'</p>'+detail('场景研究原始登记／诊断', '<pre>'+esc(json.dumps(scene_research,ensure_ascii=False,indent=2))+'</pre>')
        brief=editorial.get('scene_brief_20k',{})
        candidate=brief.get('minimal_CPU_candidate',{})
        if candidate:body+='<p class="note">CPU信用分配原型将安全优势与“通过候选子集内中心化”的EP优势分开。当前实际SafetyEP为kappa0的相对成本中心化：全失败但成本不同时可有方向，同成本仍无相对PG方向；不能把旧CPU推演当作当前部署保证。低EP或停车并非自动错误，必须先检查同组是否有通过指标的可行替代。'+link(root/'analysis'/'SCENE_SAFETY_EFFICIENCY_BRIEF_20K.json','20k解释、局部方法边界与CPU原型')+'</p>'
    else:body+='<p class="muted">场景条件诊断／表征方案由研究主进程准备，正式SCENE_SAFETY_EFFICIENCY_RESEARCH.json待发布；当前是研究目标与判据，尚非新策略成绩。</p>'
    body+='<h3>数据覆盖与反馈：支持诊断，不替代场景解释</h3><div class="plots-two">'+data_chart+panel_chart+'</div><p>固定Hard的5k→10k下降0.118841点；同10k Full−Hard为+0.283893，配对区间[+0.011398,+0.520610]且NC/DAC/TTC均更高，支持本轮机会成本停止决策。'+adaptive_sentence+'其TRAIN先升后降，不能把反馈panel回升当泛化证明。</p>'
    if adaptive_amendment:
        old=adaptive_amendment.get('previous_mixture',{})
        mix=lambda node:'/'.join(str(round(100*node.get(k,0))) for k in ['uniform_fraction','low_score_fraction','regression_fraction'])
        body+='<div class="note"><b>Adaptive新阶段已登记：</b>自 '+number(adaptive_amendment.get('effective_from_optimizer_update'))+' 更新起（stage '+number(adaptive_amendment.get('effective_stage_index'))+'），均匀／低分／退化采样从 '+mix(old)+' 改为 '+mix(adaptive_amendment)+'。旧stage0–16原样保留，后续结果属于新混合策略，不能归因于原35/40/25。额外4096诊断不并入既有反馈bank；IL父模型、optimizer与seed保持。'+link(root/'ADAPTIVE_COVERAGE_POLICY_AMENDMENT.json','阶段策略修订收据')+'</div>'
    if adaptive_analysis:
        coverage=adaptive_analysis.get('coverage',{});feedback=coverage.get('feedback1024',{});extra=coverage.get('additional3072_feedback_unseen',{});timing=adaptive_analysis.get('feedback_time_summary',{})
        body+=table(['已完成至'+number(coverage.get('completed_through_updates'))+'更新的曝光','证据','含义'],[
            ['旧反馈1024占比',fmt(feedback.get('exposure_fraction'),2,percent=True)+'%；最高'+number(feedback.get('maximum_exposures'))+'次','反馈子集被大量重复，不能代替总体覆盖'],
            ['全人口已见场景',number(coverage.get('population_unique_exposed'))+' / 103,288（'+fmt(coverage.get('population_coverage'),2,percent=True)+'%）','选择分布与均匀全量路线不同'],
            ['额外3072中未被RL采样',number(extra.get('never_exposed'))+' / 3,072','称feedback-unseen；并非全部training-unseen'],
            ['正常反馈等待中位数',fmt((timing.get('steady_after1k_gap_median_seconds')or 0)/60,2)+'分钟','墙钟含export/容量/评测，不等于独占GPU成本']])
        body+='<div class="note">至9k，旧panel占57.83%的实际曝光；8k曾有36.5分钟异常反馈间隔、6次pre-child资源拒绝，不能当作模型/scorer错误。旧Full／Adaptive窗口的低zero-PG比例不能外推到SafetyEP，应按算法实际mask与canonical同分分开统计。'+link(root/'analysis'/'ADAPTIVE_TRAIN_PROCESS_20261030.json','完整曝光、日志与反馈耗时')+'</div>'
        replay=adaptive_analysis.get('mixture_replay_summary',{}).get('9000',{})
        if replay:
            replay_rows=[]
            for key,label in [('actual35_40_25','实际35/40/25'),('replay80_10_10','CPU回放80/10/10')]:
                values=replay.get(key,{})
                replay_rows.append([label,number(values.get('scene_exposures')),number(values.get('unique_scenes')),fmt(values.get('feedback1024_fraction'),2,percent=True)+'%',number(values.get('maximum_scene_exposures'))])
            body+=detail('固定旧bank的CPU覆盖回放，非新策略提分结果',table(['口径','同等曝光','唯一场景','反馈panel占比','单场景最高重复'],replay_rows)+'<p>回放固定旧反馈bank，仅说明机械覆盖可改善，不模拟新策略产生的reward；不能视作训练实绩，也不据此停止Adaptive。'+link(root/'analysis'/'ADAPTIVE_80_10_10_FROZEN_BANK_REPLAY_20261030.json','完整CPU回放')+'</p>')
    else:body+='<p class="muted">stage曝光／反馈耗时的CPU汇总正在整理；未完成的10k feedback不提前登记为完成。</p>'
    diagnostic=editorial.get('train4096',{})
    if diagnostic:
        diagnostic_rows=[]
        for arm,node in diagnostic.get('runs',{}).items():
            spec=node.get('spec',{});label=spec.get('label');evaluation=next((r for r in snapshot['results']if r.get('label')==label),{}) if label else {}
            state_labels={'EVALUATING':'GPU评测进行中','WAITING_NATIVE_10K':'等待10k原生checkpoint','COMPLETE':'已完成汇总','RUNNING':'运行中'}
            diagnostic_state='完整SUMMARY已到' if evaluation.get('publishable_complete') else state_labels.get(node.get('status'),node.get('status','待记录'))
            diagnostic_rows.append([esc(arm),esc(diagnostic_state),number(spec.get('world')) if spec else '待native10k',
                number(evaluation.get('scenes'))+' / 4,096' if evaluation else '待原生checkpoint',
                fmt(evaluation.get('display_PDMS'),6) if evaluation.get('publishable_complete') else '—（未完整不发布均分）'])
        body+='<h3>固定TRAIN4096独立诊断执行进度</h3><p>IL70与FULL10k先行GPU评测，ADAPTIVE按自己的10k原生checkpoint推进，当前状态以下表为准。有限三模型，拆旧1024与额外3072；新增panel不进入adaptive反馈。该面板是feedback-unseen诊断，不称独立Test或全体training-unseen。</p>'+table(['模型','独立owner状态','shards','进度','完整均分'],diagnostic_rows)+'<p>'+link(root/'owners'/'TRAIN4096'/'STATE.json','TRAIN4096独立STATE')+'</p>'
        diag_complete=[r for r in complete if r.get('cohort')=='TRAIN_DIAGNOSTIC' and r.get('scenes')==4096]
        diag_base=next((r for r in diag_complete if r.get('evaluation_kind')=='IL_BASELINE'),None)
        diag_full=next((r for r in diag_complete if r.get('arm')=='PPO_FULL' and update(r)==10000),None)
        if diag_base and diag_full:
            component_deltas={m:100*(diag_full.get('means',{}).get(m,0)-diag_base.get('means',{}).get(m,0))for m in metrics}
            full_test=points.get(('PPO_FULL',10000))
            diag_adaptive=next((r for r in diag_complete if r.get('arm')=='PPO_ADAPTIVE'),None)
            adaptive_note=('ADAPTIVE完整4096为'+fmt(diag_adaptive['PDMS'],6)+'，与另外两模型按相同人口比较。' if diag_adaptive else 'ADAPTIVE完整4096尚待检验。')
            body+='<div class="note"><b>已完成模型即时报告：</b>TRAIN4096上IL70 '+fmt(diag_base['PDMS'],6)+'、Full10k '+fmt(diag_full['PDMS'],6)+'，点差 '+signed(diag_full['PDMS']-diag_base['PDMS'],6)+'。伴随EP '+signed(component_deltas['ego_progress'],5)+' pp、DAC '+signed(component_deltas['drivable_area_compliance'],5)+' pp、NC '+signed(component_deltas['no_at_fault_collisions'],5)+' pp。'+('同一Full10k在完整NAVTEST相对IL70观察提高 '+signed(full_test['PDMS']-base,6)+' 点；' if full_test else '')+'训练诊断人口与Test的方向不同，不能用panel下降替代Test结论。'+adaptive_note+'</div>'
        if editorial.get('train4096_diagnosis'):
            diagnosis=editorial['train4096_diagnosis'];estimator_rows=[]
            for group,group_label,estimator,estimator_label in [('all4096','4096全panel','scene_mean','场景均值'),('extra3072','反馈外3072','scene_mean','场景均值'),('all4096','TRAIN总体推算','population_log_reweighted_TRAIN_projection','按log人口重加权投影')]:
                for pair,values in diagnosis.get('subgroups',{}).get(group,{}).get('paired',{}).items():
                    score=values.get(estimator,{}).get('score',{});interval=score.get('cluster_bootstrap_95pct',[])
                    estimator_rows.append([esc(pair.replace('_minus_',' − ')),group_label+'<small>'+estimator_label+'</small>',signed(score.get('delta_points')), '['+', '.join(signed(v)for v in interval)+']' if len(interval)==2 else '待记录'])
            body+=table(['已完整模型配对','估计对象','PDMS点差','162-log聚类95%区间'],estimator_rows)
            body+='<p class="note">当前Full10k−IL70的4096场景均值下降区间低于零，但TRAIN总体重加权投影区间跨零；其额外3072区间也跨零。它们回答不同人口问题，TRAIN投影不是原生Test，不据此下退化停训结论。额外场景只对Adaptive做曝光分层，不能称Full的training-unseen。'+link(root/'analysis'/'TRAIN4096_DIAGNOSIS.json','完整诊断、log等权与不确定性')+'</p>'
            adaptive_rows=[]
            for group,label in [('feedback1024','旧反馈1024'),('extra3072','反馈外3072'),('all4096','完整4096')]:
                item=diagnosis.get('subgroups',{}).get(group,{}).get('paired',{}).get('PPO_ADAPTIVE_minus_PPO_FULL',{}).get('scene_mean',{}).get('score',{})
                interval=item.get('cluster_bootstrap_95pct',[])
                if item:adaptive_rows.append([label,signed(item.get('delta_points'),5),'['+', '.join(signed(v,5)for v in interval)+']' if len(interval)==2 else '待记录'])
            if adaptive_rows:body+='<h3>4096 final：面板特化迹象，尚无广泛增益证据</h3>'+table(['ADAPTIVE−FULL，同人口','PDMS点差','95%区间'],adaptive_rows)+'<p>旧反馈panel有正向配对支持，反馈外3072与全4096区间跨零；新增场景没有明确增益，也不能写成额外场景已退化。后续数据策略应与场景条件安全—效率诊断结合，不能只追反复评测的panel。</p>'
    new_pilots=[arm for arm in display_arms if arm not in ARMS]
    pilots_complete=all((arm,10000)in points for arm in ['PPO_MEMORYLR20','PPO_SAFETY_EP'])
    body+='<h3>两个独立10k pilot：'+('首轮完整评测已完成，后续决策见路线表' if pilots_complete else '均已全8rank实际训练，梯度有限' if len(active_pilots)>=2 else '已登记执行进度见路线表' if new_pilots else '准备中，尚无启动收据')+'</h3>'+table(['研究问题','唯一主要改动','保持不变／判据'],[
        ['给定IL知识能否更充分适配轨迹任务','memory学习率5e−6 → 2e−5','固定IL父模型与RL配方其余项；检验适配强度，不能代表补齐全部驾驶知识。'],
        ['安全与通过候选内效率能否分别分配信用','安全／可行EP分量优势：PPO_SAFETY_EP候选','不叠加memory LR试验；同预算检查新增风险、必要等待和条件效率。']])+'<p>两项由研究主进程实现与注册，实际是否启动看独立登记／worker收据。PPO_SAFETY_EP是待验证的候选指标条件信用分解：通过者内部才分配效率信用，全未通过时用指标缺失成本给安全方向；它没有新增场景输入，也不代表已具备场景风险推理。新IL达标后的RL对照另起实验；现有已封存模型与旧训练不热改。</p>'
    if pilot_startups:
        pilot_rows=[]
        for arm,value in pilot_startups.items():
            registration=pilot_registrations.get(arm,{})
            startup_updates=min((row.get('updates',0) for row in value.get('rows',[])),default=None)
            guard=registration.get('lease_guard_UTC');expiry=registration.get('lease_expiry_UTC')
            boundary=('8h临时许可：'+str(guard)+'前保护checkpoint；'+str(expiry)+'到期' if guard else '既有许可主机；按当前有效grant运行')
            superseded_hosts=snapshot.get('resource_authorization_request',{}).get('hosts',[])
            if snapshot.get('resource_finite_window_superseded') and runs.get(arm,{}).get('host') in superseded_hosts:
                boundary='最新REQUEST临时不限时；原8h保护线／到期仅为历史，不代表部署容量已核实'
            pilot_rows.append([esc(arm),'全8rank有限梯度；启动证据 '+number(startup_updates)+' 更新' if arm in active_pilots else esc(value.get('status','待收据')),
                number(registration.get('pilot_budget_updates'))+' cap<small>1k／5k／10k评审；scheduler '+number(registration.get('native_scheduler_max_updates'))+' 不等于研究已授权跑满</small>',
                esc(boundary),link(root/f'STARTUP_{arm}_20261030.json','实际启动')+' · '+link(root/f'EXPERIMENT_{arm}_REGISTRATION.json','独立注册')])
        no_pilot_score=not any(latest.get(arm) for arm in pilot_startups)
        body+=table(['独立pilot','实际优化证据（不是成绩）','首轮预算','运行时间边界','来源'],pilot_rows)+'<p>两项均固定IL70、冻结Qwen与输入，测试既有知识的适配和轨迹信用分配；没有注入新知识。'+('目前尚无已完成的正式成绩。' if no_pilot_score else '已完成的正式成绩见路线表，匹配配对见下表。')+'v7科学snapshot独立封存，旧v1–v6保持。</p>'
        review=editorial.get('safety_ep_operator_review',{})
        if review:body+=detail('PPO_SAFETY_EP首次派发：child前operator拒绝，非科学训练失败','<p>首次attempt在参数白名单检查阶段被拒，RUNNING不存在、native root不存在；operator单文件审查修复后attempt002已实际优化。不是loss或scorer失败，科学snapshot未更换。'+link(root/'jobs'/'PPO_SAFETY_EP'/'formal_to001000_attempt001'/'PRECHILD_OPERATOR_REVIEW.json','child前审查与修复收据')+'</p>')
    pilot_review=editorial.get('new_pilot_matched_review',{})
    if pilot_review:
        complete_pairs=[(key,row) for key,row in pilot_review.get('contrasts',{}).items() if row.get('status')=='COMPLETE']
        body+='<h3>新pilot同预算复查：'+number(len(complete_pairs))+' / '+number(pilot_review.get('target_pairs',6))+'组完整</h3>'
        if complete_pairs:
            def pilot_metric(row,metric):
                item=row.get('safety_metrics',{}).get(metric,{})
                interval=item.get('CI95_delta_percentage_points',[])
                return signed(item.get('delta_percentage_points'),5)+('<small>['+', '.join(signed(v,5)for v in interval)+']</small>' if len(interval)==2 else '<small>区间待记录</small>')
            pair_rows=[]
            for key,row in complete_pairs:
                target=row.get('target',key.split('_minus_',1)[0]);baseline=row.get('baseline',key.split('_minus_',1)[-1])
                pair_rows.append([esc(target.replace('_NAVTEST',''))+'<small>− '+esc(baseline.replace('_NAVTEST',''))+'</small>',signed(row.get('delta_PDMS'))+'<small>'+ci_text(row)+'</small>',
                    pilot_metric(row,'no_at_fault_collisions'),pilot_metric(row,'time_to_collision_within_bound'),pilot_metric(row,'ego_progress'),esc(row.get('review',{}).get('recommendation_zh','按总分和组件区间联合判断'))])
            body+=table(['同预算完整人口配对','ΔPDMS［95%区间］','ΔNC pp［区间］','ΔTTC pp［区间］','ΔEP pp［区间］','结论／风险'],pair_rows)
        else:body+='<p class="muted">尚无双方完整且已配对的结果，不填分数或区间；每组完成后独立展示，不必等待六组齐备。</p>'
        body+='<p>'+esc(pilot_review.get('knowledge_scope_zh',''))+' '+esc(pilot_review.get('interpretation_zh',''))+' '+link(root/'analysis'/'NEW_PILOT_MATCHED_REVIEW.json','有限CPU配对分析与实际进度')+'</p>'
    actions=[['P0／已登记','两项同采样5k偏好pilot','Replay−Full隔离采样；CFM−Replay隔离辅助损失','910有效偏好场景；历史pair不混on-policy PG，当前成绩看完整证据'],
        ['已完成／后续诊断' if fixed_probe_complete else 'P0／有限探针','固定theta、同初始latent的ODE8／SDE8','选定cohort的噪声损害已观察，是否推广／减弱仍待验证' if fixed_probe_complete else '排除历史theta变化后比较候选可行性与效率','不把不同更新模型排名；不把best-of8当部署'],
        ['P0','场景条件安全—效率与表征诊断','合法等待、低风险停滞、确认风险／观测不确定分开','IL补知识、RL做适配与轨迹强化，不把局部奖励去干扰当场景策略'],
        ['已完成／继续拆解','解释20k匹配结果，保留Full控制','SAFE总分阶段增益、BATCH/CISPO的NC/TTC风险','不把10k领先外推；不因一个总分替代场景分组'],
        ['已完成／收据','停止NFT、固定hard-only、BATCH与CISPO后续预算','逐臂STOPPED、退出与checkpoint收据','保留旧结果与历史好点；不是模型失败'],
        ['已完成／后续解释','三模型固定TRAIN4096诊断','旧反馈1024、反馈外3072与实际曝光分层','面板特化迹象，不等于反馈外场景退化'],
        ['已在做','独立运行ANCHOR40单因素','降低anchor压力是否改善后续canonical结果','只改β，不叠SAFE/归一化；成绩按该臂完整结果读数'],
        ['已登记／阶段2','Adaptive自17,001起80/10/10覆盖策略','反馈panel特化与反馈外场景的实际曝光分层','区别旧35/40/25；覆盖回放不是提分结果']]
    body+=table(['优先级','下一步','用什么证据决定','执行边界'],actions)
    body+='<p>资源安排依据用户授权的证据决策调整；其余健康路线不按小分差自动剪枝。'+link(root/'RESEARCH_ALLOCATION_POLICY_20261030.json','当前研究资源决策')+'</p>'
    chapter('06','process','场景条件安全—效率：信息、表征与下一步','核心问题是根据足够的场景信息决定怎样安全且高效地驾驶，而不只是调整奖励系数。',body)

    # 07: details are preserved, but folded away from the argument.
    test_rows=[[esc(r.get('arm')),number(update(r)),fmt(r['PDMS'],6),number(r.get('scenes')),link(r.get('evidence'),'SUMMARY')]for r in sorted(test,key=lambda r:(r.get('arm',''),update(r)))]
    evaluation_rows=[]
    for r in sorted(snapshot['results'],key=lambda r:(str(r.get('cohort')),str(r.get('arm')),update(r))):
        evaluation_rows.append([esc(r.get('arm','')),esc(r.get('cohort','')),number(update(r)),number(r.get('scenes'))+' / '+number(r.get('expected_scenes')),
            fmt(r.get('display_PDMS'),6) if r.get('publishable_complete') else '—（未完整不发布均值）',link(r.get('evidence'),'原始结果')])
    run_rows=[[esc(a),esc(runs[a].get('status')),number(runs[a].get('updates')),esc(json.dumps(runs[a].get('rank_updates',{}))),
        number(runs[a].get('new_candidates')),fmt(runs[a].get('recent_clipfrac'),2,percent=True)+'%',h.gradient_fmt(runs[a].get('recent_gradient')),
        esc(runs[a].get('host')),link(runs[a].get('job'),'worker'),link(runs[a].get('source'),'source')]for a in display_arms if a in runs]
    inputs=[[link(r['path'],Path(r['path']).name),h.utc(r['mtime']),number(r['bytes']),'<code>'+r['sha256'][:20]+'…</code>']for r in snapshot['inputs']]
    identity='<p>共同原始World IL70k父模型；全量NAVTRAIN103,288包含旧DEV1,696，后者在全量臂中只是训练内诊断。困难臂15,171；NAVTEST12,146、44物理log，与TRAIN隔离。固定训练seed42／推理seed2101，单候选ODE K10 / shift5；SAFE_COMFORT训练塑形分独立保存，正式部署仍按canonical评分。</p><p>完整评测需要COMPLETE、计数等于声明人口、invalid=0、证据seal正确。partial只显示进度；固定TRAIN1024/4096不与完整Test或全TRAIN均值混用。NAVTEST已用于研究假设／资源选择，相关决策透明记录；不将其token回流为训练名单。</p>'
    currently_excluded=snapshot.get('withdrawn_hosts',[])
    ops='<p>当前许可按最新grant和临时时间窗，不把历史撤回当作永久禁用。当前未在有效许可中的历史撤回主机：'+esc('、'.join(currently_excluded) or '无，已被当前授权覆盖')+'。vlawm-zt2与tsinghua-zt2是不同节点。09:31的vla-zt3撤回迁移时，SSH/API被拒、旧进程退出未直接确认；这是历史收据，不替代后续再授权与库存审计。迁移曾回滚未提交尾部26/41/57，不称无损；新mutable日志独立，旧目录保留。pre-child资源恢复与真实科学错误分别记录。</p>'
    ops+='<p>'+link(root/'resources'/'LATEST_RESOURCE_AUTHORIZATION.json','当前授权指针')+' · '+link(root/'resources'/'LATEST_RESOURCE_ADDITION.json','最新节点准入／部署指针')+' · '+link(snapshot.get('host_withdrawal_migration_path'),'旧机撤回迁移结果')+' · '+link(snapshot.get('resource_addition_deployment_path'),'最新准入／部署收据')+' · '+link(root/'RESOURCE_CAPACITY_POLICY.json','容量恢复规则')+'</p>'
    if snapshot.get('historical_addition_inventory'):
        ops+='<p>已授权既有节点的库存保留各次独立准入证据，最新指针不覆盖累计库存：</p>'+table(['历史准入库存','覆盖主机'],[[link(item['path'],'库存原件'),esc('、'.join(item['hosts']))]for item in snapshot['historical_addition_inventory']])
    activation=snapshot.get('resource_activation',{})
    if activation.get('hosts'):
        activation_rows=[[esc(host),esc(entry.get('status','待记录')),number(entry.get('physical_GPU_count')),number(len(entry.get('headroom32GiB_accounted_gpu_indices',[]))),esc(entry.get('inventory_at','待记录')),link(entry.get('inventory_evidence'),'库存证据')]for host,entry in activation['hosts'].items()]
        ops+='<h3>最近可核查库存：与最新权限分开</h3>'+table(['主机','当时收据状态','物理GPU','当时32GiB合格卡','库存证据时间','来源'],activation_rows)+'<p>库存是对应时间的只读快照，不证明当前空闲或新部署。最新许可型ACTIVATED的host列表只表示授权激活，不当作GPU库存。'+link(snapshot.get('resource_activation_path'),'最新授权激活收据，无报告侧host操作')+'</p>'
    if snapshot.get('resource_finite_window_superseded'):
        ops+='<p>最新REQUEST明确四exact hosts临时不限时，并覆盖此前8小时窗口。effective grant内残留的旧expires字段标为historical_only，报告不再用它推断当前到期；本页不改变任何worker租约。'+link(snapshot.get('resource_authorization_request_path'),'最新用户REQUEST原文')+'</p>'
    ops+=detail('最近阶段事件','<pre>'+esc(json.dumps(snapshot.get('events',[])[-15:],ensure_ascii=False,indent=2))+'</pre>')
    body=detail('科学身份、人口与防误读规则',identity)+detail('全部完整NAVTEST与原始SUMMARY',table(['路线','更新','PDMS','场景','来源'],test_rows))
    body+=detail('TRAIN诊断与尚未完整的评测进度',table(['路线','人口','更新','已完成 / 应有','完整分数','来源'],evaluation_rows))
    body+=detail('训练过程明细：各rank、fresh候选、clip、梯度与worker',table(['路线','状态','共同更新','各rank更新','fresh候选','ratio越界%','梯度范数','主机','worker','科学source'],run_rows))
    ops+='<p>'+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'PPO_HARD.json','固定hard-only停止依据')+' · '+link(root/'owners'/'CAMPAIGN'/'arm_decisions'/'PPO_HARD.STOPPED.json','HARD原生停止收据')+' · '+link(root/'analysis'/'PORTFOLIO_10K_RECOMMENDATIONS.json','10k研究建议汇总')+' · '+link(root/'analysis'/'ADV_CLAMP_NEAR10K.json','优势cap诊断')+'</p>'
    body+=detail('资源变更、迁移、等待与失败证据',ops)
    body+=detail('输入文件时间、哈希与报告源码',table(['输入文件','原证据修改时间','字节','SHA256前缀'],inputs)+'<p>'+link(root/'reporting'/'report_snapshot.json','派生快照')+' · '+link(root/'reporting'/'research_report.py','展示层源码')+' · '+link(root/'reporting'/'render_report.py','采集与watch源码')+' · '+link(root/'reporting'/'OWNER.json','唯一报告owner')+' · '+link(root/'reporting'/'REPORT_CONFIG.json','刷新偏好')+' · '+link(h.REVIEW_ROOT/'literature_and_recommendations.json','论文与算法调研索引')+'</p>')
    chapter('07','evidence','证据附录','复核材料保留在这里；展开后可查看完整人口、原始结果、运行身份和运营记录。',body)

    css='''
    :root{--ink:#15243b;--muted:#617087;--line:#dde5ef;--blue:#2563eb;--paper:#fff}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f9;color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}header{background:#12243d;color:white;padding:44px max(24px,calc((100vw - 1360px)/2));border-bottom:4px solid #10b981}header .eyebrow{font-size:12px;letter-spacing:2px;color:#9db6d7}h1{font-size:32px;line-height:1.35;margin:12px 0}header p{color:#c5d4e8;max-width:960px;margin:8px 0}.header-meta{display:flex;gap:26px;flex-wrap:wrap;font-size:12px;color:#b9cce3;margin-top:20px}nav{position:sticky;top:0;z-index:5;background:#fffffff5;border-bottom:1px solid var(--line);backdrop-filter:blur(8px);display:flex;gap:24px;padding:12px max(24px,calc((100vw - 1360px)/2));overflow:auto;white-space:nowrap}nav a{font-size:13px;color:#43536c}main{max-width:1408px;margin:auto;padding:25px 24px 55px}.chapter{background:white;border:1px solid var(--line);border-radius:16px;margin:0 0 24px;padding:30px;box-shadow:0 6px 26px #10223c04;scroll-margin-top:65px}.chapter-head{display:flex;gap:17px;align-items:flex-start;margin-bottom:22px}.number{font:600 14px/32px ui-monospace,monospace;color:#5b76a0;background:#edf3fc;min-width:34px;text-align:center;border-radius:8px}.chapter h2{font-size:23px;line-height:1.4;margin:0 0 5px}.deck{margin:0;color:var(--muted);font-size:14px}h3{font-size:17px;line-height:1.5;margin:12px 0 9px}p{margin:12px 0}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:22px}.metric{border:1px solid var(--line);padding:19px;border-radius:12px;background:#fafcff}.metric span{color:var(--muted);font-size:12px}.metric strong{display:block;font-size:29px;line-height:1.5;letter-spacing:-.4px}.metric small{font-size:11px;margin-top:5px}.conclusion-grid,.review-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.finding{padding:20px;border-radius:12px;border:1px solid var(--line);background:#f8fafc}.finding.success{border-top:4px solid #059669;background:#f1faf6}.finding.caution{border-top:4px solid #d97706;background:#fffbef}.finding.stop{border-top:4px solid #dc5b5b;background:#fff7f6}.finding p{font-size:14px;margin-bottom:0}.badge{display:inline-block;font-size:11px;border-radius:6px;padding:3px 8px;background:#e6edf7;color:#36557c}.badge.strong{background:#d7f2e4;color:#167249}.badge.limited{background:#fff0bd;color:#866318}.badge.action{background:#ffe2df;color:#a34338}.protocol,.note,.alert{padding:15px 18px;border-radius:9px;background:#eff5fc;color:#35506f;font-size:13px;margin-top:18px}.note{border-left:3px solid #3b82f6}.alert{background:#fff1ed;color:#9b3d22}.plots-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:start;margin-bottom:20px}figure{margin:0;padding:16px 14px 12px;background:#fbfcfe;border:1px solid #e5ebf3;border-radius:12px;min-width:0}figcaption{font-size:15px;font-weight:650;margin:0 4px 10px}figure svg{display:block;width:100%;height:auto}.figure-note{font-size:12px;color:var(--muted);margin:7px 4px 0;line-height:1.7}.downloads{font-size:11px;margin:8px 4px 0}.review-grid article{background:#f6f8fc;border-radius:10px;padding:18px}.review-grid p{font-size:13px;color:#4e6078}.process-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:24px;align-items:start}.process-text{padding:3px 10px}.process-text li{margin-bottom:9px}.process-text p{font-size:14px}.table-wrap{overflow:auto;border:1px solid #e4eaf2;border-radius:10px;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px}th{font-size:12px;font-weight:600;color:#4a5e7a;background:#f2f6fb;white-space:nowrap}th,td{padding:13px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e7edf5}tr:last-child td{border-bottom:0}td{overflow-wrap:anywhere}td b{font-size:12px}small{display:block;color:var(--muted);font-size:11px;max-width:260px;line-height:1.7;margin-top:4px}a{color:#2563eb;text-decoration:none}a:hover{text-decoration:underline}.muted{font-size:13px;color:var(--muted)}details{border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:14px 17px}summary{cursor:pointer;font-size:14px;font-weight:550;color:#35516f}details[open] summary{margin-bottom:15px}pre{font:11px/1.7 ui-monospace,monospace;background:#f6f8fb;padding:16px;border-radius:8px;white-space:pre-wrap;overflow:auto;max-height:360px}code{font-size:11px;overflow-wrap:anywhere}footer{font-size:12px;color:var(--muted);text-align:center;padding:10px 20px 35px}ul,ol{padding-left:23px}@media(max-width:1000px){.conclusion-grid,.review-grid{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}.plots-two,.process-grid{grid-template-columns:1fr}.chapter{padding:22px}header{padding:30px 24px}h1{font-size:27px}.chapter h2{font-size:21px}}@media(max-width:600px){main{padding:16px 10px}.chapter{padding:18px 14px}.metric strong{font-size:24px}nav{gap:19px;padding:10px 17px}.chapter-head{gap:10px}.plots-two{gap:18px}.metric-grid{gap:10px}.metric{padding:14px}}
    '''
    nav=[('conclusions','结论'),('routes','路线状态'),('matched','匹配结果'),('ppo-diagnosis','PPO诊断'),('nft','NFT决策'),('process','过程与下一步'),('evidence','证据附录')]
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World RL研究报告</title><style>'+css+'</style></head><body><header><div class="eyebrow">QWEN-WORLD · RL RESEARCH</div><h1>RL训练改进：场景条件安全—驾驶效率</h1><p>从场景信息、可训练表征与完整部署证据定位问题；保留可归因对照，区分阶段收益与真正的风险权衡能力。</p><div class="header-meta"><span>报告生成 '+esc(snapshot['generated_utc'])+'</span><span>最后完整评测证据 '+esc(h.utc(source_time) if source_time else '待记录')+'</span><span>常规30分钟＋阶段事件更新</span></div></header><nav>'+''.join('<a href="#'+id+'">'+name+'</a>'for id,name in nav)+'</nav><main>'+''.join(chapters)+'</main><footer>固定路径 '+esc(root/'report.html')+' · 本页只读汇总，不训练、不推理、不评分、不操作GPU或服务器</footer></body></html>'
