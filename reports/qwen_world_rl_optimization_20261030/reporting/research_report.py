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
    nft_state='已协作停止' if stopped else '停止决策已下，等待退出收据' if decision else '待停止决策／执行收据'
    analysis10=editorial.get('matched_10k',{})
    analysis20=editorial.get('matched_20k',{})
    scene_research=editorial.get('scene_safety_efficiency',{})
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
        ('匹配对照预算',number(budget),'optimizer updates；不同预算不直接排优劣'),
        ('NFT执行状态','已停' if stopped else '退出待证实' if decision else '待决策','四个完整阶段点已可判断投入方向')]
    body='<div class="note"><b>核心研究目标：根据场景信息，平衡安全与驾驶效率。</b>识别何时必须让行、何时存在安全通行窗口，以及何时只是观测信息不足；不是把安全与舒适性作为主要对立轴。SafeComfort是突出EP的局部奖励去干扰，尚未实现真正的场景条件风险权衡。</div><div class="metric-grid">'+''.join('<div class="metric"><span>'+a+'</span><strong>'+b+'</strong><small>'+c+'</small></div>' for a,b,c in cards)+'</div>'
    body+='<p><b>IL与RL两线并行：</b>IL提供并补齐离线驾驶知识基础；RL在固定IL知识基础上强化轨迹行为，不全面重做IL。知识缺口反馈IL；新IL达标后，用相同RL配方另起对照，不热切换正在训练的旧父模型。</p>'
    body+='<div class="conclusion-grid"><article class="finding success">'+evidence('strong','已确认 · 完整评测与父模型配对')+'<h3>PPO带来收益，增加预算并非单调提升</h3><p>IL70基线 '+fmt(base,6)+'。PPO_FULL在1k提高 +1.639627，配对95%区间 [＋1.214409, ＋2.039287]；10k曾达到91分以上，但20k多条路线回落。保留历史最佳checkpoint，不能默认训练更久就更好。</p></article>'
    batch_pair=contrast(analysis10,'PPO_BATCHSTD','PPO_FULL')
    if budget>=20000 and ('PPO_FULL',20000) in points and ('PPO_BATCHSTD',20000) in points:
        full20=points['PPO_FULL',20000];batch20=points['PPO_BATCHSTD',20000];safe20=points.get(('PPO_SAFE_COMFORT',20000))
        safe_pair20=contrast(analysis20,'PPO_SAFE_COMFORT','PPO_FULL')
        safe_summary=('SAFE20k−FULL '+signed(safe_pair20.get('delta_PDMS'))+'，区间 '+ci_text(safe_pair20)+'，支持该阶段局部改动收益；' if safe_pair20 else 'SAFE20k观察分 '+fmt(safe20['PDMS'],5)+'；' if safe20 else '')
        body+='<article class="finding caution">'+evidence('limited','阶段更新 · 10k不能外推')+'<h3>10k的BatchStd领先未普遍持续</h3><p>10k的+0.134370有配对支持；12k点分低于Full，20k Batch−Full仅 '+signed(batch20['PDMS']-full20['PDMS'])+' 点。'+safe_summary+'这仍不能证明策略已学会按场景平衡安全与效率。</p></article>'
    elif batch_pair.get('CI_excludes_zero') and batch_pair.get('delta_PDMS',0)>0:
        body+='<article class="finding caution">'+evidence('strong','支持 · 单seed物理log配对')+'<h3>BatchStd的10k增益有区间支持</h3><p>相对PPO_FULL '+signed(batch_pair['delta_PDMS'])+' 点，95%区间 '+ci_text(batch_pair)+'。这是本轮固定seed与10k阶段的证据；SAFE−FULL、BATCH−SAFE区间仍跨零，不把数值排名当作普遍算法结论。</p></article>'
    else:body+='<article class="finding caution">'+evidence('limited','待判断 · 小差需要匹配比较')+'<h3>新增trick的优劣尚需区间支持</h3><p>中心化std、训练Comfort塑形和CISPO只比较同预算、同人口的完整结果。数值领先可以记录；配对区间未完成或跨零时，不宣布算法胜出。</p></article>'
    body+='<article class="finding stop">'+evidence('action','决策 · 低潜力路线释放预算')+'<h3>NFT缺少持续部署收益</h3><p>'+(number(update(nft))+'更新完整分 '+fmt(nft['PDMS'],6)+'，相对IL70 '+signed(nft['PDMS']-base)+'；已测最高仅 '+fmt(nft_best,6)+'。' if nft else '等待完整NFT结果。')+' '+esc(nft_state)+'。</p></article></div>'
    body+='<div class="protocol">统一读数：NAVTEST 12,146场景 · 单候选ODE K10 / shift5 · 训练seed42 / 推理seed2101。固定种子研究，不安排多种子。TRAIN panel与随机训练候选分数另列。</div>'
    if snapshot.get('warnings'):body+='<div class="alert">'+esc('；'.join(snapshot['warnings']))+'</div>'
    chapter('01','conclusions','现在可以得出的结论','先区分已观察到的提升、尚未证明的算法优势，以及已经决定调整的投入。',body)

    # 02: a compact ledger of doing / done / stopped / next.
    route_rows=[]
    for arm in display_arms:
        run=runs.get(arm,{})
        result=latest.get(arm)
        status=run.get('owner_receipt',{}).get('status') or run.get('status','未登记')
        if arm in stop_receipts:state='已协作停止';next_action='保留checkpoint、optimizer/RNG与结果；预算转向更高优先研究'
        elif arm in decisions:state='停止决策已下，等待退出收据';next_action='协作保存原生checkpoint，等待退出确认'
        elif status=='WAITING_OWN_TRAIN_FEEDBACK':state='等待自己的TRAIN反馈';next_action='bank就绪后继续；不等待其他臂或Test'
        elif status.startswith('WAITING'):state='等待容量／阶段条件';next_action='按退出事件与容量heartbeat重排'
        elif run.get('job'):
            state=('已观察全rank原生日志' if run.get('journal_ranks')==run.get('expected_ranks',8) and h.finite(run.get('updates')) else
                   'worker已启动，原生更新待日志' if run.get('job_receipts',{}).get('RUNNING.json') else '已登记派发，启动／更新待收据')
            next_action='当前目标 '+number(run.get('stage_budget_updates'))+'；周期完整评测异步'
        else:state='未派发';next_action='等待研究决策与资源'
        native_cell=('停止于 '+number(stop_receipts[arm].get('checkpoint',{}).get('updates')) if arm in stop_receipts else
                     number(run.get('updates'))+' / '+number(run.get('stage_budget_updates')))
        route_rows.append(['<b>'+esc(arm)+'</b><small>'+esc(IDEAS.get(arm,run.get('hypothesis','新登记独立pilot')))+'</small>',
            fmt(result['PDMS'],4)+'<small>'+number(update(result))+'更新 · 完整Test</small>' if result else '暂无完整Test',
            native_cell,
            esc(state),esc(next_action)])
    body=table(['路线与单因素','最近完整结果','原生日志 / 当前目标','现在在做什么','下一步'],route_rows)
    ongoing_count=sum(a in runs and a not in decisions for a in display_arms)
    anchor_result=latest.get('PPO_ANCHOR40')
    anchor_result_text=('ANCHOR40最近完整Test为'+fmt(anchor_result['PDMS'],6)+' / '+number(update(anchor_result))+'更新，单独保留其预算。' if anchor_result else 'ANCHOR40是独立单因素，尚无正式完整Test成绩。')
    stopped_names='、'.join(sorted(stop_receipts)) or '暂无'
    pending_stop_names='、'.join(sorted(set(decisions)-set(stop_receipts))) or '暂无'
    body+='<p class="note">已协作停止：'+esc(stopped_names)+'。已决定停止、待退出收据：'+esc(pending_stop_names)+'。当前 '+number(ongoing_count)+' 条未止损研究路线推进；被停止路线的全部已测checkpoint继续保留，包含Batch10k历史好点。'+anchor_result_text+'</p>'
    resources=snapshot.get('active_resources',[]);uuids={r.get('uuid') for r in resources if r.get('uuid')}
    body+='<p class="muted">最新已落盘资源记录：'+number(len(snapshot.get('current_authorized_hosts') or []))+'主机 / '+number(len(uuids))+'库存UUID；授权、准入合格和实际部署分开记录，库存不等于空闲。运营细节见折叠附录。</p>'
    if snapshot.get('resource_inventory_unknown_hosts'):body+='<p class="muted">当前重新授权但物理库存／就绪度未核实的主机：'+esc('、'.join(snapshot['resource_inventory_unknown_hosts']))+'；不填8张可用卡，不将连接失败写成部署完成。</p>'
    temporary=snapshot.get('resource_temporary_window',{})
    reauthorization=snapshot.get('resource_reauthorization_note',{})
    if reauthorization:
        granted=set(reauthorization.get('hosts',[]))<=set(snapshot.get('current_authorized_hosts') or [])
        try:window_ended=snapshot.get('generated_unix',0)>=datetime.fromisoformat(reauthorization['expires_utc'].replace('Z','+00:00')).timestamp()
        except (KeyError,ValueError):window_ended=False
        window_status=('该8小时窗口已结束，当前有效许可看最新grant。' if window_ended else '已进入当前有效grant。' if granted else 'grant／库存记录正在更新，旧撤回记录不能当作永久禁用。')
        body+='<p class="muted">用户新增8小时再授权：vla-zt3 / tsinghua-zt2，'+esc(reauthorization.get('starts_utc'))+' → '+esc(reauthorization.get('expires_utc'))+'。'+window_status+'这不表示新部署已发生；当前状态以时间窗与独立startup收据为准。</p>'
    chapter('02','routes','路线状态：已完成、进行中、停止与下一步','研究预算依据证据调整；控制进程的状态不替代模型更新或评测完成证据。',body)

    # 03: matched-budget curves and point deltas sit side by side.
    series={a:[(update(r),r['PDMS']) for r in test if r.get('arm')==a and update(r)<=budget] for a in ALGO_ARMS}
    line=curve(root,h,'research_matched_ppo','PPO算法完整NAVTEST：同预算曲线',series,
        caption='纵轴放大用于看路线差异；IL70='+fmt(base,4)+'见首卡总提升。数据路线与NFT分别另图。')
    paired_source=analysis20 if budget==20000 and analysis20.get('status')=='COMPLETE' else analysis10 if budget==10000 else {}
    base_row=points.get(('PPO_FULL',budget))
    def delta_draw(plt):
        fig,ax=plt.subplots(figsize=(7,3.9),layout='constrained')
        arms=[a for a in ALGO_ARMS if a!='PPO_FULL' and (a,budget) in points]
        for i,arm in enumerate(arms):
            delta=points[arm,budget]['PDMS']-base_row['PDMS'] if base_row else 0
            pair=contrast(paired_source,arm,'PPO_FULL');interval=pair.get('CI95_PDMS',[])
            ax.barh(i,delta,color=COLORS[arm],height=.58,alpha=.85)
            if len(interval)==2:ax.errorbar(delta,i,xerr=[[max(0,delta-interval[0])],[max(0,interval[1]-delta)]],fmt='o',color='#0f172a',capsize=4,ms=3)
            else:ax.text(delta+.012 if delta>=0 else delta-.012,i,f'{delta:+.3f}',va='center',ha='left' if delta>=0 else 'right',fontsize=9)
        ax.set_yticks(range(len(arms)),[SHORT[a] for a in arms]);ax.axvline(0,color='#475569',lw=1)
        ax.set(xlabel='PDMS points vs PPO_FULL',title=f'Matched {budget//1000}k checkpoint');ax.grid(axis='x',alpha=.18);ax.margins(x=.25)
        return fig
    delta=figure(root,h,'research_matched_deltas','相对PPO_FULL的点差与配对区间',delta_draw,
        '有区间才画误差线；没有误差线表示配对分析尚未完成，不能据条形长度宣布优劣。')
    rows=[]
    table_budgets=[u for u in (1000,5000,10000,12000,20000) if any((a,u)in points for a in ALGO_ARMS)]
    for arm in ALGO_ARMS:
        cells=[]
        for u in table_budgets:
            row=points.get((arm,u));cells.append(fmt(row['PDMS'],6) if row else '—（无完整点）')
        rows.append(['<b>'+esc(arm)+'</b>']+cells)
    body='<div class="plots-two">'+line+delta+'</div>'+table(['完整NAVTEST']+[str(u//1000)+'k' for u in table_budgets],rows)
    if budget>=20000:
        change_rows=[]
        for arm in ALGO_ARMS:
            earlier=points.get((arm,10000));later=points.get((arm,20000))
            if earlier and later:change_rows.append([esc(arm),fmt(earlier['PDMS'],6),fmt(later['PDMS'],6),signed(later['PDMS']-earlier['PDMS'],6)])
        body+=table(['同一路线','10k完整分','20k完整分','20k−10k：观察点差'],change_rows)+'<p class="note">20k结果不支持把10k的最佳checkpoint优势外推为普遍持续收益。阶段点回落与算法间差异分开解释；20k是否存在可重复的安全—效率改进，需看同预算配对及场景分组，不能只追一个总分。</p>'
    if analysis20.get('status')=='COMPLETE':
        rows20=[[esc(key.replace('_minus_',' − ')),signed(row.get('delta_PDMS')),ci_text(row),verdict(row)]for key,row in analysis20.get('contrasts',{}).items()]
        body+=table(['20k同预算配对','点差','95%区间：PDMS点','判断'],rows20)+link(editorial.get('matched_20k_path'),'20k完整配对与组件拆解')
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
        arms=[a for a in PPO_ARMS if (a,budget) in points]
        values=np.array([[100*(points[a,budget].get('means',{}).get(m,0)-base_row.get('means',{}).get(m,0)) for m in metrics]for a in arms]) if base_row else np.zeros((len(arms),5))
        limit=max(.05,float(abs(values).max())) if values.size else .05
        image=ax.imshow(values,cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(5),['NC','DAC','TTC','EP','Comfort']);ax.set_yticks(range(len(arms)),[SHORT[a] for a in arms])
        for i in range(len(arms)):
            for j in range(5):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',fontsize=9)
        fig.colorbar(image,ax=ax,label='Percentage points vs Full',shrink=.8)
        return fig
    heat=figure(root,h,'research_ppo_components','同预算安全、效率与其他评价分量',components_draw,'每格是相对PPO_FULL的均值百分点。Comfort保留为官方评价分量，核心研究问题是场景条件安全—效率权衡。')
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
    data_series={a:[(update(r),r['PDMS'])for r in test if r.get('arm')==a and update(r)<=budget]for a in ['PPO_FULL','PPO_HARD','PPO_ADAPTIVE']}
    data_chart=curve(root,h,'research_data_navtest','数据路线的完整NAVTEST',data_series,caption='只连接已完整阶段，未完成阶段不补点；纵轴放大显示数据路线差异。')
    adaptive_tenk=points.get(('PPO_ADAPTIVE',10000))
    adaptive_sentence=('Adaptive的10k完整Test为'+fmt(adaptive_tenk['PDMS'],6)+'；' if adaptive_tenk else 'Adaptive的5k Test较Full高0.157064点，但10k尚未完整；')
    body='<h3>新的研究主线：场景条件安全—驾驶效率</h3><p>先审计当前观测能支持哪些风险与交互判断，再定义场景分组、表征诊断与策略判据。不能仅凭低EP把等待判为低效，也不能把所有停滞都解释为安全；已确认风险与信息不足须分开。</p>'
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
            body+='<p><b>研究判据：</b>'+esc(objective.get('safety','按场景检查安全约束与尾部'))+' '+esc(objective.get('efficiency','在场景制约内评价可实现效率'))+'</p><p>'+esc(objective.get('all_fail_rule','8个候选全未通过不等于场景无安全解。'))+'</p><p class="muted">实施状态：'+esc(objective.get('status','研究方案准备中，未声明新策略已获收益'))+'</p>'
        preliminary=scene_research.get('preliminary_training_evidence',{})
        if preliminary:
            body+='<p>初步信用分配证据（'+esc(preliminary.get('scope','既有TRAIN候选，离线诊断'))+'）：unsafe正优势候选 '+number(preliminary.get('unsafe_positive_advantage'))+'，其中 '+number(preliminary.get('unsafe_positive_with_feasible_alternative'))+' 个有通过候选可替代，'+number(preliminary.get('unsafe_positive_EP_above_all_feasible'))+' 个EP高于所有通过候选。另有 '+number(preliminary.get('groups_with_at_least2_feasible'))+' 组至少两个通过候选，'+number(preliminary.get('groups_feasible_EP_span_ge_005'))+' 组的通过候选EP跨度≥0.05。这里“通过”仅指离线NC=DAC=TTC=1，不等于掌握全部场景风险；这些是待验证的信用分配线索。</p>'
        body+='<p>'+link(root/'analysis'/'SCENE_SAFETY_EFFICIENCY_RESEARCH.json','场景安全—效率研究目标、证据与实施状态')+'</p>'+detail('场景研究原始登记／诊断', '<pre>'+esc(json.dumps(scene_research,ensure_ascii=False,indent=2))+'</pre>')
        brief=editorial.get('scene_brief_20k',{})
        candidate=brief.get('minimal_CPU_candidate',{})
        if candidate:body+='<p class="note">CPU信用分配原型把安全优势与“通过候选子集内中心化”的EP优势分开，并保留全失败fallback和绝对成本。该简报记录的是CPU结构检查；是否形成正式pilot，看新注册／worker收据，不能把代数性质当作训练成绩或安全保证。低EP或停车并非自动错误，必须先检查同组是否有通过指标的可行替代。'+link(root/'analysis'/'SCENE_SAFETY_EFFICIENCY_BRIEF_20K.json','20k解释、局部方法边界与CPU原型')+'</p>'
    else:body+='<p class="muted">场景条件诊断／表征方案由研究主进程准备，正式SCENE_SAFETY_EFFICIENCY_RESEARCH.json待发布；当前是研究目标与判据，尚非新策略成绩。</p>'
    body+='<h3>数据覆盖与反馈：支持诊断，不替代场景解释</h3><div class="plots-two">'+data_chart+panel_chart+'</div><p>固定Hard的5k→10k下降0.118841点；同10k Full−Hard为+0.283893，配对区间[+0.011398,+0.520610]且NC/DAC/TTC均更高，支持本轮机会成本停止决策。'+adaptive_sentence+'其TRAIN先升后降，不能把反馈panel回升当泛化证明。</p>'
    if adaptive_analysis:
        coverage=adaptive_analysis.get('coverage',{});feedback=coverage.get('feedback1024',{});extra=coverage.get('additional3072_feedback_unseen',{});timing=adaptive_analysis.get('feedback_time_summary',{})
        body+=table(['已完成至'+number(coverage.get('completed_through_updates'))+'更新的曝光','证据','含义'],[
            ['旧反馈1024占比',fmt(feedback.get('exposure_fraction'),2,percent=True)+'%；最高'+number(feedback.get('maximum_exposures'))+'次','反馈子集被大量重复，不能代替总体覆盖'],
            ['全人口已见场景',number(coverage.get('population_unique_exposed'))+' / 103,288（'+fmt(coverage.get('population_coverage'),2,percent=True)+'%）','选择分布与均匀全量路线不同'],
            ['额外3072中未被RL采样',number(extra.get('never_exposed'))+' / 3,072','称feedback-unseen；并非全部training-unseen'],
            ['正常反馈等待中位数',fmt((timing.get('steady_after1k_gap_median_seconds')or 0)/60,2)+'分钟','墙钟含export/容量/评测，不等于独占GPU成本']])
        body+='<div class="note">至9k，旧panel占57.83%的实际曝光；8k曾有36.5分钟异常反馈间隔、6次pre-child资源拒绝，不能当作模型/scorer错误。也不应把少量zero-PG组描述为大量零方差浪费。'+link(root/'analysis'/'ADAPTIVE_TRAIN_PROCESS_20261030.json','完整曝光、日志与反馈耗时')+'</div>'
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
    body+='<h3>两个独立10k pilot：'+('新登记执行进度见路线表' if new_pilots else '准备中，尚无启动收据')+'</h3>'+table(['研究问题','唯一主要改动','保持不变／判据'],[
        ['给定IL知识能否更充分适配轨迹任务','memory学习率5e−6 → 2e−5','固定IL父模型与RL配方其余项；检验适配强度，不能代表补齐全部驾驶知识。'],
        ['安全与通过候选内效率能否分别分配信用','安全／可行EP分量优势：PPO_SAFETY_EP候选','不叠加memory LR试验；同预算检查新增风险、必要等待和条件效率。']])+'<p>两项由研究主进程实现与注册，实际是否启动看独立登记／worker收据。PPO_SAFETY_EP是待验证的候选指标条件信用分解：通过者内部才分配效率信用，全未通过时用指标缺失成本给安全方向；它没有新增场景输入，也不代表已具备场景风险推理。新IL达标后的RL对照另起实验；现有已封存模型与旧训练不热改。</p>'
    actions=[['P0','场景条件安全—效率与表征诊断','合法等待、低风险停滞、确认风险／观测不确定分开','IL补知识、RL做适配与轨迹强化，不把局部奖励去干扰当场景策略'],
        ['已完成／继续拆解','解释20k匹配结果，保留Full控制','SAFE总分阶段增益、BATCH/CISPO的NC/TTC风险','不把10k领先外推；不因一个总分替代场景分组'],
        ['已完成／收据','停止NFT、固定hard-only、BATCH与CISPO后续预算','逐臂STOPPED、退出与checkpoint收据','保留旧结果与历史好点；不是模型失败'],
        ['已完成／后续解释','三模型固定TRAIN4096诊断','旧反馈1024、反馈外3072与实际曝光分层','面板特化迹象，不等于反馈外场景退化'],
        ['已在做','独立运行ANCHOR40单因素','降低anchor压力是否改善后续canonical结果','只改β，不叠SAFE/归一化；成绩按该臂完整结果读数'],
        ['P1','依据4096 final与场景诊断决定数据策略','反馈panel特化与反馈外场景的实际曝光分层','覆盖回放不是提分结果；不将TRAIN投影当Test']]
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
        activation_rows=[[esc(host),esc(entry.get('status','待记录')),number(entry.get('physical_GPU_count')),number(len(entry.get('headroom32GiB_accounted_gpu_indices',[]))),number(entry.get('new_job_deployments'))]for host,entry in activation['hosts'].items()]
        ops+='<h3>本次8h再授权与准入收据</h3>'+table(['主机','收据状态','物理GPU','32GiB准入合格卡','该步骤新部署'],activation_rows)+'<p>保护checkpoint时间 '+esc(activation.get('checkpoint_guard_utc','待记录'))+'；到期 '+esc(activation.get('expires_utc','待记录'))+'。许可和库存审计不等于部署。'+link(snapshot.get('resource_activation_path'),'实际准入收据，无报告侧host操作')+'</p>'
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
