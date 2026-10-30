"""Editorial research report; evidence collection stays in render_report.py."""
from __future__ import annotations

from collections import defaultdict
import io
import json
import math
import os
from pathlib import Path
import tempfile

ARMS=['PPO_FULL','PPO_BATCHSTD','PPO_SAFE_COMFORT','CISPO_FULL','PPO_HARD','PPO_ADAPTIVE','PPO_ANCHOR40','NFT_FULL']
PPO_ARMS=ARMS[:5]
ALGO_ARMS=ARMS[:4]
COLORS=dict(zip(ARMS,['#2563eb','#059669','#7c3aed','#d97706','#64748b','#0891b2','#be185d','#dc2626']))
SHORT={'PPO_FULL':'Full','PPO_BATCHSTD':'BatchStd','PPO_SAFE_COMFORT':'SafeComfort','CISPO_FULL':'CISPO','PPO_HARD':'Hard','PPO_ADAPTIVE':'Adaptive','PPO_ANCHOR40':'Anchor40','NFT_FULL':'NFT'}
IDEAS={'PPO_FULL':'全量103,288；原始PPO对照','PPO_BATCHSTD':'仅改中心化优势的全局std分母','PPO_SAFE_COMFORT':'仅改安全候选的训练Comfort奖励','CISPO_FULL':'全量数据；仅改策略目标','PPO_HARD':'困难15,171；数据人口对照','PPO_ADAPTIVE':'训练内低分／退化／覆盖再采样','PPO_ANCHOR40':'仅将均值anchor权重β50改为40','NFT_FULL':'纯ODE采样＋NFT目标／EMA'}


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
    base=snapshot.get('ppo_full_first_stage_paired_analysis',{}).get('reference_PDMS',88.79290990699972)
    panel_base=next((r['PDMS'] for r in panel if r.get('evaluation_kind')=='IL_BASELINE'),93.64668412698353)
    editorial=snapshot.get('editorial_analysis',{})
    decision=editorial.get('nft_decision',{})
    stopped=editorial.get('nft_stopped',{})
    decisions=editorial.get('decisions',{});stop_receipts=editorial.get('stopped_receipts',{})
    nft_state='已协作停止' if stopped else '停止决策已下，等待退出收据' if decision else '待停止决策／执行收据'
    analysis10=editorial.get('matched_10k',{})
    adaptive_analysis=editorial.get('adaptive_process',{})
    budgets=sorted({update(r) for r in test if r.get('arm') in PPO_ARMS})
    budget=max((u for u in budgets if ('PPO_FULL',u) in points and sum((a,u) in points for a in PPO_ARMS)>=3),default=1000)
    matched=[points[a,budget] for a in PPO_ARMS if (a,budget) in points]
    latest={a:max((r for r in test if r.get('arm')==a),key=update,default=None) for a in ARMS}
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
    cards=[('当前最高完整点',fmt(best['PDMS'],4) if best else '待评测',esc(best_name)+' · RL '+number(update(best)) if best else ''),
        ('对IL70的观察增益',signed(best['PDMS']-base) if best else '—','同一完整NAVTEST人口；点差不等于算法因果'),
        ('匹配对照预算',number(budget),'optimizer updates；不同预算不直接排优劣'),
        ('NFT执行状态','已停' if stopped else '退出待证实' if decision else '待决策','四个完整阶段点已可判断投入方向')]
    body='<div class="metric-grid">'+''.join('<div class="metric"><span>'+a+'</span><strong>'+b+'</strong><small>'+c+'</small></div>' for a,b,c in cards)+'</div>'
    body+='<div class="conclusion-grid"><article class="finding success">'+evidence('strong','已确认 · 完整评测与父模型配对')+'<h3>PPO路线已带来实质收益</h3><p>IL70基线 '+fmt(base,6)+'。PPO_FULL在1k的增益为 +1.639627 点，配对95%区间 [＋1.214409, ＋2.039287]；当前较高预算的PPO点继续达到91分以上。先保留这个对照，再判断新增改动。</p></article>'
    batch_pair=contrast(analysis10,'PPO_BATCHSTD','PPO_FULL')
    if batch_pair.get('CI_excludes_zero') and batch_pair.get('delta_PDMS',0)>0:
        body+='<article class="finding caution">'+evidence('strong','支持 · 单seed物理log配对')+'<h3>BatchStd的10k增益有区间支持</h3><p>相对PPO_FULL '+signed(batch_pair['delta_PDMS'])+' 点，95%区间 '+ci_text(batch_pair)+'。这是本轮固定seed与10k阶段的证据；SAFE−FULL、BATCH−SAFE区间仍跨零，不把数值排名当作普遍算法结论。</p></article>'
    else:body+='<article class="finding caution">'+evidence('limited','待判断 · 小差需要匹配比较')+'<h3>新增trick的优劣尚需区间支持</h3><p>中心化std、训练Comfort塑形和CISPO只比较同预算、同人口的完整结果。数值领先可以记录；配对区间未完成或跨零时，不宣布算法胜出。</p></article>'
    body+='<article class="finding stop">'+evidence('action','决策 · 低潜力路线释放预算')+'<h3>NFT缺少持续部署收益</h3><p>'+(number(update(nft))+'更新完整分 '+fmt(nft['PDMS'],6)+'，相对IL70 '+signed(nft['PDMS']-base)+'；已测最高仅 '+fmt(nft_best,6)+'。' if nft else '等待完整NFT结果。')+' '+esc(nft_state)+'。</p></article></div>'
    body+='<div class="protocol">统一读数：NAVTEST 12,146场景 · 单候选ODE K10 / shift5 · 训练seed42 / 推理seed2101。固定种子研究，不安排多种子。TRAIN panel与随机训练候选分数另列。</div>'
    if snapshot.get('warnings'):body+='<div class="alert">'+esc('；'.join(snapshot['warnings']))+'</div>'
    chapter('01','conclusions','现在可以得出的结论','先区分已观察到的提升、尚未证明的算法优势，以及已经决定调整的投入。',body)

    # 02: a compact ledger of doing / done / stopped / next.
    route_rows=[]
    for arm in ARMS:
        run=runs.get(arm,{})
        result=latest.get(arm)
        status=run.get('owner_receipt',{}).get('status') or run.get('status','未登记')
        if arm in stop_receipts:state='已协作停止';next_action='保留checkpoint、optimizer/RNG与结果；预算转向更高优先研究'
        elif arm in decisions:state='停止决策已下，等待退出收据';next_action='协作保存原生checkpoint；退出确认前不写已停'
        elif status=='WAITING_OWN_TRAIN_FEEDBACK':state='等待自己的TRAIN反馈';next_action='bank就绪后继续；不等待其他臂或Test'
        elif status.startswith('WAITING'):state='等待容量／阶段条件';next_action='按退出事件与容量heartbeat重排'
        elif run.get('job'):state='原生训练／独立pipeline';next_action='继续当前目标 '+number(run.get('stage_budget_updates'))+'；周期完整评测异步'
        else:state='未派发';next_action='等待研究决策与资源'
        native_cell=('停止于 '+number(stop_receipts[arm].get('checkpoint',{}).get('updates')) if arm in stop_receipts else
                     number(run.get('updates'))+' / '+number(run.get('stage_budget_updates')))
        route_rows.append(['<b>'+esc(arm)+'</b><small>'+IDEAS[arm]+'</small>',
            fmt(result['PDMS'],4)+'<small>'+number(update(result))+'更新 · 完整Test</small>' if result else '暂无完整Test',
            native_cell,
            esc(state),esc(next_action)])
    body=table(['路线与单因素','最近完整结果','原生日志 / 当前目标','现在在做什么','下一步'],route_rows)
    ongoing_count=sum(a in runs and a not in decisions for a in ARMS)
    body+='<p class="note">已完成：共同父模型、全量人口、严格原生恢复、周期完整评测与独立pipeline。NFT与固定hard-only已协作停止；当前 '+number(ongoing_count)+' 条研究路线推进，不按小分差自动淘汰。ANCHOR40是新单因素，尚无正式训练成绩。</p>'
    resources=snapshot.get('active_resources',[]);uuids={r.get('uuid') for r in resources if r.get('uuid')}
    body+='<p class="muted">资源许可 '+number(len(snapshot.get('current_authorized_hosts') or []))+'主机 / '+number(len(uuids))+'库存UUID，库存不等于空闲。新增节点实际使用8张训练卡＋7张评测卡；GPU6待清。vla-zt3与tsinghua-zt2仍排除。运营细节见折叠附录。</p>'
    chapter('02','routes','路线状态：已完成、进行中、停止与下一步','研究预算依据证据调整；控制进程的状态不替代模型更新或评测完成证据。',body)

    # 03: matched-budget curves and point deltas sit side by side.
    series={a:[(update(r),r['PDMS']) for r in test if r.get('arm')==a and update(r)<=budget] for a in ALGO_ARMS}
    line=curve(root,h,'research_matched_ppo','PPO算法完整NAVTEST：同预算曲线',series,
        caption='纵轴放大用于看路线差异；IL70='+fmt(base,4)+'见首卡总提升。数据路线与NFT分别另图。')
    matching_budget=analysis10.get('matching_evidence',{}).get('RL_updates',10000)
    paired_source=analysis10 if budget==matching_budget else {}
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
    for arm in ALGO_ARMS:
        cells=[]
        for u in (1000,5000,10000):
            row=points.get((arm,u));cells.append(fmt(row['PDMS'],6) if row else '待完整评测')
        rows.append(['<b>'+esc(arm)+'</b>']+cells)
    body='<div class="plots-two">'+line+delta+'</div>'+table(['完整NAVTEST','1k','5k','10k'],rows)
    if analysis10.get('status')=='COMPLETE':
        paired_rows=[]
        for key,row in analysis10.get('contrasts',{}).items():
            if 'PPO_HARD' in key:continue
            paired_rows.append([esc(key.replace('_minus_',' − ')),signed(row.get('delta_PDMS')),ci_text(row),verdict(row)])
        body+=table(['10k配对','点差','95%区间：PDMS点','可支持的判断'],paired_rows)+link(root/'analysis'/'PORTFOLIO_10K_PAIRED.json','10k完整配对与匹配证据')
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
    heat=figure(root,h,'research_ppo_components','同预算安全、进度与舒适性变化',components_draw,'每格是相对PPO_FULL的均值百分点；不同分量的变化不是同等大小的PDMS贡献。')
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
    if batch_pair:
        components=batch_pair.get('safety_metrics',{});transitions=batch_pair.get('paired',{}).get('safety_transitions',{});zeros=batch_pair.get('paired',{})
        loss=batch_pair.get('loss_decomposition',{});unsafe_loss=loss.get('new_safety_violation_Gminus_PDMS');total_loss=batch_pair.get('G_minus_PDMS')
        fraction=unsafe_loss/total_loss if h.finite(unsafe_loss) and h.finite(total_loss) and total_loss<0 else None
        nc=transitions.get('no_at_fault_collisions',{})
        body+='<div class="note"><b>BatchStd 10k的收益与代价：</b>整体增益主要伴随EP '+signed(components.get('ego_progress',{}).get('delta_percentage_points'),5)+' pp；NC '+signed(components.get('no_at_fault_collisions',{}).get('delta_percentage_points'),5)+' pp（新增失败'+number(nc.get('new_failure'))+'、修复'+number(nc.get('repaired_failure'))+'），DAC '+signed(components.get('drivable_area_compliance',{}).get('delta_percentage_points'),5)+' pp、TTC '+signed(components.get('time_to_collision_within_bound',{}).get('delta_percentage_points'),5)+' pp。零分修复'+number(zeros.get('repaired_zero'))+'、新增'+number(zeros.get('new_zero'))+'。'+number(loss.get('new_safety_violation_degraded_scenes'))+'个新增安全违规的下降场景贡献G− '+fmt(unsafe_loss,5)+'，占该对照全部损失约'+fmt(fraction,1,percent=True)+'%。<b>继续主线，同时在20k重点检查NC风险，保留Full控制。</b></div>'
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
        body+='<h3>同12k安全复查已登记，与20k训练并行</h3><p>利用已释放评测槽提前复查Full与BatchStd的收益／NC风险是否持续，不空等20k。导出与异步派发由独立调度推进；不改变learner20k目标、不新增seed，probe失败不停止主线训练。</p>'+table(['路线','精确预算','完整PDMS','NC：0–100','证据'],probe_rows)+'<p>'+link(root/'ADDITIONAL_NAVTEST_PROBES.json','两份12,000更新完整Navtest注册')+'</p>'
    body+='<p>PPO_FULL 1k修复原零分 '+number(paired.get('repaired_zero'))+' 个，新增零分 '+number(paired.get('new_zero'))+' 个。改善场景数量少于小幅下降场景，均值仍明显上升；只数“涨了多少场景”会漏掉严重失败修复的权重。</p>'
    scale=snapshot.get('advantage_normalization',{}).get('arms',{}).get('PPO_FULL',{}).get('raw_to_centered_scale',{}).get('mean')
    body+='<div class="review-grid"><article><h3>BatchStd：主线继续匹配20k</h3><p>基线本来已做组内中心＋全8rank原始reward std。新臂只改中心化优势的全局std分母；首1k TRAIN平均尺度倍率 '+fmt(scale,3)+'。10k配对区间支持收益，下一门槛检验是否持续。</p></article><article><h3>SafeComfort：保留到下一门槛</h3><p>10k相对Full的小增益区间跨零，EP提高同时伴随安全与Comfort小降。训练塑形分不等于PDMS；不因为点分稍高就叠加到BatchStd。</p></article><article><h3>固定hard-only：停止后保留早期checkpoint</h3><p>Full−Hard在10k为+0.283893，区间[+0.011398,+0.520610]，NC/DAC/TTC均更高。该固定困难日程的机会成本已足够支持停止；不否定困难样本作为混合数据组件。</p></article></div>'
    body+='<p class="muted">两个新臂更新前的首个fresh collection在全部8rank与PPO_FULL逐值一致；后续rollout会随策略改变。近10k SAFE塑形后的PG有效组约90.5%，FULL为98.9%，奖励重排也改变了对比信号。现有日志仅含总DDP梯度范数，不据scalar loss声称anchor梯度占比。'+link(root/'analysis'/'PROCESS_DIAGNOSTIC_10K.json','近10k过程诊断')+' · '+link(root/'analysis'/'ALGORITHM_ARMS_FIRST_COLLECTION_MATCH.json','起点匹配')+' · '+link(root/'analysis'/'safe_comfort_counterfactual_u001000.json','奖励反事实与风险')+'</p>'
    if 'PPO_ANCHOR40' in runs:
        body+='<div class="note"><b>已新增PPO_ANCHOR40：</b>使用NFT释放的8卡研究槽，从同一IL70、seed42与全量数据开始，仅β50→40，保留RAWstd，不叠加SAFE或中心化std。目的是检验PG与anchor压力的近似再平衡；A cap5与Adam使它不具严格等价变换。近10k PPO阻塞仅约2.4%–2.6%，梯度clip1未触发，因此当前不优先Clip-Higher。'+link(root/'EXPERIMENT_PPO_ANCHOR40_REGISTRATION.json','正式注册与科学边界')+'</div>'
    chapter('04','ppo-diagnosis','PPO领先路线：分数为何变化，风险在哪里','安全、进度、奖励变换和更新尺度分别检查，避免只保留有利解释。',body)

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
    data_chart=curve(root,h,'research_data_navtest','数据路线的完整NAVTEST',data_series,caption='只连接已完整阶段，Adaptive未完成的10k不补点；纵轴放大显示数据路线差异。')
    adaptive_tenk=points.get(('PPO_ADAPTIVE',10000))
    adaptive_sentence=('Adaptive的10k完整Test为'+fmt(adaptive_tenk['PDMS'],6)+'；' if adaptive_tenk else 'Adaptive的5k Test较Full高0.157064点，但10k尚未完整；')
    body='<div class="plots-two">'+data_chart+panel_chart+'</div><p>固定Hard的5k→10k下降0.118841点；同10k Full−Hard为+0.283893，配对区间[+0.011398,+0.520610]且NC/DAC/TTC均更高，支持本轮机会成本停止决策。'+adaptive_sentence+'其TRAIN先升后降，不能把反馈panel回升当泛化证明。</p>'
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
    actions=[['P0','BatchStd主线＋Full控制完成匹配20k','10k正向配对区间与后续持续性','SAFE/CISPO保留到下一门槛，不盲目组合'],
        ['已完成／收据','停止NFT与固定hard-only，回收预算','逐臂STOPPED、退出与checkpoint收据','保留旧结果；不是模型失败'],
        ['P1','诊断反馈panel外的TRAIN场景','固定人口、曝光分层、基线与当前策略配对','CPU缓存审计不等于已执行模型评测'],
        ['已在做','独立运行ANCHOR40单因素','降低anchor压力是否改善后续canonical结果','只改β，不叠SAFE/归一化；尚无正式成绩'],
        ['P1','依据4096诊断决定覆盖策略','反馈外场景与实际RL曝光分层','CPU回放不当作提分结果，不盲叠技巧']]
    body+=table(['优先级','下一步','用什么证据决定','执行边界'],actions)
    body+='<p>资源安排依据用户授权的证据决策调整；其余健康路线不按小分差自动剪枝。'+link(root/'RESEARCH_ALLOCATION_POLICY_20261030.json','当前研究资源决策')+'</p>'
    chapter('06','process','训练过程与下一步：从问题选择实验','区分“训练学到了什么”“部署有没有更好”“流程是否浪费等待”。',body)

    # 07: details are preserved, but folded away from the argument.
    test_rows=[[esc(r.get('arm')),number(update(r)),fmt(r['PDMS'],6),number(r.get('scenes')),link(r.get('evidence'),'SUMMARY')]for r in sorted(test,key=lambda r:(r.get('arm',''),update(r)))]
    evaluation_rows=[]
    for r in sorted(snapshot['results'],key=lambda r:(str(r.get('cohort')),str(r.get('arm')),update(r))):
        evaluation_rows.append([esc(r.get('arm','')),esc(r.get('cohort','')),number(update(r)),number(r.get('scenes'))+' / '+number(r.get('expected_scenes')),
            fmt(r.get('display_PDMS'),6) if r.get('publishable_complete') else '—（未完整不发布均值）',link(r.get('evidence'),'原始结果')])
    run_rows=[[esc(a),esc(runs[a].get('status')),number(runs[a].get('updates')),esc(json.dumps(runs[a].get('rank_updates',{}))),
        number(runs[a].get('new_candidates')),fmt(runs[a].get('recent_clipfrac'),2,percent=True)+'%',h.gradient_fmt(runs[a].get('recent_gradient')),
        esc(runs[a].get('host')),link(runs[a].get('job'),'worker'),link(runs[a].get('source'),'source')]for a in ARMS if a in runs]
    inputs=[[link(r['path'],Path(r['path']).name),h.utc(r['mtime']),number(r['bytes']),'<code>'+r['sha256'][:20]+'…</code>']for r in snapshot['inputs']]
    identity='<p>共同原始World IL70k父模型；全量NAVTRAIN103,288包含旧DEV1,696，后者在全量臂中只是训练内诊断。困难臂15,171；NAVTEST12,146、44物理log，与TRAIN隔离。固定训练seed42／推理seed2101，单候选ODE K10 / shift5；SAFE_COMFORT训练塑形分独立保存，正式部署仍按canonical评分。</p><p>完整评测需要COMPLETE、计数等于声明人口、invalid=0、证据seal正确。partial只显示进度；固定TRAIN1024/4096不与完整Test或全TRAIN均值混用。NAVTEST已用于研究假设／资源选择，相关决策透明记录；不将其token回流为训练名单。</p>'
    ops='<p>当前资源许可按最新指针，vla-zt3与tsinghua-zt2仍撤回；vlawm-zt2是另一节点。vla-zt3旧侧SSH/API被拒，退出仍无法直接确认。迁移回滚未提交尾部26/41/57，不称无损；新mutable日志独立，旧目录保留。运营重试只针对确认的pre-child资源拒绝，模型/scorer错误不自动当作科学重试。</p>'
    ops+='<p>'+link(root/'resources'/'LATEST_RESOURCE_AUTHORIZATION.json','当前授权指针')+' · '+link(root/'resources'/'LATEST_RESOURCE_ADDITION.json','新增节点部署指针')+' · '+link(snapshot.get('host_withdrawal_migration_path'),'旧机撤回迁移结果')+' · '+link(snapshot.get('resource_addition_deployment_path'),'新节点实际部署')+' · '+link(root/'RESOURCE_CAPACITY_POLICY.json','容量恢复规则')+'</p>'
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
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="1800"><title>Qwen-World RL研究报告</title><style>'+css+'</style></head><body><header><div class="eyebrow">QWEN-WORLD · RL RESEARCH</div><h1>RL训练改进：结果、诊断与研究决策</h1><p>从完整部署分数与训练过程找问题；保留可归因对照，把预算留给值得继续验证的方法。</p><div class="header-meta"><span>报告生成 '+esc(snapshot['generated_utc'])+'</span><span>最后完整评测证据 '+esc(h.utc(source_time) if source_time else '待记录')+'</span><span>常规30分钟＋阶段事件更新</span></div></header><nav>'+''.join('<a href="#'+id+'">'+name+'</a>'for id,name in nav)+'</nav><main>'+''.join(chapters)+'</main><footer>固定路径 '+esc(root/'report.html')+' · 本页只读汇总，不训练、不推理、不评分、不操作GPU或服务器</footer></body></html>'
