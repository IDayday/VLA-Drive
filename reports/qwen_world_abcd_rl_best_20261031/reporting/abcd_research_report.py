"""Read-only ABCD current-best research section for the fixed living report."""
from pathlib import Path


ARCHITECTURES={
    'A':('C160＋独立动作actor','WorldStream为IL辅助，部署移除；动作attention LoRA单独适配。'),
    'B':('P／A共享20层actor','共享attention会影响P前缀，不能缓存已受LoRA影响的prefix图。'),
    'C':('独立world流＋actor与world→actor桥','部署保留world与桥；world attention／KV与actor的scope分开核对。'),
    'D':('原生Qwen语言层直接生成连续动作','不存在旧独立动作头；原生文本attention／时间接口与论文对应关系单独审查。'),
}


def render_early_process(root,project,analysis,h):
    """Draw only the sealed, all-rank prefix in the CPU evidence summary."""
    arms=analysis.get('arms',{})
    if not arms:return ''
    from research_report import figure
    ordered=sorted(arms.values(),key=lambda x:x.get('architecture_id',''))
    complete=[]
    for arm in ordered:
        counts=arm.get('actual_counts',{})
        if counts.get('analyzed_updates_per_rank',0)>0 and arm.get('finite',{}).get('all_examined_numeric_fields_finite'):
            complete.append(arm)
    if not complete:return ''
    colors={'A':'#2563eb','B':'#0d9488','C':'#d97706','D':'#7c3aed'}
    def lr_plot(plt):
        fig,axes=plt.subplots(1,2,figsize=(6.4,3.2))
        for arm in complete:
            key=arm['architecture_id'];counts=arm['actual_counts'];limit=min(counts['analyzed_updates_per_rank'],counts['common_completed_journal_prefix'])
            points=arm.get('learning_rate',{}).get('by_update_used_lr',{})
            points=sorted((int(u),p['mean'])for u,p in points.items()if int(u)<=limit and p.get('count')==8 and p.get('finite')and h.finite(p.get('mean')))
            if points:
                ax=axes[1 if key=='D'else 0];ax.plot([x[0]for x in points],[x[1]for x in points],label=key,color=colors[key],linestyle={'A':'-','B':'--','C':':'}.get(key,'-'),linewidth=2)
        for ax,title in zip(axes,['A / B / C: peak 5e-5','D: peak 1e-6']):
            ax.set(title=title,xlabel='Optimizer update',ylabel='Actual used LR');ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(frameon=False,fontsize=8)
        fig.tight_layout();return fig
    def pg_plot(plt):
        fig,ax=plt.subplots(figsize=(6.4,3.2));keys=[];fractions=[]
        for arm in complete:
            credit=arm.get('credit_groups',{});groups=credit.get('groups',0)
            if groups and h.finite(credit.get('effective_PG_groups')):keys.append(arm['architecture_id']);fractions.append(100*credit['effective_PG_groups']/groups)
        bars=ax.bar(keys,fractions,color=[colors[k]for k in keys],width=.55)
        for bar,value in zip(bars,fractions):ax.text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.2f}%',ha='center',fontsize=10)
        ax.set(ylim=(0,108),ylabel='Groups with nonzero PG credit (%)',xlabel='Architecture / distinct objective');ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();return fig
    body='<h3>已验证的早期学习过程：首128更新、八rank共同完整前缀</h3><p>四组均已有真实八rank学习与100更新的完整native checkpoint。下列固定窗口每组包含1,024个rank-update、256个fresh G8组／2,048候选；实时步数见首表，图表不外推到窗口之外。四组数值均finite，但作用于参数不等于已经提高部署分数。</p>'
    body+='<div class="two">'+figure(root,h,'abcd_early_actual_lr','实际学习率：四组独立LoRA优化器',lr_plot,'仅画step前实际LR；A/B/C曲线重合，D使用独立纵轴。LinearLR第100更新实际使用0.9901×base，第101更新起为base constant；不以scheduler下一步值替代实际使用值。')+figure(root,h,'abcd_early_pg_credit','非零PG信用组率：训练信号而非得分',pg_plot,'按实际优势／mask分类，非loss标量或参数梯度测量。不同目标、架构和训练率不能按组率排名；成功饱和也可使PG信用为零。')+'</div>'
    rows=[]
    for arm in complete:
        c=arm.get('credit_groups',{});f=arm.get('finite',{})
        rows.append([arm['architecture_id'],h.number(c.get('groups')),h.number(c.get('effective_PG_groups')),h.number(c.get('zero_PG_groups')),h.number(c.get('successful_EP1_PGzero_groups')),h.number(c.get('canonical_all8zero_groups')),h.number(c.get('positive_unsafe_with_safe_alternative')),h.number(f.get('updates_above_grad_clip'))+' / '+h.number(arm.get('actual_counts',{}).get('optimizer_rank_rows'))])
    body+=h.table(['架构','fresh组','非零PG信用组','PG信用零组','其中安全EP1饱和组','8候选canonical全零组','有安全替代仍获正A的unsafe候选','梯度超clip阈值的rank-update'],rows)
    d=next((arm for arm in complete if arm.get('architecture_id')=='D'),None)
    if d:
        m=d.get('metrics',{});value=lambda key:m.get(key,{}).get('mean')
        blocked=value('d_PPO_blocked_fraction')
        clip=value('clipfrac')
        body+='<p><b>D目标单独诊断：</b>joint clipfrac（|ratio−1|>0.2） '+h.fmt(100*clip if h.finite(clip)else None,3)+'%，按A符号实际阻断率 '+h.fmt(100*blocked if h.finite(blocked)else None,3)+'%，两者定义不同。own-IL joint Gaussian KL均值 '+h.fmt(value('d_reference_KL_joint'),6)+'，乘0.001后的KL项 '+(f"{value('d_weighted_reference_KL'):.3e}"if h.finite(value('d_weighted_reference_KL'))else '未记录')+'，标量大小不代表梯度份额。继承日志中的dimension-mean概率／KL代理不是D实际目标，不与ABC物理mean anchor直接比较。D的unsafe正信用是canonical标量中心化的过程观察，是否造成安全损失须等完整评测。</p>'
    body+='<p class="muted">ABC此窗口有安全替代时unsafe正信用为0；PG信用零组主要是成功EP1饱和。全零canonical、无可行候选与PG信用零不是同义词。窗口并非全量场景发生率，也未证明记忆语义、最终安全或泛化改善。'+h.relative_link(root,str(project/'analysis'/'ABCD_EARLY_NATIVE_PROCESS.json'),'有限CPU过程证据')+' · '+h.relative_link(root,str(project/'analysis'/'ABCD_FOUR_ACTUAL_NATIVE_STARTUP.json'),'实际八rank启动收据')+'</p>'
    return body


def render_abcd_research(root,research,h):
    if not research.get('parents'):return ''
    esc,fmt,number,table=h.esc,h.fmt,h.number,h.table
    project=Path(research['root']);parents=research['parents'].get('selected',{})
    link=lambda path,label:h.relative_link(root,str(path),label)
    runs=research.get('runs',[])
    def architecture(row):
        explicit=row.get('architecture_id',row.get('architecture'))
        if explicit in ARCHITECTURES:return explicit
        label=str(row.get('id',''))
        for key in ARCHITECTURES:
            if label in(key,'ARCH_'+key)or label.startswith('ARCH_'+key+'_'):return key
        return None
    by_arch={architecture(row):row for row in runs if architecture(row)}
    cards=[];status_rows=[]
    for key,(title,scope)in ARCHITECTURES.items():
        parent=parents.get(key,{})
        cards.append('<article><span>'+key+' · '+esc(title)+'</span><strong>'+fmt(parent.get('PDMS'),5)+'</strong><small>当前best IL '+number(parent.get('updates'))+'更新 · 完整12146</small></article>')
        run=by_arch.get(key,{})
        if not run:status='父模型已保护；结构适配／资格与注册准备中'
        elif run.get('worker_returncode')==75:status='准入拒绝，按实际prechild证据等待资源'
        elif run.get('worker_returncode')not in(None,0):status='执行失败待审，保留原receipt'
        elif run.get('worker_receipt_status')=='RUNNING'and h.finite(run.get('updates'))and run['updates']>0:status='已见实际原生更新（'+number(run.get('journal_ranks'))+'/'+number(run.get('expected_ranks'))+' rank）'
        elif run.get('worker_receipt_status')=='RUNNING':status='child已开始；等待实际optimizer日志'
        elif run.get('worker_returncode')==0:status='原生子进程已退出；完整阶段结果独立核对'
        else:status={'WAITING_RESOURCE':'等待资源／依赖准入','REGISTERED':'已登记，待实际启动证据'}.get(run.get('status'),str(run.get('status','已登记，待实际启动证据')))
        job=run.get('job');attempt=Path(job).name if job else '尚未派发'
        status_rows.append([key,esc(status),number(run.get('updates')),esc(run.get('host','未派发／待记录')),esc(run.get('algorithm')or run.get('research_method')or run.get('method')or('设计：SafetyEP＋PPO/W3；待正式注册'if key in'ABC'else '设计：JaT启发action-only PPO；待正式注册')),link(job,attempt)if job else esc(attempt)])
    body='<div class="notice"><b>新增主研究：ABCD各自以当前最高分IL checkpoint启动RL。</b>按用户要求选当前已完整NAVTEST的best，不等待未来100k端点；C选择12k而非其最新18k。四条架构的动作路径与梯度scope不同，不能直接复用旧World的52.7M scope。旧四条RL实验不因这张表自动停止或迁移。</div>'
    body+='<div class="cards">'+''.join(cards)+'</div>'
    body+=table(['架构','本轮实际状态','共同完整更新','host','方法（设计／注册分开）','当前attempt'],status_rows)
    if runs:
        body+='<p>四组已正式登记；实际训练更新只取八个rank各自已提交metrics的共同最小数，每100步checkpoint的latest指针不作为实时步数。当前attempt由独立native owner选定；同一job的RESULT优先于RUNNING，失败旧attempt保留为历史。'+link(project/'RUNS.json','正式RUNS')+' · '+link(project/'owners'/'ABCD_NATIVE'/'STATE.json','当前native owner状态')+'</p>'
    evaluator=research.get('controllers',{}).get('ABCD_EVALUATION',{})
    if evaluator:
        body+='<p>独立异步评测owner当前文件状态 '+esc(evaluator.get('state',{}).get('status','未记录'))+'（记录PID '+number(evaluator.get('owner',{}).get('pid'))+'），到门槛后分别发布DEV与完整NAVTEST；最多32分片是调度上限，实际分片数以每项spec为准。'+link(evaluator['state_path'],'评测进度')+'</p>'
    body+='<p class="protocol">新IL父模型统一完整NAVTEST12,146／44物理log、invalid0、single clean-x ODE K10／shift5／seed2101。起点IL预算分别不同；跨架构绝对分数不能作为训练充分性或架构优劣的单变量因果比较。各自RL收益应对各自锁定父模型与匹配预算报告。</p>'
    body+='<h3>父模型选择已锁定，权重保护与恢复能力分开</h3>'
    parent_rows=[]
    for key,parent in parents.items():
        native=parent.get('native_full_restore_available')
        parent_rows.append([esc(key),number(parent.get('updates')),fmt(parent.get('PDMS'),6),
            '完整native＋已评分export'if native else '已评分export完整；原native／IL optimizer副本缺失',
            link(parent.get('pin_manifest',''),'硬链接与原始seals')+' · '+link(parent.get('summary',''),'完整IL成绩')])
    body+=table(['架构','选中IL更新','父模型PDMS','实际可用资产','证据'],parent_rows)
    body+='<p>父权重已硬链接到新项目parents目录，原COMPLETE／export metadata保持原样，新增权重拷贝为0。A／C另保护完整native恢复状态；B／D当前只能依靠已封存部署权重与其完整祖先metadata，不能伪装成可恢复IL checkpoint。新RL为fresh optimizer／RNG，不承继旧IL Adam；实际加载仍需验证模型字节与部署closure。</p>'
    body+='<p class="notice">'+('当前使用锁定选择文档；后续IL出现更高分不会热切换这些RL父模型。'if research.get('parents_locked')else '当前已完成best选择与保护，root正在锁定本轮父身份；未将选择元数据当作RL已启动证据。')+'本次使用已观察NAVTEST选择父模型，明确是研究选择而非盲测；没有把它伪装成旧指南的100k／20个DEV点选择规则，也不将NAVTEST场景名单回流训练。</p>'
    body+='<p>'+link(research['parent_path'],'当前父模型选择与身份')+' · '+link(project/'PARENT_CANDIDATES.json','25个完整候选点')+'</p>'
    scope_audit=research.get('analyses',{}).get('ABC_META_SCOPE_AUDIT.json',{}).get('architectures',{})
    design_d=research.get('analyses',{}).get('D_PAPER_RL_DESIGN.json',{})
    body+='<h3>按结构适配，而不是只更换checkpoint路径</h3>'+table(['架构','原生动作路径','RL适配需要核查','当前META LoRA计数'],[[key,esc(title),esc(scope),number(design_d.get('architecture_facts',{}).get('LoRA_parameters')if key=='D'else scope_audit.get(key,{}).get('LoRA_trainable_numel'))]for key,(title,scope)in ARCHITECTURES.items()])
    body+='<p class="muted">META计数是CPU结构审计，不是实际GPU梯度或资格通过。ABCD保留各自数据／processor／参数闭包，旧103k训练人口不默认迁移；正式方法、预算、scope与source以本轮注册和实际原生日志为准。</p>'
    if design_d:
        body+='<p><b>'+('当前已登记的首版方法：'if runs else '当前已实现、待集成的首版方法：')+'</b>A／B／C为SafetyEP优势＋原PPO/W3，保留本组IL物理mean anchor beta50；D为JaT启发的action-only、每步24坐标joint-Gaussian PPO，clip0.2、equal-covariance native KL系数0.001、LoRA LR1e-6，D文本attention LoRA '+number(design_d.get('architecture_facts',{}).get('LoRA_parameters'))+' 参数。D并非完整AUGUstus／CoT论文复现，也不声称是最终轨迹精确似然；其算法与ABC同时不同，跨架构不能作单因素归因。'+link(project/'analysis'/'D_PAPER_RL_DESIGN.json','D论文、官方配置与本地差异')+'</p>'
    body+=render_early_process(root,project,research.get('analyses',{}).get('ABCD_EARLY_NATIVE_PROCESS.json',{}),h)
    analyses=research.get('analyses',{})
    if analyses:body+='<details><summary>本轮设计／source核查入口：提案与实际训练分开</summary><p>'+' · '.join(link(project/'analysis'/name,name)for name in analyses)+'</p><p>旧export的单文件差异为D的checkpoint_short_suffix激活checkpoint执行开关；当前执行另新增execution.py helper，已按精确hash完成四架构CPU迁移核对。旧seals不变，其他source变化仍拒绝；C／D首次因helper未映射而在actor前退出的attempt001保留，不计为算法结果，当前状态仅取owner所指新attempt。</p></details>'
    complete=[row for row in research.get('results',[])if row.get('publishable_complete')and row.get('cohort')=='NAVTEST']
    if complete:
        body+='<h3>新ABCD已完整RL成绩</h3>'+table(['架构','RL更新','完整PDMS','相对各自IL观察点差','证据'],[[esc(row.get('architecture_id',row.get('arm','待身份'))),number(row.get('updates')),fmt(row.get('PDMS'),6),fmt(row['PDMS']-parents[row['architecture_id']]['PDMS'],6)if row.get('architecture_id')in parents else '待匹配',link(row.get('evidence',''),'完整SUMMARY')]for row in complete])
    else:body+='<p class="notice"><b>当前尚无本轮ABCD完整RL成绩。</b>父模型分数、CPU结构检查、配置或worker启动均不计为RL提分；未完成人口不发布均值。</p>'
    observations=research.get('candidates',{}).get('candidates',[])
    body+='<details><summary>全部完整父模型候选：按架构与IL更新展开</summary>'+table(['架构','IL更新','完整PDMS','是否当前best','证据'],[[esc(row.get('architecture')),number(row.get('updates')),fmt(row.get('PDMS'),6),'是'if row.get('summary')==parents.get(row.get('architecture'),{}).get('summary')else '—',link(row.get('summary',''),'SUMMARY')]for row in sorted(observations,key=lambda row:(row['architecture'],row['updates']))])+'</details>'
    guide=Path(research['source_worktree'])/'reports/qwen_world_four_frameworks_il_rl_20261029'
    body+='<p class="muted">原始结构指南：'+link(guide/'README.md','总设计')+' · '+link(guide/'AC_IMPLEMENTATION.md','A／C')+' · '+link(guide/'B_IMPLEMENTATION.md','B')+' · '+link(guide/'D_IMPLEMENTATION.md','D与来源')+'。旧指南预算不是本轮新run的实际执行收据。</p>'
    return body
