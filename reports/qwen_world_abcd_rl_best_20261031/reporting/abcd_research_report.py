"""Read-only ABCD current-best research section for the fixed living report."""
from pathlib import Path


ARCHITECTURES={
    'A':('C160＋独立动作actor','WorldStream为IL辅助，部署移除；动作attention LoRA单独适配。'),
    'B':('P／A共享20层actor','共享attention会影响P前缀，不能缓存已受LoRA影响的prefix图。'),
    'C':('独立world流＋actor与world→actor桥','部署保留world与桥；world attention／KV与actor的scope分开核对。'),
    'D':('原生Qwen语言层直接生成连续动作','不存在旧独立动作头；原生文本attention／时间接口与论文对应关系单独审查。'),
}


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
