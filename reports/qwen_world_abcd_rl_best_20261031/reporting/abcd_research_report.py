"""Read-only ABCD current-best research section for the fixed living report."""
from pathlib import Path


ARCHITECTURES={
    'A':('C160＋独立动作actor','WorldStream为IL辅助，部署移除；动作attention LoRA单独适配。'),
    'B':('P／A共享20层actor','共享attention会影响P前缀，不能缓存已受LoRA影响的prefix图。'),
    'C':('独立world流＋actor与world→actor桥','部署保留world与桥；world attention／KV与actor的scope分开核对。'),
    'D':('原生Qwen语言层直接生成连续动作','不存在旧独立动作头；原生文本attention／时间接口与论文对应关系单独审查。'),
}


def render_replanning(root,research,h):
    """A compact argument and decision history, ahead of experiment details."""
    analyses=research.get('analyses',{})
    retrospective=analyses.get('RL_DEVELOPMENT_RETROSPECTIVE_20261031.json',{})
    next_research=analyses.get('RL_NEXT_ALGORITHM_RESEARCH_20261031.json',{})
    project=Path(research['root'])
    def prose(value):
        if isinstance(value,str):return h.esc(value)
        if isinstance(value,list):return ' '.join(h.esc(v)for v in value if isinstance(v,str))
        if isinstance(value,dict):return ' '.join(prose(value[k])for k in ['title_zh','change_zh','rule_zh','reason_zh','description_zh','primary','secondary']if k in value)
        return ''
    abcd_complete=[r for r in research.get('results',[])if r.get('publishable_complete')and r.get('cohort')=='NAVTEST']
    target=('本轮ABCD已有完整NAVTEST达到92，具体模型与匹配基线见下节。'if any(r.get('PDMS',0)>=92 for r in abcd_complete)else '92目标仍未达成。')
    body='<div class="notice"><b>当前判断：尚无可信超越SafetyEP的World方法突破；'+target+'</b>World最高观察点Memory10k=91.37616，比Safety10k仅+0.02627，配对95%CI[-0.11956,+0.15879]；零分反而净增19且NC／TTC下降。这个结果不能当作成功改进。接下来按问题与判据推进，优先完成可区分假设的对照。</div>'
    population=analyses.get('SAFETY_FULL_POPULATION_AUDIT_20261031/SUMMARY.json',{})
    population_complete=population.get('status')=='COMPLETE'and population.get('population',{}).get('scenes')==12146
    training=analyses.get('SAFETY_FULL_TRAINING_AUDIT_20261031/SUMMARY.json',{})
    training_complete=training.get('status')=='COMPLETE_ALL62640_GROUPS_ACTUAL_RECORDS'and training.get('actual_record_validation',{}).get('all_expected_files_read')is True and training.get('actual_record_validation',{}).get('errors')==0
    body+='<div class="'+('notice'if population_complete and training_complete else 'warning')+'"><b>'+('全量证据已齐：'if population_complete and training_complete else '正在补齐全量证据：')+'</b>SafetyEP全12,146场景审计'+('已完整，见下方全人口分档。'if population_complete else '进行中，未完成前不画全人口结论。')+('全部62,640个真实G8组／501,120候选也已逐件验证。'if training_complete else '全训练组census单独汇总。')+'旧1,280个TRAIN组窗口和24组梯度回放只解释局部材料；全量候选信用统计也不等于全训练参数梯度归因。ABC/D首1k不能替代原Safety全训练。修复零／低分与高分不退化必须同时检验。</div>'
    body+='<p class="protocol">全量分档固定为0、0&lt;s&lt;0.5、0.5≤s&lt;0.9、s≥0.9（s为0–1原始score）。IL70与Safety10k各自定义固定群体，成员数、修复／新增和高分保持分别计算，不能混作同一队列。</p>'
    relation=analyses.get('SAFETY_FAILURE_RELATION_20261031/SUMMARY.json',analyses.get('SAFETY_FAILURE_RELATION_20261031.json',{}))
    relation_status=str(relation.get('status',''))
    relation_complete=relation_status.startswith('COMPLETE')and not any(k in relation_status for k in ['FAIL','MISSING','PARTIAL'])
    body+='<p class="warning"><b>效率指标的解释边界：</b>canonical EP会在NC／DAC零门情况下被scorer清零，EP下降不等同车辆更保守或实际少开了多少米。'+('完整raw_progress_meters关系摘要已发布；原始距离与canonical EP分别阅读，仍不据单项归因。'if relation_complete else '全部17×12,146观测的raw_progress_meters关系审计另行进行，完成前不据EP单项作物理行为结论。')+'</p>'
    if relation_complete:
        conclusions=relation.get('conclusions_zh',[])
        if isinstance(conclusions,list):body+='<p>'+' '.join(h.esc(v)for v in conclusions[:3]if isinstance(v,str))+'</p>'
        relation_rows=[]
        for label,contrast in relation.get('contrasts',{}).items():
            zero=contrast.get('new_zero',{});drop=contrast.get('any_safety_metric_drop',{});witness=contrast.get('observed_dominance',{})
            name='Memory10k vs Safety10k'if label.startswith('PPO_SAFETY_SUCCESS_MEMORY')else 'Safety30k vs Safety10k'if label.startswith('PPO_SAFETY_EP_u030000')else label
            relation_rows.append([h.esc(name),h.number(drop.get('scenes')),h.number(drop.get('risk_drop_raw_progress_up_gt0p1m'))+' / '+h.number(drop.get('risk_drop_raw_progress_flat_within0p1m'))+' / '+h.number(drop.get('risk_drop_raw_progress_down_gt0p1m')),h.number(zero.get('scenes')),h.number(zero.get('risk_drop_raw_progress_up_gt0p1m'))+' / '+h.number(zero.get('risk_drop_raw_progress_flat_within0p1m'))+' / '+h.number(zero.get('risk_drop_raw_progress_down_gt0p1m')),h.number(witness.get('new_zero_with_observed_safe_at_least_same_raw_progress_and_comfort'))])
        if relation_rows:
            body+='<h3>完整原始progress核对：不是把EP下降一概叫“保守”</h3><p>已核对 '+h.number(relation.get('policies'))+'个policy人口的 '+h.number(relation.get('raw_scene_records_read'))+'份原始scene记录。以下上升／持平／下降按raw_progress_meters、0.1m容差分列，canonical EP另外记；每个newzero都被官方EP清零不代表车辆真的停止。</p>'+h.table(['完整对照','任一NC/DAC/TTC下降场景','其中raw进度↑／≈／↓','新增zero','其中raw进度↑／≈／↓','新增zero中存在同进度且不低舒适的strict-safe已观察输出'],relation_rows)+'<p class="protocol">跨17个已观察policy找到的单scene安全替代，说明不能把所有新错归为不可避免的安全—效率前沿；它不证明当前θ能采到该动作，也不证明一个参数向量能同时保持全部witness。不用NAV IDs构造训练队列；raw进度更长本身也不等于合理通行。</p>'
        source='SAFETY_FAILURE_RELATION_20261031/SUMMARY.json'if 'SAFETY_FAILURE_RELATION_20261031/SUMMARY.json'in analyses else 'SAFETY_FAILURE_RELATION_20261031.json'
        body+='<p>'+h.relative_link(root,str(project/'analysis'/source),'原始progress与失败关系正式审计')+'</p>'
        matrix_path=project/'analysis'/'SAFETY_FAILURE_RELATION_20261031'/'raw_progress_events_matrix.npz'
        if matrix_path.is_file():body+='<details><summary>原始progress／失败事件矩阵</summary><p>'+h.relative_link(root,str(matrix_path),'已校验的17×12,146原始矩阵')+'；仅用于完整NAV诊断，不进入训练sampler。</p></details>'
    body+=render_full_population(root,research,h)
    body+=render_full_training(root,research,h)
    body+=render_zero_group_research(root,research,h)
    summary=retrospective.get('executive_summary_zh',[])
    if summary:body+='<p>'+prose(summary[:2]if isinstance(summary,list)else summary)+'</p>'
    else:body+='<p>已经建立了从完整评测、候选信用到真实Memory梯度的证据链；后续组件虽在代码与梯度上生效，仍未稳定转成更优驾驶策略。这使我们能排除具体配方，而不能把某一算法名称、更多梯度或静态候选更好当成提升保证。</p>'
    rows=[]
    decisions={'BEST_PRESERVED_OLD_RECIPE_CLOSED':'旧配方已结束；保留best','OLD_VARIANTS_CLOSED':'原变体已结束','CLOSED_WITH_BEST_AND27K_REVIEW':'已结束；保留best与27k复核','COOPERATIVE_STOP_BEST_PRESERVED':'已协作停止；保留best','BEST10K_PRESERVED_ORIGINAL_CONTINUATION_LATER_CLOSED':'保留10k；原长训停止','STOPPED_BY_USER_SCOPE_WITH_COMPLETED_POINTS_PRESERVED':'按用户研究收缩停止，非算法否决','FINITE_DIAGNOSIS_COMPLETE':'有限机制诊断已完成','EXACT_RECIPES_NOT_PRIORITY':'这些具体配方不优先续长','COMPLETED_NO_EXTENSION':'已完成，不续长','BEST_PRESERVE_OR_DEFER_EXACT_RECIPE':'保留模型；此配方无新增预算','ACTIVE_NEW_ARCHITECTURE_RESEARCH_NOT_SCORE_PROVEN':'当前ABCD在研；等待匹配完整NAV'}
    for route in retrospective.get('route_rows',[]):
        evidence=[]
        points=sorted(route.get('matched_evidence',[]),key=lambda p:p.get('updates',0),reverse=True)
        for point in points[:2]:
            if not isinstance(point,dict):continue
            delta=point.get('delta_PDMS');ci=point.get('CI95_PDMS')
            text=(str(point.get('target','目标'))+' 相对 '+str(point.get('baseline','对照')))
            if h.finite(delta):text+=' Δ'+f'{delta:+.4f}'
            if isinstance(ci,list)and len(ci)==2 and all(h.finite(v)for v in ci):text+='，CI['+', '.join(f'{v:+.4f}'for v in ci)+']'
            if point.get('source_review'):evidence.append(h.relative_link(root,point['source_review'],text))
            else:evidence.append(h.esc(text))
        mechanism_value=route.get('mechanism_evidence_zh');mechanism=prose(mechanism_value[:1]if isinstance(mechanism_value,list)else mechanism_value)
        if mechanism:evidence.append(mechanism)
        decision=decisions.get(route.get('disposition'),'按完整证据复核')
        rows.append(['<b>'+h.esc(route.get('title_zh',route.get('id','路线')))+'</b><small>'+prose(route.get('hypothesis_zh'))+'</small>','<br>'.join(evidence)or'尚无完整结果',prose(route.get('conclusion_zh')),'<b>'+h.esc(decision)+'</b><small>'+prose(route.get('continue_recommendation'))+'</small>'])
    if not rows:
        rows=[['SafetyEP：分解安全／可行内效率信用','10k相对原FULL +0.3304；CI排除0；存在NC代价','有收益证据，长训未持续；保留10k对照','保留基线，不默认续满50k'],
            ['成功Memory：真实成功轨迹保持','10k+0.02627且CI跨0；zero修复15／新增34','辅助确实激活，尚未证明部署收益','保留模型；不据最高均值自动续长'],
            ['FPO与时间覆盖','FPO10k−2.6051；时间扩展5k仍低于原FPO','当前终点代理与时间配方未转为学习收益','这两个具体配方不直接延长'],
            ['可行内EP／Comfort质量信用','5k−0.5848；EP约损失1.2pp','补足信用不等于更优信用','结束此固定q配方'],
            ['ABCD：从结构与各自best父模型重新验证','已完成1k DEV；NAV与续训状态见下方实时表','尚不能用DEV或跨父模型分数归因','当前进行中的注册对照；等完整同人口结果']]
    body+='<h3>开发路线总表：假设 → 有效证据 → 结论 → 取舍</h3>'+h.table(['路线与研究假设','已有有效证据','当前结论与证据强度','是否继续／建议'],rows)
    credit=analyses.get('ABCD_MATCHED_TRAIN1K_CREDIT.json',{})
    if credit.get('status')=='COMPLETE':
        signal_rows=[]
        for key,arm in credit.get('arms',{}).items():
            groups=arm.get('groups',{});signs=arm.get('candidate_sign_counts',{})
            signal_rows.append([h.esc(key),h.number(groups.get('groups')),h.number(groups.get('zero_PG')),h.number(groups.get('no_feasible_and_zero_PG')),h.number(signs.get('positive_unsafe_with_safe_alternative'))])
        body+='<details><summary>新证据：PG信用多，不代表更符合安全—效率目标</summary><p>四组首1k使用相同顺序的2,000个TRAIN scene group；这是不同策略随训练变化的采样记录，未冻结同θ，也不是参数梯度或部署分数比较。</p>'+h.table(['架构','同序TRAIN组','PG信用零组（总计）','其中无严格可行候选','有严格可行替代仍正A的unsafe候选'],signal_rows)+'<p>D的总PG信用零组更少，但出现308个有严格可行替代仍为正A的unsafe候选；ABC为0。TTC的canonical软分与SafetyEP严格门不同，不能把这308个全称为canonical错误。D同时改变架构、父模型、目标、学习率和噪声截断，当前不能据此单独归因或排序。'+h.relative_link(root,str(project/'analysis'/'ABCD_MATCHED_TRAIN1K_CREDIT.json'),'同TRAIN顺序信用核对')+'</p></details>'
    bottlenecks=retrospective.get('core_bottlenecks',[])
    if bottlenecks:
        body+='<h3>候选瓶颈：以全量事实约束，仍需因果对照</h3><div class="research-grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">'
        for item in bottlenecks[:3]:
            contrast=item.get('next_minimal_contrast',{})
            body+='<article class="callout"><h3>'+h.esc(item.get('name_zh',item.get('id','问题')))+'</h3><p>'+prose(item.get('evidence_zh'))+'</p><p class="muted">尚未证明：'+prose(item.get('not_proven_zh'))+'</p>'
            if contrast:body+='<p>最小可区分对照：'+prose(contrast.get('single_change'))+'；关键判据：'+prose(contrast.get('decisive_metrics'))+'</p>'
            body+='</article>'
        body+='</div>'
    corrections=retrospective.get('experiment_design_corrections',[])
    if corrections:
        correction_rows=[[prose(item.get('issue_zh')),prose(item.get('actual_correction_zh')),prose(item.get('lesson_zh'))]for item in corrections if isinstance(item,dict)]
        body+='<details><summary>这轮研究设计要纠正什么：运营修复与性能收益分开</summary>'+h.table(['原来的安排／问题','已做的纠正','下一轮规则'],correction_rows)+'</details>'
    if next_research:
        body+='<details><summary>前一轮算法备选：文档建议与最新落实状态分开</summary>'
        for key in ['executive_summary_zh','main_conclusions_zh','conclusions_zh']:
            if next_research.get(key):body+='<p>'+prose(next_research[key])+'</p>';break
        papers={p.get('id'):p for p in next_research.get('papers',[])if isinstance(p,dict)}
        body+='<div class="research-grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">'
        for direction in next_research.get('directions',[])[:3]:
            body+='<article class="callout"><h3>P'+str(direction.get('priority',''))+' · '+h.esc(direction.get('title',direction.get('id','候选')))+'</h3><small>文档时点为研究提案；当前代码／注册／训练以新研究进度为准</small><p>'+prose(direction.get('hypothesis'))+'</p><p><b>只改什么：</b>'+prose(direction.get('single_core_change'))+'</p><p><b>先满足：</b>'+prose(direction.get('trigger',[])[:1])+'</p><p class="muted"><b>主要风险：</b>'+prose(direction.get('risks',[])[:1])+'</p>'
            prototype=analyses.get('D_SAFETY_CREDIT_CPU_CHECK.json',{})
            if direction.get('id')=='D_SAFETY_CREDIT'and prototype.get('status')=='PASS':body+='<small>独立CPU原型3项通过；尚未注册／派GPU，不是完整GPU资格或性能结果。'+h.relative_link(root,str(project/'analysis'/'D_SAFETY_CREDIT_CPU_CHECK.json'),'CPU收据')+'</small>'
            source_links=[]
            for source in direction.get('sources',[])[:2]:
                paper=papers.get(source,{})
                if paper.get('paper_url'):source_links.append('<a href="'+h.esc(paper['paper_url'])+'">'+h.esc(paper.get('title',source))+'</a>')
            if source_links:body+='<p class="muted">原始研究参考：'+' · '.join(source_links)+'；本地方案为待检验适配。</p>'
            body+='</article>'
        body+='</div><p><b>最新执行要求：</b>按用户要求将研究点逐项落实为代码与有对照实验，开发可并行；每个arm的单变量、真实候选／曝光预算与完整评测分别登记，不把所有组件堆成一个无法归因的loss。现有live继续，原文档“先一项”的旧安排不限制新授权；仍不默认50k或增加训练seed。</p>'
        body+='<p>'+h.relative_link(root,str(project/'analysis'/'RL_NEXT_ALGORITHM_RESEARCH_20261031.json'),'算法文献、可证伪假设与触发条件')+'；该研究文档不授予新GPU或训练预算，实际启动仍以RUNS与worker／原生日志为准。</p>'
        body+='</details>'
    if retrospective:body+='<p>'+h.relative_link(root,str(project/'analysis'/'RL_DEVELOPMENT_RETROSPECTIVE_20261031.json'),'完整开发复盘与来源')+'</p>'
    body+='<p class="protocol">完整同人口部署结果支持性能判断；同候选CPU回放和真实梯度只解释局部机制。研究中同时存在IL知识、场景曝光、候选探索、信用、优化及部署差异；不把不同观测集合的persistent zero合并成一个比例，也不将执行修复记为算法提分。</p>'
    return body


def render_full_training(root,research,h):
    analyses=research.get('analyses',{});journal=analyses.get('SAFETY_FULL_TRAINING_AUDIT_20261031/JOURNAL_CENSUS.json',{})
    final=analyses.get('SAFETY_FULL_TRAINING_AUDIT_20261031/SUMMARY.json',{});project=Path(research['root'])
    verified=final.get('status')=='COMPLETE_ALL62640_GROUPS_ACTUAL_RECORDS'and final.get('actual_record_validation',{}).get('all_expected_files_read')is True and final.get('actual_record_validation',{}).get('errors')==0
    if verified:
        p=final['population'];n=p['actual_groups'];c=final['verified_counts'];rates=final.get('rates_per_group',{});boundary=final.get('boundary',{})
        body='<h3>全部实际训练组：'+h.number(n)+'个G8逐件验证，非窗口外推</h3><p>截至sealed '+h.number(boundary.get('seal',{}).get('updates'))+'更新／inner2，八rank '+h.number(final.get('journal_prefix',{}).get('rank_optimizer_rows'))+'条更新日志与全部behavior对应；'+h.number(n)+'组、'+h.number(p.get('candidates'))+'候选、'+h.number(p.get('unique_scene_tokens'))+'个唯一scene，缺失／错误0，保存的优势逐值误差0。每组只计一次，末组只有2／4次inner实际提交；不是全103,288 eligible人口一遍。</p>'
        categories=[('mixed：可行与失败并存','category_mixed_feasible_failed','安全信用＋可行子集EP'),('全部可行，EP仍有相对差异','category_all_feasible_progress_only','按EP分配相对信用'),('全部可行且EP1，Safety信用饱和','category_all_feasible_EP1_saturated','PG0；不等于canonical满分'),('全失败，平均cost仍有差异','category_all_fail_cost_varied','已有相对A；正A仍可不安全'),('全失败，平均cost打平','category_all_fail_cost_tie','PG0；须继续查组件／时序信息')]
        body+=h.table(['互斥组类别','实际G8组数','占62,640组','当前信用含义'],[[name,h.number(c.get(key)),h.fmt(100*c[key]/n,3)+'%',meaning]for name,key,meaning in categories])
        body+='<p><b>三种“零”分开：</b>canonical八条全0为 '+h.number(c.get('canonical_allzero'))+'组（'+h.fmt(100*c['canonical_allzero']/n,3)+'%），其中 '+h.number(c.get('canonical_allzero_but_PG_active'))+'组仍有Safety信用、'+h.number(c.get('canonical_allzero_and_PG0'))+'组A也全0。实际PG0共 '+h.number(c.get('PG0'))+'组＝'+h.number(c.get('allsafe_EP1_saturated'))+'个可行EP1饱和＋'+h.number(c.get('allfail_cost_tie'))+'个失败cost-tie；其中 '+h.number(c.get('Comfort_varied_when_EP1_saturated'))+'个饱和组仍有Comfort差异，不能统称成功满分，更不能施加失败负A。</p>'
        anchor=final.get('PG0_local_anchor_objective',{});global_context=anchor.get('global_collection_context',{})
        if global_context:
            a=anchor.get('all_PG0',{})
            body+='<p><b>局部PG0不等于整步无PG：</b>全部 '+h.number(global_context.get('collections'))+'个global collection中，八rank同时PG0为 '+h.number(global_context.get('collections_with_all8rank_PG0'))+'。局部PG0的 '+h.number(a.get('rows'))+'条inner记录中 '+h.number(a.get('weighted_anchor_positive'))+'条anchor objective为正；这是标量活动审计，不是梯度范数、anchor主导或回退因果的证明。</p>'
        signal=[('有严格safe替代仍获正A的unsafe',c.get('positive_unsafe_with_safe_alternative'),'mixed中为0；与D canonical信用区别需D内隔离'),('所有情形的正A unsafe候选',c.get('positive_unsafe'),'集中在无可行替代的fallback；least-cost不叫安全'),('其中NC0',c.get('positive_unsafe_NC0'),'这是candidate数，不是scene组；可与下面事件重叠'),('其中NC0.5',c.get('positive_unsafe_NC05'),'不能沿用局部“未见”作全局结论'),('其中DAC0 / TTC0',h.number(c.get('positive_unsafe_DAC0'))+' / '+h.number(c.get('positive_unsafe_TTC0')),'事件可重叠，不相加'),('严格可行但负A的候选',c.get('negative_feasible'),'相对效率信用；低EP可能是必要等待，不能全判错')]
        body+='<details><summary>全量候选信用风险：少数失败方向也须审查</summary>'+h.table(['事件','candidate计数（总501,120）','含义／边界'],[[name,h.number(value)if h.finite(value)else str(value),note]for name,value,note in signal])+'<p>上述是候选信用，不是参数梯度或与NAV失败的因果join；不把无可行替代时的正A当作安全监督。</p></details>'
        fine=final.get('all_costtie_fine_feedback',{}).get('counts',{})
        if fine:
            body+='<p><b>97个cost-tie并非97个“完全无信息”：</b>'+h.number(fine.get('A_equal_scalar_cost_different_component_vectors'))+'组平均cost相同但组件向量不同；'+h.number(fine.get('B_same_component_vector_different_real_fine_feedback'))+'组组件相同但真实时序反馈不同，完全相同的可用安全反馈组为 '+h.number(fine.get('C_same_components_and_all_available_fine_safety_feedback'))+'。其中 '+h.number(fine.get('DAC_only_tie_with_duration_difference_and_valid_t0'))+'组是t0合法、NC/TTC通过但DAC失败，出界时长可分。更短出界仍是失败，不是成功teacher；不把raw progress或更晚碰撞擅自当更安全。</p>'
        body+='<p>'+h.relative_link(root,str(project/'analysis'/'SAFETY_FULL_TRAINING_AUDIT_20261031'/'SUMMARY.json'),'全训练原件验证、分阶段与完整ledger')+'；NAV与TRAIN是不同人口／推理过程，不用无关token作join。</p>'
        return body
    if journal.get('status')!='COMPLETE_ALL8RANK_SEALED_PREFIX':return '<p class="muted">全训练组census尚待正式产物，不将局部窗口频率外推至所有训练。</p>'
    boundary=journal.get('boundary',{});counts=journal.get('counts',{});project=Path(research['root'])
    body='<h3>全训练组审计：日志封口已齐，原始行为验证中</h3><p>已扫描完整八rank、截至原生sealed checkpoint '+h.number(boundary.get('seal',{}).get('updates'))+'更新，'+h.number(counts.get('optimizer_rows'))+'条rank-update；每个fresh G8只在首次inner计一次，共 '+h.number(counts.get('groups'))+'组／'+h.number(journal.get('candidate_count'))+'候选，'+h.number(journal.get('unique_scene_tokens'))+'个实际scene。末个collection仅提交2／4次inner；这也不表示已遍历全部103,288 eligible训练人口。</p>'
    body+='<p class="warning"><b>仍待行为原件逐一核对：</b>日志flags中的all-fail／cost-tie等部分分类为inferred，尚不能将它们当作最终全量rawA／adv-mask结论。全零canonical、实际A全零、严格全失败与全八rank整batch无PG必须分开。即使局部PG为零，仍可能有其它组、anchor或aux梯度。</p><p>'+h.relative_link(root,str(project/'analysis'/'SAFETY_FULL_TRAINING_AUDIT_20261031'/'JOURNAL_CENSUS.json'),'全日志封口census与分类限制')+'</p>'
    return body


def render_zero_group_research(root,research,h):
    analyses=research.get('analyses',{});review=analyses.get('ZERO_REWARD_GROUP_RESEARCH_20261031.json',{});project=Path(research['root'])
    if not str(review.get('status','')).startswith('COMPLETE'):return '<p class="muted">全零训练组的原始论文复核正在汇总；未把group-std、过滤或更精确ratio称作凭空产生监督。</p>'
    body='<h3>全零训练组：先看缺什么信号，再选论文方法</h3><p>全量已确认140个R全0组中59个已有Safety信用，81个A也全0；不能把6007个局部PG0全当失败。平均cost打平也不等于组件／时序完全相同。扩大候选、失败负学习、真实成功见证与成功行为保持解决不同问题，不能相互替代。</p>'
    tax_names={'Z1_COMPONENT_SIGNAL':'R全0，但已有组件成本差异','Z1B_HIDDEN_COMPONENT_OR_TIME':'标量打平，仍有未利用的组件／时序','Z2_TRUE_OBSERVED_COST_TIE':'本组尚无已验证的合法排序','Z3_STOCHASTIC_SUPPORT_ODE_FAILURE':'随机候选成功，固定ODE仍失败','Z4_NORMALIZATION_ONLY':'只改归一化／过滤','SATURATED_SUCCESS':'可行EP1信用饱和，非失败'}
    body+='<details><summary>六类信号条件：不能混叫“零梯度”</summary>'+h.table(['情形','当前含义','真正还需什么信号'],[[tax_names.get(item.get('id'),item.get('id','')),h.esc(item.get('current_training','')),h.esc(item.get('needed_signal',''))]for item in review.get('taxonomy',[])])+'</details>'
    body+='<p><b>G16必须双预算看：</b>组内候选8→16，轨迹horizon仍8、world仍8。相同候选数会减少scene曝光和optimizer更新；相同scene数则翻倍候选／评分成本。全部16条仍同成本时，不会自动产生Safety相对优势；找到SDE成功也不保证固定ODE学会。</p>'
    body+=h.table(['设计比较点（均未计为训练结果）','新候选','scene曝光','optimizer更新'],[['G8 u1000','16,000','2,000','1,000'],['G16 u500：同候选预算','16,000','1,000','500'],['G16 u1000：同曝光／更新预算','32,000','2,000','1,000']])
    route_rows=[]
    for item in review.get('priority_routes',[]):route_rows.append([h.esc(item.get('title','')),h.esc(item.get('signal','')),h.esc(item.get('keep_high','')),h.esc(item.get('budget',''))])
    body+='<details><summary>对应的三类研究机制与算力代价</summary>'+h.table(['方法方向','信号从哪里来','高分保持／限制','预算与额外成本'],route_rows)+'</details>'
    paper_rows=[]
    for paper in review.get('papers',[]):
        title='<a href="'+h.esc(paper.get('paper_url',''))+'">'+h.esc(paper.get('title',paper.get('id','论文')))+'</a><small>'+h.esc(paper.get('date',''))+'</small>'
        paper_rows.append([title,h.esc(paper.get('advantage_source','')),h.esc(paper.get('compute','')),h.esc(paper.get('limitations',''))])
    body+='<details><summary>六篇原始论文：提供哪种信息，哪些不能迁移</summary>'+h.table(['原始研究','新增信号','算力成本','迁移边界'],paper_rows)+'</details>'
    body+='<p>NGRPO／Sign的失败负学习不提供成功轨迹，且二值reward、真实score-function梯度的结论不能直接套到本地clamped维均值W3代理。SIL／Scaf／FRS要真实witness或额外指导；CLEAR式已学分布保持也不是永久IL70 mean anchor的同义词，不能凭局部PG0＋anchor标量推断回退因果。</p><p>'+h.relative_link(root,str(project/'analysis'/'ZERO_REWARD_GROUP_RESEARCH_20261031.json'),'完整论文与代码核验、病例覆盖')
    if analyses.get('G16_IMPLEMENTATION_DESIGN.json'):body+=' · '+h.relative_link(root,str(project/'analysis'/'G16_IMPLEMENTATION_DESIGN.json'),'G16实现与双预算设计')
    body+='；新代码／注册／实际训练的当前进度见独立研究，不把这份设计文档计为实验收益。</p>'
    return body


def render_exploration_diagnostics(root,research,h):
    """Selected TRAIN diagnostics stay outside all formal score tables."""
    project=Path(research['root']);analyses=research.get('analyses',{});body='';registered={r.get('id')for r in research.get('runs',[])}
    census=analyses.get('HIGH_SCORE_CREDIT_CENSUS.json',{});verified=analyses.get('HIGH_SCORE_CREDIT_BEHAVIOR_VERIFICATION.json',{})
    if verified.get('status')=='COMPLETE_ALL8700_ACTUAL_BEHAVIOR':
        c=verified.get('counts',{});allc=census.get('counts',{});pairs=verified.get('canonical_dominance_pair_statistics',{})
        body+='<h3>优势分配：高分保持还缺哪一类信用</h3><p>全部62,640组中，canonical≥0.9候选 '+h.number(allc.get('canonical_high_candidates'))+' 条：正A '+h.number(allc.get('high_positive_A'))+'、负A '+h.number(allc.get('high_negative_A'))+'、零A '+h.number(allc.get('high_zero_A'))+'。针对全部 '+h.number(c.get('groups'))+' 个含负high的组，已逐件读取 '+h.number(c.get('candidates'))+' 条实际评分记录；这是整个选定集合，不是随机样本。</p>'
        body+=h.table(['实际核查','候选／pair计数','科学含义'],[
            ['canonical≥0.9且负A',h.number(c.get('negative_high')),'全部NC/DAC/TTC/Comfort=1、EP<1；不表示被标为失败'],
            ['其中全可行组／mixed组',h.number(c.get('negative_high_in_allfeasible'))+' / '+h.number(c.get('negative_high_in_mixed')),'低于可行子集EP均值，效率差额超过非负安全保护'],
            ['同组存在更低canonical正A见证',h.number(c.get('negative_high_with_lower_canonical_positive')),'部分相对信用与最终canonical排序不一致'],
            ['已核实的反序pair',h.number(c.get('lower_canonical_positive_pairs')),'全部为Comfort1／较低EP负A，对Comfort0／较高EP正A'],
            ['这些组rawA全0／PG0',h.number(c.get('rawA0_groups'))+' / '+h.number(c.get('PG0_groups')),'属于已有PG的组；新增饱和信用不能覆盖这一问题']])
        delta=pairs.get('delta_canonical',{}).get('mean')
        if h.finite(delta):body+='<p>这些pair中，正A候选相对负A候选的canonical分数平均 '+h.fmt(delta,6)+'（单候选0..1单位；不是全局PDMS变化）。负相对优势本身可以合理；较低canonical获正号才是需要隔离验证的目标差异。</p>'
        body+='<p class="warning"><b>尚未证明部署退化的因果：</b>rawA符号不等于参数梯度方向，仍受clipping、anchor、跨rank平均、共享表征和部署ODE支持影响。这里的高分候选也不同于固定检查点的高分scene队列。'+h.relative_link(root,str(project/'analysis'/'HIGH_SCORE_CREDIT_BEHAVIOR_VERIFICATION.json'),'完整8700组核查与限制')+'</p>'
        if analyses.get('SATURATED_CREDIT_HANDOFF.json'):
            phase2=bool(registered&{'SAFETY_SATURATED_G8','SAFETY_PROTECTED_HIGH_G8','SAFETY_INVERSION_CANCEL_G8'})
            body+='<details><summary>三类信用机制覆盖不同问题：'+('已正式登记，实际更新另列'if phase2 else 'CPU准备，待正式登记')+'</summary><p>饱和信用只针对全safe、EP1、原A0但canonical因Comfort不同的组；不触及上述active负high。Protected只把strict4、canonical≥0.9的原负A置0，允许组均值正偏，较低canonical正A仍保留。Inversion同时抵消高分负A与更低canonical正A的相反信用，近似守恒组和；正信用预算耗尽时仍可留下负high。三者独立对照，不混称一个算法，均保留原normalization尺度；尚无新训练收益结论。</p><p>'
            docs=[('SATURATED_CREDIT_HANDOFF.json','饱和信用开发收据'),('PROTECTED_CREDIT_HANDOFF.json','Protected开发收据'),('INVERSION_RUNTIME_HANDOFF.json','Inversion开发收据')]
            body+=' · '.join(h.relative_link(root,str(project/'analysis'/name),title)for name,title in docs if name in analyses)
            body+='</p><p class="muted">Inversion可能在保留原mask=1时把整组实际A变0；其true PG0应读inversion_credit_post_raw_PG0_groups／new_PG0_groups，不能沿用仅按mask统计的旧zero_PG字段。</p></details>'
    probe=research.get('support_probe',{});state=probe.get('state',{});summary=probe.get('summary',{});manifest=probe.get('manifest',{})
    if state or summary:
        count=manifest.get('count');ranks=state.get('ranks',{});done=sum(r.get('completed_tokens',0)for r in ranks.values())
        complete=(state.get('status')=='COMPLETE'and summary.get('status')=='COMPLETE'and summary.get('count')==count==312)
        body+='<h3>固定Safety10k支持探针：'+('312场景完整完成'if complete else '实际诊断进度')+'</h3><p>156个全历史选定失败scene＋156个同物理log高分对照；冻结同一Safety10k，仅生成／评分，不优化actor。'+('已完成 '+h.number(summary.get('costs',{}).get('new_generated_candidates'))+' 个新候选／官方评分，零optimizer更新。'if complete else '已记录完成 '+h.number(done)+' / '+h.number(count)+' 场景；不发布partial均值。')+'</p>'
        if complete:
            cohorts=summary.get('cohorts',{});failure=cohorts.get('historical_failure',{});high=cohorts.get('matched_high_control',{})
            names={'A_late':'A：末段SDE [7,8,9]','B_mixed':'B：分散SDE [3,6,8]','C_shared_root':'C：末段SDE＋同初始root','D_ODE':'D：16个独立root ODE'}
            rows=[]
            for key,title in names.items():
                f=failure.get(key,{});g=high.get(key,{})
                if f and g:rows.append([title,h.number(f.get('scenes_with_quality'))+' / 156',h.number(f.get('feasible_count'))+' / '+h.number(f.get('candidates')),h.number(f.get('canonical_zero'))+' / '+h.number(f.get('candidates')),h.number(g.get('strict_quality_count'))+' / '+h.number(g.get('candidates'))])
            body+=h.table(['诊断采样方式（每scene 16条）','历史失败：找到quality的scene','历史失败：可行候选','历史失败：zero候选','匹配高分对照：quality候选'],rows)
            body+='<p>quality定义为NC=DAC=TTC=Comfort1且canonical≥0.9，DDC另报。候选质量与scene支持覆盖须分开看：例如B显著减少失败候选，但并未同幅扩大“至少找到一条quality”的scene数。A的16条相对其嵌套前8条新增 '+h.number(summary.get('paired_support_contrasts',{}).get('historical_failure',{}).get('A16_good_A_first8_not'))+' 个quality场景；嵌套前8不是独立执行的G8。</p>'
            spread=analyses.get('SUPPORT_COVERAGE_PHYSICAL_SPREAD.json',{})
            if spread.get('status')=='COMPLETE':
                f=spread.get('cohorts',{}).get('historical_failure',{})
                body+='<p><b>支持覆盖与扰动幅度：</b>156个历史失败scene中，晚段前8条quality覆盖 '+h.number(f.get('A8_quality_scenes'))+'，16条覆盖 '+h.number(f.get('A16_quality_scenes'))+'；已观测8late＋8mixed候选的并集覆盖 '+h.number(f.get('mixture8plus8_quality_scenes'))+'，尚不是已训练的混合采样策略。晚段候选相对ODE平均XY差 '+h.fmt(f.get('A_xy_delta_meters',{}).get('mean'),3)+'m，mixed '+h.fmt(f.get('B_xy_delta_meters',{}).get('mean'),3)+'m。更接近ODE、更高候选分和新增有效支持不能混为同一指标。'+h.relative_link(root,str(project/'analysis'/'SUPPORT_COVERAGE_PHYSICAL_SPREAD.json'),'物理扰动与覆盖原件')+'</p>'
        body+='<p class="muted">这是历史选择的TRAIN局部支持诊断，不是全量分布、NAVTEST成绩或best-of16部署。B虽同η和三步，改时间位置也改变实际扰动能量；D首候选使用训练私有seed42，不能冒充正式seed2101评测。'+h.relative_link(root,probe.get('summary_path',''),'支持探针完整SUMMARY')+' · '+h.relative_link(root,str(project/'data'/'probes'/'FROZEN_SAFETY10K_SUPPORT_20261031'/'MANIFEST.json'),'冻结选择与四种采样协议')+'</p>'
    comparisons=research.get('pilot_comparisons',{})
    if comparisons:
        total=len(research.get('runs',[]))*2
        body+='<p><b>正式性能仍独立等完整人口：</b>'+h.number(len(research.get('runs',[])))+'臂500／1,000共'+h.number(total)+'个NAVTEST输入，当前配对reader状态 '+h.esc(comparisons.get('status','待记录'))+'，就绪 '+h.number(comparisons.get('ready'))+' / '+h.number(total)+'；既有旧baseline smoke不计新成绩。'+h.relative_link(root,str(project/'analysis'/'PILOT_COMPARISONS'/'STATE.json'),'异步配对状态')+'</p>'
    return body


def render_exploration_research(root,research,h):
    project=Path(research.get('root','/mnt/project/qwen-world-safety-exploration-20261031'));runs=research.get('runs',[]);analyses=research.get('analyses',{})
    body='<div class="notice"><b>最新执行要求：研究点逐项落实为代码与有对照实验。</b>新研究位于旧World IL70体系的独立目录，与ABCD架构实验分开；现有live不热改。代码完成、CPU检查、正式注册、child启动、八rank真实更新和完整性能分层记录。</div><p>组内G16不改变轨迹horizon8或world8。online priority／learning-value、DenseDAC tie信用等分别开发；不能把准备状态写成已训练，也不能把新信用当成已提分。每个arm的数据合同、单变量与预算以本轮登记为准。</p>'
    registered={r.get('id')for r in runs}
    priorities=[('1','SAFETY_INVERSION_CANCEL_G8','Inversion：同组canonical反序','抵消strict4高分负A与更低canonical的strict3正A；保组和，优先直接检验局部目标冲突'),('2','SAFETY_PROTECTED_HIGH_G8','Protected：高分负信用保护','仅将strict4高分负A置0；其它正A不动，单独检验去除负信用'),('3','SAFETY_SATURATED_G8','Saturated：补信用饱和','仅safe EP1、原A0且canonical可分时恢复组内canonical信用；不覆盖active负high')]
    phase2=[row for row in priorities if row[1]in registered]
    if phase2:
        body+='<h3>七臂问题对照：第二阶段优先验证信用方向</h3><p>原四臂继续分别检验G8控制、G16探索、窄DAC失败细分与在线数据课程；新三臂独立改变信用分配。原四臂保持initial_v1，新三臂使用credit_v2；不热改正在运行的旧source，也不把准入或代码就绪当作学习完成。</p>'
        body+=h.table(['Phase2优先级','正式arm／要解决的问题','唯一机制与判据'],[[p,h.esc(arm)+'<small>'+h.esc(title)+'</small>',h.esc(note)]for p,arm,title,note in phase2])
    body+=render_exploration_diagnostics(root,research,h)
    development=[]
    for filename,title in [('GROUP_FOUNDATION_HANDOFF.json','G8/G16基础、形状与恢复'),('CURRICULUM_HANDOFF.json','在线priority／learning-value课程'),('DENSE_DAC_HANDOFF.json','DenseDAC真实出界时长信用')]:
        document=analyses.get(filename,{})
        if document:development.append([title,'代码／CPU就绪（仅开发收据）'if 'READY'in str(document.get('status',''))else h.esc(document.get('status','待状态')),h.relative_link(root,str(project/'analysis'/filename),'代码与检查证据')])
    if development:body+=h.table(['研究点','开发阶段','证据'],development)+'<p class="muted">上表是代码交付阶段；是否已正式注册、开始真实优化或提分，以以下运行与完整结果为准。</p>'
    registry=research.get('registry',{})
    if runs:
        body+='<p><b>'+h.number(len(runs))+'组已正式注册。</b>共同fresh IL70／Adam／seed42／update0；训练人口 '+h.number(registry.get('training_population'))+'，旧DEV不独立；NAVTEST不作训练反馈。有限门槛500／1,000更新，不自动50k。owner／worker与日志未出现前，只记录登记，不称在训。</p>'
        commits=sorted({r.get('source_commit')for r in runs if r.get('source_commit')})
        if commits:body+='<p class="muted">新World注册source commit：'+h.esc(' / '.join(commits))+'；与ABCD的运行source分开，以实际job回执核对执行路径。</p>'
    for role,controller in research.get('controllers',{}).items():
        body+='<p class="muted">'+{'RESEARCH_NATIVE':'第一阶段原生协调','PHASE2_NATIVE':'第二阶段原生协调','RESEARCH_EVALUATION':'共享评测协调'}.get(role,role)+'owner文件状态 '+h.esc(controller.get('state',{}).get('status','待记录'))+'，记录PID '+h.number(controller.get('owner',{}).get('pid'))+'；此为CPU协调状态，learner实际更新另列。</p>'
    methods={'SAFETY_G8_CONTROL':'SafetyEP G8 uniform控制','SAFETY_G16':'只将组内G8改为G16','SAFETY_DAC_DENSE_G8':'G8＋窄DenseDAC时长信用；保baseStd','SAFETY_ACTIVE_DATA_G8':'G8＋在线课程；≥75%uniform floor','SAFETY_SATURATED_G8':'G8＋仅饱和safe组canonical信用','SAFETY_PROTECTED_HIGH_G8':'G8＋strict4高分负A置0','SAFETY_INVERSION_CANCEL_G8':'G8＋canonical反序信用对抵消'}
    rows=[]
    for run in runs:
        status=('准入等待（prechild75，非科学失败）'if run.get('worker_returncode')==75 else '执行失败待审'if run.get('worker_returncode')not in(None,0)else '原生已退出'if run.get('worker_returncode')==0 else '八rank共同更新已见'if run.get('worker_receipt_status')=='RUNNING'and run.get('journal_ranks')==run.get('expected_ranks')and h.finite(run.get('updates'))and run['updates']>0 else 'child已启动，待实际更新'if run.get('worker_receipt_status')=='RUNNING'else str(run.get('status','已登记，待启动')))
        if run.get('worker_receipt_status')!='RUNNING'and run.get('worker_returncode')is None:
            status={'REGISTERED_NOT_STARTED':'已注册，待实际启动证据','DISPATCHED':'已派发，待worker回执','TRAINING':'协调状态已推进，待worker／实际更新','WAITING_RESOURCE':'等待资源准入','RESOURCE_REQUEUE':'准入等待／资源重排，非科学失败'}.get(run.get('status'),status)
        arm_label=h.esc(run.get('name_zh',run.get('id','')))
        if run.get('host'):arm_label+='<small>'+h.esc(run['host'])+'</small>'
        if run.get('job'):arm_label+='<small>'+h.relative_link(root,run['job'],Path(run['job']).name)+'</small>'
        rows.append([arm_label,h.esc(run.get('algorithm')or run.get('research_method')or methods.get(run.get('id'),'待方法元数据')),h.number(run.get('group_size')),h.esc(status),h.number(run.get('updates')),h.number(run.get('new_candidates')),h.number(run.get('research_budget_updates',run.get('cap_updates',run.get('target_updates'))))])
    if rows:body+=h.table(['研究arm','方法','组内G','实际状态','共同更新','记录的fresh候选','登记预算上限'],rows)
    else:body+='<p class="warning">当前尚无该新研究的正式RUNS／实际训练证据；不将开发任务或设计文档当作已启动。</p>'
    process=[]
    for run in runs:
        window=run.get('process_window',{})
        if window:process.append([h.esc(run.get('id','')),h.number(window.get('start_update'))+'–'+h.number(window.get('end_update')),h.number(window.get('fresh_G_groups')),h.fmt(window.get('fresh_ratio_max_abs_error'),6),'全finite'if window.get('all_recorded_gradient_norms_finite')else '待核',h.number(window.get('dense_activated_groups'))])
    if process:body+='<details open><summary>实际训练过程：八rank共同日志尾窗，不是性能分数</summary>'+h.table(['arm','共同更新窗口','完整fresh组','fresh |ratio−1| 最大值','记录的总梯度范数','Dense触发组'],process)+'<p class="muted">仅有界日志尾窗，fresh collection只取首次inner且要求全部八rank齐全；ratio误差0代表fresh ratio=1。未记录的Dense字段留待记录，不能当零；四次inner不重复计算候选或触发次数。</p></details>'
    startup=analyses.get('ACTUAL_NATIVE_STARTUP.json',{})
    dense=next((r for r in startup.get('runs',[])if r.get('arm')=='SAFETY_DAC_DENSE_G8'),{})
    if dense and h.finite(dense.get('dense_activated_groups')):body+='<p><b>Dense并非尚未激活：</b>'+h.esc(h.utc(startup.get('at_unix')))+'启动快照中，实际 '+h.number(dense.get('dense_activated_groups'))+' / '+h.number(dense.get('unique_local_first_inner_rows'))+' 个fresh local组触发；这是当时累计快照，后续尾窗另列，不把触发当成修复成绩。</p>'
    active=next((r.get('curriculum_native',{})for r in runs if r.get('id')=='SAFETY_ACTIVE_DATA_G8'),{})
    if active:
        body+='<p><b>课程状态已进入原生恢复边界：</b>sealed '+h.number(active.get('updates'))+'更新的八rank保存摘要，共 '+h.number(active.get('total_scene_exposures'))+' 次scene曝光，其中uniform '+h.number(active.get('uniform_exposures'))+'、priority replay '+h.number(active.get('priority_exposures'))+'；八rank分别满足≥75%uniform。这里是实际已保存的选择／恢复状态，不是proxy已证明learning value或提分。'+h.relative_link(root,active.get('evidence',''),'八rank transaction摘要')+'</p>'
    proof_rows=[]
    for name,document in analyses.items():
        if not isinstance(document,dict):continue
        status=document.get('status')
        if not status:continue
        label='CPU检查通过（仅该收据）'if status=='PASS'or 'CPU_READY'in str(status)else '设计／待注册'if 'DESIGN'in str(status)or 'NOT_REGISTERED'in str(status)else str(status)
        proof_rows.append([h.esc(document.get('title_zh',name)),h.esc(label),h.relative_link(root,str(project/'analysis'/name),'开发证据')])
    if proof_rows:body+='<details><summary>逐项开发与CPU证据；不等于GPU资格或收益</summary>'+h.table(['条目','代码／检查阶段','来源'],proof_rows)+'</details>'
    complete=[r for r in research.get('results',[])if r.get('publishable_complete')]
    if complete:body+=h.table(['arm','更新','cohort','完整场景','完整PDMS','证据'],[[h.esc(r.get('arm','')),h.number(r.get('updates')),h.esc(r.get('cohort','')),h.number(r.get('scenes')),h.fmt(r.get('PDMS'),5),h.relative_link(root,r.get('evidence',''),'SUMMARY')]for r in complete])
    else:body+='<p>本研究尚无完整新成绩，目标仍是raw canonical PDMS与零／低分修复、高分保持共同改善。</p>'
    if research.get('registry'):body+='<p>'+h.relative_link(root,str(project/'RUNS.json'),'新研究RUNS')+'</p>'
    body+='<p class="muted">固定根目录 '+h.esc(project)+'；报告只读，不执行调度或更改现有learner。</p>'
    return body


def render_full_population(root,research,h):
    """Render a sealed population summary; never read or subset its ledger."""
    audit=research.get('analyses',{}).get('SAFETY_FULL_POPULATION_AUDIT_20261031/SUMMARY.json',{})
    if audit.get('status')!='COMPLETE'or audit.get('population',{}).get('scenes')!=12146:return ''
    checkpoints=audit.get('checkpoints',{});contrasts=audit.get('contrasts',{});updates=audit.get('population',{}).get('safety_updates',[])
    if not updates:return ''
    base='PPO_SAFETY_EP_u010000_NAVTEST';latest=f'PPO_SAFETY_EP_u{max(updates):06d}_NAVTEST';memory='PPO_SAFETY_SUCCESS_MEMORY_u010000_NAVTEST'
    pair=contrasts.get(latest+'_minus_'+base,{})
    matrix=pair.get('transition_matrix4x4',[])
    if len(matrix)!=4 or any(len(row)!=4 for row in matrix)or sum(sum(row)for row in matrix)!=12146:return ''
    from research_report import figure
    project=Path(research['root']);bins=['zero','low_positive_lt05','mid_05_to09','high_ge09'];bin_names=['0','0<s<.5','.5<=s<.9','s>=.9']
    def short(label):
        if label==base:return 'Safety10k'
        if label==latest:return 'Safety'+str(max(updates)//1000)+'k'
        if label==memory:return 'Memory10k'
        if label=='IL70_NAVTEST':return 'IL70'
        if '_u'in label:
            prefix,tail=label.rsplit('_u',1)
            return prefix.removeprefix('PPO_')+' '+str(int(tail[:6])//1000)+'k'
        return label
    def ci_text(c):
        ci=c.get('existing_CI95_PDMS',{})
        if isinstance(ci,dict):ci=ci.get('CI95_PDMS')
        return '['+', '.join(f'{v:+.4f}'for v in ci)+']'if isinstance(ci,list)and len(ci)==2 else '未新增CI'
    def matrix_plot(plt):
        fig,ax=plt.subplots(figsize=(6.5,4.3));plot=ax.imshow(matrix,cmap='Blues',vmin=0)
        top=max(max(row)for row in matrix)
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):ax.text(j,i,f'{value:,}',ha='center',va='center',color='white'if value>top*.55 else '#172b43',fontsize=10)
        ax.set_xticks(range(4),bin_names);ax.set_yticks(range(4),bin_names)
        ax.set(xlabel=short(latest)+' score bin',ylabel='Safety10k score bin',title='Full population transitions (12,146 scenes)');fig.colorbar(plot,ax=ax,label='Scene count',shrink=.8);fig.tight_layout();return fig
    def mass_plot(plt):
        fig,ax=plt.subplots(figsize=(6.5,4.3));points=[]
        for u in sorted(updates):
            label=f'PPO_SAFETY_EP_u{u:06d}_NAVTEST';c=contrasts.get(label+'_minus_'+base,{})
            if c:points.append(('S'+str(u//1000),c))
        if contrasts.get(memory+'_minus_'+base):points.append(('M10',contrasts[memory+'_minus_'+base]))
        valid=[(name,c['gain_loss_mass'])for name,c in points if all(h.finite(c.get('gain_loss_mass',{}).get(key))for key in ['G_plus_PDMS','G_minus_PDMS','net_PDMS'])]
        xx=list(range(len(valid)));ax.bar(xx,[m['G_plus_PDMS']for _,m in valid],color='#059669',label='Gain mass');ax.bar(xx,[m['G_minus_PDMS']for _,m in valid],color='#e76f51',label='Loss mass');ax.scatter(xx,[m['net_PDMS']for _,m in valid],s=22,color='#172b43',label='Net',zorder=3)
        ax.set_xticks(xx,[name for name,_ in valid],rotation=45,ha='right',fontsize=8);ax.axhline(0,color='#64748b',lw=.8);ax.set(ylabel='Global PDMS contribution vs Safety10k',xlabel='Complete checkpoint (S: Safety, M: Memory)');ax.legend(frameon=False,fontsize=8,ncol=3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();return fig
    population=audit['population'];longitudinal=audit.get('longitudinal',{});zeros=pair.get('zeros',{});mass=pair.get('gain_loss_mass',{})
    body='<h3 id="safety-population">Safety完整人口：零／低分修复与高分保持必须同时看</h3><p>'+h.number(population.get('complete_populations'))+'份完整人口 × 12,146场景／44物理log，共 '+h.number(population.get('total_scene_observations'))+'次场景观测；Safety包含 '+h.number(longitudinal.get('k'))+'个完整节点，最高到 '+h.number(max(updates))+' 更新。这是全NAV指标审计，不是旧TRAIN窗口的外推。</p>'
    body+='<div class="cards">'+''.join('<article><span>'+title+'</span><strong>'+h.number(value)+'</strong><small>'+note+'</small></article>'for title,value,note in [('始终零分',longitudinal.get('persistent_zero'),'每个列明的'+h.number(longitudinal.get('k'))+'次Safety观测均为0'),('曾经零分',longitudinal.get('ever_zero'),'覆盖全部12,146个场景'),('发生修复后复发',longitudinal.get('relapsed_zero'),'至少一次0→正→0'),('最新零分',longitudinal.get('latest_zero'),'Safety'+str(max(updates)//1000)+'k完整人口')])+'</div>'
    body+='<p><b>从Safety10k到'+str(max(updates)//1000)+'k：</b>总分变化 '+f"{pair['delta_PDMS']:+.5f}"+'，复用既有95%配对CI '+ci_text(pair)+'；修复 '+h.number(zeros.get('repaired'))+' 个零分，同时新增 '+h.number(zeros.get('new_zero'))+' 个。零分数量减少仍可伴随总分下降；不能只优化zero计数。</p>'
    original_gain=contrasts.get(base+'_minus_PPO_FULL_u010000_NAVTEST',{})
    if original_gain:
        zz=original_gain.get('zeros',{});attribution=original_gain.get('algebraic_score_attribution',{}).get('component_PDMS_mass',{})
        body+='<p><b>Safety最初的收益也不等同于更少失败：</b>10k比原FULL10k '+f"{original_gain['delta_PDMS']:+.5f}"+' PDMS，但zero '+h.number(zz.get('baseline'))+'→'+h.number(zz.get('target'))+'（修复'+h.number(zz.get('repaired'))+'／新增'+h.number(zz.get('new_zero'))+'）。完整score代数归账中，EP贡献 '+h.fmt(attribution.get('ego_progress'),5)+'，NC '+h.fmt(attribution.get('no_at_fault_collisions'),5)+'、TTC '+h.fmt(attribution.get('time_to_collision_within_bound'),5)+'；这是分数恒等分解，不是参数梯度或驾驶因果。</p>'
    memory_pair=contrasts.get(memory+'_minus_'+base,{})
    if memory_pair:
        mm=memory_pair.get('gain_loss_mass',{});high=memory_pair.get('by_fixed_cohort',{}).get('Safety10k_high_ge09',{})
        body+='<p><b>Memory的新高为何不足以叫突破：</b>全人口gain '+h.fmt(mm.get('G_plus_PDMS'),5)+' 被loss '+h.fmt(mm.get('G_minus_PDMS'),5)+' 抵消；固定Safety10k高分队列均值 '+h.fmt(high.get('delta_PDMS'),5)+'，其中 '+h.number(high.get('below09'))+' 个跌破0.9、'+h.number(high.get('zero'))+' 个变zero。保持表继续用固定成员，而不是只统计目标模型新产生了多少高分。</p>'
    body+='<div class="two">'+figure(root,h,'safety_population_transition','全12,146场景转移：基准行 → 目标列',matrix_plot,'行固定为Safety10k四档，列为最新完整Safety四档；每个格子是场景数，总和12,146。0、0.5、0.9边界不重分类。')+figure(root,h,'safety_population_gain_loss','全人口gain与loss mass：改善被哪里抵消',mass_plot,'G+与G−均以全12,146为分母，单位是全局PDMS点；黑点为净变化。仅使用完整对照；S为Safety、M10为Memory10k。不同模型的点不连成连续训练曲线。')+'</div>'
    cohort_mass=pair.get('by_fixed_cohort',{})
    mass_rows=[]
    for bin_key,title in zip(bins,['原zero','原0<s<0.5','原0.5≤s<0.9','原s≥0.9']):
        row=cohort_mass.get('Safety10k_'+bin_key,{}) if bin_key!='high_ge09'else cohort_mass.get('Safety10k_high_ge09',{})
        mm=row.get('global_PDMS_mass',{})
        if row:mass_rows.append([title,h.number(row.get('scenes')),h.fmt(mm.get('G_plus_PDMS'),5),h.fmt(mm.get('G_minus_PDMS'),5),h.fmt(mm.get('net_PDMS'),5)])
    if mass_rows:body+=h.table(['Safety10k固定档位','场景数','全局G+','全局G−','全局净贡献（PDMS点）'],mass_rows)
    body+='<h3>高分保持：两种固定队列分开，不换成员</h3>'
    for cohort,origin,title in [('IL70_high_ge09','IL70_NAVTEST','固定IL70 s≥0.9：主保持诊断'),('Safety10k_high_ge09',base,'固定Safety10k s≥0.9：条件次要诊断')]:
        original=checkpoints.get(origin,{}).get('fixed_cohorts',{}).get(cohort,{})
        control=checkpoints.get(base,{}).get('fixed_cohorts',{}).get(cohort,{})
        rows=[]
        for label in ([origin,base,latest,memory]if origin!=base else[base,latest,memory]):
            group=checkpoints.get(label,{}).get('fixed_cohorts',{}).get(cohort,{})
            if not group:continue
            b=group.get('bins',{});pdms=group.get('PDMS')
            rows.append([short(label),h.number(group.get('scenes')),h.fmt(pdms,5),h.fmt(pdms-original['PDMS'],5)if h.finite(pdms)and h.finite(original.get('PDMS'))else '未齐',h.fmt(pdms-control['PDMS'],5)if h.finite(pdms)and h.finite(control.get('PDMS'))else '未齐',h.number(b.get('high_ge09')),h.number(sum(b.get(k,0)for k in bins[:3])),h.number(b.get('zero'))])
        body+='<p><b>'+title+'</b></p>'+h.table(['目标模型','固定成员','群体PDMS','相对本队列定义模型','相对Safety10k','仍≥0.9','低于0.9','变zero'],rows)
    body+='<p class="protocol">队列均值差是描述性保持诊断，未新增bootstrap；Safety10k是已观察峰值，以它筛出的高分群体有条件选择与上行空间限制。转移和分数代数归账不直接证明遗忘、anchor或某个组件的物理因果。</p>'
    witnesses=research.get('analyses',{}).get('SAFETY_FULL_POPULATION_AUDIT_20261031/OBSERVED_POLICY_WITNESSES.json',{})
    if witnesses.get('status')=='COMPLETE':
        wz=witnesses.get('cohorts',{}).get('safety10k_zero',{});wp=witnesses.get('cohorts',{}).get('persistent_safety_zero_13obs',{})
        body+='<p><b>零分不等于已证明无解：</b>Safety10k的 '+h.number(wz.get('scenes'))+'个zero里，在17个已观察policy输出中有 '+h.number(wz.get('any17_strict_safe_witness'))+'个存在严格safe见证、'+h.number(wz.get('any17_ge09_witness'))+'个可达≥0.9；13次Safety都为0的 '+h.number(wp.get('scenes'))+'个场景中，也有 '+h.number(wp.get('external4_strict_safe_witness'))+'个被另外4个已观察policy中的输出安全解决。未看到见证不证明物理不可行；这些NAV见证不转入TRAIN，也不证明同一个参数向量能同时选择全部好动作。'+h.relative_link(root,str(project/'analysis'/'SAFETY_FULL_POPULATION_AUDIT_20261031'/'OBSERVED_POLICY_WITNESSES.json'),'完整policy见证审计')+'</p>'
    bin_rows=[]
    for label in population.get('labels',[]):
        cp=checkpoints.get(label,{})
        if cp.get('scenes')==12146:bin_rows.append([short(label),h.fmt(cp.get('PDMS'),5)]+[h.number(cp.get('bins',{}).get(key))for key in bins])
    body+='<details><summary>全部完整checkpoint分档与可核查ledger</summary>'+h.table(['完整人口','PDMS','score=0','0<s<0.5','0.5≤s<0.9','s≥0.9'],bin_rows)+'<p>'+' · '.join(h.relative_link(root,item['path'],name+'（'+h.number(item.get('rows'))+'行）')for name,item in audit.get('artifacts',{}).items()if name in('ledger.csv','ledger.jsonl'))+'</p><p>'+h.relative_link(root,str(project/'analysis'/'SAFETY_FULL_POPULATION_AUDIT_20261031'/'SUMMARY.json'),'完整全人口SUMMARY、校验与全部对照')+'</p></details>'
    return body


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


def render_completed_methods(root,research,snapshot,h):
    """Keep completed World methods separate from new ABCD DEV/NAVTEST."""
    wanted={'PPO_SAFETY_SUCCESS_MEMORY':('成功经验Memory','在SafetyEP上复用当前run已验证成功轨迹，增加独立历史辅助回放。'),
        'PPO_SAFETY_FPO':('FPO目标替换','以条件flow-matching损失差构造终点代理ratio，替换原转移PG。'),
        'PPO_FPO_FULLTIME':('FPO时间上界对照','仅将FPO采样时间上界0.8扩到1.0，同时改变原区间密度与权重。'),
        'PPO_SAFETY_FEASIBLE_SCORE':('可行内质量信用','在可行候选中用q=(5EP+2Comfort)/7替代仅EP的效率信用。')}
    nav=[r for r in snapshot.get('results',[])if r.get('publishable_complete')and r.get('cohort')=='NAVTEST']
    controls={r.get('updates'):r for r in nav if r.get('arm')=='PPO_SAFETY_EP'}
    latest={arm:max((r for r in nav if r.get('arm')==arm),key=lambda r:r.get('updates',0),default={})for arm in wanted}
    review=research.get('analyses',{}).get('NEW_METHODS_COMPLETED_REVIEW_20261031.json',{})
    def matched(arm,updates,baseline='PPO_SAFETY_EP'):
        for node in review.get('methods',{}).get(arm,{}).get('matched_results',[]):
            if node.get('updates')!=updates:continue
            candidate=node.get('comparisons',{}).get(baseline,node).get('NAVTEST',{})
            if candidate.get('status')=='COMPLETE':return candidate
        return {}
    def interval(ci):return '['+', '.join(f'{v:+.5f}'for v in ci)+']'if isinstance(ci,list)and len(ci)==2 else '配对区间待更新'
    candidates=[r for r in nav if r.get('arm')in(*wanted,'PPO_SAFETY_EP')]
    if not candidates:return ''
    best=max(candidates,key=lambda r:r['PDMS']);rows=[]
    for arm,(name,mechanism)in wanted.items():
        r=latest[arm]
        if not r:continue
        baseline=controls.get(r.get('updates'),{});delta=r['PDMS']-baseline['PDMS']if baseline else None
        pair=matched(arm,r.get('updates'));zeros=pair.get('zeros',{})
        conclusion=({'PPO_SAFETY_SUCCESS_MEMORY':'最高观察点；CI跨0，未证明总分优势，NC/TTC与zero交换不利',
            'PPO_SAFETY_FPO':'此配方10k明确退化，不直接续长；不能由此否定全部Flow优化方法',
            'PPO_FPO_FULLTIME':'同预算也低于原FPO，扩大时间覆盖未转为部署收益',
            'PPO_SAFETY_FEASIBLE_SCORE':'信用增加未转为高分；EP损失约1.198pp'}[arm])
        rows.append([name+'<small>'+h.esc(mechanism)+'</small>',h.number(r.get('updates')),h.fmt(r['PDMS'],5),(f'{delta:+.5f}'if h.finite(delta)else '待同预算')+'<small>'+interval(pair.get('CI95_PDMS'))+'</small>',h.number(zeros.get('repaired'))+' / '+h.number(zeros.get('new_zero')),conclusion,h.relative_link(root,r.get('evidence',''),'完整NAVTEST')])
    memory=matched('PPO_SAFETY_SUCCESS_MEMORY',10000)
    body='<div class="notice"><b>已有World方法最新结论：最高完整NAVTEST '+h.fmt(best['PDMS'],5)+'，尚未达到92。</b>这是同IL70的已完成方法研究，与本轮ABCD各自IL父模型／DEV人口分开。成功经验Memory10k比原Safety10k仅高约0.02627，95%配对CI '+interval(memory.get('CI95_PDMS'))+'；不能称为显著或稳定提升。</div>'
    body+=h.table(['方法／机制','已完整更新','完整NAVTEST PDMS','同预算Safety点差／95%CI','零分修复／新增','当前结论','证据'],rows)
    if review:
        body+='<p><b>收益来自哪里、代价是什么：</b>Memory10k虽刷新观察最高点，零分仍从256增到275（修复15、新增34），NC与TTC退化区间均不含0。FPO的1k高分队列局部收益没有持续到10k，全局退化2.60511点；时间扩展5k仍比原FPO低0.25738点。补Comfort的q信用激活了更多训练信号，但约1.2pp的EP损失更大。辅助确实激活／梯度方向确实改变，均不能代替部署收益。</p>'
        conclusions=review.get('what_we_learned_zh',[])
        if isinstance(conclusions,list):body+='<details><summary>完整方法结论与因果边界</summary><p>'+'</p><p>'.join(h.esc(s)for s in conclusions if isinstance(s,str))+'</p></details>'
        body+='<p>'+h.relative_link(root,str(Path(research['root'])/'analysis'/'NEW_METHODS_COMPLETED_REVIEW_20261031.json'),'完整方法复核、组件与边界')+'</p>'
    body+='<p><b>优先级：</b>先恢复ABCD的原生续训与完整NAVTEST，按各自IL父模型作同人口配对；保留Safety10k与Memory10k，按零分新增／修复和固定IL高分保持审查，不自动延长最高均值配方。未曝光、真实all-fail打平局、已有安全见证但ODE失败需分别处理。FPO、时间扩展及质量信用的这些完整负结果不支持直接续长或盲叠组件；新方案须另作单变量注册。</p>'
    return body


def render_abcd_research(root,research,h,world_snapshot=None):
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
        elif run.get('failure_kind')=='NATIVE_RESUME_IDENTITY_MIGRATION_REQUIRED':status='5k续训identity校验失败；停于1k，待恢复'
        elif run.get('worker_returncode')not in(None,0):status='执行失败待审，保留原receipt'
        elif run.get('worker_receipt_status')=='RUNNING'and run.get('updates')==1000 and(run.get('stage_target_updates')or 0)>1000:status='续训child已开始；共同更新仍1k，等待新推进'
        elif run.get('worker_receipt_status')=='RUNNING'and h.finite(run.get('updates'))and run['updates']>0:status='已见实际原生更新（'+number(run.get('journal_ranks'))+'/'+number(run.get('expected_ranks'))+' rank）'
        elif run.get('worker_receipt_status')=='RUNNING':status='child已开始；等待实际optimizer日志'
        elif run.get('worker_returncode')==0:status='原生子进程已退出；完整阶段结果独立核对'
        else:status={'WAITING_RESOURCE':'等待资源／依赖准入','REGISTERED':'已登记，待实际启动证据','RESUME_FROM_NATIVE1000_REGISTERED':'1k恢复已登记；待新child与实际推进','EXECUTION_RECOVERY_REGISTERED':'恢复已登记；待实际启动证据'}.get(run.get('status'),str(run.get('status','已登记，待实际启动证据')))
        job=run.get('job');attempt=Path(job).name if job else '尚未派发'
        status_rows.append([key,esc(status),number(run.get('updates')),esc(run.get('host','未派发／待记录')),esc(run.get('algorithm')or run.get('research_method')or run.get('method')or('设计：SafetyEP＋PPO/W3；待正式注册'if key in'ABC'else '设计：JaT启发action-only PPO；待正式注册')),link(job,attempt)if job else esc(attempt)])
    body='<div class="notice"><b>新增主研究：ABCD各自以锁定时最高分IL checkpoint启动RL。</b>按用户要求选当时已完整NAVTEST的best，不等待未来100k端点；C锁定当时的12k best，而非当时最新18k。四条架构的动作路径与梯度scope不同，不能直接复用旧World的52.7M scope。原World四组的已完成结果另列，保持父模型与数据人口的区别。</div>'
    body+='<details><summary>原World四组完整方法复核：配对结果、机制与代价</summary>'+render_completed_methods(root,research,world_snapshot or{},h)+'</details>'
    failed=[r for r in runs if r.get('failure_kind')=='NATIVE_RESUME_IDENTITY_MIGRATION_REQUIRED']
    recovering=[r for r in runs if r.get('status')in('RESUME_FROM_NATIVE1000_REGISTERED','EXECUTION_RECOVERY_REGISTERED')and r.get('worker_receipt_status')!='RUNNING']
    failed_nav=[r for r in research.get('evaluation_operations',[])if r.get('failure_kind')=='NAVTEST_DATA_MANIFEST_MISSING_BEFORE_MODEL_LOAD']
    if failed or failed_nav or recovering:
        body+='<div class="warning"><b>当前执行故障与恢复：</b>'+('有'+number(len(failed))+'组5k原生恢复在identity一致性校验处退出；已保留1k完整checkpoint，没有把失败启动算作新学习。'if failed else '')+('有'+number(len(recovering))+'组从1k恢复已登记，尚未看到当前新child与超过1k的共同更新；不计为已恢复训练。'if recovering else '')+('有'+number(len(failed_nav))+'组NAVTEST因原指定目录的manifest.json不存在，在模型加载前失败；恢复／重新入队状态按当前owner读取，尚无本轮完整Test成绩。'if failed_nav else '')+' DEV完整结果单独列示；不能用其替代NAVTEST或92目标。</div>'
        identity_check=research.get('analyses',{}).get('ABCD_NATIVE_RESUME_IDENTITY_CPU_CHECK.json',{})
        if (failed or recovering)and identity_check.get('status')=='PASS':
            body+='<p>恢复校验的CPU根因已定位：迁移provenance在内存为tuple、JSON落盘为list，canonical哈希与原配方均相同，却被Python对象比较拒绝；发生于restore／新update前。这是执行兼容故障，不是算法学习负结果。两个定向CPU恢复测试已通过；CPU修复不等于四组已恢复训练。'+link(project/'analysis'/'ABCD_NATIVE_RESUME_IDENTITY_CPU_CHECK.json','精确恢复CPU核对')+'</p>'
    body+='<div class="cards">'+''.join(cards)+'</div>'
    body+=table(['架构','本轮实际状态','共同完整更新','host','方法（设计／注册分开）','当前attempt'],status_rows)
    if runs:
        body+='<p>四组已正式登记；实际训练更新只取八个rank各自已提交metrics的共同最小数，每100步checkpoint的latest指针不作为实时步数。当前attempt由独立native owner选定；同一job的RESULT优先于RUNNING，失败旧attempt保留为历史。'+link(project/'RUNS.json','正式RUNS')+' · '+link(project/'owners'/'ABCD_NATIVE'/'STATE.json','当前native owner状态')+'</p>'
    evaluator=research.get('controllers',{}).get('ABCD_EVALUATION',{})
    if evaluator:
        state=evaluator.get('state',{})
        body+='<p>独立异步评测owner当前文件状态 '+esc(state.get('status','未记录'))+'（记录PID '+number(evaluator.get('owner',{}).get('pid'))+'）；跨cohort总worker上限 '+number(state.get('max_workers'))+'，单cohort分片上限 '+number(state.get('max_shards'))+'，两者分别记录。实际world和完成分片以每项spec／owner为准。'+link(evaluator['state_path'],'评测进度')+'</p>'
    nav_progress=[]
    for row in research.get('results',[]):
        if row.get('cohort')!='NAVTEST':continue
        operation=next((op for op in research.get('evaluation_operations',[])if op.get('evaluation')==row.get('evaluation')),{});statuses=operation.get('ranks_by_status',{})
        nav_progress.append([esc(row.get('architecture_id',row.get('arm',''))),number(row.get('updates')),esc(operation.get('status',row.get('status','待状态'))),number(row.get('scenes'))+' / '+number(row.get('expected_scenes')),number(statuses.get('COMPLETE',0))+' / '+number(operation.get('world')),'完整 '+fmt(row.get('PDMS'),5)if row.get('publishable_complete')else '尚未完整，不发布均值'])
    if nav_progress:body+='<h3>本轮NAVTEST实时进度</h3>'+table(['架构','RL更新','评测owner状态','已提交scene','完成分片／实际world','成绩'],nav_progress)
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
        body+='<h3>新ABCD已完整NAVTEST成绩</h3>'+table(['架构','RL更新','完整PDMS','相对各自IL观察点差','证据'],[[esc(row.get('architecture_id',row.get('arm','待身份'))),number(row.get('updates')),fmt(row.get('PDMS'),6),fmt(row['PDMS']-parents[row['architecture_id']]['PDMS'],6)if row.get('architecture_id')in parents else '待匹配',link(row.get('evidence',''),'完整SUMMARY')]for row in complete])
    else:body+='<p class="notice"><b>当前尚无本轮ABCD完整NAVTEST成绩，DEV另列。</b>父模型分数、CPU结构检查、配置或worker启动均不计为Test提分；未完成人口不发布均值。</p>'
    dev=[row for row in research.get('results',[])if row.get('publishable_complete')and row.get('cohort')=='DEV']
    if dev:
        body+='<h3>已完整DEV：1,696场景，单独人口</h3>'+table(['架构','RL更新','完整DEV PDMS','证据'],[[esc(row.get('architecture_id',row.get('arm'))),number(row.get('updates')),fmt(row['PDMS'],6),link(row.get('evidence',''),'DEV SUMMARY')]for row in sorted(dev,key=lambda r:(r.get('architecture_id',''),r.get('updates',0)))])+'<p class="muted">DEV与NAVTEST的场景、物理log不同；不与首卡的父模型Test分数相减，也不把DEV高分当作Test泛化收益。</p>'
    if failed or failed_nav:
        body+='<details><summary>本轮故障证据与恢复边界</summary><p>'+' · '.join(link(r['failure_evidence'],str(r.get('architecture_id',r.get('id',r.get('arm',''))))+' '+('native恢复'if r in failed else 'NAV数据路径'))for r in failed+failed_nav)+'</p><p>这些是执行与数据路径故障；没有产生可用于算法优劣判断的5k续训结果或NAVTEST分数。原1k模型、DEV与失败attempt全部保留。</p></details>'
    observations=research.get('candidates',{}).get('candidates',[])
    body+='<details><summary>全部完整父模型候选：按架构与IL更新展开</summary>'+table(['架构','IL更新','完整PDMS','是否当前best','证据'],[[esc(row.get('architecture')),number(row.get('updates')),fmt(row.get('PDMS'),6),'是'if row.get('summary')==parents.get(row.get('architecture'),{}).get('summary')else '—',link(row.get('summary',''),'SUMMARY')]for row in sorted(observations,key=lambda row:(row['architecture'],row['updates']))])+'</details>'
    guide=Path(research['source_worktree'])/'reports/qwen_world_four_frameworks_il_rl_20261029'
    body+='<p class="muted">原始结构指南：'+link(guide/'README.md','总设计')+' · '+link(guide/'AC_IMPLEMENTATION.md','A／C')+' · '+link(guide/'B_IMPLEMENTATION.md','B')+' · '+link(guide/'D_IMPLEMENTATION.md','D与来源')+'。旧指南预算不是本轮新run的实际执行收据。</p>'
    return body
