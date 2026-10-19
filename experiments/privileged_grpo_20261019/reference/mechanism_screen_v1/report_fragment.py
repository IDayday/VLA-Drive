import html
import json
from pathlib import Path


def render(root):
    root=Path(root)
    a=json.loads((root/'ANALYSIS.json').read_text());v=json.loads((root/'VALIDATION.json').read_text())
    assert v['status']=='ACCEPTED'
    conclusion=json.loads((root/'CONCLUSION.json').read_text())['html']
    names={'A':'原始 SFT 输入','B':'原始输入＋修正请求','C':'B＋真实失败轨迹','D':'B＋匹配的跨场景完整轨迹','E':'C＋完整 guidance'}
    performance='';safety='';feature='';comparisons='';seed_table=''
    for arm in 'ABCDE':
        r=a['aggregate'][arm]
        performance+=f'<tr><td>{arm}：{names[arm]}</td><td>{r["pdms_mean"]:.4f}</td><td>{r["pdms_first"]:.4f}</td><td>{r["pdms_max"]:.4f}</td><td>{r["rule_pass_fraction"]:.1%}</td><td>{r["pairwise_ADE_m"]:.3f}</td><td>{r["pdms_std"]:.4f}</td></tr>'
        safety+='<tr><td>'+arm+'</td>'+''.join(f'<td>{r["mean_metrics"][k]:.4f}</td>' for k in ['NC','DAC','TTC','EP','CF','DDC'])+f'<td>{r["all_zero_groups"]}/96</td></tr>'
        seed_table+='<tr><td>'+arm+'</td>'+''.join(f'<td>{r["pdms_mean"]:.4f}</td>' for r in a['per_seed'][arm])+'</tr>'
    for name in ['B-A','C-B','C-D','D-A','E-C']:
        r=a['contrasts'][name]['pdms_mean'];low,high=r['bootstrap95']
        seeds=', '.join(f'{x:+.4f}' for x in a['contrasts'][name]['pdms_by_seed'])
        comparisons+=f'<tr><td>{name}</td><td>{r["mean"]:+.4f}</td><td>[{low:+.4f}, {high:+.4f}]</td><td>{r["positive"]}/{r["negative"]}/{r["ties"]}</td><td>{seeds}</td></tr>'
        f=a['feature_differences'][name];g=a['paired_geometry'][name]
        feature+=f'<tr><td>{name}</td><td>{f["query"]["relative_l2"]:.4f}</td><td>{f["projected_context"]["relative_l2"]:.4f}</td><td>{f["initial_velocity"]["relative_l2"]:.4f}</td><td>{g["path_shift_m"]["mean"]:.3f}</td><td>{g["endpoint_dx_m"]["mean"]:+.3f}</td><td>{g["endpoint_dy_m"]["mean"]:+.3f}</td></tr>'
    d=json.loads((root/'DONORS.json').read_text())['assignments']
    boundary_max=max(max(abs(x-y) for x,y in zip(r['true_waypoint_token_ends'],r['donor_waypoint_token_ends'])) for r in d)
    exact=sum(r['waypoint_boundary_l1']==0 for r in d)
    following=a['donor_following'];fl,fh=following['cosine']['bootstrap95']
    svg=''
    if (root/'paired_effects.svg').exists():
        source=(root/'paired_effects.svg').read_text();svg=source[source.index('<svg'):]
    return f'''<h2>冻结 Base 的 A–E 输入机制筛查</h2>
<p class="note">{conclusion}</p>
<p>实验固定原始0804_16 action-only Base，全部参数冻结，无任何训练更新。使用既有32个内部场景，来自32个原始驾驶记录；每场景、每条件使用3个种子批次，每批8候选，五组共{a['counts']['candidates']:,}条轨迹。每一批A–E共享同一个初始高斯噪声和全部10步SDE转移噪声，噪声bank哈希逐组一致；没有通过采样种子或提示词挑选最优结果。固定141条测试集未使用。</p>
<p>B–E统一请求为 <code>Refine the driving plan. Output the complete corrected future trajectory in the original ego frame.</code>，避免B出现“下面的轨迹”却没有轨迹的悬空指代。C/D在相同位置插入同格式8路点数值，E再追加原始完整指导。这个请求与先前短程对照略有不同，C/E是本轮锁定的新输入，不能把旧分数直接代入。A首批256候选与旧基线逐条核对，轨迹与分项评分完全一致。</p>
<table class="probe"><tr><th>条件</th><th>平均PDMS</th><th>固定首候选PDMS</th><th>平均Oracle@8</th><th>通过率</th><th>两两路点距离/m</th><th>组内PDMS标准差</th></tr>{performance}</table>
<p>先在每场景平均3批次，再对32场景等权平均。Oracle取每批8条中的最高分再平均，绝不是Oracle@24，也不代表可部署的选择器；首候选也对3批次平均。通过率为PDMS≥0.8且NC/DAC/TTC均为1。</p>
{svg}
<table class="probe"><tr><th>预定配对比较</th><th>平均PDMS差</th><th>按场景bootstrap95%区间</th><th>正/负/持平场景数</th><th>三个种子批次的平均差</th></tr>{comparisons}</table>
<p>每个场景先平均24候选再比较；5000次bootstrap以原始驾驶记录为单位，保留场景内全部种子和候选，不把768候选视为独立样本。五项对比均为探索性筛查，区间未做多重比较调整；跨零不证明等价。单场景固定供体、仅3种子且反复查看过该32场景，尚不足以作独立泛化或唯一机制结论。</p>
<details><summary>D组为何是合理但错配的轨迹对照</summary>
<p>从现有训练侧失败轨迹池中排除32评估驾驶记录后，使用769条候选供体，选出32条不重复的完整模型轨迹，来自23个驾驶记录。接收场景与供体具有相同导航指令；按首路点速度和总路程匹配尺度，静止/缓行使用显式低速范围。供体数值原样保留，不打乱路点、不拼接、不添加随机坐标。匹配未使用接收场景GT、目标评分或新模型输出。</p>
<p>32/32场景C/D的完整多模态input token数量与history/action特殊token位置完全相同，文本插入前缀和格式相同；{exact}/32场景每个路点的token结束位置也完全相同，其余最大相差{boundary_max}个token。供体与真实轨迹的平均路点距离为{sum(x['trajectory_ADE_m'] for x in d)/32:.3f}米，范围{min(x['trajectory_ADE_m'] for x in d):.3f}–{max(x['trajectory_ADE_m'] for x in d):.3f}米。D保留了粗导航与运动尺度，因此它破坏的是场景对应关系，并非消除所有规划信息。轨迹本身合理也不意味着适合接收场景；这正是干预，但仍可能存在场景不一致引发的响应。</p>
<p>完整映射与匹配审计见<code>DONORS.json</code>；精确输入见<code>PROMPTS.json</code>；实际多模态token核验见<code>runs/attempt01/inputs.jsonl</code>、<code>token_matching.jsonl</code>。</p></details>
<details open><summary>输入变化如何传到VLM、DiT和输出轨迹</summary>
<table class="probe"><tr><th>比较</th><th>action query相对L2差</th><th>投影后条件相对L2差</th><th>初始DiT速度相对L2差</th><th>配对路点位移/m</th><th>终点纵向变化/m</th><th>终点横向变化/m</th></tr>{feature}</table>
<p>相对L2差的分母为比较右侧张量范数。VLM每个场景/条件只计算一次；DiT速度差在相同初始噪声、相同时间步计算并跨3批次平均。路点位移比较匹配的种子、候选与时间点。不同模块的相对差数值不能直接互比重要性。这些记录验证“输入→action query→DiT条件/速度→轨迹”的传递，不证明模型理解了修正语义；未进行中间表征替换，不能据此识别唯一的内部计算机制。</p>
<p>将D−C的平均输出位移投影到“供体轨迹−真实输入轨迹”的方向：平均余弦{following['cosine']['mean']:.4f}，探索性95%区间[{fl:.4f}, {fh:.4f}]；投影系数均值{following['projection_coefficient']['mean']:.4f}。这描述是否沿供体几何方向变化，不是驾驶质量，也不是单独的内容理解证明。逐场景值见<code>DONOR_FOLLOWING.json</code>。</p></details>
<details><summary>安全分项、种子波动与复核入口</summary>
<table class="probe"><tr><th>条件</th><th>NC</th><th>DAC</th><th>TTC</th><th>EP</th><th>CF</th><th>DDC</th><th>全零奖励组</th></tr>{safety}</table>
<table class="probe"><tr><th>条件</th><th>种子批次0</th><th>种子批次1</th><th>种子批次2</th></tr>{seed_table}</table>
<p>单PPU-ZW810E，耗时{a['wall_seconds']/60:.2f}分钟，峰值分配显存{a['peak_gpu_gib']:.2f}GiB。无新增权重，原始Base见证参数不变。新增代码、锁定计划、输入和证据均在<code>mechanism_screen_v1/</code>：<code>PLAN.json</code>、<code>prepare.py</code>、<code>run.py</code>、<code>analyze.py</code>、<code>ANALYSIS.json</code>、<code>VALIDATION.json</code>、<code>SCENE_RESULTS.json</code>、<code>GROUP_RESULTS.csv</code>。运行目录保存逐候选轨迹/分项、公共噪声哈希及32份表征NPZ，详见<code>runs/attempt01/</code>。B−A和E−C仍同时改变文本长度和语义；E包含GT对齐信息，不能把它的效果解释为纯事件诊断收益。</p></details>'''
