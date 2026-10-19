import json
import re
from pathlib import Path

def render(root):
    a=json.loads((root/'ANALYSIS.json').read_text());c=json.loads((root/'CONCLUSION.json').read_text());prep=a['preparation'];runtime=a['runtime'];r=''
    labels={'A':'原始 SFT 输入','C':'真实失败轨迹＋修正请求','D':'同导航、近尺度错配轨迹＋修正请求','F':'语义显著冲突的完整轨迹＋修正请求'}
    for arm in 'ACDF':
        x=a['aggregate'][arm];r+=f'<tr><td>{arm} · {labels[arm]}</td><td>{x["pdms_mean"]:.6f}</td><td>{x["pdms_max"]:.6f}</td><td>{x["rule_pass_fraction"]:.2%}</td><td>{x["pairwise_ADE_m"]:.4f}</td></tr>'
    contrasts=''
    for n in ['F-C','F-D','F-A']:
        x=a['contrasts'][n]['pdms_mean'];lo,hi=x['bootstrap95'];ss=a['contrasts'][n]['pdms_by_seed'];contrasts+=f'<tr><td>{n}</td><td>{x["mean"]:+.6f}</td><td>[{lo:+.6f}, {hi:+.6f}]</td><td>{", ".join(f"{v:+.6f}" for v in ss)}</td><td>{x["positive"]} / {x["negative"]} / {x["ties"]}</td></tr>'
    f=a['feature_differences']['F-C'];g=a['paired_geometry']['F-C'];follow=a['donor_following'];metric_rows=''
    for m in a['aggregate']['F']['mean_metrics']:
        metric_rows+=f'<tr><td>{m}</td>'+''.join(f'<td>{a["aggregate"][arm]["mean_metrics"][m]:.6f}</td>' for arm in 'ACDF')+'</tr>'
    figure=(root/'effects.svg').read_text();figure=figure[figure.index('<svg'):]
    for ident in set(re.findall(r'id="([^"]+)"',figure)):
        figure=figure.replace('id="'+ident+'"','id="semantic_'+ident+'"').replace('#'+ident+'"','#semantic_'+ident+'"').replace('#'+ident+')','#semantic_'+ident+')')
    return f'''<section id="semantic-conflict"><h2>轨迹粗运动语义是否驱动修复：F组配对筛查</h2>
<p class="note"><strong>{c['headline']}</strong>{c['interpretation']}</p>
<p>同一原始冻结 action-query＋DiT Base、32 个内部场景、每场景三个固定种子批次、每批8候选，NAVSIM v1.1 PDMS（0–1）。F只替换C的轨迹数值，不添加guidance；A/C/D复用已验收配对缓存。没有本轮参数更新，固定141条测试继续留出。</p>
<table class="probe"><tr><th>输入条件</th><th>平均 PDMS</th><th>平均 Oracle@8</th><th>通过率</th><th>组内轨迹 ADE / 米</th></tr>{r}</table>
<p>平均分使用全部候选；Oracle@8为每批8条最大值再平均；通过率要求PDMS≥0.8且NC/DAC/TTC均为1。三批不是24候选一起取最大。</p>
{figure}
<table><tr><th>配对比较</th><th>平均 PDMS 差</th><th>场景bootstrap 95%区间</th><th>三个批次分别差值</th><th>改善 / 下降 / 相同场景</th></tr>{contrasts}</table>
<p>以32个不同原始驾驶记录为独立单位，先在场景内平均3×8候选，再做5000次bootstrap。F−C为预设主要比较；其余为辅助比较，未作多重检验校正。这是固定小集合的机制筛查。</p>
<h3>干预是否足够强</h3>
<p>F供体来自同一769条训练侧其他驾驶记录的Base预测池，完整保留8路点，绝非GT或随机拼接。{prep['heading_conflicts']}/32场景末航向差≥30°；{prep['scale_conflicts']}/32场景路程差≥8米且加1米后的路程比≥4；两类可重叠，{prep['navigation_different']}/32供体导航标签不同。输入轨迹平均ADE为{prep['input_ADE_mean']:.3f}米，范围{prep['input_ADE_range'][0]:.3f}–{prep['input_ADE_range'][1]:.3f}米。</p>
<p>输出F与C的同噪声配对轨迹平均只相差<strong>{g['path_shift_m']['mean']:.4f}米</strong>，末航向平均绝对差{g['endpoint_yaw_shift_deg']['mean']:.3f}°。从action query到投影条件、初始DiT速度的平均相对L2差依次为{f['query']['relative_l2']:.4%}、{f['projected_context']['relative_l2']:.4%}、{f['initial_velocity']['relative_l2']:.4%}，这些是敏感性描述，不构成独立因果中介识别。</p>
<p>F−C的平均输出位移与“错误供体−真实失败轨迹”方向的余弦为{follow['cosine']['mean']:.4f}（95%区间{follow['cosine']['bootstrap95'][0]:.4f}至{follow['cosine']['bootstrap95'][1]:.4f}）；投影系数为{follow['projection']['mean']:.6f}（{follow['projection']['bootstrap95'][0]:.6f}至{follow['projection']['bootstrap95'][1]:.6f}）。该量用于检查输出是否跟随供体，不把微小位移的方向统计当作理解证明。</p>
<details><summary>匹配、合理性、验收与来源</summary>
<p>预先按末航向/路程差定义冲突，输入ADE≥3米。供体分段速度≤30m/s、离散速度向量加速度≤8m/s²、相邻环绕航向变化≤0.4rad；这些仅筛除明显异常，不保证供体在接收场景安全。以一对一匹配选32条不重复供体，最大化有上限的语义差，次要约束路点token边界；不按新输出、接收场景GT或评分挑选供体。F保留当前场景原导航指令，没有把供体导航文字输入模型。</p>
<p>32/32的C/F完整token总数、history/action位置、图像及历史状态哈希一致；各路点内部token边界并非全部相同，因此仍有局部位置差异。32场景重新计算C的action query及投影条件，与旧缓存逐元素一致；首场景全部三个种子共24条C轨迹和评分精确重放。F的96组768候选全部完成，11个高斯噪声张量逐组哈希与旧C一致，所有分项、统计与表征差异从原始记录复核，零遗漏/重复/非有限轨迹。</p>
<p>单PPU-ZW810E耗时{runtime['elapsed_s']:.1f}秒（run.py主流程含加载；不含之前供体准备和Python导入），峰值分配显存{runtime['peak_gpu_allocated_gib']:.2f}GiB，训练参数为0；额外评分24条用于重放验收。参数见证保持不变，任务退出后设备释放另有凭据。</p>
<p>实验入口：<code>20261018-privileged-feedback-expert/semantic_conflict_v1/</code>。计划/匹配：<code>PLAN.json</code>、<code>prepare.py</code>、<code>DONORS.json</code>、<code>PROMPTS.json</code>、<code>PREPARATION.json</code>；运行身份和每轨迹评分：<code>runs/attempt01/CONFIG.json</code>、<code>rollouts.jsonl</code>、<code>replay.jsonl</code>、<code>features/</code>；分析与复核：<code>analyze.py</code>、<code>ANALYSIS.json</code>、<code>SCENE_RESULTS.json</code>、<code>VALIDATION.json</code>。沿用原Base SHA256 075ed30723bdac61cd05e3eacec94267631e4a8ff342f189fc6e82649bdb38b2。</p>
<table><tr><th>分项</th><th>A</th><th>C</th><th>D</th><th>F</th></tr>{metric_rows}</table></details>
<details open><summary>结论边界</summary><p>{c['limitation']}</p><p>这一轮选用语义冲突的合理轨迹，未额外运行乱码组。F仍带有轨迹字段、数字格式及粗结构，不能排除模型识别“这是一条轨迹”等语义。模型也可能理解冲突而主动忽略，或优先依赖图像和导航。32场景已反复用于探索，每场景只使用一个固定供体，三个种子批次只覆盖采样随机性；不能推广到其他数据或GRPO训练后的专家。</p></details></section>'''
