"""Render this probe into the task's single coherent HTML report."""
import json
from pathlib import Path


def render(root):
    root=Path(root)
    if not (root/'VALIDATION.json').exists():return ''
    a=json.loads((root/'ANALYSIS.json').read_text())
    validation=json.loads((root/'VALIDATION.json').read_text())
    assert validation['status']=='ACCEPTED'
    label={('initial','plain'):'原始提示 Base（8候选SDE）',('initial','without'):'失败轨迹，无 guidance，更新前',('final','without'):'失败轨迹，无 guidance，更新后',('initial','with'):'失败轨迹＋guidance，更新前',('final','with'):'失败轨迹＋guidance，更新后'}
    order=[('initial','plain'),('initial','without'),('final','without'),('initial','with'),('final','with')]
    records={(r['role'],r['stage'],r['arm']):r for r in a['aggregate']}
    performance='';diversity='';safety=''
    for stage,arm in order:
        r=records['dev',stage,arm]
        performance+=f'<tr><td>{label[stage,arm]}</td><td>{r["pdms_first"]:.4f}</td><td>{r["pdms_mean"]:.4f}</td><td>{r["pdms_max"]:.4f}</td><td>{r["rule_pass_fraction"]:.1%}</td></tr>'
        diversity+=f'<tr><td>{label[stage,arm]}</td><td>{r["pairwise_ADE_m"]:.3f}</td><td>{r["pairwise_FDE_m"]:.3f}</td><td>{r["pairwise_final_yaw_deg"]:.2f}</td><td>{r["pdms_std"]:.4f}</td><td>{r["zero_variance_groups"]}/32</td></tr>'
        safety+='<tr><td>'+label[stage,arm]+'</td>'+''.join(f'<td>{r["mean_metrics"][k]:.4f}</td>' for k in ['NC','DAC','TTC','EP','CF','DDC'])+'</tr>'
    delta=a['paired']['dev']['pdms_mean']
    gain_rows=''
    for key,name in [('without','无 guidance 的训练增益'),('with','有 guidance 的训练增益'),('guidance_extra_gain','两组训练增益之差（有−无）')]:
        r=delta[key];low,high=r['bootstrap95']
        gain_rows+=f'<tr><td>{name}</td><td>{r["mean"]:+.4f}</td><td>[{low:+.4f}, {high:+.4f}]</td></tr>'
    train_rows=''
    for arm,name in [('without','无 guidance'),('with','有 guidance')]:
        r=records['train','train',arm]
        monitor=a['paired']['train_monitor']['pdms_mean'][arm]
        train_rows+=f'<tr><td>{name}</td><td>{r["zero_variance_groups"]}/64</td><td>{r["all_zero_groups"]}/64</td><td>{r["all_one_groups"]}/64</td><td>{r["pairwise_ADE_m"]:.3f}</td><td>{monitor["mean"]:+.4f}</td></tr>'
    g0=records['dev','initial','with'];g1=records['dev','final','with']
    extra=delta['guidance_extra_gain'];lo,hi=extra['bootstrap95']
    return f'''<h2>G=8、多场景积累与分学习率的训练侧对照</h2>
<p><strong>两组固定预算短程对照已完成并验收。</strong>各从同一个 Base 出发，使用相同 64 场景单遍训练，每场景 8 条轨迹，每 4 个场景累积后更新一次，共 16 次更新。VLM（含视觉、语言、共享 action query embedding）学习率 2e-7，动作头和历史状态模块 1e-6；保持官方 PDMS 奖励及 KL，未加 SFT、BC 或奖励整形。</p>
<p class="note"><strong>本轮观察：两组平均 PDMS 均提高，尚未看到 guidance 的额外训练收益。</strong>有 guidance 与无 guidance 的训练增益之差为 {extra['mean']:+.4f}，探索性95%区间为 [{lo:+.4f}, {hi:+.4f}]。guidance 组 Oracle@8 从 {g0['pdms_max']:.4f} 降至 {g1['pdms_max']:.4f}，NC 均值从 {g0['mean_metrics']['NC']:.4f} 降至 {g1['mean_metrics']['NC']:.4f}，全零奖励组从 {g0['all_zero_groups']}/32 增至 {g1['all_zero_groups']}/32；平均分改善未同步带来候选上限与碰撞指标改善。这些是本轮观察，不是稳定退化的统计结论。</p>
<p>内部评估使用训练侧另 32 个场景，与更新数据不共享原始驾驶记录；另在更新数据中的 16 个预定场景检查训练内变化。固定 141 条测试样本未参与。两组评估均使用同一组随机种子，末次 checkpoint 固定为第 16 次更新，不按评估分数选择。总计 {a['counts']['candidate_scores']:,} 条候选，耗时 {a['wall_seconds']/60:.2f} 分钟，峰值分配显存 {a['peak_gpu_allocated_gib']:.2f} GiB。</p>
<table class="probe"><tr><th>内部评估条件</th><th>固定首候选 PDMS</th><th>8候选平均 PDMS</th><th>Oracle@8</th><th>候选通过率</th></tr>{performance}</table>
<p>候选通过率使用原失败收集规则的反面：PDMS≥0.8 且 NC/DAC/TTC 均为1。Oracle@8 使用真实评分选取每场景最高分，仅描述候选上限。表内 Base 也使用同样10步SDE与8候选，不与历史单次ODE采样混为同预算比较。</p>
<table class="probe"><tr><th>配对比较：8候选平均PDMS</th><th>平均变化</th><th>按32个驾驶记录 bootstrap 95%区间</th></tr>{gain_rows}</table>
<p>这里区分 guidance 直接输入的效果与训练后的变化。单遍、单种子和小样本结果只用于决定下一步投入；bootstrap 不覆盖训练随机种子不确定性，训练侧留出也不代表 Base 未见数据。尚无稳定强专家或正常场景能力保留结论。</p>
<p>这轮只对照是否提供 guidance，两组均使用 G=8、4场景积累与分模块学习率，不能分别归因这三项设置的收益。16个训练内监测场景的变化与内部留出不一致，亦不支持声称已充分拟合。若继续投入，优先检查低分与全零组的候选覆盖、guidance内容及其输入响应，再决定是否扩大训练预算；本轮没有据此更改奖励或追加训练。</p>
<details><summary>内部评估的安全、进度和舒适性分项均值</summary><table class="probe"><tr><th>条件</th><th>NC</th><th>DAC</th><th>TTC</th><th>EP</th><th>CF</th><th>DDC</th></tr>{safety}</table><p>每场景8条候选等权，32个场景再等权平均；DDC虽不进入当前PDMS加权项，仍独立报告。分项均值不等于每种事件的发生率。</p></details>
<details open><summary>8 条轨迹的简单多样性记录</summary>
<table class="probe"><tr><th>内部评估条件</th><th>两两平均路点距离 / m</th><th>两两终点距离 / m</th><th>终点航向差 / °</th><th>组内PDMS标准差</th><th>奖励零方差组</th></tr>{diversity}</table>
<p>每组8条轨迹对应28对。路点距离为同时间点xy欧氏距离，先平均8个路点再平均28对；终点距离只取末路点。航向差按[-π,π]环绕后取绝对值。表中数值再平均32个场景；不是相对GT的ADE/FDE。几何多样性增大可能伴随更差轨迹，必须结合PDMS与安全分项解释。</p></details>
<details><summary>训练组奖励退化、训练内变化与可复核产出</summary>
<table class="probe"><tr><th>训练条件</th><th>零方差组</th><th>全零组</th><th>全满分组</th><th>两两路点距离 / m</th><th>16场景训练内平均PDMS变化</th></tr>{train_rows}</table>
<p>零方差组策略优势设为0，仍计入固定4场景平均与KL项，没有补采样或删除难例。每场景先完成反向传播并积累梯度，4场景期间参数保持不变；窗口末统一裁剪梯度、执行一次AdamW。有限梯度、视觉语言动作参数真实更新、旧／新概率回放、分组隔离、完整2944条评分和多样性复算均通过验收。</p>
<p>任务目录 <code>grpo_probe_v2/</code>：计划 <code>PLAN.json</code>；固定清单 <code>DATA.json</code>；运行 <code>runs/attempt01/</code>；逐候选轨迹与分项 <code>rollouts.jsonl</code>；逐场景多样性 <code>groups.jsonl</code> 与 <code>GROUP_DIVERSITY.csv</code>；梯度与更新 <code>updates.jsonl</code>；分析与验收 <code>ANALYSIS.json</code>、<code>VALIDATION.json</code>。两组最终权重分别在运行目录的 <code>checkpoints/without/</code> 和 <code>checkpoints/with/</code>，附SHA256；仅权重快照，未保存可续训的优化器状态。加载前须先将模型转为FP32再严格载入，避免BF16构造参数舍入微小更新；约定见 <code>CHECKPOINT_LOADING.json</code>。</p></details>'''
