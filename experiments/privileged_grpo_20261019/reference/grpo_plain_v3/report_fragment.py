"""Unified report of the three input conditions under matched short GRPO budgets."""
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def render(root):
    root=Path(root)
    assert read(root/'VALIDATION.json')['status']=='ACCEPTED'
    current=read(root/'ANALYSIS.json')
    previous=read(root.parent/'grpo_probe_v2/ANALYSIS.json')
    previous_result=read(root.parent/'grpo_probe_v2/runs/attempt01/RESULT.json')
    names={'plain':'原始 SFT 输入','without':'原始输入＋失败轨迹＋修正请求','with':'原始输入＋失败轨迹＋修正请求＋guidance'}
    records={}
    for analysis in [previous,current]:
        for row in analysis['aggregate']:
            records[row['role'],row['stage'],row['arm']]=row
    performance='';diversity='';safety='';training=''
    gains={}
    for arm in ['plain','without','with']:
        analysis=current if arm=='plain' else previous
        before=records['dev','initial',arm];after=records['dev','final',arm]
        gain=analysis['paired']['dev']['pdms_mean'][arm];gains[arm]=gain
        low,high=gain['bootstrap95']
        performance+=f'<tr><td>{names[arm]}</td><td>{before["pdms_mean"]:.4f}</td><td>{after["pdms_mean"]:.4f}</td><td>{gain["mean"]:+.4f}</td><td>[{low:+.4f}, {high:+.4f}]</td><td>{before["pdms_first"]:.4f} → {after["pdms_first"]:.4f}</td><td>{before["pdms_max"]:.4f} → {after["pdms_max"]:.4f}</td></tr>'
        for stage,stage_name in [('initial','更新前'),('final','更新后')]:
            row=records['dev',stage,arm]
            diversity+=f'<tr><td>{names[arm]}，{stage_name}</td><td>{row["pairwise_ADE_m"]:.3f}</td><td>{row["pairwise_FDE_m"]:.3f}</td><td>{row["pairwise_final_yaw_deg"]:.2f}</td><td>{row["pdms_std"]:.4f}</td><td>{row["zero_variance_groups"]}/32</td><td>{row["all_zero_groups"]}/32</td></tr>'
            safety+='<tr><td>'+names[arm]+'，'+stage_name+'</td>'+''.join(f'<td>{row["mean_metrics"][k]:.4f}</td>' for k in ['NC','DAC','TTC','EP','CF','DDC'])+f'<td>{row["rule_pass_fraction"]:.1%}</td></tr>'
        row=records['train','train',arm];monitor=analysis['paired']['train_monitor']['pdms_mean'][arm]
        training+=f'<tr><td>{names[arm]}</td><td>{row["zero_variance_groups"]}/64</td><td>{row["all_zero_groups"]}/64</td><td>{row["all_one_groups"]}/64</td><td>{row["pairwise_ADE_m"]:.3f}</td><td>{monitor["mean"]:+.4f}</td></tr>'
    cross=''
    for arm in ['without','with']:
        comparison=current['comparison']['dev']['pdms_mean'][arm]
        gain=comparison['plain_minus_other_training_gain'];final=comparison['plain_minus_other_final']
        low,high=gain['bootstrap95']
        cross+=f'<tr><td>原始输入 − {names[arm]}</td><td>{final["mean"]:+.4f}</td><td>{gain["mean"]:+.4f}</td><td>[{low:+.4f}, {high:+.4f}]</td></tr>'
    conclusion=read(root/'CONCLUSION.json')['html']
    replay=current['base_replay']
    return f'''<h2>三种输入条件下的短程 GRPO 对照</h2>
<p class="note">{conclusion}</p>
<p>三组均从指定 <code>0804_16-action-only</code> Base 权重重新开始，使用<strong>完全相同的64个失败场景与顺序</strong>单遍训练。每场景8条轨迹，4场景梯度积累后统一裁剪和更新，固定16次更新、512条训练候选。VLM视觉、语言及共享action query embedding共同训练，学习率2e-7；动作和历史状态模块1e-6。奖励为官方NAVSIM v1.1 PDMS，EP/TTC/CF权重5/5/2；组内标准化优势，KL系数0.01，无SFT/BC损失或奖励整形。三组仅改变输入条件；本实验不能分别归因G8、梯度积累与分学习率的收益。</p>
<p>原始输入组仅提供三视图图像、历史状态、导航指令及固定特殊token；不输入失败轨迹、修正请求、guidance、GT轨迹或GT对齐差异。失败轨迹组增加初始轨迹和修正请求；完整guidance组再增加评分、诊断和GT对齐特权信息。评分器所需reference只用于计算奖励，不作为原始输入组的模型条件。</p>
<p>内部评估为相同32个场景，与64个更新场景不共享原始驾驶记录；另监测训练内预定16个场景。固定141条测试集未使用。所有评估采用配对种子、8候选、10步SDE（eta=0.1），固定选择末次权重，未按评估挑选checkpoint。原始输入组重跑48场景初始基线：{replay['candidate_count']}条候选与前次输出核对通过，轨迹最大绝对差 {replay['trajectory_max_abs_error']:.2g}，评分最大绝对差 {replay['metric_max_abs_error']:.2g}。同一模型因输入不同，训练前分数可以不同。</p>
<table class="probe"><tr><th>模型输入条件</th><th>训练前平均PDMS</th><th>训练后平均PDMS</th><th>配对增益</th><th>增益95%区间</th><th>固定首候选PDMS</th><th>Oracle@8</th></tr>{performance}</table>
<p>平均PDMS先平均每场景8候选，再平均32场景。Oracle@8使用真实评分选择每场景最高分，仅表示候选上限；固定首候选不使用评分选择。区间按32个原始驾驶记录进行5000次配对bootstrap，不覆盖训练种子不确定性。训练侧留出不代表Base未见数据。</p>
<details><summary>原始输入组与另两组的配对差异及解释边界</summary>
<table class="probe"><tr><th>比较方向</th><th>训练后平均分之差</th><th>训练增益之差</th><th>增益差95%区间</th></tr>{cross}</table>
<p>不同组在训练和评估时都采用各自输入条件，因此训练后分数比较不等于隔离训练条件的因果效应。原始输入组是查看前两组结果后新增的探索对照，不是独立确认。两组旧实验的增益差（guidance有−无）为 {previous['paired']['dev']['pdms_mean']['guidance_extra_gain']['mean']:+.4f}，仍未显示额外指导训练收益。只有一遍、一个种子和少量场景，不能据此宣称稳定强专家、正常场景能力保留或指导信息普遍无效。</p></details>
<details><summary>安全、进度、舒适性与候选通过率</summary>
<table class="probe"><tr><th>条件与阶段</th><th>NC</th><th>DAC</th><th>TTC</th><th>EP</th><th>CF</th><th>DDC</th><th>通过率</th></tr>{safety}</table>
<p>通过率采用原失败筛选的反面：PDMS≥0.8且NC/DAC/TTC均为1。分项为全部候选的均值，不等于所有事件的发生率；DDC虽不进入当前PDMS加权项，仍独立报告。</p></details>
<details open><summary>8条轨迹的多样性记录</summary>
<table class="probe"><tr><th>条件与阶段</th><th>两两路点距离/m</th><th>两两终点距离/m</th><th>终点航向差/°</th><th>组内PDMS标准差</th><th>零方差组</th><th>全零组</th></tr>{diversity}</table>
<p>每组8条轨迹共28对，距离按匹配时间点xy欧氏距离计算，再平均8个路点与28对；终点仅取末路点，航向差环绕到[-π,π]。表内再平均32个场景，不是相对GT的ADE/FDE。几何多样性不等于安全或修复质量。</p></details>
<details><summary>训练内监测、工程验收与成本</summary>
<table class="probe"><tr><th>条件</th><th>训练零方差组</th><th>全零组</th><th>全满分组</th><th>两两路点距离/m</th><th>16训练内场景PDMS变化</th></tr>{training}</table>
<p>零方差组的策略优势设为0，仍保留KL并计入4场景等权平均，不补采样、不删除难例。训练内16场景为预先选定且富集稀有类别的监测集，其变化不代表全部64训练场景。概率回放、有限梯度、视觉/语言/action query/DiT实际更新、4场景间参数不变、参考参数见证不变及完整轨迹评分均通过验收。每批rollout只更新一次，不能据此声称已验证多轮复用的PPO裁剪行为。</p>
<p>前两组共{previous['counts']['candidate_scores']:,}候选，耗时{previous['wall_seconds']/60:.2f}分钟，峰值分配显存{previous['peak_gpu_allocated_gib']:.2f}GiB；原始输入组新增{current['counts']['candidate_scores']:,}候选，耗时{current['wall_seconds']/60:.2f}分钟，峰值{current['peak_gpu_allocated_gib']:.2f}GiB。均使用单张PPU-ZW810E。新增组费用包含重跑初始基线，不将复用的比较结果计成新增计算；未启动长周期训练。</p></details>
<details><summary>复核入口与权重加载</summary>
<p>失败轨迹和guidance两组：<code>grpo_probe_v2/</code>；原始输入组：<code>grpo_plain_v3/</code>。各自计划<code>PLAN.json</code>、相同数据清单<code>DATA.json</code>、分析<code>ANALYSIS.json</code>、验收<code>VALIDATION.json</code>、逐组多样性<code>GROUP_DIVERSITY.csv</code>。运行目录均为<code>runs/attempt01/</code>，内含<code>rollouts.jsonl</code>、<code>groups.jsonl</code>、<code>updates.jsonl</code>；原始输入组另有<code>inputs.jsonl</code>和<code>BASE_REPLAY.json</code>。SHA256保存在配置、结果与验收记录中。</p>
<p>三份最终权重分别在对应运行目录的<code>checkpoints/plain/</code>、<code>checkpoints/without/</code>、<code>checkpoints/with/</code>。权重为FP32主参数，构造模型后须先转FP32再严格加载，避免BF16构造参数舍入小更新；VLM前向仍用BF16 autocast。只保存权重，未保存优化器状态，不能作为原AdamW状态的精确恢复。具体见各目录<code>CHECKPOINT_LOADING.json</code>。</p></details>'''
