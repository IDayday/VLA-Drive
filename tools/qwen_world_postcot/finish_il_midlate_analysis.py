"""Publish only frozen trajectory TRAIN-L1 / full-Test paired observations."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.qwen_world_postcot.analyze_il_loss_pdms import correlation


def scope_statistics(rows, minimum):
    chosen=[r for r in rows if r['update']>=minimum]
    loss=np.array([r['x'] for r in chosen]);score=np.array([r['PDMS'] for r in chosen])
    dl,ds=np.diff(loss),np.diff(score)
    return dict(minimum_update=minimum,n=len(chosen),updates=[r['update'] for r in chosen],
        levels=correlation(loss,score),adjacent_changes=correlation(dl,ds),
        L1_down_PDMS_up=int(((dl<0)&(ds>0)).sum()),L1_down_PDMS_down=int(((dl<0)&(ds<0)).sum()),
        L1_up_PDMS_up=int(((dl>0)&(ds>0)).sum()),L1_up_PDMS_down=int(((dl>0)&(ds<0)).sum()),
        adjacent_transitions=len(dl),net_PDMS=float(score[-1]-score[0]))


def plot(result,curve,output,new_updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['points_by_window']['500'];steps=np.array([r['update']/1000 for r in rows])
    losses=np.array([r['x'] for r in rows]);scores=np.array([r['PDMS'] for r in rows])
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    bins=curve['bins1000']
    axs[0,0].plot([r['update']/1000 for r in bins],[r['x_L1'] for r in bins],color='#d95f02')
    axs[0,0].set(xlabel='IL optimizer updates (k)',ylabel='Normalized clean-x L1',title='Complete joint TRAIN: non-overlapping 1k means')
    axs[0,1].plot(steps,scores,'o-',color='#1b7837')
    axs[0,1].set(xlabel='IL optimizer updates (k)',ylabel='Full Navtest PDMS (0-100)',title='Fixed checkpoint observations; no score interpolation')
    primary=steps>=40
    axs[1,0].scatter(losses[primary],scores[primary],c=steps[primary],cmap='viridis',s=45)
    metrics=scope_statistics(rows,40000)
    axs[1,0].set(xlabel='Trailing500 normalized TRAIN trajectory L1',ylabel='Navtest PDMS',
        title=f"40k+: n={metrics['n']}, r={metrics['levels']['pearson']:.3f}, rho={metrics['levels']['spearman']:.3f}")
    dl,ds=np.diff(losses[primary]),np.diff(scores[primary])
    axs[1,1].scatter(dl,ds,c=steps[primary][1:],cmap='viridis',s=45)
    axs[1,1].axhline(0,color='gray',lw=.8);axs[1,1].axvline(0,color='gray',lw=.8)
    axs[1,1].set(xlabel='Adjacent change in TRAIN trajectory L1',ylabel='Adjacent change in PDMS (points)',
        title='40k+ adjacent changes (descriptive)')
    for i,row in enumerate(rows):
        offset=(5,7 if i%2 else -12);label=f"{row['update']//1000}k"
        axs[0,1].annotate(label,(steps[i],scores[i]),xytext=offset,textcoords='offset points',fontsize=8)
        if primary[i]:axs[1,0].annotate(label,(losses[i],scores[i]),xytext=offset,textcoords='offset points',fontsize=8)
        if row['update'] in new_updates:
            axs[0,1].scatter([steps[i]],[scores[i]],marker='D',s=60,facecolors='none',edgecolors='#b2182b',zorder=5)
        primary_index=i-int((~primary).sum())
        if primary_index>0:axs[1,1].annotate(label,(dl[primary_index-1],ds[primary_index-1]),xytext=offset,textcoords='offset points',fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.2);ax.margins(.13)
    fig.suptitle('Qwen-World IL: trajectory loss and real single-candidate planning scores')
    fig.supxlabel('One training seed42 / inference seed2101. Official v1.1, shift5 ODE K10. Diamond = newly evaluated.',fontsize=9)
    fig.savefig(output/'trajectory_L1_PDMS.png',dpi=180);fig.savefig(output/'trajectory_L1_PDMS.svg');plt.close(fig)
    svg=output/'trajectory_L1_PDMS.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def finish(root,report):
    registration=json.loads((root/'PINNED.json').read_text())['trajectory_loss_analysis']
    output=Path(registration['final_analysis_output'])
    command=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python','-m',
        'tools.qwen_world_postcot.analyze_il_loss_pdms','--run',registration['frozen_run'],
        '--scores',registration['score_sources'],'--output',str(output)]
    with (root/'loss_analysis.log').open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    result=json.loads((output/'analysis.json').read_text())
    curve=json.loads((Path(registration['frozen_run'])/'TRAJECTORY_LOSS_CURVE.json').read_text())
    scopes={width:{str(minimum):scope_statistics(rows,minimum) for minimum in [30000,40000,50000]}
            for width,rows in result['points_by_window'].items()}
    summary=dict(primary_minimum_update=40000,primary_window=500,statistics=scopes,
        warning='Dependent checkpoints and common time trend; descriptive correlations, not independent replicates, causal estimates, or a loss-based model selector.')
    (report/'trajectory_correlations.json').write_text(json.dumps(summary,indent=2)+'\n')
    (report/'loss_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (report/'training_trajectory_curve.json').write_text(json.dumps(curve,indent=2)+'\n')
    (report/'checkpoint_losses.csv').write_text((output/'checkpoint_losses.csv').read_text())
    plot(result,curve,report,registration['new_updates'])
    rows=result['points_by_window']['500'];primary=scopes['500']['40000']
    lines=['# Qwen-World IL中后期轨迹L1与完整Navtest','',
        '所有新请求均已全量完成：canonical NAVSIMv1.1，12146场景，单候选clean-x十步shift5 ODE，seed2101。'
        '训练seed42不变。新增55k/62k/63k，复用其余九个已封存结果；Test已被多次观察。', '',
        '## 训练L1是否持续降低','',
        '训练有效日志预先固定到63k：8rank逐更新L1副本完全一致、cursor=32×update、optimizer计数连续且skipped0。'
        '这不是把8rank当8次独立训练平均。统计为原始归一化8×3 clean-x L1，乘总loss系数10之前；不是速度MSE，也不是米制最终轨迹误差。','',
        f"30k到63k的非重叠1000步窗口：{curve['midlate_30k_1k_trend']['decreases']}次下降，"
        f"{curve['midlate_30k_1k_trend']['increases']}次上升。完整joint阶段仍有"
        f"{curve['full_joint_1k_trend']['increases']}个上升窗口，不能说每一步都下降。", '',
        '| IL更新 | 末500步轨迹L1 | 完整Navtest PDMS | 本轮新增 |','|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['update']} | {row['x']:.9f} | {row['PDMS']:.6f} | {'是' if row['update'] in registration['new_updates'] else '复用'} |")
    lines+=['','## 相关性：分开看水平值与相邻变化','',
        '预登记主区间40k–63k，补充30k–63k及50k–63k。Pearson/Spearman为负表示低loss伴随高PDMS；'
        '相邻变化另算，用来减少仅因训练时间共同变化产生的表面关联。样本点少且依赖，不使用独立样本p值宣称因果。','',
        '| 起始更新 | n | 水平Pearson | 水平Spearman | 相邻变化Pearson | L1下降且分数升/降 |',
        '|---|---:|---:|---:|---:|---:|']
    fmt=lambda value:'未定义' if value is None else f'{value:.6f}'
    for minimum in [30000,40000,50000]:
        item=scopes['500'][str(minimum)]
        lines.append(f"| {minimum} | {item['n']} | {fmt(item['levels']['pearson'])} | {fmt(item['levels']['spearman'])} | "
                     f"{fmt(item['adjacent_changes']['pearson'])} | {item['L1_down_PDMS_up']} / {item['L1_down_PDMS_down']} |")
    lines+=['',f"主区间{primary['adjacent_transitions']}个相邻变化中，L1下降且PDMS上升{primary['L1_down_PDMS_up']}次，"
        f"L1下降但PDMS也下降{primary['L1_down_PDMS_down']}次。因此不能用训练L1替代Test评价或保证继续续训必然提分。", '',
        '100/500/1000更新窗口的全部敏感性分析见trajectory_correlations.json。它们只改变日志汇总窗口，不改变模型或评测。'
        '在线TRAIN重建误差来自变化中的权重、样本、噪声；不称固定检查点Test监督loss，也不能单凭它证明过拟合或归因某个辅助任务。', '',
        '![训练L1与全量Navtest](trajectory_L1_PDMS.png)','',
        '## 身份、预算与缺失检查点','',
        '新评测总计36438条单候选场景、额外3条native恢复/导出一致性；原有九个评分直接复用，不重新计为新增评分。'
        '各新模型JSON保留真实分项、相对IL40及预登记前点的配对log bootstrap、安全/零分转移和实际worker成本。'
        '固定训练仍按100k预算继续，本轮不改变优化器/数据/损失，不自动更换RL父模型或追加新seed。', '',
        '原生54k、56–59k及61k当前不在保留检查点或已核实的导出清单中；不从相邻权重插值、不重训伪造。'
        '全部可用且未评测的中后期原生55k/62k/63k已在新分数产生前锁定。私有原始日志、轨迹、权重均不上传。', '',
        '入口：tools.qwen_world_postcot.freeze_il_loss_journals、analyze_il_loss_pdms及latest_snapshot_owner；'
        'request.json登记真实路径与恢复身份，loss_analysis.log和各GPU invocation完整记录保留在资产根。', '']
    (report/'README.md').write_text('\n'.join(lines))
                           