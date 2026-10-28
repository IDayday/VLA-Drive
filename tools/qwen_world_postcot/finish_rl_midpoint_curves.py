"""Combine complete frozen Test observations; no new model/scorer execution."""
import csv
import json
from pathlib import Path
import numpy as np

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import fingerprint,file_sha
from starVLA.rl.qwen_world_postcot.data import POPULATION
from tools.qwen_world_postcot.summarize import load,compare,METRICS


def row_metrics(identity,rows,*,route,update,kind):
    values={name:100*float(np.mean([r['feedback']['metrics'][name]for r in rows.values()]))for name in METRICS}
    return dict(route=route,updates=update,point_kind=kind,PDMS=values.pop('score'),**values,
        evaluation_identity_sha256=identity['sha256'],scenes=len(rows))


def finish(evaluation,report):
    evaluation,report=Path(evaluation),Path(report);registration=json.loads((evaluation/'PINNED.json').read_text());curve=registration['RL_intermediate_curve']
    population=json.loads(POPULATION.read_text());base_id,base=load(curve['baseline_evaluation'],population=population)
    base_score=100*float(np.mean([r['feedback']['metrics']['score']for r in base.values()]))
    points=[];intervals={};source_hashes={};patterns={}
    for route in ('ATTENTION','MEMORY','TIME'):
        previous_id,previous,previous_step=base_id,base,0;route_points=[]
        points.append(row_metrics(base_id,base,route=route,update=0,kind='fixed_IL40k_parent'))
        for step in [10000,20000,40000,50000]:
            path=(Path(curve['final_evaluations'][route])if step==50000 else evaluation/f'{route}_RL_{step:06d}'/'evaluation')
            identity,rows=load(path,population=population)
            paired=compare(identity,rows,previous_id,previous)
            delta=np.array([rows[t]['feedback']['metrics']['score']-previous[t]['feedback']['metrics']['score']for t in base])
            paired.update(G_plus_PDMS=100*float(np.maximum(delta,0).mean()),G_minus_PDMS=100*float(np.minimum(delta,0).mean()),
                from_update=previous_step,to_update=step,route=route,
                interpretation='Paired fixed-seed physical-log bootstrap; no training/inference-seed uncertainty.')
            paired['sha256']=fingerprint(paired);intervals[f'{route}_{previous_step:06d}_to_{step:06d}']=paired
            point=row_metrics(identity,rows,route=route,update=step,kind='reused_final'if step==50000 else'new_common_midpoint')
            point['delta_vs_IL_parent_PDMS']=point['PDMS']-base_score
            points.append(point);route_points.append(point['PDMS']);source_hashes[str(path/'identity.json')]=file_sha(path/'identity.json')
            previous_id,previous,previous_step=identity,rows,step
        patterns[route]=dict(common_updates=[0,10000,20000,40000,50000],PDMS=[base_score]+route_points,
            observed_nondecreasing=all(b>=a for a,b in zip([base_score]+route_points,route_points)),
            caution='Descriptive finite-grid observation, not independent replications or a stability guarantee.')
        del previous_id,previous,identity,rows
        for step,path in curve['historical_observations'][route].items():
            identity,rows=load(path,population=population)
            # Same contract/context checks, without treating historical points
            # as additional newly scored experiments or training replicates.
            compare(identity,rows,base_id,base,bootstrap=1)
            points.append(row_metrics(identity,rows,route=route,update=int(step),kind='reused_historical_off_grid'))
            source_hashes[str(Path(path)/'identity.json')]=file_sha(Path(path)/'identity.json')
            del identity,rows
    output=dict(schema='world_RL_midpoint_fixed_Test_curves_v1',points=points,common_grid_intervals=intervals,
        descriptive_patterns=patterns,source_evaluation_identities=source_hashes,training_seed=42,inference_seed=2101,
        scenes_per_point=12146,canonical='NAVSIM_v1.1',inference='single_candidate_cleanx_K10_shift5',
        logical_new_Test_scene_evaluations=9*12146,extra_native_export_parity_checks=9,
        historical_scoring_is_reused_not_free=True,Test_previously_observed=True,
        limitations=['One training seed and one inference seed.',
            'Common10k/20k/40k registered before these scores; sparse checkpoints do not identify the true maximum.',
            'Optimizer/scene/score budgets match at common steps; actual compute differs across parameter scopes.',
            'Partial means, oracle candidates and reward curves are not deployment results.'])
    output['sha256']=fingerprint(output);atomic_json(report/'RL_learning_curves.json',output)
    fields=['route','updates','point_kind','PDMS']+list(METRICS[1:])+['delta_vs_IL_parent_PDMS','scenes','evaluation_identity_sha256']
    with(report/'RL_learning_curves.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(points)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for route,color in [('ATTENTION','tab:blue'),('MEMORY','tab:orange'),('TIME','tab:green')]:
        selected=sorted([p for p in points if p['route']==route and p['point_kind']!='reused_historical_off_grid'],key=lambda p:p['updates'])
        ax.plot([p['updates']/1000 for p in selected],[p['PDMS']for p in selected],'-o',color=color,label=route)
        other=[p for p in points if p['route']==route and p['point_kind']=='reused_historical_off_grid']
        ax.scatter([p['updates']/1000 for p in other],[p['PDMS']for p in other],marker='x',color=color,alpha=.65)
    ax.set(xlabel='RL optimizer updates (k)',ylabel='Full Navtest PDMS (0–100)',title='Common checkpoints and previously scored observations')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(report/'RL_learning_curves.png',dpi=160);plt.close(fig)
    lines=['# Qwen-World RL：中间权重完整Navtest','',
        '三条路线均从同一IL40k父模型开始。此次预先固定10k/20k/40k，各新增3个检查点，共9次完整12146场景评测。'
        'canonical v1.1、单候选clean-x十步shift5 ODE、seed2101；无新训练、额外seed、best-of-N或部署选轨。'
        '50k终点和已完成的非整数历史补点直接复用。Test历史已反复观察，不称盲测。','',
        '| 路线 | IL父模型 | RL10k | RL20k | RL40k | RL50k | 50k−IL |','|---|---:|---:|---:|---:|---:|---:|']
    for route,p in patterns.items():
        v=p['PDMS'];lines.append(f'| {route} | '+ ' | '.join(f'{x:.6f}'for x in v)+f' | {v[-1]-v[0]:+.6f} |')
    lines+=['','![完整Test学习曲线](RL_learning_curves.png)','',
        '实线为事先锁定的共同步数，叉号为历史已有补点。不能把稀疏曲线的最高点称为训练全过程最高点。', '',
        '| 路线 | 更新区间 | 净变化PDMS点 | 配对95%区间 | G+ | G− | 新增零分 | 零分恢复 |','|---|---|---:|---|---:|---:|---:|---:|']
    for p in intervals.values():
        lo,hi=[100*x for x in p['CI95']]
        lines.append(f'| {p["route"]} | {p["from_update"]//1000}k→{p["to_update"]//1000}k | {100*p["mean_delta"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {p["G_plus_PDMS"]:+.6f} | {p["G_minus_PDMS"]:+.6f} | {p["new_zero"]} | {p["repaired_zero"]} |')
    lines+=['','区间按物理log聚类bootstrap，保留场景权重；不把G8候选当作训练复现，不包含训练/推理seed不确定性。'
        'NC/DAC/TTC新增违规与恢复保存在每个区间JSON；不能只以均分或零分净数量判断安全保持。', '',
        '新评测逻辑预算109314条单候选场景，另有9次native/export一致性检查。实际耗时、重试及共享GPU worker成本保留在各模型evaluation/jobs收据；'
        '共享GPU worker时长不等于独占GPU计算时间。未重跑历史已完成轨迹和评分。', '',
        '新检查点的逐组原生一致性、完整子项与相对IL配对报告见同目录各模型JSON。复现入口为queued_snapshot_owner与PINNED.json；'
        '本工作区模型、采样器和官方评分计算未修改。']
    (report/'README.md').write_text('\n'.join(lines)+'\n')
