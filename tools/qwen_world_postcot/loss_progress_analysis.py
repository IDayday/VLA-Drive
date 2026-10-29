"""CPU-only complete Test comparisons; never rank different-update snapshots."""
import argparse
from pathlib import Path
import numpy as np
from tools.qwen_world_postcot.summarize import load, compare, summarize
from tools.qwen_world_postcot.user_interim_owner import read, write, fingerprint

BASE=Path('/mnt/project/qwen-world-postcot-20261025')
ARMS=('PHYSICAL','TIME_RESIDUAL','DELTA','HUBER')


def paired_result(evaluation, baseline):
    identity,rows=load(evaluation);base_id,old=load(baseline)
    pair=compare(identity,rows,base_id,old)
    delta=np.array([rows[t]['feedback']['metrics']['score']-old[t]['feedback']['metrics']['score'] for t in rows])
    return dict(delta_PDMS_points=100*pair['mean_delta'],CI95_points=[100*x for x in pair['CI95']],
        G_plus_points=100*float(np.maximum(delta,0).mean()),G_minus_points=100*float(np.minimum(delta,0).mean()),
        paired=pair,baseline=str(baseline),evaluation=str(evaluation))


def five_k(control,output):
    control=Path(control);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ready=(control.parent/'COMPLETE.json').exists()
    results={};lines=['# 四种损失：相同5k更新的完整Navtest','',
        'canonical v1.1；12146场景；单候选clean-x K10 shift5 ODE；seed2101。Test历史已观察。'
        '四种损失均从通用初始化开始，非63k切换损失的旧预实验。固定10k筛选终点不变。','',
        '| 损失 | Navtest PDMS | 相对原L1同预算5k |','|---|---:|---:|']
    if ready:
        c=summarize(control);results['CONTROL']=c;lines.append(f'| 原L1 | {c["PDMS"]:.6f} | 0 |')
    else:lines.append('| 原L1 | 评测中 | 尚不能计算 |')
    for arm in ARMS:
        root=BASE/'fresh_loss_IL_20261029'/arm/f'NAVTEST/update005000/{arm}_IL_005000'
        if not(root/'COMPLETE.json').exists():
            results[arm]=dict(status='PENDING_FULL_TEST');lines.append(f'| {arm} | 尚未完成 | — |');continue
        done=read(root/'COMPLETE.json')
        if done['scenes']!=12146 or not done['all_owned_workers_exited'] or read(root/'native_export_parity.json')['status']!='PASS':
            raise ValueError('Not a complete qualified single-candidate result')
        row=summarize(root/'evaluation')
        if ready:row['matched_control']=paired_result(root/'evaluation',control)
        row.pop('sha256');row['sha256']=fingerprint(row);results[arm]=row
        delta=f'{row["matched_control"]["delta_PDMS_points"]:+.6f}'if ready else '等待L1控制'
        lines.append(f'| {arm} | {row["PDMS"]:.6f} | {delta} |')
    if results.get('PHYSICAL',{}).get('status')=='COMPLETE':
        for arm in ('DELTA','HUBER'):
            if results.get(arm,{}).get('status')=='COMPLETE':
                results[arm]['versus_PHYSICAL']=paired_result(results[arm]['evaluation'],results['PHYSICAL']['evaluation'])
                results[arm].pop('sha256');results[arm]['sha256']=fingerprint(results[arm])
    lines+=['','训练loss定义不同，不能用加权L1、Huber或增量loss数值直接排名。'
        '5k只是早期学习观察；不提前终止任意一组，仍按预登记10k全Test筛选。'
        '不同训练步数的最新检查点用于各组学习曲线，不代替同预算对照。',
        '配对区间按物理log聚类，保留场景权重；不是训练/推理多seed不确定性。','']
    row=dict(schema='world_fresh_loss_matched5k_Test_v1',results=results,control_ready=ready,
        all_four_complete=all(results.get(a,{}).get('status')=='COMPLETE'for a in ARMS),
        fixed_selection_endpoint=10000,training_seed=42,inference_seed=2101,Test_observed=True)
    row['sha256']=fingerprint(row);write(output/'matched5k.json',row)
    (output/'README.md').write_text('\n'.join(lines));return row


def finish(root,requests,report):
    root,report=Path(root),Path(report);report.mkdir(parents=True,exist_ok=True);results={}
    lines=['# Qwen-World：四种新损失与最新IL检查点','',
        '完整12146场景、canonical v1.1、seed2101、单候选clean-x十步shift5 ODE。'
        '最新检查点的更新数不同，分数不能直接作为同预算方法排名。','',
        '| 检查点 | 更新数 | PDMS | 相对本组上个已测点 |','|---|---:|---:|---:|']
    for label,request in requests.items():
        done=read(request.root/'COMPLETE.json')
        if done['status']!='COMPLETE'or not done['all_owned_workers_exited']:raise ValueError('Incomplete Test')
        result=summarize(request.output)
        previous=request.pin.get('previous_evaluation')
        if previous:result['previous_same_run']=paired_result(request.output,previous)
        result.pop('sha256');result['sha256']=fingerprint(result);results[label]=result;write(report/(label+'.json'),result)
        delta=f'{result["previous_same_run"]["delta_PDMS_points"]:+.6f}'if previous else '—'
        lines.append(f'| {label} | {request.pin["updates"]} | {result["PDMS"]:.6f} | {delta} |')
    matched=five_k(requests['CONTROL_IL_005000'].output,report/'matched5k')
    lines+=['','[同预算5k损失对照](matched5k/README.md)单独报告。时间权重5k若尚未完成明确保留待补，'
        '不以4k替代。各组共同100k scheduler与10k筛选规则不改变。'
        '原RL均已完成50k，无新RL检查点，本次不重复历史RL。Test已用于观察及损失筛选，不称盲测。','']
    (report/'README.md').write_text('\n'.join(lines));write(report/'registration.json',read(root/'PINNED.json'))
    return results


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();five_k(a.control,a.output)


if __name__=='__main__':main()
