"""Derived full-Test and training readouts for the registered coordinate study."""
import json
from pathlib import Path
import pickle

import numpy as np

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import file_sha
from tools.qwen_world_postcot.summarize import summarize,load
from tools.qwen_world_postcot.compare_saved_il_trajectories import load_model,fingerprint


def finish(evaluation,report):
    evaluation,report=Path(evaluation),Path(report)
    root=Path(json.loads((evaluation/'PINNED.json').read_text())['coordinate_loss_study']['root'])
    registration=json.loads((root/'REGISTRATION.json').read_text())
    baseline=evaluation/'CONTROL_065000/evaluation';physical=evaluation/'PHYSICAL_065000/evaluation'
    result=summarize(physical,baseline)
    _,control_rows=load(baseline);_,physical_rows=load(physical)
    delta=np.array([physical_rows[t]['feedback']['metrics']['score']-control_rows[t]['feedback']['metrics']['score'] for t in control_rows])
    result['G_plus_PDMS']=100*float(np.maximum(delta,0).mean())
    result['G_minus_PDMS']=100*float(np.minimum(delta,0).mean())
    atomic_json(report/'PHYSICAL_minus_CONTROL.json',result)
    raw=Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
    manifest=json.loads((raw/'manifest.json').read_text());seal=json.loads((raw/'COMPLETE').read_text())
    if seal['manifest_sha256']!=file_sha(raw/'manifest.json') or manifest['records_sha256']!=file_sha(raw/'records.pkl'):
        raise ValueError('Unsealed raw Scene GT')
    records=pickle.load(open(raw/'records.pkl','rb'));tokens=sorted(manifest['tokens'])
    if len(tokens)!=12146 or set(tokens)!=set(records):raise ValueError('Wrong Test GT population')
    valid={t:np.abs(np.cumsum(records[t]['raw_future_dt_seconds'])-np.arange(1,9)*.5)<.06 for t in tokens}
    geometry={};training={};costs={}
    for arm in ('CONTROL','PHYSICAL'):
        label=arm+'_065000';directory=evaluation/label
        spec=dict(name=label,kind='single_scene_json',root=str(directory/'evaluation'),summary=str(directory/'SUMMARY.json'),
            checkpoint=arm+' IL63000+2000',seed=2101,grid='clean-x shift5 10-step ODE')
        geometry[arm],_=load_model(spec,tokens,records,valid,4)
        rows=[json.loads(line) for line in (root/'runs'/arm/'metrics_rank0.jsonl').read_text().splitlines()]
        if [r['updates'] for r in rows]!=list(range(63001,65001)) or any(r['cursor']!=32*r['updates'] for r in rows):
            raise ValueError('Study update/exposure journal not continuous')
        for rank in range(1,8):
            other=[json.loads(line) for line in (root/'runs'/arm/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
            if [r['updates'] for r in other]!=list(range(63001,65001)):
                raise ValueError('Missing peer training updates')
            if any(a['sample_ids']==b['sample_ids'] for a,b in zip(rows,other)):
                raise ValueError('Rank scene sampling unexpectedly duplicated')
        training[arm]=dict(logical_new_updates=2000,scene_exposures=64000,logged_full_update_seconds=sum(r['slowest_rank_seconds'] for r in rows),
            first100=dict(weighted_trajectory_loss=float(np.mean([r['losses']['x'] for r in rows[:100]])),
                physical_MAE_by_time_channel=np.mean([r['trajectory_geometry']['physical_MAE_by_time_channel'] for r in rows[:100]],axis=0).tolist()),
            last100=dict(weighted_trajectory_loss=float(np.mean([r['losses']['x'] for r in rows[-100:]])),
                physical_MAE_by_time_channel=np.mean([r['trajectory_geometry']['physical_MAE_by_time_channel'] for r in rows[-100:]],axis=0).tolist()),
            warning='Training noisy reconstruction, not deployed output. Weighted scalar losses have different definitions across arms.')
        for row in rows:
            if row['optimizer_global_steps']!=row['updates'] or row['optimizer_skipped_steps']!=0:
                raise ValueError('Skipped or miscounted native optimizer update')
        costs[arm]=json.loads((root/'jobs'/('train_'+arm)/'RESULT.json').read_text())
    # Scene/order and common auxiliary exposure must match, not only counts.
    def rank_rows(arm,rank):return [json.loads(l) for l in (root/'runs'/arm/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
    for rank in range(8):
        for a,b in zip(rank_rows('CONTROL',rank),rank_rows('PHYSICAL',rank)):
            if any(a[k]!=b[k] for k in ('sample_ids','candidate_bank_hash_used','lr_used','valid_supervision_counts')):
                raise ValueError('Matched training population/LR/bank exposure differs')
    atomic_json(report/'geometry.json',geometry);atomic_json(report/'training.json',training)
    atomic_json(report/'registration.json',registration)
    atomic_json(report/'calibration.json',json.loads((root/'calibration/CALIBRATION.json').read_text()))
    atomic_json(report/'resume_qualification.json',json.loads((root/'RESUME_QUALIFICATION.json').read_text()))
    worker_costs={}
    for p in (root/'jobs').glob('*/RESULT.json'):
        r=json.loads(p.read_text());worker_costs[p.parent.name]=dict(status=r['status'],returncode=r['returncode'],
            worker_wall_seconds=r['end_unix']-r['start_unix'],gpus=len(r['gpus']),
            shared_GPU_process_hours=(r['end_unix']-r['start_unix'])*len(r['gpus'])/3600,
            note='Worker wall times GPU count, not exclusive CUDA kernel time; includes setup/save and failed attempts.')
    atomic_json(report/'costs.json',dict(training_invocations=worker_costs,
        untracked_failure='calibration attempt001 failed before model child startup: missing interpreter; retained launch traceback, exact elapsed worker time unavailable.',
        evaluation_receipts=str(evaluation),calibration_and_qualification_not_free=True))
    paired=result['paired'];change=100*paired['mean_delta'];ci=[100*x for x in paired['CI95']]
    judgment=('支持本窗口内改善' if change>=.5 and ci[0]>0 else
              '当前窗口未获得净提升' if change<=0 and ci[1]<0 else '证据不足：未证明稳定净提升')
    lines=['# Qwen-World：坐标损失权重匹配续训','',
        '两组从相同IL63k原生模型、Adam、scheduler、数据游标、全部RNG开始；分别完整续训2000步/global32，共64000次scene曝光。训练seed42，完整Navtest12146，评价seed2101，单候选clean-x十步shift5 ODE，canonical v1.1。无best-of-N，无DEV/Test拟合权重或候选回流。Test历史多次被观察，不能声称完全盲测。','',
        'CONTROL保留原归一化L1；PHYSICAL只改变三个坐标的固定权重。工程容差为0.5m/0.25m/2度，提前登记，非论文最优值；输出层梯度量级仅在32个TRAIN场景校准一次。两组均保持原8点时间等权、三维动作编码、噪声日程、全模型训练、四个辅助任务及其系数；原父候选库对两组共同固定，避免异步刷新改变数据。未加入Huber、角度环绕损失、运动学或RL修改。','',
        '| 组 | Navtest PDMS | 纵向MAE cm | 横向MAE cm | 航向MAE ° | ADE cm |','|---|---:|---:|---:|---:|---:|']
    for arm,g in geometry.items():
        v=g['all'];lines.append(f'| {arm} | {g["PDMS"]:.6f} | {100*v["x_L1_m"]:.3f} | {100*v["y_L1_m"]:.3f} | {v["heading_MAE_deg"]:.4f} | {100*v["ADE_m"]:.3f} |')
    lines += ['',f'PHYSICAL−CONTROL：{change:+.6f} PDMS点，44物理log配对95%区间[{ci[0]:+.6f}, {ci[1]:+.6f}]；G+={result["G_plus_PDMS"]:+.6f}，G−={result["G_minus_PDMS"]:+.6f}点。',
        f'本窗口判断：**{judgment}**。只有一个训练/推理seed；2000步是有限续训窗口，不据一次阴性结果否定整类加权损失。', '',
        '新失败、恢复、原高分→零分及NC/DAC/TTC转移见PHYSICAL_minus_CONTROL.json；相对原IL63k见每组JSON。几何误差为最终请求与原始GT的差，非控制器模拟结果。TRAIN物理误差逐时刻/通道及费用另表，不用加权训练loss自身下降充当成功。','',
        '资格验证包含原L1值/梯度对照、物理尺度与mask、多micro归约、资源撤回合同，以及真实八卡连续3步对中断1步再恢复2步的模型/Adam/scheduler/RNG/builder逐值检查。资格权重未进入正式训练。源码、登记、命令和失败证据保留；旧IL/RL没有被热修改或停止。','',
        '恢复：原运行目录使用相同REGISTRATION和arm，追加`--resume <本组最近合法checkpoint>`；不可用qualification目录初始化正式实验。启动入口`python -m tools.qwen_world_postcot.coordinate_study_owner --root <登记根>`只有一个owner，已启动后不可重复执行。','']
    (report/'README.md').write_text('\n'.join(lines))
