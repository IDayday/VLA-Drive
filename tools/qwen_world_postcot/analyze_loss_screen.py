"""Read complete matched Test records; do not run a model, scorer, or fit weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.qwen_world_postcot.summarize import load, compare
from starVLA.rl.qwen_world_postcot.contracts import fingerprint


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    base = Path('/mnt/project/qwen-world-postcot-20261025')
    control = base/'coordinate_loss_20261028/NAVTEST/CONTROL_065000'
    reference_identity, reference = load(control/'evaluation')
    control_score = 100*np.mean([r['feedback']['metrics']['score'] for r in reference.values()])
    results = dict(CONTROL=dict(PDMS=float(control_score), delta_vs_control_PDMS=0.0))
    candidates = dict(PHYSICAL=base/'coordinate_loss_20261028/NAVTEST/PHYSICAL_065000',
        TIME_RESIDUAL=base/'loss_designs_20261028/NAVTEST/TIME_RESIDUAL_065000',
        DELTA=base/'loss_designs_20261028/NAVTEST/DELTA_065000',
        HUBER=base/'loss_designs_20261028/NAVTEST/HUBER_065000')
    for name, directory in candidates.items():
        if not (directory/'COMPLETE.json').exists():
            results[name] = dict(status='INCOMPLETE', evaluated_scenes=len(list((directory/'evaluation/scenes').glob('*.json'))),
                PDMS=None, reason='Do not report a partial-population mean.')
            continue
        seal = json.loads((directory/'COMPLETE.json').read_text())
        if seal.get('all_owned_workers_exited') is not True:
            raise ValueError('Active workers cannot form a complete comparison')
        identity, rows = load(directory/'evaluation')
        paired = compare(identity, rows, reference_identity, reference)
        delta = np.array([rows[t]['feedback']['metrics']['score']-reference[t]['feedback']['metrics']['score'] for t in reference])
        row = dict(status='COMPLETE', PDMS=100*float(np.mean([r['feedback']['metrics']['score'] for r in rows.values()])),
            delta_vs_control_PDMS=100*paired['mean_delta'], CI95_PDMS=[100*x for x in paired['CI95']],
            G_plus_PDMS=100*float(np.maximum(delta,0).mean()), G_minus_PDMS=100*float(np.minimum(delta,0).mean()),
            paired=paired, evaluation=str(directory/'evaluation'), inference_identity_sha256=identity['sha256'])
        row['sha256'] = fingerprint(row); results[name] = row
        (output/(name+'_minus_CONTROL.json')).write_text(json.dumps(row,indent=2)+'\n')
    stats = dict(schema='world_IL_late_loss_screen_interim_v1',results=results,training_seed=42,inference_seed=2101,
        scenes=12146,control_checkpoint='Same63k native parent, unchanged L1,2000 added updates',
        scene_exposures_per_arm=64000,full_TRAIN_scenes=90890,approximate_TRAIN_passes=64000/90890,
        action_peak_lr=1e-4,action_lr_first=3.180475315182563e-5,action_lr_last=2.882979433651155e-5,
        interpretation='Tests a late loss switch with inherited optimizer state, not loss choice from generic initialization.',
        training_seed_uncertainty_included=False,Test_historically_observed=True)
    stats['sha256']=fingerprint(stats);(output/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    lines=['# Qwen-World 损失设计：2k续训初筛','',
        '各组从相同63k模型、Adam、scheduler、RNG和数据游标恢复，续训2000步/global32。'
        '四个辅助任务及固定候选库一致。完整12146 Navtest、canonical v1.1、单候选十步shift5 ODE、seed2101；Test已多次观察。', '',
        '| 方法 | 完整Navtest PDMS | 相对匹配CONTROL | 配对95%区间（点） |', '|---|---:|---:|---|',
        f'| CONTROL原归一化L1 | {control_score:.6f} | — | — |']
    for name in candidates:
        row=results[name]
        if row.get('status')!='COMPLETE':lines.append(f'| {name} | 未完成 | — | — |');continue
        lo,hi=row['CI95_PDMS'];lines.append(f'| {name} | {row["PDMS"]:.6f} | {row["delta_vs_control_PDMS"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['',
        '当前完成的方法没有高于匹配CONTROL的点估计。是否有明确回退应看各自区间，不能把区间跨零称作等价。'
        '新失败、恢复、G+/G−与各安全项转移见配对JSON。一个训练/推理seed不提供训练复现的不确定性。', '',
        '2000步对应64000次scene曝光，约0.704次完整TRAIN人口，只有原100k预算的2%。'
        '动作LR由3.1805e-5到2.8830e-5；Qwen语言/视觉对应3.1805e-6/3.1805e-7起。'
        '这不是从零Adam、完整warmup起跑的训练。新损失面对的是已按旧目标塑造的表示与优化器状态。', '',
        '该设计能判断晚期切换是否立即有益、实现是否稳定、是否出现明显副作用；不能据此排除从头采用新损失的价值。'
        '保留Adam是为匹配既有模型续训，不等于已排除优化器过渡影响。也没有证据仅凭末端分数认定发生了数值不稳定。', '',
        '从头的确认实验应使用相同通用Qwen与相同随机memory/actor初始化、相同TRAIN/辅助任务、batch、100k总预算与观察点；'
        '唯一改变轨迹损失，Adam和scheduler都从同一起点开始。20k/40k/60k/80k/100k完整Test曲线及固定终点分开报告，不能只挑单个峰值。'
        '晚期输出层梯度校准的scale不能未经检查照搬到随机初始化；若需要匹配尺度，在共同TRAIN初始化批次一次校准并登记。', '',
        '不建议把所有阴性初筛方案立即展开100k：先补齐Huber，与原损失选择最有依据的一种构造做完整匹配确认。'
        '这份报告是已有轨迹的CPU派生统计，未增加模型前向或评分，也未启动新100k算法矩阵。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{x:v.get(x)for x in ['status','PDMS','delta_vs_control_PDMS','CI95_PDMS']}for k,v in results.items()},indent=2))


if __name__=='__main__':main()
