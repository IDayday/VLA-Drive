"""Render measured results; incomplete training cannot produce a final report."""
import argparse
import json
import numpy as np
from iqe.contracts import require
from iqe.io import atomic_bytes, atomic_json, file_hash, read_json
from .config import CELLS, load


def generate(config):
    final = read_json(config.root/'FINAL_RESULTS.json')
    choice = read_json(config.root/'MODEL_CHOICE.json')
    verdict = read_json(config.root/'VERDICT.json')
    bound = read_json(config.root/'EXPERIMENT_BINDING.json')
    margin = read_json(config.root/'SELECTOR_MARGIN.json')
    runtime = {}
    for cell in CELLS:
        folder = config.root/'scorers'/cell
        result = read_json(folder/'result.json')
        require(result['status'] == 'COMPLETE' and result['optimizer_steps'] == config.updates,
                'incomplete Scorer cannot be reported as a final model')
        logs = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        require([r['optimizer_step'] for r in logs] == list(range(1, config.updates+1)), 'missing/duplicated actual optimizer exposures')
        require(all(r['global_batch_actual'] == config.global_batch and np.isfinite(r['loss'])
                    and np.isfinite(r['gradient_norm']) for r in logs), 'invalid production updates')
        seconds = [r['compute_update_seconds'] for r in logs]
        runtime[cell] = dict(optimizer_steps=result['optimizer_steps'], actual_scene_presentations=sum(r['global_batch_actual'] for r in logs),
                            update_seconds_sum=float(sum(seconds)), update_mean_seconds=float(np.mean(seconds)),
                            update_p50_seconds=float(np.median(seconds)), update_p95_seconds=float(np.quantile(seconds, .95)),
                            scope='measured optimizer updates including fetch; validation/checkpoint/startup excluded',
                            last_process_elapsed_seconds=result['elapsed_seconds'],
                            initialization_hash=file_hash(folder/'SCORER_INITIALIZATION.json'),
                            best_checksum=result['best_checkpoint_hash'])
    evidence = dict(experiment_hash=file_hash(config.root/'EXPERIMENT_BINDING.json'),
                    model_choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
                    final_result_hash=file_hash(config.root/'FINAL_RESULTS.json'),
                    runtime=runtime, external_scorer_weights_loaded=False, ACTIVE_updated=False,
                    scientific_qualification=verdict['empirical_heldout_qualification'])
    for name in ('ONLINE_AND_LATENCY', 'ROUNDTRIP'):
        path = config.root/'qualification'/(name+'.json')
        evidence[name] = dict(path=str(path), status=read_json(path)['status'], checksum=file_hash(path)) if path.exists() else {'status': 'WAITING_INPUTS'}
    resume = config.root/'qualification/real_resume_R11/REPORT.json'
    evidence['REAL_RESUME'] = dict(status=read_json(resume)['status'], checksum=file_hash(resume)) if resume.exists() else {'status': 'WAITING_INPUTS'}
    atomic_json(config.root/'MEASURED_COST_AND_EVIDENCE.json', evidence, immutable=True)
    lines = ['# 从零训练的 Scorer：固定候选池结果', '',
             '以下 PDMS 使用百分制。模型选择仅使用 stage_val，margin 仅使用 selector_cal；'
             '完整固定 dev 在模型与规则锁定后读取。没有新 navtest，也没有更新 ACTIVE。', '',
             f"实验：`{config.experiment_id}`；代码起点：`{bound['source_commit']}`。",
             '四组均随机初始化，Base、三个专家、W/A/ego、样本清单和采样定义保持固定。', '',
             '| 模型 | 验证选择步 | dev 选轨 PDMS | 相对 Base | 相对最好固定策略 | Oracle 捕获 | 高分转零分 | 新增 NC / DAC / TTC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    selected_steps = {row['cell']: row['step'] for row in choice['cells']}
    for cell in CELLS:
        r = final['cells'][cell]
        capture = 'NA' if r['oracle_capture'] is None else f"{100*r['oracle_capture']:.2f}%"
        safety = r['safety']
        risk = ' / '.join(str(safety[k]['new_failures']) for k in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound'))
        lines.append(f"| {cell} | {selected_steps[cell]} | {100*r['selected_01']:.4f} | {100*r['selected_gain_01']:+.4f} | "
                     f"{100*r['selected_over_best_fixed_01']:+.4f} | {capture} | {r['high_to_zero']} | {risk} |")
    selected = choice['selected']['cell']
    r = final['cells'][selected]
    ci = r['gain_ci']
    lines += ['', f"预先锁定的选择结果：**{selected}**；dev {r['scenes']} 场景、{r['logs']} logs。",
              f"同集合 Base={100*r['base_01']:.4f}，Oracle={100*r['oracle_01']:.4f}；"
              f"选轨相对 Base 差值的配对 log bootstrap 95% 区间：[{100*ci['lower_95']:.4f}, {100*ci['upper_95']:.4f}]。",
              '单 seed 结果；没有逐场景安全保证，平均收益不代替新增失败场景检查。', '',
              f"可选 margin：delta={margin['selected_delta']}，always_Base={margin['always_Base']}；"
              f"dev PDMS={100*final['margin']['selected_01']:.4f}。始终回 Base 不算 Scorer 成功。", '',
              '| 模型 | D_S 选轨收益 | dev 选轨收益 | D_S 零分候选平均预测 | dev 零分候选平均预测 |',
              '|---|---:|---:|---:|---:|']
    for cell in CELLS:
        f, d = final['fit_diagnostics'][cell], final['cells'][cell]
        def number(value):
            return 'NA' if value is None else f'{100*value:.4f}'
        lines.append(f"| {cell} | {100*f['selected_gain_01']:+.4f} | {100*d['selected_gain_01']:+.4f} | "
                     f"{number(f['prediction_diagnostics']['zero_true_mean_prediction'])} | "
                     f"{number(d['prediction_diagnostics']['zero_true_mean_prediction'])} |")
    lines += ['', '训练开销见 `MEASURED_COST_AND_EVIDENCE.json`；它按真实逐步记录累计，'
              '中断恢复不会丢掉此前更新的成本或曝光。验证、checkpoint 和启动成本与更新计时分开。', '',
              f"经验留出资格通过：**{verdict['empirical_heldout_qualification']}**；"
              f"存在新增风险场景：**{verdict['risk_regression']}**。",
              '资格未通过时保留研究结果，不更新部署，不把代码通过或训练完成写成科学成功。', '',
              '运行和恢复命令：仓库 `reports/iqe_robust_scorer_20261014/RUNBOOK.md`。']
    atomic_bytes(config.root/'RESULTS.md', ('\n'.join(lines)+'\n').encode(), immutable=True)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(load(args.config))))


if __name__ == '__main__':
    main()
