"""Publish completed registered follow-up stages; never run actor or scorer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.analysis.simwam_recipe.registered_followup import require_registration, matched_control
from starVLA.rl.simwam_recipe.contracts import ROOT, file_sha


DESCRIPTIONS = {
    'R_CANONICAL_X1_SAFETY25': (
        'SAFETY25', 'X1',
        '只改变固定困难场景中安全失败人口的曝光配额；SFT、LoRA、X1 anchor、源SDE、reward和optimizer共同设置不变。训练场景顺序不同，训练随机均分不可直接比较；下表部署评价人口相同。'),
    'R_CANONICAL_LAST_ODE': (
        'LAST_ODE', 'Canonical',
        '唯一科学配置差异为sde_exclude_last_step=true：三个随机步骤从[7,8,9]前移到[6,7,8]，第9步为普通Euler ODE。SFT、LoRA、困难场景顺序、reward、原anchor和optimizer不变，不叠加X1。窗口改变同时影响采样状态与被优化时间位置，不能归因于只删除最后一次epsilon。')}


def summarize(variant, update, output, matched_comparison_dir=None):
    display, control_display, difference = DESCRIPTIONS[variant]
    if update not in (5000, 10000, 15000, 20000):
        raise ValueError('Only the four existing fixed observations are admitted')
    root = require_registration(variant)
    control, control_root = matched_control(variant)
    label = f'{variant}_update{update:06d}'
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    sources = {}

    def read(path):
        path = Path(path); sources[str(path)] = file_sha(path)
        return json.loads(path.read_text())

    marker = read(root/'milestone_queue'/(label+'.complete.json'))
    if marker['status'] != 'COMPLETE' or marker['label'] != label:
        raise ValueError('No completed fixed stage')
    checkpoint = root/'runs'/(variant+'_seed42_v1')/'checkpoints'/f'update{update:06d}_inner0'
    seal = read(checkpoint/'COMPLETE')
    if (seal['phase'], seal['updates'], seal['world'], seal['next_inner']) != ('READY', update, 8, 0):
        raise ValueError('Wrong sealed policy boundary')
    if file_sha(checkpoint/'files.json') != seal['files_sha256']:
        raise ValueError('Checkpoint seal changed')
    export = read(root/'exports'/label/'COMPLETE')
    if export['updates'] != update: raise ValueError('Exported policy has a different age')
    values = {}
    for cohort, n in [('DEV', 1696), ('TRAIN_DIAGNOSTIC', 512)]:
        paired_key = cohort+'_matched_control'
        if matched_comparison_dir is None:
            paired_path = marker['results'][paired_key]
        else:
            paired_path = Path(matched_comparison_dir)/(cohort+'_versus_'+control+'.json')
        data = {
            'SFT': read(Path(ROOT)/'evaluations'/('SFT_update000000_'+cohort)/'summary.json'),
            control: read(control_root/'evaluations'/f'{control}_update{update:06d}_{cohort}'/'summary.json'),
            variant: read(marker['results'][cohort]),
            'versus_control': read(paired_path)}
        actual = data[variant]['identity']
        if actual['weight_sha256'] != export['weight_sha256'] or actual['qualification_only']:
            raise ValueError('Evaluation does not use the locked formal export')
        for value in data.values():
            if value['status'] != 'COMPLETE_POPULATION' or value['scenes'] != n or value['seed'] != 2101:
                raise ValueError('Incomplete population or changed evaluation seed')
            for key in ('population_identity', 'tokens_sha256', 'scenes', 'cohort', 'seed', 'modes',
                        'profile', 'feature_cache', 'scoring'):
                if value['identity'][key] != actual[key]:
                    raise ValueError('Unmatched evaluation contract: '+key)
            if set(value['modes']) != {'E_NATIVE', 'E_SHIFT5'}:
                raise ValueError('Both fixed single-candidate grids are required')
        if data['versus_control']['identity'] != actual:
            raise ValueError('Paired comparison uses a different candidate identity')
        expected_sft = Path(ROOT)/'evaluations'/('SFT_update000000_'+cohort)
        if Path(data[variant]['reference']).resolve() != expected_sft.resolve():
            raise ValueError('Wrong SFT reference')
        expected_ref = control_root/'evaluations'/f'{control}_update{update:06d}_{cohort}'
        if Path(data['versus_control']['reference']).resolve() != expected_ref.resolve():
            raise ValueError('Wrong matched control')
        values[cohort] = data
    result = dict(schema='simwam_registered_followup_fixed_stage_v2',
        status='COMPLETE_REGISTERED_STAGE_NOT_NAVTEST', captured_utc=datetime.now(timezone.utc).isoformat(),
        optimizer_update=update, variant=variant, control=control, checkpoint_seal=seal,
        export=export, evaluations=values, sources_sha256=sources, analysis_source_sha256=file_sha(__file__),
        new_predictions_from_assembly=0, new_scores_from_assembly=0,
        limits=['TRAIN is a fixed targeted diagnostic slice, not the full training population.',
                'Two grids are reported separately; never best-of-two. Single training42 and inference2101.',
                'Only final locked Navtest establishes the primary target outcome; DEV above92 does not.',
                'Cost retains evaluator invocations; aggregation does not repeat inference or scoring.'])
    lines = [f'# {display} 固定{update}更新：完整单候选ODE阶段评价', '',
        result['captured_utc']+'。DEV1696/16 logs、固定定向TRAIN512/343 logs；canonical v1.1、原始观测、单候选、seed2101。没有搜索或GT选择。', '',
        '这是已登记20k计划中的固定观察点，不是Navtest。主目标仍是原十步均匀ODE的完整Navtest提升及安全保持；不能把DEV超过92写成目标已实现。两组训练seed均42，按log聚类的区间不包含训练或推理seed重复的不确定性。DEV仅保证RL更新未见，SFT可能见过；TRAIN是固定定向诊断集，不是总体无偏估计。', '',
        difference, '',
        f'| 人口 / 网格 | SFT | 同阶段{control_display} | {display} | 对SFT差值 / 95%区间 | 对{control_display}差值 / 95%区间 |',
        '|---|---:|---:|---:|---|---|']
    for cohort, data in values.items():
        for mode in ('E_NATIVE', 'E_SHIFT5'):
            scores = [data[k]['modes'][mode]['mean_percent']['score'] for k in ('SFT', control, variant)]
            cells = []
            for key, baseline in [(variant, scores[0]), ('versus_control', scores[1])]:
                ci = data[key]['modes'][mode]['paired_comparison']['PDMS_log_cluster_bootstrap']['CI95']
                cells.append(f'{scores[2]-baseline:+.4f} [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}]')
            lines.append(f'| {cohort} / {mode} | '+' | '.join(f'{s:.4f}' for s in scores)+' | '+' | '.join(cells)+' |')
    fields = ['no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
              'ego_progress', 'comfort', 'driving_direction_compliance']
    lines += ['', '## 分项与安全转移', '',
        '| 人口 / 网格 / 模型 | NC | DAC | TTC | EP | Comfort | DDC | 零分 / 联合安全场景数 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for cohort, data in values.items():
        for mode in ('E_NATIVE', 'E_SHIFT5'):
            for key, name in [('SFT', 'SFT'), (control, control_display), (variant, display)]:
                v = data[key]['modes'][mode]
                lines.append(f'| {cohort} / {mode} / {name} | '+' | '.join(f'{v["mean_percent"][f]:.4f}' for f in fields)+f' | {v["zero_scenes"]} / {v["fully_safe_scenes"]} |')
    lines += ['', '新增/恢复按每个分项完全通过定义；NC=0.5不算完全通过。固定safe人口由各参照分别定义。G+/G−是场景原始0—1分差的正/负部分总和，不是百分点。', '',
        '| 人口 / 网格 / 参照 | NC 新增/恢复 | DAC 新增/恢复 | TTC 新增/恢复 | 固定safe新增联合违规/人口 | 非零→零/零→非零 | G+ / G− |',
        '|---|---|---|---|---|---|---|']
    for cohort, data in values.items():
        for mode in ('E_NATIVE', 'E_SHIFT5'):
            for key, name in [(variant, 'SFT'), ('versus_control', control_display)]:
                p = data[key]['modes'][mode]['paired_comparison']; cells = []
                for field in fields[:3]:
                    t = p['safety_transitions'][field]; cells.append(f'{t["new_violation"]}/{t["recovery"]}')
                lines.append(f'| {cohort} / {mode} / {name} | '+' | '.join(cells)+f' | {p["reference_safe_new_joint_violation"]}/{p["reference_safe_population"]} | {p["nonzero_to_zero"]}/{p["zero_to_nonzero"]} | {p["positive_gain_sum"]:.4f}/{p["negative_degradation_sum"]:.4f} |')
    lines += ['', '## 评价成本与复查', '',
        '| 人口，两网格分表 | 逻辑单候选评分 | 实际推理（含核验） | 累计worker wall秒 | 单项job跨度秒 |',
        '|---|---:|---:|---:|---:|']
    for cohort, data in values.items():
        c = data[variant]['cost']
        lines.append(f'| {cohort} | {c["logical_single_candidate_evaluations"]} | {c["physical_predictions_including_diagnostics"]} | {c["cumulative_worker_wall_seconds"]:.2f} | {c["job_wall_seconds"]:.2f} |')
    lines += ['', '两个人口并行，不能把各job跨度相加当作端到端wall；共享worker时间也不是独占GPU计算小时。合并导出及本报告聚合未单独恢复完整wall，不伪造为0。详见[完整结果与成本](results.json)。不重新运行模型或评分，只读取控制器已完成并校验的摘要。', '',
        '实际可执行汇总入口，输出必须使用新目录：', '', '```bash',
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
        'python -m scripts.analysis.simwam_recipe.summarize_followup_stage \\',
        f'  --variant {variant} --update {update} \\']
    if matched_comparison_dir is not None:
        lines.append(f'  --matched-comparison-dir {str(Path(matched_comparison_dir).resolve())} \\')
    lines += ['  --output "$NEW_PUBLIC_STAGE_OUTPUT"', '```', '']
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'], output=str(output), new_scores=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--variant', choices=sorted(DESCRIPTIONS), required=True)
    parser.add_argument('--update', type=int, required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--matched-comparison-dir', help='Existing read-only paired reports; their actual reference identities are checked')
    args = parser.parse_args(); summarize(args.variant, args.update, args.output, args.matched_comparison_dir)
