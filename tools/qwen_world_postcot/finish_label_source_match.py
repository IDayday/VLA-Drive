"""Matched update-budget label comparison from complete, already scored outputs."""
import json
from pathlib import Path

import numpy as np

from tools.qwen_world_postcot.summarize import load, compare, METRICS
from tools.qwen_world_postcot.user_interim_owner import read, write, fingerprint


def finish(root, requests, report):
    root, report = Path(root), Path(report)
    pinned = read(root/'PINNED.json'); audit = read(root/'MATCHING_AUDIT.json')
    body = dict(audit)
    if body.pop('sha256') != fingerprint(body) or audit['sha256'] != pinned['matching_audit_sha256']:
        raise ValueError('Changed comparison matching audit')
    report.mkdir(parents=True, exist_ok=True)
    steps = '/'.join(str(request.pin['updates']) for request in requests.values())
    count = len(requests)
    results = {}; lines = ['# 原轨迹与优化轨迹标签：同预算 Qwen-World IL', '',
        '两组都是通用Qwen＋新建memory/actor的完整100k预算IL，使用同一架构、seed42、global32、'
        f'原L1和四个辅助任务。此处比较相同{steps}更新的完整Navtest。'
        '两组前15k八rank的场景顺序、LR、辅助曝光与实际候选bank已核对。配置仅独立bank registry路径不同。', '',
        'canonical NAVSIM v1.1、完整12146场景、单候选clean-x十步shift5 ODE、seed2101。'
        'Test历史已被观察；不称盲测。优化标签已有结果直接复用，不增加其推理或评分。', '',
        '| 更新数 | 原轨迹IL | 优化轨迹IL | 优化−原轨迹（点） | 配对95%区间（点） |',
        '|---|---:|---:|---:|---|']
    if audit.get('candidate_bank_content_matches_after15k') is False:
        lines[2] += ('从15001到20000两组各自刷新了post-head辅助候选bank，内容不同；'
                     '场景顺序/LR/optimizer计数仍匹配。因此此点比较完整标签训练方案，不能只归因主轨迹标签。')
    for label, request in requests.items():
        done = read(request.root/'COMPLETE.json')
        optimized = Path(request.pin['matched_optimized_evaluation'])
        old_done = read(optimized.parent/'COMPLETE.json')
        if any(d.get('status') != 'COMPLETE' or d.get('all_owned_workers_exited') is not True
               for d in (done, old_done)):
            raise ValueError('Both evaluations must be complete with workers exited')
        original_id, original_rows = load(request.output)
        optimized_id, optimized_rows = load(optimized)
        paired = compare(optimized_id, optimized_rows, original_id, original_rows)
        original_score = 100*float(np.mean([r['feedback']['metrics']['score'] for r in original_rows.values()]))
        optimized_score = 100*float(np.mean([r['feedback']['metrics']['score'] for r in optimized_rows.values()]))
        if abs(optimized_score-request.pin['matched_optimized_PDMS']) > 1e-9:
            raise ValueError('Reused optimized-label score differs from its locked result')
        delta = np.array([optimized_rows[t]['feedback']['metrics']['score']-original_rows[t]['feedback']['metrics']['score'] for t in original_rows])
        metrics = {key: dict(original=100*float(np.mean([r['feedback']['metrics'][key] for r in original_rows.values()])),
                            optimized=100*float(np.mean([r['feedback']['metrics'][key] for r in optimized_rows.values()])))
                   for key in METRICS}
        row = dict(updates=request.pin['updates'], original_PDMS=original_score,
            optimized_PDMS=optimized_score, optimized_minus_original_PDMS=100*paired['mean_delta'],
            CI95_PDMS=[100*x for x in paired['CI95']], G_plus_PDMS=100*float(np.maximum(delta,0).mean()),
            G_minus_PDMS=100*float(np.minimum(delta,0).mean()), paired_optimized_vs_original=paired,
            native_metrics=metrics, original_evaluation=str(request.output), optimized_evaluation=str(optimized),
            original_identity_sha256=original_id['sha256'], optimized_identity_sha256=optimized_id['sha256'],
            original_native_parity=read(request.proof), optimized_native_parity=read(optimized.parent/'native_export_parity.json'))
        row['sha256'] = fingerprint(row); results[label] = row; write(report/(label+'.json'),row)
        lo,hi=row['CI95_PDMS']
        lines.append(f'| {row["updates"]} | {original_score:.6f} | {optimized_score:.6f} | '
                     f'{row["optimized_minus_original_PDMS"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['', '区间按完整物理log聚类bootstrap，保留场景权重。paired字段方向统一为优化标签减原轨迹；'
        '包含新增失败、恢复、G+/G−及全部原生分项。一个训练/推理seed不提供训练seed复现证据。', '',
        '这些点只判断早期同预算学习；完整100k终点尚未完成，不能将早期阴性结果当作优化标签最终无效。'
        '原轨迹13k原生payload不在保留列表，本次不拿其他更新数替代它。优化标签历史98分是标签评分，不是模型部署分数。', '',
        '初始化/模型/优化器/teacher合同一致；源码有可选标签overlay、资源/身份、诊断接口等适配，未宣称源码逐字节相同。'
        '前15k实际bank相同，后续各自异步刷新仍需按实际记录核对，不能预先宣称整个100k只有标签差异。', '',
        f'本次新增{12146*count}条逻辑场景评测、{count}次native/export一致性；复用的优化标签评测也有历史费用。'
        '逐调用共享GPU worker时间在资产根保留，不当作独占GPU计算时间。没有重新训练或新增seed。']
    write(report/'matching_audit.json',audit); write(report/'registration.json',pinned)
    summary=dict(schema='world_original_vs_optimized_labels_matched_Test_v1',results=results,
        comparison='optimized_labels_minus_original_GT',training_seed=42,inference_seed=2101,
        scenes_per_model=12146,new_original_GT_scene_evaluations=12146*count,reused_optimized_scene_evaluations=12146*count,
        logical_native_parity_checks=count,Test_previously_observed=True)
    summary['sha256']=fingerprint(summary);write(report/'summary.json',summary)
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    return results
