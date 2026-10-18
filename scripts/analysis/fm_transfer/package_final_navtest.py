"""Package complete locked Navtest analyses, never launch/reselect a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.analysis.fm_transfer.repartition_cost_report import report as cost_report
from starVLA.rl.flow_grpo.transactions import atomic_json

MODELS = ['A_DDP_TIME_SFT', 'A_DDP_TIME_RL', 'B_SOURCE_ACTION_DIT_SFT', 'B_SOURCE_ACTION_DIT_RL']
MODES = ['E_NATIVE', 'E_SHIFT5']


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_complete(x):
    if (x.get('status') != 'COMPLETE_POPULATION' or x.get('scenes') != 12146
            or x.get('logs') != 136 or x.get('seed', x.get('identity', {}).get('seed')) != 2101):
        raise ValueError('Full 12146-scene/136-log seed2101 result is required')


def match_means(dual, canonical):
    for mode in MODES:
        for name, value in dual['versions']['v1.1'][mode]['mean'].items():
            if abs(value - canonical['modes'][mode]['mean'][name]) > 1e-12:
                raise ValueError('Independent canonical summaries disagree')


def collect(root):
    root = Path(root)
    locked = read(root/'final_navtest/LOCK.json')
    if locked['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST' or not locked['formal_budget_complete']:
        raise ValueError('Final endpoint was not locked')
    dual_root = root/'pdms_versions/analysis_scaleout'
    canonical_root = root/'scaleout_20261018/analysis'
    inputs, results, costs, canonical = {}, {}, {}, {}
    def load(path):
        x = read(path); require_complete(x); inputs[str(path)] = sha(path); return x
    for name in MODELS:
        d = load(dual_root/(name+'.json'))
        c = load(canonical_root/(name+'_final_NAVTEST.json'))
        if (d['identity']['weight_sha256'] != locked['models'][name]['weight_sha256']
                or c['identity']['weight_sha256'] != locked['models'][name]['weight_sha256']):
            raise ValueError('Wrong fixed model')
        match_means(d, c)
        results[name], canonical[name] = d, c
        costs[name] = cost_report(root/'evaluations_scaleout'/(name+'_final_NAVTEST'),
                                 root/'pdms_versions/v1.0/rescores_scaleout'/name)
    for variant in ['A_DDP_TIME', 'B_SOURCE_ACTION_DIT']:
        rl, sft = variant+'_RL', variant+'_SFT'
        name = rl+'_vs_'+sft
        d = load(dual_root/(name+'.json'))
        c = load(canonical_root/(rl+'_final_NAVTEST_paired_own_SFT.json'))
        match_means(d, c)
        if d['identity'] != results[rl]['identity'] or d['reference_identity'] != results[sft]['identity']['sha256']:
            raise ValueError('Wrong own-SFT comparison')
        for version in ['v1.0', 'v1.1']:
            for mode in MODES:
                row = d['versions'][version][mode]
                delta = row['mean']['score'] - results[sft]['versions'][version][mode]['mean']['score']
                if abs(delta-row['paired']['PDMS_log_cluster_bootstrap']['mean']) > 1e-12:
                    raise ValueError('Pair and independent means disagree')
                if version == 'v1.1':
                    p = c['modes'][mode]['paired_comparison']
                    for key in ['safety_transitions', 'reference_safe_new_joint_violation', 'nonzero_to_zero', 'zero_to_nonzero']:
                        if row['paired'][key] != p[key]:
                            raise ValueError('Independent safety summaries disagree')
        results[name] = d
    original = load(root/'pdms_versions/analysis/ORIGINAL_F_SFT.json')
    results['ORIGINAL_F_SFT'] = original
    for variant in ['A_DDP_TIME', 'B_SOURCE_ACTION_DIT']:
        name = variant+'_SFT_vs_ORIGINAL_F_SFT'; d = load(dual_root/(name+'.json'))
        if d['identity'] != results[variant+'_SFT']['identity'] or d['reference_identity'] != original['identity']['sha256']:
            raise ValueError('Wrong original SFT comparison')
        results[name] = d
    return dict(schema='locked_full_Navtest_publication_v1', status='COMPLETE_VERIFIED_RESULTS',
        scenes=12146, logs=136, inference_seed=2101, training_seed=42,
        modes=MODES, versions=['v1.0','v1.1'], lock_sha256=sha(root/'final_navtest/LOCK.json'),
        inputs=inputs, results=results, costs=costs, additional_predictions=0, additional_scores=0,
        bounds='Single fixed training/inference seed, historically observed Navtest, two grids reported separately, no best-of-two.')


def ci(x):
    return f"{100*x['mean']:+.6f} [{100*x['CI95'][0]:+.6f}, {100*x['CI95'][1]:+.6f}]"


def markdown(data):
    r = data['results']
    lines = ['# 固定终点完整 Navtest：SFT→RL 与双评分版本', '',
        '全部为12146场景/136 logs，训练seed42、推理seed2101。每种网格输出单条轨迹；',
        '官方v1.0与canonical v1.1使用同一请求，分列报告，不取较高版本或较高网格。',
        '以下是固定终点，非中途最高点。置信区间按log聚类，未包含训练/推理seed不确定性。', '',
        '| 模型 | 网格 | v1.0 PDMS | v1.1 PDMS |', '|---|---|---:|---:|']
    for name in ['ORIGINAL_F_SFT']+MODELS:
        for mode in MODES:
            lines.append(f"| {name} | {mode} | {r[name]['versions']['v1.0'][mode]['mean_percent']['score']:.6f} | {r[name]['versions']['v1.1'][mode]['mean_percent']['score']:.6f} |")
    lines += ['', '## RL相对自身SFT的增量', '',
        '| 路线 | 网格 | v1.0增量 [95% CI]，百分点 | v1.1增量 [95% CI]，百分点 |', '|---|---|---:|---:|']
    for v in ['A_DDP_TIME','B_SOURCE_ACTION_DIT']:
        p = r[v+'_RL_vs_'+v+'_SFT']
        for mode in MODES:
            lines.append(f"| {v} | {mode} | {ci(p['versions']['v1.0'][mode]['paired']['PDMS_log_cluster_bootstrap'])} | {ci(p['versions']['v1.1'][mode]['paired']['PDMS_log_cluster_bootstrap'])} |")
    lines += ['', '## Canonical v1.1安全转移', '',
        '| 路线/网格 | NC新增/恢复 | DAC新增/恢复 | TTC新增/恢复 | 固定SFT-safe中新联合违规 | 非零→零/零→非零 |',
        '|---|---:|---:|---:|---:|---:|']
    for v in ['A_DDP_TIME','B_SOURCE_ACTION_DIT']:
        p = r[v+'_RL_vs_'+v+'_SFT']
        for mode in MODES:
            x=p['versions']['v1.1'][mode]['paired']; tr=x['safety_transitions']
            transitions=['%d / %d'%(tr[k]['new_violation'],tr[k]['recovery']) for k in
                         ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound']]
            lines.append(f"| {v}/{mode} | {' | '.join(transitions)} | {x['reference_safe_new_joint_violation']} / {x['reference_safe_population']} | {x['nonzero_to_zero']} / {x['zero_to_nonzero']} |")
    lines += ['', '## 成本与边界', '',
        '| 模型 | 逻辑轨迹数 | 实际推理（含入口检查） | 累计共享GPU worker小时 | v1.0实际已记评分调用 |',
        '|---|---:|---:|---:|---:|']
    for name, x in data['costs'].items():
        g=x['inference']; s=x['v1_0_scoring']
        lines.append(f"| {name} | {g['logical_single_candidate_evaluations']} | {g['physical_predictions_including_diagnostics']} | {g['cumulative_worker_wall_seconds']/3600:.6f} | {s['physical_scoring_calls']} |")
    lines += ['', '共享GPU worker时间之和不等于端到端wall或独占GPU计算时间。旧/新分片的',
        '调用、加载和评分成本均保留；中断评分中未记入的在途计算成本明确为未知，不填0。',
        '原始JSON包含全部分项、低尾、G+/G−、两版配对结果、输入hash和详细成本。',
        'B换头同时重新进行SFT，A也有新的监督训练与困难清单；跨路线比较不能单独归因架构。',
        '92分判定只看已登记主协议canonical v1.1 E_NATIVE，并同时披露安全代价。',
        '这些全量数值需结合已有训练/实现证据解释，不把均分增量自动称为安全保持成功。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default='/mnt/project/fm-transfer-20261016')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve previous publication package')
    data=collect(args.root)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'results.json',data)
    (out/'README.md').write_text(markdown(data))
    for path in data['inputs']:
        source=Path(path);shutil.copyfile(source,out/source.name)
    receipt=dict(status=data['status'],files={p.name:sha(p) for p in sorted(out.iterdir())},
                 source_sha256=sha(__file__),additional_predictions=0,additional_scores=0)
    atomic_json(out/'COMPLETE.json',receipt)
    print(json.dumps(dict(status=receipt['status'],output=str(out))))


if __name__=='__main__':main()
