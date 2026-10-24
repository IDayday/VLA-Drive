"""Publish only complete, sealed native Navtest aggregates; never raw scenes."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collect(control):
    plan = read(control/'registration.json')
    complete = read(control/'COMPLETE.json')
    assert complete['status'] == 'COMPLETE' and (control/'lease_closed.json').exists()
    assert read(control/'worker_exit_verification.json')['status'] == 'PASS'
    summaries, pairs = {}, {}
    for label in plan['models']:
        summaries[label] = summary = read(control/'analysis'/f'{label}_summary.json')
        pairs[label] = pair = read(control/'analysis'/f'{label}_paired_vs_IL53k.json')
        for value in [summary, pair]:
            body = dict(value)
            assert body.pop('sha256') == fingerprint(body)
            assert value['status'] == 'COMPLETE' and value['scenes'] == 12146 and value['logs'] == 136
            assert value['seed'] == 2101
        assert summary['full_population'] and not summary['qualification']
        assert summary['metric'] == plan['metric'] and summary['inference'] == plan['inference']
        assert summary['logical_single_candidates'] == 12146
        assert complete['models'][label]['PDMS'] == summary['PDMS']
        assert all(math.isfinite(v) for v in summary['means'].values())
    return plan, complete, summaries, pairs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--repository')
    parser.add_argument('--report-relative')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=10800)
    parser.add_argument('--expected-head')
    args = parser.parse_args()
    control = Path(args.control).resolve()
    try:
        deadline = time.monotonic()+args.deadline_seconds
        while not ((control/'COMPLETE.json').exists() and (control/'lease_closed.json').exists()):
            if (control/'failure.json').exists():
                raise RuntimeError('Evaluation failed; refuse partial publication')
            if not args.wait or time.monotonic() > deadline:
                raise TimeoutError('Full Navtest and lease closure not available')
            time.sleep(60)
        plan, complete, summaries, pairs = collect(control)
        if args.validate_only:
            print(json.dumps(dict(status='PASS', models=list(summaries), scenes_each=12146)))
            return
        repository = Path(args.repository).resolve()
        report = repository/args.report_relative
        assert report.is_relative_to(repository) and report.is_dir()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repository, text=True).strip()
        assert branch == 'research/qwen-sourceflow-safe-grpo-20261024'
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip() == args.expected_head
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip()
        for label in summaries:
            for suffix in ['summary', 'paired_vs_IL53k']:
                source = control/'analysis'/f'{label}_{suffix}.json'
                assert source.stat().st_size < 200000
                shutil.copyfile(source, report/source.name)
        shutil.copyfile(control/'COMPLETE.json', report/'COMPLETE.json')
        admission = read(control/'evaluation_admission.json')
        cost = dict(schema='native_latest_NAVTEST_cost_aggregate_v1',
                    job_wall_seconds=complete['at']-read(control/'launch.json')['at'],
                    physical_GPUs=admission['physical_GPUs'], worker_slots=admission['worker_slots'],
                    logical_single_candidate_evaluations=sum(v['logical_single_candidates'] for v in summaries.values()),
                    physical_inference_including_retries=sum(v['physical_inference_including_retries'] for v in summaries.values()),
                    cumulative_worker_wall_seconds=sum(v['worker_wall_seconds'] for v in summaries.values()),
                    GPU_semantics='Shared GPU worker occupancy, not exclusive GPU compute time',
                    model_loading_worker_seconds=sum(v['model_loading_seconds'] for v in summaries.values()),
                    forward_worker_seconds=sum(v['forward_worker_seconds'] for v in summaries.values()),
                    scoring_wait_worker_seconds=sum(v['scoring_wait_worker_seconds'] for v in summaries.values()),
                    export_and_parity_worker_seconds=sum(read(p)['wall_seconds'] for p in (control/'jobs').glob('*_export*/result.json')),
                    lease_closed=True, all_native_evaluation_children_reaped_with_exit0=True)
        (report/'cost_summary.json').write_text(json.dumps(cost, indent=2)+'\n')
        text = '# 新权重完整Navtest结果\n\n'
        text += '全部12146场景/136logs；canonical NAVSIM v1.1；单候选10步shift5 ODE；seed2101。所有推理与评分已完成，非候选oracle或训练reward。\n\n'
        text += '|模型|固定更新|PDMS|相对共同IL53k|95% log-bootstrap区间（点）|\n|---|---:|---:|---:|---|\n'
        for label, value in summaries.items():
            pair = pairs[label]
            bootstrap = pair['PDMS_log_cluster_bootstrap']
            delta = bootstrap['mean']*100
            bounds = [100*v for v in bootstrap['CI95']]
            text += f"|{label}|{plan['models'][label]['updates']}|{value['PDMS']:.6f}|{delta:+.6f}|[{bounds[0]:+.4f}, {bounds[1]:+.4f}]|\n"
        text += '\nIL行代表继续IL相对53k的变化，不能写成RL收益。新安全组是2k/1.3k早期观察；预算不一致，不能据此排名完整方法。IL53k已被用户指定且历史Navtest已多次观察，本轮也明确记录为用户要求的中期测量；正式50k预算与DEV选择不变。区间仅含场景加权log重采样，不含训练或推理seed不确定性。\n\n'
        text += '|模型|NC|DAC|TTC|EP|新增零分|零分恢复|原安全人口新增联合违规|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for label, value in summaries.items():
            pair, metrics = pairs[label], value['means']
            safety = pair.get('fixed_IL_safe_new_joint_violation', pair.get('reference_safe_new_joint_violation'))
            values = [100*metrics[k] for k in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress']]
            text += f"|{label}|{values[0]:.4f}|{values[1]:.4f}|{values[2]:.4f}|{values[3]:.4f}|{pair['nonzero_to_zero']}|{pair['zero_to_nonzero']}|{safety}|\n"
        text += '\nNC/DAC/TTC/EP按0—100报告；新增与恢复均相对同一个IL53k、同场景同seed。NC=0.5不算完全通过。各分项新增违规/恢复、G+/G−及均值变化见每个`*_paired_vs_IL53k.json`，低尾和成本见对应summary，不能只用零分净数量替代安全分析。\n\n'
        text += f"实际使用{cost['physical_GPUs']}个物理GPU、{cost['worker_slots']}个推理worker，端到端{cost['job_wall_seconds']/60:.2f}分钟；总逻辑评测{cost['logical_single_candidate_evaluations']}条、物理推理{cost['physical_inference_including_retries']}条。共享worker累计wall不等于独占GPU计算时间，详见[cost_summary.json](cost_summary.json)。\n\n"
        text += '四个RL权重均先通过原始图像、非零LoRA与合并原部署类的ODE一致性验证。模型源码434cba8与分析源码ee200bc固定；原始权重/轨迹/场景图保留在私有登记目录，不上传Git。训练未被停止，评测lease已关闭，原CPU调度器已恢复。\n'
        (report/'RESULTS.md').write_text(text)
        with (report/'README.md').open('a') as stream:
            stream.write('\n\n完整评测已完成，见[RESULTS.md](RESULTS.md)及聚合JSON。\n')
        subprocess.run(['git', 'add', '--', args.report_relative], cwd=repository, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=repository, check=True)
        subprocess.run(['git', 'commit', '-m', 'reports: complete five new snapshot Navtest evaluations with paired safety and costs'], cwd=repository, check=True)
        subprocess.run(['git', 'push', 'origin', branch], cwd=repository, check=True)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
        remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', branch], cwd=repository, text=True).split()[0]
        assert head == remote
        (control/'PUBLICATION_COMPLETE.json').write_text(json.dumps(dict(status='PUSHED_VERIFIED', branch=branch, sha=head,
                    at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), report=str(report)), indent=2)+'\n')
    except BaseException:
        if not args.validate_only:
            (control/'publication_failure.json').write_text(json.dumps(dict(error=traceback.format_exc(), automatic_retry=False), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
