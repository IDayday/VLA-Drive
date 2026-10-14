"""Summarize actual two-line curves; stochastic opportunity stays out of ODE tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from starVLA.rl.constrained_flow_noise.contracts import VARIANTS, file_sha, fingerprint, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from .lock_final import read, sealed


def rows(path):
    if not path.exists():
        return []
    # A live writer may be between bytes of the last JSON line. Retain only
    # complete lines and record completeness through the all-rank intersection.
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith('\n')]


def compute_accounting(collections, curve, monitor, identity):
    """Count completed logical work, including padding but not invented retries.

    A collection can be scored before either inner update finishes. Count its
    rollout once and count only the inner updates present on all eight ranks.
    These are velocity API invocations/examples, not CUDA kernel/FLOP counts.
    """
    result = dict(training_rollout_forward_calls=0, training_rollout_forward_examples=0,
                  frozen_reference_forward_calls=0, frozen_reference_forward_examples=0,
                  PG_retention_forward_calls=0, PG_retention_backward_calls=0,
                  PG_retention_forward_examples=0, PG_retention_backward_examples=0,
                  GT_replay_forward_calls=0, GT_replay_backward_calls=0,
                  GT_replay_forward_examples=0, GT_replay_backward_examples=0,
                  PG_scene_exposures_over_completed_inner_updates=0,
                  GT_scene_exposures_over_completed_inner_updates=0)
    for collection in collections.values():
        count = collection['velocity_evaluations']
        for source, target in [('rollout', 'training_rollout'), ('frozen_reference', 'frozen_reference')]:
            for unit in ('calls', 'examples'):
                result[target+'_forward_'+unit] += count[source+'_'+unit]
    seen = set()
    for update in curve:
        key = update['collection'], update['inner']
        if key in seen or update['collection'] not in collections:
            raise ValueError('Duplicate inner update or missing completed collection')
        seen.add(key)
        count = collections[key[0]]['velocity_evaluations']
        if not 1 <= key[1] <= count['inner_epochs']:
            raise ValueError('Inner update outside the registered collection')
        for source, target in [('PG_reference', 'PG_retention'), ('GT_replay', 'GT_replay')]:
            for direction in ('forward', 'backward'):
                result[target+'_'+direction+'_calls'] += count['per_inner_'+source+'_'+direction+'_calls']
                result[target+'_'+direction+'_examples'] += count['per_inner_'+source+'_examples']
        result['PG_scene_exposures_over_completed_inner_updates'] += update['accepted_groups']
        result['GT_scene_exposures_over_completed_inner_updates'] += count['per_inner_GT_replay_forward_calls']
    # Monitor uses the same G16, K10, chunk1 no-grad rollout, with no updates or
    # reference velocity calls. Keep this cost separate from the training stream.
    if identity['kernel']['rollout_chunk'] != 1:
        raise ValueError('Monitor call accounting requires the locked chunk1 layout')
    result['monitor_rollout_forward_examples'] = sum(x['candidates'] for x in monitor)*identity['kernel']['steps']
    result['monitor_rollout_forward_calls'] = result['monitor_rollout_forward_examples']
    result['canonical_training_candidate_scores'] = sum(x['candidates'] for x in collections.values())
    result['canonical_monitor_candidate_scores'] = sum(x['candidates'] for x in monitor)
    result['canonical_contract'] = 'One invocation per candidate: [fixed official reference, that candidate]'
    result['scope'] = ('Completed collection records and all-rank completed inner updates; actual padded calls included. '
                       'Repeated/partially failed attempts are not inferred from final logical records. '
                       'Their measured log time is reported separately; a retry requires its preserved attempt receipts.')
    result['excluded_one_time_graph_setup'] = dict(
        forward_calls_per_cold_rank=4, definition='3 eager warmups plus 1 CUDA graph capture',
        source='starVLA/rl/flow_grpo/velocity_graph.py:NoGradVelocityGraph',
        note='Add once per graph-creating rank/process invocation; not a candidate or training exposure.')
    return result


def summarize(run, qualification=False):
    run = Path(run)
    if not (run/'run_identity.json').exists():
        return dict(status='NOT_STARTED', path=str(run)), []
    identity = sealed(run/'run_identity.json')
    if identity['qualification_only'] != qualification:
        raise ValueError('Formal and qualification curves must not be mixed')
    logs = [rows(run/f'training_rank{rank}.jsonl') for rank in range(8)]
    indexed = [{x['update']: x for x in values} for values in logs]
    common = sorted(set.intersection(*(set(x) for x in indexed)))
    curve = []
    for update in common:
        values = [x[update] for x in indexed]
        shared = ('collection', 'inner', 'raw_exposure', 'accepted_exposure', 'accepted_groups', 'raw_groups', 'LR', 'lambda_snapshot', 'dual')
        if any(any(x[k] != values[0][k] for k in shared) for x in values):
            raise ValueError('Training log ranks disagree at update ' + str(update))
        item = {k: values[0][k] for k in ('update', 'collection', 'inner', 'raw_exposure', 'accepted_exposure', 'accepted_groups', 'raw_groups')}
        for key in ('PG', 'GT_replay', 'reference_mean_KL', 'reference_variance_KL', 'reference_KL_weighted', 'loss',
                    'ratio_mean', 'clip_fraction', 'sigma_mean', 'entropy_per_dimension', 'KL_coefficient', 'pre_clip_grad_norm'):
            entries = [x.get(key) for x in values]
            if all(x is None for x in entries):
                item[key] = None  # e.g. C_SDE entropy lives in the covariance-aware monitor.
            elif any(x is None for x in entries):
                raise ValueError('A metric is missing on only some ranks: ' + key)
            else:
                item[key] = float(np.mean(entries))
        item['LR_actor'] = values[0]['LR'][0]
        item['LR_noise'] = values[0]['LR'][1] if len(values[0]['LR']) > 1 else None
        item['update_wall_seconds_max_rank'] = max(x['seconds'] for x in values)
        item['update_GPU_seconds_sum_ranks'] = sum(x['seconds'] for x in values)
        for j, name in enumerate(('NC', 'DAC', 'TTC', 'road_margin', 'object_margin')):
            item['lambda_snapshot_'+name] = values[0]['lambda_snapshot'][j]
            item['dual_first16_'+name] = values[0]['dual']['cost_means'][j]
        curve.append(item)
    collection_map = {int(p.stem): read(p) for p in sorted((run/'collection_summaries').glob('*.json'))}
    collections = list(collection_map.values())
    monitor = [read(p) for p in sorted((run/'monitor').glob('raw*/summary.json')) if (p.parent/'COMPLETE').exists()]
    dev = []
    for path in sorted((run/'DEV').glob('raw*/report.json')):
        report = sealed(path)
        dev.append(dict(raw_exposure=int(path.parent.name.removeprefix('raw')), report_sha256=report['sha256'],
                        PDMS=100*report['metrics']['score']['current'],
                        PDMS_change_points=100*report['metrics']['score']['paired']['mean'],
                        NC=100*report['metrics']['no_at_fault_collisions']['current'],
                        DAC=100*report['metrics']['drivable_area_compliance']['current'],
                        TTC=100*report['metrics']['time_to_collision_within_bound']['current'],
                        EP=100*report['metrics']['ego_progress']['current'],
                        zeros=report['zeros'], safety=report['safety'], motion=report['motion'],
                        arrival=report['arrival'], safe_DEV_selection_eligible=report['safe_DEV_selection_eligible']))
    result_path = run/'result_rank0.json'
    result = read(result_path) if result_path.exists() else None
    totals = {k: sum(x[k] for x in collections) for k in
              ('scenes', 'candidates', 'accepted_groups', 'zero_PDMS_advantage_groups', 'zero_final_advantage_groups',
               'all_zero_PDMS_groups', 'progress_only_official_reward_groups', 'official_safety_cost_varies_groups',
               'geometric_cost_varies_groups', 'unused_effective_surplus_groups')}
    seconds = {k: sum(x['seconds'][k] for x in collections) for k in
               ('rollout_seconds', 'scoring_seconds', 'reference_seconds', 'scoring_cpu_seconds')}
    monitor_seconds = {k: sum(x['seconds'][k] for x in monitor) for k in seconds}
    count = totals['scenes']
    exposure_means = {k: sum(x[k]*x['scenes'] for x in collections)/count if count else None for k in
                      ('PDMS_mean', 'joint_safe_candidate_rate', 'pairwise_ADE_mean_m', 'request_second_difference_mean_m')}
    summary = dict(status=result['status'] if result else 'IN_PROGRESS', path=str(run),
                   variant=identity['variant'], run_identity=identity['sha256'], qualification_only=qualification,
                   original_SFT_sha256=identity['SFT_sha256'], condition_identity=identity['condition_identity'],
                   common_complete_rank_updates=len(common), latest_all_rank_update=common[-1] if common else 0,
                   replay_reference_only_updates=sum(x['accepted_groups'] == 0 for x in curve),
                   real_PG_updates=sum(x['accepted_groups'] > 0 for x in curve),
                   raw_exposure_at_latest_update=curve[-1]['raw_exposure'] if curve else 0,
                   accepted_exposure_at_latest_update=curve[-1]['accepted_exposure'] if curve else 0,
                   observed_collection_totals=totals, unfiltered_stream_exposure_weighted=exposure_means,
                   seconds=seconds, monitor_seconds=monitor_seconds,
                   compute_accounting=compute_accounting(collection_map, curve, monitor, identity),
                   logged_update_GPU_seconds_including_reexecuted_attempts=sum(x['seconds'] for rank in logs for x in rank),
                   duplicated_update_log_rows_by_rank=[len(x)-len(y) for x, y in zip(logs, indexed)],
                   time_accounting='component sums are worker measurements; GPU allocation time is reported separately from launcher receipts',
                   fixed_TRAIN_stochastic_monitor=monitor, original_single_ODE_DEV=dev,
                   stochastic_scope='G16 candidate statistics are exploration evidence only, never single-candidate ODE scores',
                   terminal_state=result)
    return summary, curve


def plot_curves(root, summaries, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for name, data in curves.items():
        if data:
            x = [r['raw_exposure'] for r in data]
            axes[0, 0].plot(x, [r['accepted_exposure']/r['raw_exposure'] for r in data], label=name)
            axes[0, 1].plot(x, [r['reference_KL_weighted'] for r in data], label=name)
            axes[0, 2].plot(x, [r['sigma_mean'] for r in data], label=name)
        monitor = summaries[name].get('fixed_TRAIN_stochastic_monitor', [])
        dev = summaries[name].get('original_single_ODE_DEV', [])
        if monitor:
            x = [m['behavior']['raw_exposure'] for m in monitor]
            axes[1, 0].plot(x, [100*m['PDMS_mean'] for m in monitor], 'o-', label=name)
            axes[1, 1].plot(x, [100*m['joint_safe_candidate_rate'] for m in monitor], 'o-', label=name)
        if dev:
            axes[1, 2].plot([r['raw_exposure'] for r in dev], [r['PDMS'] for r in dev], 'o-', label=name)
    titles = ['Accepted / raw exposure', 'Conditional reference KL proxy', 'Mean discrete transition std',
              'Fixed TRAIN stochastic G16 PDMS', 'Fixed TRAIN stochastic joint safety (%)', 'Full DEV original single ODE PDMS']
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel('Unfiltered raw scene exposure')
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend()
        else:
            axis.text(.5, .5, 'No completed observation yet', ha='center', va='center', transform=axis.transAxes)
    fig.savefig(root/'learning_curves.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--qualification-run', action='append', help='VARIANT=PATH; smoke only, explicitly excluded from formal results')
    args = parser.parse_args()
    require_dispatch()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Snapshots are immutable: ' + str(out))
    out.mkdir(parents=True)
    runs = ({x.split('=', 1)[0]: Path(x.split('=', 1)[1]) for x in args.qualification_run} if args.qualification_run else
            {v: Path(args.run_root)/f'formal_{v}_seed42_v1' for v in VARIANTS})
    summaries, curves = {}, {}
    for name, path in runs.items():
        if name not in VARIANTS:
            raise ValueError('Only the two registered methods are supported')
        summaries[name], curves[name] = summarize(path, qualification=bool(args.qualification_run))
        if curves[name]:
            with (out/(name+'_updates.csv')).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(curves[name][0]))
                writer.writeheader()
                writer.writerows(curves[name])
    summary = dict(schema='cfn_two_line_progress_v1', variants=summaries, training_seed=42, inference_seed=2101,
                   qualification_only=bool(args.qualification_run), source_sha256=file_sha(__file__),
                   completed_formal_comparison=not args.qualification_run and all(
                       x['status'] == 'COMPLETE_FIXED_RAW_BUDGET' and len(x.get('original_single_ODE_DEV', [])) == 4 for x in summaries.values()))
    summary['sha256'] = fingerprint(summary)
    atomic_json(out/'summary.json', summary)
    if args.plot:
        plot_curves(out, summaries, curves)
    print(json.dumps(dict(status='SNAPSHOT_SAVED', completed_formal_comparison=summary['completed_formal_comparison'], output=str(out))))


if __name__ == '__main__':
    main()
