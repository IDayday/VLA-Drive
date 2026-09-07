#!/usr/bin/env python3
"""Read-only postmortem: locked banks, training logs and saved gradient audits.

No model inference, official metric implementation, training or label sampling.
Outputs are descriptive evidence, not hyperparameter/epoch selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic; do not silently drop it')
    return dict(count=len(values), mean=float(values.mean()),
                p10=float(np.quantile(values, .1)), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def token_statistics(values):
    """Centered energy rank plus amplitude; rank alone is not diversity proof."""
    x = np.asarray(values, dtype=np.float64)
    centered = x - x.mean(1, keepdims=True)
    gram = centered @ centered.swapaxes(-1, -2)
    eigen = np.linalg.eigvalsh(gram).clip(0)
    total = eigen.sum(-1)
    prob = eigen / total[:, None].clip(1e-30)
    rank = np.exp(-(prob * np.log(prob.clip(1e-30))).sum(-1))
    rank[total < 1e-24] = 0
    normalized = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-30)
    cosine = normalized @ normalized.swapaxes(-1, -2)
    mask = ~np.eye(x.shape[1], dtype=bool)
    content = x - x.mean(0, keepdims=True)
    content_centered = content - content.mean(1, keepdims=True)
    return dict(scene_count=len(x), token_count=x.shape[1],
                raw_rms=float(np.sqrt(np.mean(x*x))),
                slot_centered_rms=float(np.sqrt(np.mean(centered*centered))),
                slot_centered_to_raw_rms=float(np.sqrt(np.mean(centered*centered)/np.mean(x*x))) if np.any(x) else 0.,
                cross_scene_content_rms=float(np.sqrt(np.mean(content*content))),
                cross_scene_slot_content_rms=float(np.sqrt(np.mean(content_centered*content_centered))),
                mean_token_cosine_matrix=cosine.mean(0).tolist(),
                mean_pairwise_cosine=float(cosine[:, mask].mean()),
                centered_energy_effective_rank=distribution(rank),
                largest_centered_energy_fraction=distribution(prob[:, -1]))


def shared_initialization_audit(old_path, lite_path):
    import torch
    states = [torch.load(p, map_location='cpu', weights_only=False, mmap=True)['trainable_state_dict']
              for p in (old_path, lite_path)]
    old, lite = states
    groups = {}
    for key in sorted(old.keys() & lite.keys()):
        if old[key].shape != lite[key].shape:
            continue
        name = '.'.join(key.split('.')[:2])
        group = groups.setdefault(name, dict(common_same_shape_tensors=0, equal_after_float32_cast=0,
                                             common_parameter_values=0))
        group['common_same_shape_tensors'] += 1
        group['equal_after_float32_cast'] += int(torch.equal(old[key].float(), lite[key].float()))
        group['common_parameter_values'] += old[key].numel()
    return dict(old_path=str(old_path), old_sha256=sha256(old_path), lite_path=str(lite_path),
                lite_sha256=sha256(lite_path), groups=groups,
                note='Comparisons across methods, not within the paired Base/VQA initializer; constant zeros/norms may be equal.')


def train_events(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(run/'lightning_logs/version_0'), size_guidance={'scalars': 0}).Reload()
    records, curves = {}, []
    for tag in accumulator.Tags()['scalars']:
        events = accumulator.Scalars(tag)
        # Retain the last write at a step for deterministic resumed logging.
        events = {int(e.step): e for e in events}
        points = sorted(events.items())
        per_epoch = {}
        for step, event in points:
            epoch = step // 807 + 1
            per_epoch.setdefault(epoch, []).append(float(event.value))
        epoch_rows = {str(k): dict(n=len(v), mean=float(np.mean(v))) for k, v in per_epoch.items()}
        records[tag] = dict(count=len(points), first_step=points[0][0], last_step=points[-1][0],
                            first=float(points[0][1].value), last=float(points[-1][1].value),
                            final_1000_step_mean=float(np.mean([e.value for s, e in points if s >= 20789])) if points[-1][0] >= 20789 else None,
                            epochs=epoch_rows)
        curves.extend(dict(tag=tag, epoch=int(k), **v) for k, v in epoch_rows.items())
    return records, curves


def gradient_audit(run):
    rows, hashes = [], {}
    for path in sorted((run/'run_metadata').glob('same_batch_gradients_rank*.jsonl')):
        hashes[str(path)] = sha256(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['scope'] != 'same_batch_rank_local_unclipped':
                raise ValueError('Unexpected gradient reduction scope')
            for name, group in row['groups'].items():
                rows.append(dict(step=row['optimizer_step'], rank=row['rank'], group=name, **group))
    frame = pd.DataFrame(rows)
    results = {}
    for label, lower, upper in [('all', 0, 21789), ('late', 16000, 21789), ('ramp', 0, 2179)]:
        selected = frame[(frame.step >= lower) & (frame.step < upper)]
        results[label] = {}
        for group, data in selected.groupby('group'):
            valid = data[data.cosine_defined]
            results[label][group] = dict(
                rank_batch_count=len(data), steps=int(data.step.nunique()), ranks=int(data['rank'].nunique()),
                wm_to_plan_ratio=distribution(data.weighted_wm_to_plan_ratio),
                cosine=distribution(valid.cosine), negative_cosine_fraction=float((valid.cosine < 0).mean()),
                cosine_below_minus_point2_fraction=float((valid.cosine < -.2).mean()))
    return dict(scope='distribution_of_same_batch_rank_local_unclipped_comparisons_not_global_gradient_cosine',
                files=hashes, windows=results), frame


def selected_calibration(bank, audit):
    """Observed selected/oracle factors only; no invented all-candidate labels."""
    frame = pd.read_csv(audit/'per_scene_candidate_quality.csv').set_index('token').loc[bank['tokens'].astype(str)]
    names = bank['component_names'].astype(str)
    indices = bank['selected_indices']
    p = bank['component_probabilities'][np.arange(len(indices)), indices]
    result = {}
    for i, name in enumerate(names):
        target = frame['selected_'+name].to_numpy().copy()
        valid = target != 2 if name == 'time_to_collision_within_bound' else np.ones(len(target), dtype=bool)
        if name in ('no_at_fault_collisions', 'driving_direction_compliance'):
            target[target == .5] = 0
        probability = p[:, i]
        error = probability[valid]-target[valid]
        failures = valid & (target < .5)
        result[name] = dict(valid_count=int(valid.sum()), mean_prediction=float(probability[valid].mean()),
                            mean_training_mapped_target=float(target[valid].mean()),
                            mae=float(abs(error).mean()), squared_error=float((error*error).mean()),
                            failing_selected_count=int(failures.sum()),
                            predicted_probability_on_failing_selected=float(probability[failures].mean()) if failures.any() else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-run', 'lite-run', 'old-bank', 'lite-bank', 'evaluation-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite postmortem artifacts')
    args.output.mkdir(parents=True)
    result = dict(schema_version=1, evidence_type='posthoc_read_only_saved_artifact_analysis',
                  no_training=True, no_navtest_parameter_selection=True, runs={})
    curves = []
    for name, run, bank_path, audit_name in [('old', args.old_run, args.old_bank, 'old_epoch27'),
                                          ('lite', args.lite_run, args.lite_bank, 'lite_epoch27')]:
        print('Reading', name, flush=True)
        cfg_path = run/'run_metadata/resolved_hydra_config.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        event_records, event_curves = train_events(run)
        curves.extend(dict(run=name, **row) for row in event_curves)
        with np.load(bank_path, allow_pickle=False) as source:
            bank = {key: source[key] for key in source.files if key != 'proposals'}
        registers = bank['planning_registers']
        token_stats = {'all16': token_statistics(registers)}
        if name == 'lite':
            token_stats.update(global8=token_statistics(registers[:, :8]), local8=token_statistics(registers[:, 8:]))
            total = np.square(registers-registers.mean(1, keepdims=True)).sum()
            within = (np.square(registers[:, :8]-registers[:, :8].mean(1, keepdims=True)).sum()
                      + np.square(registers[:, 8:]-registers[:, 8:].mean(1, keepdims=True)).sum())
            token_stats['within_group_share_of_slot_variance'] = float(within/total)
        meta = run/'run_metadata'
        result['runs'][name] = dict(
            run=str(run), config_sha256=sha256(cfg_path), bank=str(bank_path), bank_sha256=sha256(bank_path),
            vlm=cfg['agent']['vlm_config'], world_model=cfg['agent']['world_model'],
            action_head=cfg['agent']['action_head_config'], loss=cfg['agent']['loss'],
            planning_registers_config=cfg['agent'].get('planning_registers'),
            scene_fusion=cfg['agent']['scene_fusion'], initialization=cfg['agent']['initialization'],
            optimizer=json.loads((meta/'training_parameter_audit.json').read_text()),
            data_protocol=json.loads((meta/'train_val_protocol.json').read_text()),
            step_budget=json.loads((meta/'formal_step_budget.json').read_text()),
            logged_training_metrics=event_records,
            event_sources={str(p): sha256(p) for p in (run/'lightning_logs/version_0').glob('events*')},
            readout=token_stats, semantic_gate=distribution(bank['semantic_gate']),
            selected_component_calibration=selected_calibration(bank, args.evaluation_root/audit_name/'scoring'))
        if name == 'lite':
            precision = json.loads((meta/'PRECISION_CONTRACT.json').read_text())
            result['runs'][name]['precision'] = {k: v for k, v in precision.items() if k != 'updates'}
    gradients, gradient_rows = gradient_audit(args.lite_run)
    result['same_batch_gradients'] = gradients
    result['shared_initialization_comparison'] = shared_initialization_audit(
        result['runs']['old']['initialization']['shared_trainable_init_path'],
        result['runs']['lite']['initialization']['shared_trainable_init_path'])
    pd.DataFrame(curves).to_csv(args.output/'logged_epoch_curves.csv', index=False)
    gradient_rows.to_csv(args.output/'same_batch_gradient_observations.csv', index=False)
    # Plot in English for portable font rendering; main interpretation is Chinese.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(curves)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, tag, title in zip(axes.flat,
            ['train/trajectory_loss','train/final_score_loss','train/score',
             'train/noc_loss','train/ttc_loss','train/best_score'],
            ['Trajectory loss','Six-head scorer BCE','Logged train selected PDMS',
             'NC BCE','TTC BCE','Logged train Oracle@64']):
        for name in ('old', 'lite'):
            sub = frame[(frame.run == name) & (frame.tag == tag)]
            ax.plot(sub.epoch, sub['mean'], label=name)
        ax.set_title(title); ax.set_xlabel('Dataset epoch'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Means of logged rank-0 train batches, NOT validation; B/rank old=8, Lite=4')
    fig.savefig(args.output/'training_curves.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for group in ('vision_qv_lora', 'planning_registers', 'planning_readout', 'readout_output'):
        sub = gradient_rows[gradient_rows.group == group]
        for ax, key in zip(axes, ['weighted_wm_to_plan_ratio','cosine']):
            summary = sub.groupby('step')[key].median()
            ax.plot(summary.index, summary.values, label=group)
    axes[0].set_ylabel('Median rank-local weighted WM / plan gradient norm')
    axes[1].set_ylabel('Median rank-local cosine, before clipping'); axes[1].axhline(0, color='gray', lw=.8)
    for ax in axes: ax.set_xlabel('Optimizer step'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(args.output/'same_batch_gradients.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, name in zip(axes[:2], ('old', 'lite')):
        matrix = result['runs'][name]['readout']['all16']['mean_token_cosine_matrix']
        heat = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_title(name+' mean cosine: 12,146 scenes')
        ax.set_xlabel('Planning slot'); ax.set_ylabel('Planning slot')
        fig.colorbar(heat, ax=ax, shrink=.75)
    for name in ('old', 'lite'):
        sub = frame[(frame.run == name) & (frame.tag == 'train/register_effective_rank')]
        axes[2].plot(sub.epoch, sub['mean'], label=name)
    axes[2].set_title('Centered energy rank: logged training')
    axes[2].set_xlabel('Dataset epoch'); axes[2].set_ylabel('Effective rank'); axes[2].legend(); axes[2].grid(alpha=.2)
    fig.savefig(args.output/'readout_collapse.png', dpi=170); plt.close(fig)
    result['script_sha256'] = sha256(__file__)
    (args.output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
