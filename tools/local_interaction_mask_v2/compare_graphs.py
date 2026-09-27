"""Compare all-hidden graph trajectories on identical matched tracks, with full coverage counts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import torch
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow
from starVLA.model.modules.joint_world.local_masks import stable_noise
from tools.local_interaction_mask_v2.graph_runtime import LocalCorpus, batch_current, trajectory_metrics
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.public_baseline import sha256


def common_track_slots(first, second):
    def mapping(sample):
        records = [row for row in sample['mapping']['assignments'] if row['used_for_local_loss']]
        result = {row['track_id']: row['local_slot'] for row in records}
        if len(result) != len(records): raise ValueError('Repeated matched track identity')
        return result
    a, b = mapping(first), mapping(second); tracks = sorted(set(a) & set(b))
    return tracks, [0] + [a[track] for track in tracks], [0] + [b[track] for track in tracks]


def load_model(path, corpus):
    saved = torch.load(path, map_location='cpu', weights_only=False); identity = saved['identity']
    if identity['cache_identity'] != corpus.manifest['identity_sha256']: raise ValueError('Graph/current identity differs')
    config = identity['config']
    model = JointTrajectoryFlow(corpus[0]['current']['context'].shape[-1], **{key: value for key, value in config.items()
        if key in ('dim', 'heads', 'layers', 'steps', 'scale_m', 'agent_scale_m', 'trajectory_mode', 'edge_feature_dim')}).cuda()
    model.load_state_dict(saved['model'], strict=True); model.eval().requires_grad_(False)
    return model, config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('first-checkpoint', 'baseline-checkpoint', 'first-cache', 'baseline-cache', 'targets', 'meta-root', 'index', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--same-graph', action='store_true', help='Required for the primary ALL/MASK pair')
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    corpora = [LocalCorpus(cache, args.targets, args.meta_root) for cache in (args.first_cache, args.baseline_cache)]
    index = json.loads(Path(args.index).read_text()); logs = {row['token']: row['log'] for row in index}
    if len(logs) != len(index): raise ValueError('Duplicate comparison index')
    for corpus in corpora:
        if {row['token'] for row in corpus.records} != set(logs): raise ValueError('Graph comparison population differs')
    lookups = [{row['token']: i for i, row in enumerate(corpus.records)} for corpus in corpora]
    models, configs = zip(*(load_model(path, corpus) for path, corpus in zip(
        (args.first_checkpoint, args.baseline_checkpoint), corpora)))
    if configs[0] != configs[1]: raise ValueError('Paired graph sampling/model configurations differ')
    if args.same_graph and corpora[0].manifest['identity_sha256'] != corpora[1].manifest['identity_sha256']:
        raise ValueError('Primary comparison must use exactly the same frozen graph rules')
    rows = []
    for token, log in logs.items():
        row = {'token': token, 'log': log, 'status': 'ok'}
        try:
            samples = [corpus[lookup[token]] for corpus, lookup in zip(corpora, lookups)]
            if args.same_graph:
                for key in ('source_slot_ids', 'active_actor_mask', 'predictable_actor_mask', 'edge_mask', 'edge_features'):
                    if not torch.equal(samples[0]['payload']['local_graph'][key], samples[1]['payload']['local_graph'][key]):
                        raise ValueError('Primary paired current graphs differ')
            tracks, first_slots, base_slots = common_track_slots(*samples)
            positions = [first_slots, base_slots]
            truths = [sample['xy'][:, slots] for sample, slots in zip(samples, positions)]
            validity = [sample['valid'][:, slots] for sample, slots in zip(samples, positions)]
            if not torch.equal(validity[0], validity[1]) or not torch.equal(truths[0][validity[0]], truths[1][validity[1]]):
                raise ValueError('Common track future labels/validity differ')
            if not torch.equal(samples[0]['target_current_xy'][:, first_slots], samples[1]['target_current_xy'][:, base_slots]):
                raise ValueError('Common track current label anchors differ')
            row['common_matched_tracks'] = len(tracks)
            row['common_neighbor_valid_points'] = int(validity[0][:, 1:].sum())
            row['full_current_roi_targets'] = samples[0]['mapping']['full_current_gt']
            row['full_current_future_points'] = samples[0]['mapping']['full_valid_future_points']
            for name, model, sample, slots in zip(('first', 'baseline'), models, samples, positions):
                current = batch_current([sample]); graph = current['local_graph']
                noise = stable_noise([token], graph.source_slot_ids, model.steps, configs[0].get('sampling_seed', 2037), device='cuda')
                prediction, _ = model.sample(noise, **current, sampling_steps=configs[0].get('sampling_steps', 10))
                values = trajectory_metrics(prediction[:, slots], sample['xy'][:, slots].cuda(), sample['valid'][:, slots].cuda(),
                    current['current_xy'][:, slots], sample['target_current_xy'][:, slots].cuda())
                row.update({name + '_' + key: value for key, value in values.items()})
                row[name + '_active_neighbors'] = int(graph.active_actor_mask[:, 1:].sum())
                row[name + '_matched_neighbors'] = sample['mapping']['selected_with_accepted_association']
        except Exception as error: row.update(status='failed', error=repr(error))
        rows.append(row)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (output / 'scenes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, keys); writer.writeheader(); writer.writerows(rows)
    failed = sum(row['status'] != 'ok' for row in rows)
    summary = {'scenes': len(rows), 'logs': len(set(logs.values())), 'failed': failed, 'valid': failed == 0,
        'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': [sha256(path) for path in (args.first_checkpoint, args.baseline_checkpoint)],
        'current_manifest_sha256': [sha256(Path(path)/'manifest.json') for path in (args.first_cache, args.baseline_cache)],
        'label_fingerprints': [corpus.label_fingerprint for corpus in corpora],
        'same_graph_required': args.same_graph, 'common_tracks': sum(row.get('common_matched_tracks', 0) for row in rows),
        'interpretation': 'Both models evaluated on the same matched GT track/time cohort; coverage is reported separately, not full-target accuracy',
        'conditioning': 'current inputs + all-hidden source-stable noise, no GT future in either sampling forward', 'metrics': {}}
    for actor in ('ego', 'agents'):
        for group in ('', '_dynamic', '_static'):
            prefix = actor + group; count = sum(row.get('first_' + prefix + '_valid_points', 0) for row in rows)
            fde_count = sum(row.get('first_' + prefix + '_fde_count', 0) for row in rows)
            values = {'valid_points': count, 'FDE_targets': fde_count}
            for name in ('first', 'baseline'):
                values[name + '_ADE_m'] = sum(row.get(name + '_' + prefix + '_error_sum', 0) for row in rows)/count if count else None
                values[name + '_FDE_m'] = sum(row.get(name + '_' + prefix + '_fde_sum', 0) for row in rows)/fde_count if fde_count else None
            values['paired_ADE_delta_m'] = values['first_ADE_m']-values['baseline_ADE_m'] if count else None
            summary['metrics'][prefix] = values
    atomic_json(output / 'summary.json', summary); atomic_json(output / 'status.json', {'status': 'failed' if failed else 'complete'})
    if failed: raise RuntimeError('Comparison failures retained; do not interpret partial metrics as complete')


if __name__ == '__main__': main()
