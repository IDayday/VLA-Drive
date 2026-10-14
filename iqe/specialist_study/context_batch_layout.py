"""Replay an attested original microbatch when regrouping changes AMP numerics.

Only the cache builder uses this fallback. The deployment path remains one shared
forward, and no old W/A, Base prediction, tolerance, or feature identity changes.
"""
from dataclasses import asdict
from pathlib import Path

from iqe.contracts import require
from iqe.data.manifests import load_scenes
from iqe.data.sources import load_observation
from iqe.io import read_json, digest, file_hash


TOLERANCE = 1e-5


def errors(scene_id, raw, values, prior):
    return {'scene_id': scene_id,
            'raw_max_abs': float((raw.detach().cpu()-prior['base_raw']).abs().max()),
            'world_max_abs': float((values['world'].detach().cpu()-prior['world']).abs().max()),
            'action_max_abs': float((values['action'].detach().cpu()-prior['action']).abs().max())}


def within_tolerance(row):
    return all(row[k] <= TOLERANCE for k in ('raw_max_abs', 'world_max_abs', 'action_max_abs'))


def original_batch_map(plan):
    require(plan['identity'] == digest({k: v for k, v in plan.items() if k != 'identity'}),
            'original context batch plan changed')
    require(plan['batch_composition_unchanged'], 'original layout is not attested')
    result = {}
    for task in plan['tasks']:
        for batch in task['batches']:
            ids = tuple(batch['ids'])
            require(0 < len(ids) <= plan['cache_batch'] and len(set(ids)) == len(ids),
                    'invalid original batch')
            for sid in ids:
                require(sid not in result, 'duplicate original batch ownership')
                result[sid] = ids
    return result


def initial_shard_batch_map(scene_ids, cache_batch, launches, completions):
    """Reconstruct only an attested *first*, complete cache extraction.

    The legacy builder sliced the manifest by rank, removed cached records, then
    grouped by cache_batch. Therefore new == scenes for every rank is essential:
    a resumed/partial extraction cannot establish the original AMP batch layout.
    """
    require(scene_ids and len(set(scene_ids)) == len(scene_ids), 'invalid initial scene order')
    require(type(cache_batch) is int and cache_batch > 0, 'invalid initial cache batch')
    topology, owners, config_paths = None, set(), set()
    for worker in launches:
        command = worker['command']
        require(command[1:4] == ['-m', 'iqe.internal_baseline.cli', 'cache-features'],
                'unrecognized original cache launch')
        args = command[4:]
        require(len(args) == 6 and set(args[::2]) == {'--config', '--shard', '--shards'},
                'original cache launch has partial/unknown options')
        options = dict(zip(args[::2], args[1::2]))
        shard, shards = int(options['--shard']), int(options['--shards'])
        require(0 <= shard < shards <= 8 and worker['shard'] == shard,
                'invalid original cache shard')
        require(topology is None or topology == shards, 'mixed original shard topology')
        topology = shards
        require(shard not in owners, 'duplicate original cache launch')
        owners.add(shard); config_paths.add(options['--config'])
    require(topology is not None and owners == set(range(topology))
            and set(completions) == owners and len(config_paths) == 1,
            'incomplete original cache launch/completion receipts')
    result = {}
    for shard in range(topology):
        assigned = scene_ids[shard::topology]
        receipt = completions[shard]
        require(receipt['status'] == 'COMPLETE' and receipt.get('new') == len(assigned)
                and receipt.get('scenes') == len(assigned),
                'original extraction was partial or reused records; layout is not identifiable')
        for at in range(0, len(assigned), cache_batch):
            batch = tuple(assigned[at:at+cache_batch])
            for sid in batch: result[sid] = batch
    return result


class OriginalBatchReplay:
    def __init__(self, config):
        path = Path(config.reuse_root)/'runtime/context_pool_v1/PLAN.json'
        self.plan = read_json(path)
        self.batches = original_batch_map(self.plan)
        binding = read_json(config.root/'BASE_BINDING.json')
        require(self.plan['Base_hash'] == binding['checkpoint_hash'], 'original layout uses another Base')
        # This includes the original observation manifest and numerical feature
        # code; neither path existence nor an unverified batch-size guess suffice.
        for dependency, checksum in self.plan['inputs'].items():
            require(file_hash(dependency) == checksum, 'original batch dependency changed: '+dependency)
        self.scenes = {s.scene_id: s for s in load_scenes(Path(config.reuse_root)/'scenes.json')}
        self.plan_hash = file_hash(path)
        self.provenance = {'primary_plan': str(path), 'primary_plan_hash': self.plan_hash,
                           'primary_scenes': len(self.batches)}
        self.reused_records = {}
        self.verified_reused = set()
        if self.plan.get('reused_scenes', 0):
            self._add_inherited_layout(Path(config.reuse_root), binding)

    def _add_inherited_layout(self, destination, current_binding):
        """Recover inherited batches from the content-bound import and receipts."""
        reuse_path = destination/'CONTEXT_REUSE.json'
        require(str(reuse_path) in self.plan['inputs'], 'reuse report was not bound by original plan')
        report = read_json(reuse_path)
        require(report['status'] == 'COMPLETE' and report['schema'] == 'verified_joint_context_reuse_v1'
                and Path(report['destination_root']) == destination, 'invalid context reuse provenance')
        source = Path(report['source_root']); inputs = report['inputs']
        paths = {'source_binding': source/'BASE_BINDING.json',
                 'destination_binding': destination/'BASE_BINDING.json',
                 'source_manifest': source/'scenes.json',
                 'destination_manifest': destination/'scenes.json'}
        for name, dependency in paths.items():
            require(file_hash(dependency) == inputs[name], 'inherited context dependency changed: '+name)
        from iqe.internal_baseline import features
        require(file_hash(features.__file__) == inputs['feature_code'], 'inherited feature implementation changed')
        source_binding = read_json(paths['source_binding'])
        destination_binding = read_json(paths['destination_binding'])
        for name in ('checkpoint_hash', 'contract_hash', 'inference_runtime'):
            require(source_binding[name] == destination_binding[name] == current_binding[name],
                    'inherited layout Base/runtime changed: '+name)
        source_config = read_json(source/'resolved_config.json')
        require(digest(source_config) == source_binding['config_hash']
                and Path(source_config['output_root']) == source, 'inherited resolved config changed')
        source_scenes = load_scenes(source/'scenes.json')
        records = report['records']; ids = [r['scene_id'] for r in records]
        require(len(set(ids)) == len(ids) == report['count'] == self.plan['reused_scenes']
                and digest(sorted(ids)) == inputs['scene_ids_hash']
                and set(ids) == {s.scene_id for s in source_scenes}, 'inherited scene coverage changed')
        require(set(ids).isdisjoint(self.batches), 'new and inherited batch ownership overlap')
        self.reused_records = {r['scene_id']: r for r in records}
        for scene in source_scenes:
            require(scene.scene_id in self.scenes
                    and asdict(scene) == asdict(self.scenes[scene.scene_id])
                    and digest(asdict(scene)) == self.reused_records[scene.scene_id]['scene_hash'],
                    'inherited observation/target/provenance changed: '+scene.scene_id)
        launch_paths = sorted((source/'runtime').glob('*_CACHE_LAUNCH.json'))
        launches = [worker for p in launch_paths for worker in read_json(p)['workers']]
        complete_paths = sorted(source.glob('cache_worker_*.json'))
        completions = {int(p.stem.removeprefix('cache_worker_')): read_json(p) for p in complete_paths}
        inherited = initial_shard_batch_map([s.scene_id for s in source_scenes],
                                           source_config['cache_batch'], launches, completions)
        self.batches.update(inherited)
        require(set(self.batches) == set(self.scenes), 'original context layouts do not cover the manifest')
        self.reused_source, self.reused_destination = source, destination
        receipt_paths = [reuse_path, source/'resolved_config.json', *paths.values(), *launch_paths, *complete_paths]
        self.provenance['inherited'] = {'source_root': str(source), 'scenes': len(inherited),
            'layout': 'original_manifest_stride_then_cache_batch; all first-pass workers new==scenes',
            'cache_batch': source_config['cache_batch'], 'shards': len(completions),
            'receipts': {str(p): file_hash(p) for p in receipt_paths}}
        self.provenance['layout_hash'] = digest(self.batches)

    def batch_for(self, scene_id):
        require(scene_id in self.batches, 'scene has no attested original batch: '+scene_id)
        batch = self.batches[scene_id]
        # Validate only accessed payloads, not all 30 GB of the inherited cache.
        for sid in batch:
            if sid not in self.reused_records or sid in self.verified_reused: continue
            record = self.reused_records[sid]
            for root in (self.reused_source, self.reused_destination):
                path = root/'features'/(sid+'.pt')
                receipt = read_json(str(path)+'.COMPLETE.json')
                require(receipt['scene_id'] == sid and receipt['key'] == record['key']
                        and receipt['checksum'] == record['checksum']
                        and path.stat().st_size == record['bytes'] and file_hash(path) == record['checksum'],
                        'inherited accessed payload/receipt changed: '+sid)
            self.verified_reused.add(sid)
        return batch

    def replay(self, scene, adapter, capture, prior, rejected):
        ids = self.batch_for(scene.scene_id)
        rows = [self.scenes[sid] for sid in ids]
        require(all(s.split_role != 'final_test' for s in rows), 'final_test cannot enter cache replay')
        require(self.scenes[scene.scene_id].observation_hash == scene.observation_hash,
                'original batch observation changed')
        planner, values = capture.encode([load_observation(s, adapter.contract) for s in rows])
        prediction = adapter.predict_s0(planner)
        index = ids.index(scene.scene_id)
        item = {k: v[index] for k, v in values.items()}
        actual = errors(scene.scene_id, prediction.raw[index], item, prior)
        evidence = {'scene_id': scene.scene_id, 'original_scene_ids': list(ids),
                    'original_plan_hash': self.plan_hash, 'layout_provenance': self.provenance,
                    'regrouped_errors': rejected,
                    'original_batch_errors': actual, 'tolerance': TOLERANCE,
                    'old_context_modified': False, 'capture_code_modified': False,
                    'status': 'PASS' if within_tolerance(actual) else 'FAIL'}
        return item, actual, evidence
