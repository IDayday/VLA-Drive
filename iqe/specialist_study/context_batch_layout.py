"""Replay an attested original microbatch when regrouping changes AMP numerics.

Only the cache builder uses this fallback. The deployment path remains one shared
forward, and no old W/A, Base prediction, tolerance, or feature identity changes.
"""
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

    def replay(self, scene, adapter, capture, prior, rejected):
        require(scene.scene_id in self.batches, 'drifted scene has no attested original batch')
        ids = self.batches[scene.scene_id]
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
                    'original_plan_hash': self.plan_hash, 'regrouped_errors': rejected,
                    'original_batch_errors': actual, 'tolerance': TOLERANCE,
                    'old_context_modified': False, 'capture_code_modified': False,
                    'status': 'PASS' if within_tolerance(actual) else 'FAIL'}
        return item, actual, evidence
