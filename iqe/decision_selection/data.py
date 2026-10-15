"""Strict existing-bank joins and fit-only log partitioning; no new scoring."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
import math
import numpy as np
import torch
from iqe.contracts import require, CandidateRecord, strict_record
from iqe.io import read_json, atomic_json, digest, file_hash
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.data import ReusedContext
from .models import CONTEXT_KEYS

ROLES = ('scorer_fit', 'stage_val', 'selector_cal', 'dev_report')


def canonical_bank(config, role, *, verify_records=False):
    require(role in ROLES, 'final_test/navtest are forbidden')
    source = config.source()
    meta, arrays = load_bank(source, 'deploy', role)
    scenes = scenes_by_role(source)[role]
    registry = read_json(source.root/'SERVING_CANDIDATE_POOL.json')
    ids = ['expert_0']+sorted(i for i in registry['expert_ids'] if i != 'expert_0')
    original = meta['provenance']['sources']
    require(set(ids) == set(original) and len(ids) == 4, 'fixed K4 identity')
    require(meta['scene_ids'] == [s.scene_id for s in scenes] and meta['source_logs'] == [s.source_log_id for s in scenes], 'scene/log order mismatch')
    require(meta['score_scale'] == 'zero_one' and meta['invalid_prediction_count'] == 0, 'complete quality/trajectory bank')
    order = [original.index(i) for i in ids]
    data = {k: v[:, order].copy() for k, v in arrays.items()}
    require(data['valid'].all() and np.isfinite(data['scores']).all(), 'missing labels cannot become zero')
    require(((data['scores'] >= 0) & (data['scores'] <= 1)).all(), 'official [0,1] scores')
    require(np.isfinite(data['trajectories']).all(), 'invalid deployed candidate')
    require(data['trajectories'].shape[:2] == data['scores'].shape, 'candidate shape')
    if verify_records:
        base = read_json(source.root/'BASE_BINDING.json')
        hashes = {'expert_0': base['checkpoint_hash']} | {r['expert_id']:r['model_hash'] for r in registry['experts']}
        for n, scene in enumerate(scenes):
            require(scene.split_role == ('incremental_fit' if role == 'scorer_fit' else role), 'source role leakage')
            require(len(meta['records'][n]) == 4, 'incomplete candidate row')
            references = set()
            for j, index in enumerate(order):
                row = meta['records'][n][index]; candidate = strict_record(CandidateRecord, row['candidate']); label = row['label']
                require(candidate.scene_id == scene.scene_id and candidate.expert_id == ids[j], 'candidate ID join')
                require(candidate.expert_checkpoint_hash == hashes[ids[j]] and candidate.feature_contract_hash == base['contract_hash'], 'generator/feature identity')
                require(candidate.finite_valid and label['score_valid'] and label['candidate_key'] == candidate.key, 'invalid/mismatched label')
                require(label['metric_context_hash'] == scene.metric_context_hash and label['score_protocol_hash'] == meta['score_protocol_hash'], 'score context/protocol join')
                require(abs(label['total_score_01']-float(data['scores'][n,j])) <= 6e-8, 'bank score serialization changed')
                references.add(label['official_reference_hash'])
                for c, name in enumerate(meta['components']):
                    valid = label['component_valid_masks'].get(name, False)
                    require(bool(data['component_valid'][n,j,c]) == valid, 'component mask changed')
                    if valid: require(abs(label['named_components'][name]-float(data['components'][n,j,c])) <= 6e-8, 'component label changed')
            require(len(references) == 1, 'official reference varies across candidates')
    # Preserve the score mask separately. Never feed this field to either model.
    meta = {k:v for k,v in meta.items() if k != 'records'}
    meta.update(expert_ids=ids, canonical_order=order)
    return meta, data, scenes


def split_components(scenes, seed, fraction=.8):
    """Union log/group/parent identities before stable hash partitioning."""
    parent = {}
    def find(key):
        parent.setdefault(key,key)
        if parent[key] != key: parent[key] = find(parent[key])
        return parent[key]
    def keys(s):
        # Groups and parents share a namespace, linking derived observations.
        return [('log',s.source_log_id), ('group',s.source_group_id)] + ([('group',s.parent_group_id)] if s.parent_group_id else [])
    for s in scenes:
        ks = keys(s)
        for k in ks[1:]: parent[find(k)] = find(ks[0])
    sets = {}
    for s in scenes: sets.setdefault(find(keys(s)[0]), []).append(s)
    groups = sorted(sets.values(), key=lambda rows: digest([seed, sorted({x.source_log_id for x in rows}), sorted({x.source_group_id for x in rows})]))
    count = max(1, min(len(groups)-1, round(fraction*len(groups))))
    require(len(groups) >= 2, 'at least two independent groups required')
    fit = {s.scene_id for g in groups[:count] for s in g}
    val = {s.scene_id for g in groups[count:] for s in g}
    require(fit and val and not fit & val, 'empty/overlapping internal split')
    return fit, val


def split_inner(config):
    source = config.source(); scenes = scenes_by_role(source)['scorer_fit']
    require(all(s.split_role == 'incremental_fit' for s in scenes), 'internal split from D_S only')
    fit, val = split_components(scenes, config.partition_seed)
    maximum = math.ceil(config.diagnostic_updates*len(scenes)/len(fit))
    require(maximum <= config.refit_limit, 'predeclared refit ceiling exceeded by actual split; no silent truncation')
    def entry(ids):
        rows = [s for s in scenes if s.scene_id in ids]
        return dict(scene_ids=[s.scene_id for s in rows], source_logs=[s.source_log_id for s in rows],
                    groups=sorted({s.source_group_id for s in rows}), scenes=len(rows), logs=len({s.source_log_id for s in rows}))
    result = dict(schema='fixed_DS_inner_log_partition_v1', partition_seed=config.partition_seed,
                  source_partition_hash=file_hash(source.root/'DATA_PARTITION.json'),
                  source_scene_hash=file_hash(source.root/'scenes.json'),
                  fit=entry(fit), validation=entry(val), full_scene_ids=[s.scene_id for s in scenes],
                  max_refit_updates=maximum, refit_rule='ceil(T_inner*N_full/N_inner)',
                  candidate_scores_used_for_partition=False, R00_inner_validation='IN_SAMPLE_BACKGROUND_ONLY',
                  Base_saw_DS=True, experts_fitted_DS=False,
                  probe_trigger='both internal-val selected gains <=0 OR both high-to-zero>0 after successful64-scene fits',
                  probe_fit_limit=2048, probe_val_limit=512, probe_updates_limit=2000,
                  primary_pool='fixed_K4', old_dev='previously_observed_exploratory_not_blind',
                  bootstrap_seed=config.bootstrap_seed, bootstrap_repetitions=config.bootstrap_repetitions)
    result['hash'] = digest(result)
    atomic_json(config.root/'INNER_SPLIT_MANIFEST.json', result, immutable=True)
    return result


def read_split(config):
    r = read_json(config.root/'INNER_SPLIT_MANIFEST.json')
    require(r['hash'] == digest({k:v for k,v in r.items() if k != 'hash'}), 'inner partition corrupt')
    require(r['partition_seed'] == config.partition_seed and r['source_partition_hash'] == file_hash(config.source().root/'DATA_PARTITION.json'), 'inner split/source changed')
    return r


class FrozenContexts:
    """Bounded cache of whitelisted W/A/ego, with source cache hash validation."""
    def __init__(self, config, scenes, *, preload=True):
        self.source = config.source(); self.reader = ReusedContext(self.source, capacity=0)
        first = self._read(scenes[0]); size = sum(v.numel()*v.element_size() for v in first.values())+4096
        self.capacity = max(1, min(len(scenes), int(config.feature_memory_gib*1024**3//size)))
        self.loaded = OrderedDict(); self.shapes = {k:list(v.shape) for k,v in first.items()}
        if preload and self.capacity == len(scenes):
            def one(s):
                x = ReusedContext(self.source, capacity=0).read(s)
                return {k:x[k].detach() for k in CONTEXT_KEYS}
            with ThreadPoolExecutor(max_workers=4) as workers:
                for s,x in zip(scenes,workers.map(one,scenes)): self.loaded[s.scene_id] = x

    def _read(self, scene):
        require(scene.split_role != 'final_test', 'forbidden training context role')
        x = self.reader.read(scene)
        return {k:x[k].detach() for k in CONTEXT_KEYS}

    def read(self, scene):
        if scene.scene_id not in self.loaded:
            self.loaded[scene.scene_id] = self._read(scene)
            while len(self.loaded) > self.capacity: self.loaded.popitem(last=False)
        self.loaded.move_to_end(scene.scene_id)
        return self.loaded[scene.scene_id]

    def batch(self, scenes, device):
        rows = [self.read(s) for s in scenes]
        return {k:torch.stack([r[k] for r in rows]).to(device) for k in CONTEXT_KEYS}
