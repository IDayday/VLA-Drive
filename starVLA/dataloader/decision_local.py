"""Small frozen-token/trajectory dataset, patterned on the existing frozen-W probe.

Only immutable compact chunks are read. Targets never enter ModelInputs.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from starVLA.model.modules.decision_local.consequence_contract import ModelInputs, PredictionQueries, TrainingTargets, OracleRelations


def read(path):
    return json.loads(Path(path).read_text())


class FrozenConsequenceData:
    def __init__(self, features, labels, *, require_complete=True):
        self.features, self.labels = Path(features), Path(labels)
        self.feature_identity = read(self.features/'identity.json')
        self.label_identity = read(self.labels/'identity.json')
        if self.label_identity['anchor_cache_identity'] != self.feature_identity['identity']:
            raise ValueError('Label/anchor cache mismatch')
        if require_complete:
            for root in (self.features, self.labels):
                if not (root/'COMPLETE.json').exists(): raise ValueError('Complete population required')
        self.index = self.feature_identity['index']
        self.locations = {}
        for path in sorted((self.labels/'records').glob('*.json')):
            r, f = read(path), read(self.features/'records'/path.name)
            if r['identity'] != self.label_identity['identity'] or f['identity'] != self.feature_identity['identity'] or r['tokens'] != f['tokens']:
                raise ValueError('Chunk population mismatch')
            for offset, token in enumerate(r['tokens']):
                if token in self.locations: raise ValueError('Duplicate scene')
                self.locations[token] = (path.stem, offset)
        if {r['token'] for r in self.index} != set(self.locations):
            raise ValueError('No scene may be silently dropped')
        self.handles = OrderedDict()

    def _chunk(self, root, name, ident):
        key = (str(root), name)
        if key not in self.handles:
            handle = safe_open(str(root/'chunks'/(name+'.safetensors')), framework='pt', device='cpu')
            if handle.metadata()['identity'] != ident: raise ValueError('Foreign chunk')
            self.handles[key] = handle
            if len(self.handles) > 32: self.handles.popitem(last=False)
        self.handles.move_to_end(key)
        return self.handles[key]

    def batch(self, ids, device, *, epoch=0, evaluation=False):
        rows = []
        for i in ids:
            token = self.index[i]['token']; chunk, at = self.locations[token]
            f = self._chunk(self.features, chunk, self.feature_identity['identity'])
            l = self._chunk(self.labels, chunk, self.label_identity['identity'])
            # Two anchor seeds alternate per scene/epoch, independent of arm and
            # head initialization. Exactly one exposure per original scene.
            nanchors = f.get_slice('q0').get_shape()[1]
            phase = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            anchor = 0 if evaluation else (phase+epoch) % nanchors
            row = {k: f.get_slice(k)[at] for k in ('W', 'H_A', 'state')}
            row.update({k: f.get_slice(k)[at, anchor] for k in ('q0', 'q0_encoded')})
            row.update({k: l.get_slice(k)[at, anchor] for k in ('poses', 'in_grid', 'kinematic_valid', 'relations', 'valid', 'common_valid')})
            row.update({k: l.get_slice(k)[at] for k in ('ego', 'future_valid', 'body')})
            rows.append(row)
        batch = {}
        for k in rows[0]:
            t = torch.stack([r[k] for r in rows])
            if torch.device(device).type == 'cuda': t = t.pin_memory()
            batch[k] = t.to(device, non_blocking=True)
        return batch


def inputs_queries_targets(batch):
    inputs = ModelInputs.from_mapping({k: batch[k] for k in ModelInputs.__dataclass_fields__})
    T = inputs.q0.shape[1]
    queries = PredictionQueries(batch['poses'], torch.arange(1, T+1, device=inputs.q0.device)*.5,
                                batch['in_grid'], batch['kinematic_valid'])
    # Current road/current obstacles and actual future obstacles. Both P arms
    # receive exactly this same target population; CV is an oracle probe only.
    relation = torch.cat((batch['relations'][:, 0], batch['relations'][:, 2, ..., 1:]), -1)
    valid = torch.cat((batch['valid'][:, 0], batch['valid'][:, 2, ..., 1:]), -1)
    kinematic = batch['kinematic_valid'].clone(); kinematic[:, 0] = True
    valid = valid & kinematic[..., None, None]
    return inputs, queries, TrainingTargets(relation, valid, batch['ego'], batch['future_valid'])


def oracle_relations(batch, kind):
    sources = {'E_CURRENT_GT': 0, 'E_CV_GT': 1, 'E_FUTURE_GT': 2, 'E_MASK_ONLY': 2}
    if kind not in sources: raise ValueError('Not an oracle experiment')
    source = sources[kind]
    values = torch.cat((batch['relations'][:, 0], batch['relations'][:, source, ..., 1:]), -1)
    # Same common support for every privileged arm, even current channels.
    valid = torch.cat((batch['common_valid'], batch['common_valid'][..., 1:]), -1)
    kinematic=batch['kinematic_valid'].clone();kinematic[:,0]=True
    valid=valid&kinematic[...,None,None]
    if kind == 'E_MASK_ONLY': values = torch.zeros_like(values)
    return OracleRelations(values, valid, {'E_CURRENT_GT':'CURRENT','E_CV_GT':'CV','E_FUTURE_GT':'FUTURE','E_MASK_ONLY':'MASK_ONLY'}[kind])
