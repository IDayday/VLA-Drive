from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import scene
from iqe.contracts import CandidateRecord, ScoreRecord
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.specialist_study import validation_reuse


def prepared(tmp_path, monkeypatch):
    config = SimpleNamespace(root=tmp_path, identity='fixture_config')
    rows = [scene(i, role='stage_val') for i in range(2)]
    trajectory = {'horizon': 8, 'dt': .5, 'raw_representation': 'normalized_xy_sincos', 'frame': 'ego_relative'}
    atomic_json(tmp_path/'BASE_BINDING.json', {'contract_hash': 'fixture_contract'})
    atomic_json(tmp_path/'OFFLINE_BASE_CONTRACT.json', {'trajectory': trajectory})
    cp = tmp_path/'experts/E11/ema_003000.pt'
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(b'explicit test fixture, not a real checkpoint')
    item = {'cell': 'E11', 'step': 3000, 'endpoint': 'ema', 'checkpoint': str(cp),
            'checksum': file_hash(cp), 'model_hash': 'fixture_state'}
    atomic_json(str(cp)+'.COMPLETE.json', {k: item[k] for k in ('checksum', 'model_hash', 'step', 'endpoint')})
    atomic_json(cp.parent/'validation_003000.json', {'endpoints': {'ema': {
        'candidate_hash': item['model_hash'], 'scenes': 2, 'fixed_policy_01': [.5, .75]}}})
    proto = {'protocol_hash': 'fixture_protocol', 'navsim_root': 'fixture_navsim',
             'navsim_source_hash': 'fixture_source', 'config': {}}
    monkeypatch.setattr(validation_reuse, 'protocol', lambda _: proto)
    paths = []
    for s, value in zip(rows, [.6, .9]):
        path = tmp_path/'candidates/E11_ema/fixture_state'/(s.scene_id+'.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, trajectory=np.zeros((8, 3), np.float32))
        c = CandidateRecord(1, s.scene_id, 'E11_ema', item['model_hash'], 'fixture_contract',
            file_hash(path), str(path), 'normalized_xy_sincos', 'ego_relative', 8, .5, True,
            'fixture_code', config.identity, 'fixture_time')
        atomic_json(path.with_suffix('.json'), asdict(c))
        seed = int(digest({'scene_id': s.scene_id, 'seed': 42, 'repetition': 0})[:8], 16)
        request = {'candidate': asdict(c), 'scene': asdict(s), 'protocol_hash': proto['protocol_hash'],
                   'navsim_root': proto['navsim_root'], 'source_hash': proto['navsim_source_hash'],
                   'config': proto['config'], 'seed': seed, 'eval_seed': 42, 'repetition': 0}
        p = tmp_path/'scores'/(digest([c.key, s.metric_context_hash, proto['protocol_hash'], seed])+'.json')
        label = ScoreRecord(1, c.key, proto['protocol_hash'], s.metric_context_hash, 'fixture_reference',
                            value, {}, {}, 'fixture_backend', 'fixture_equivalence', 42, 0, True, None)
        atomic_json(p, asdict(label))
        atomic_json(p.with_suffix('.COMPLETE.json'), {'checksum': file_hash(p), 'request_hash': digest(request)})
        paths.append((path, p))
    return config, item, rows, paths


def test_completed_selection_reuses_exact_records_without_forward_or_score(tmp_path, monkeypatch):
    config, item, scenes, paths = prepared(tmp_path, monkeypatch)
    hashes = [(file_hash(t), file_hash(s)) for t, s in paths]
    a = validation_reuse.completed_endpoint_records(config, item, scenes)
    b = validation_reuse.completed_endpoint_records(config, item, scenes)
    assert a == b and [r.total_score_01 for _, r in a] == [.6, .9]
    assert hashes == [(file_hash(t), file_hash(s)) for t, s in paths]
    audit = read_json(next((tmp_path/'runtime/validation_reuse').glob('*.json')))
    assert audit['model_forward_calls'] == audit['official_backend_calls'] == 0


@pytest.mark.parametrize('failure', ['observation', 'context', 'trajectory', 'receipt', 'score', 'checkpoint', 'population'])
def test_reuse_rejects_incomplete_or_changed_dependencies(tmp_path, monkeypatch, failure):
    config, item, scenes, paths = prepared(tmp_path, monkeypatch)
    if failure == 'observation': scenes[0] = replace(scenes[0], observation_hash='changed')
    if failure == 'context': scenes[0] = replace(scenes[0], metric_context_hash='changed')
    if failure == 'trajectory': paths[0][0].write_bytes(b'corrupt')
    if failure == 'receipt': paths[0][1].with_suffix('.COMPLETE.json').unlink()
    if failure == 'score': paths[0][1].write_text('{}')
    if failure == 'checkpoint': item['checksum'] = 'wrong'
    if failure == 'population': scenes = scenes[:1]
    with pytest.raises(Exception): validation_reuse.completed_endpoint_records(config, item, scenes)


def test_pool_columns_do_not_load_or_recompute_selected_endpoints(tmp_path, monkeypatch):
    from iqe.specialist_study import experiments
    config, item, scenes, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, 'scenes_by_role', lambda _: {'stage_val': scenes})
    monkeypatch.setattr(experiments, 'verified_base_scores', lambda *_: {
        s.scene_id: {'total_score_01': .5, 'named_components': {}, 'component_valid_masks': {}} for s in scenes})
    monkeypatch.setattr(experiments, 'load_endpoint', lambda *_: pytest.fail('pool must not execute an endpoint again'))
    _, scores, _ = experiments.stage_val_columns(config, [item], 'cpu')
    assert np.allclose(scores, [[.5, .6], [.5, .9]])
