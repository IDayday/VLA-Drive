from dataclasses import asdict
from types import SimpleNamespace
import pytest

from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.specialist_study.rich_labels import pending_indices
from iqe.specialist_study.rich_precompute import collect_completed
from .test_deploy_precompute import setup


def test_fixed_source_order_dedup_and_capacity_do_not_use_scores():
    hashes = [{'base'}, {'base'}, {'base'}]; counts = [4, 11, 12]
    first = [SimpleNamespace(trajectory_hash=h) for h in ('base', 'new', 'third')]
    assert pending_indices(first, hashes, counts, 12) == [1]
    assert counts == [4, 12, 12]
    second = [SimpleNamespace(trajectory_hash=h) for h in ('x', 'y', 'third')]
    assert pending_indices(second, hashes, counts, 12) == [0]
    assert counts == [5, 12, 12]


def test_completed_rich_collection_never_calls_model_and_rejects_corruption(tmp_path):
    config, scenes, item, _, model, _, original = setup(tmp_path)
    source = item['reuse_rich_source']
    atomic_json(tmp_path/'SERVING_CANDIDATE_POOL.json', {'hash': 'pool'})
    p = {'config_hash': config.identity, 'scenes_hash': digest([asdict(s) for s in scenes]),
         'Base_binding_hash': file_hash(tmp_path/'BASE_BINDING.json'), 'pool_hash': 'pool',
         'columns': [source], 'scene_ids': [s.scene_id for s in scenes]}
    p['hash'] = digest(p)
    atomic_json(tmp_path/'runtime/rich_precompute/PLAN.json', p)
    receipt = tmp_path/'runtime/rich_precompute'/source['expert_id']/source['model_hash']/'batch_000000.json'
    value = read_json(receipt); value['binding']['plan_hash'] = p['hash']; atomic_json(receipt, value)
    assert collect_completed(config, scenes, source['expert_id'], source['model_hash'], 2) == original
    assert model.calls == 1
    with pytest.raises(Exception, match='observations'):
        collect_completed(config, list(reversed(scenes)), source['expert_id'], source['model_hash'])
    (tmp_path/'checkpoint').write_bytes(b'changed')
    with pytest.raises(Exception, match='checkpoint'):
        collect_completed(config, scenes, source['expert_id'], source['model_hash'])


def test_corrupt_prepared_column_receipt_is_not_silently_rescored(tmp_path, monkeypatch):
    from iqe.specialist_study import rich_labels
    config = SimpleNamespace(root=tmp_path, identity='config')
    atomic_json(tmp_path/'SERVING_CANDIDATE_POOL.json', {'hash': 'pool'})
    atomic_json(tmp_path/'runtime/rich_precompute/PLAN.json', {'fixture': True})
    root = rich_labels.folder(config)
    atomic_json(root/'PLAN.json', {'config_hash': 'config', 'pool_hash': 'pool',
                'rich_plan_hash': file_hash(tmp_path/'runtime/rich_precompute/PLAN.json'), 'job_names': ['column']})
    monkeypatch.setattr(rich_labels, 'binding', lambda *args: {'fixed': True})
    atomic_json(root/'column.json', {'tampered': True})
    atomic_json(root/'column.COMPLETE.json', {'checksum': 'wrong', 'binding_hash': digest({'fixed': True})})
    with pytest.raises(Exception, match='checksum'):
        rich_labels.prepared_or_score(config, [], [], 'column')


def test_publishing_requests_is_the_exact_existing_job_without_worker_launch(tmp_path, monkeypatch):
    from iqe.internal_baseline import bank
    config, scenes, _, _, _, _, candidates = setup(tmp_path)
    config.navsim_root = '/fixture/navsim'
    config.scoring_attempts = 2
    proto = {'protocol_hash': 'protocol', 'navsim_root': '/fixture/navsim',
             'nuplan_root': '/fixture/nuplan', 'navsim_source_hash': 'source', 'config': {'fixed': True}}
    monkeypatch.setattr(bank, 'protocol', lambda config: proto)
    monkeypatch.setattr(bank.subprocess, 'run', lambda *a, **k: pytest.fail('publisher launched scoring'))
    path, job = bank.prepare_score_job(config, scenes, candidates, 'column')
    items = []
    for scene, candidate in zip(scenes, candidates):
        seed = int(digest({'scene_id': scene.scene_id, 'seed': 42, 'repetition': 0})[:8], 16)
        request = {'candidate': asdict(candidate), 'scene': asdict(scene), 'protocol_hash': 'protocol',
                   'navsim_root': '/fixture/navsim', 'source_hash': 'source', 'config': {'fixed': True},
                   'seed': seed, 'eval_seed': 42, 'repetition': 0}
        response = config.root/'scores'/(digest([candidate.key, scene.metric_context_hash, 'protocol', seed])+'.json')
        items.append({'request': request, 'response': str(response)})
    expected = {'items': items, 'navsim_root': '/fixture/navsim', 'nuplan_root': '/fixture/nuplan', 'attempts_per_invocation': 2}
    assert job == read_json(path) == expected
    assert path.parent.name == 'column_'+digest(items)[:12]
    before = path.read_bytes()
    assert bank.prepare_score_job(config, scenes, candidates, 'column')[0] == path
    assert path.read_bytes() == before
