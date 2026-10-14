from types import SimpleNamespace
import pytest
from iqe.io import atomic_json
from iqe.specialist_study import context_parallel as module


def test_resharding_is_disjoint_and_covers_fixed_manifest():
    ids = ['scene_'+str(i) for i in range(23)]
    for count in (1, 2, 6, 8):
        shards = module.assignments(ids, count)
        assert sum(map(len, shards)) == len(ids)
        assert {x for shard in shards for x in shard} == set(ids)
        assert all(shards[i] == ids[i::count] for i in range(count))
    with pytest.raises(ValueError): module.assignments(ids, 9)
    with pytest.raises(ValueError): module.assignments(['same', 'same'], 2)


def test_two_complete_shards_cannot_finish_six_worker_topology(tmp_path, monkeypatch):
    plan = {'workers': 6, 'hash': 'locked', 'assignments': [[] for _ in range(6)], 'scene_ids': []}
    config = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(module, 'read_plan', lambda *_: (tmp_path, plan))
    monkeypatch.setattr(module, 'ordered_scenes', lambda *_: [])
    for index in (0, 1):
        atomic_json(tmp_path/f'worker_{index}/COMPLETE.json', {'status': 'COMPLETE', 'plan_hash': 'locked',
                    'scene_receipt_hashes': {}, 'numeric_comparison': []})
    with pytest.raises(FileNotFoundError): module.verify(config, 'topology')
    assert not (tmp_path/'SPATIAL_CONTEXT_COMPLETE.json').exists()
