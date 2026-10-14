from dataclasses import replace
import numpy as np
import pytest

from iqe.contracts import ScoreRecord
from iqe.io import atomic_json, read_json
from iqe.specialist_study import bank
from .test_deploy_precompute import setup


def test_parallel_bank_assembly_keeps_all_bytes_order_and_masks(tmp_path, monkeypatch):
    config, scenes, _, _, _, _, candidates = setup(tmp_path)
    monkeypatch.setattr(bank, 'pool', lambda c: {'hash': 'pool'})
    monkeypatch.setattr(bank, 'protocol', lambda c: {'protocol_hash': 'protocol'})
    rows = []
    for scene, candidate in zip(scenes, candidates):
        masks = dict.fromkeys(bank.COMPONENTS, True); masks[bank.COMPONENTS[-1]] = False
        components = dict.fromkeys(bank.COMPONENTS, .5); components[bank.COMPONENTS[-1]] = None
        label = ScoreRecord(1, candidate.key, 'protocol', scene.metric_context_hash, 'ref', .5,
                            components, masks, 'fixture', 'fixture', 42, 0, True, None)
        rows.append([(candidate, label)])
    first = bank.save_bank(config, 'deploy', 'scorer_fit', scenes, rows, {'sources': ['expert_0']}, read_workers=1)
    folder = config.root/'banks/deploy/scorer_fit'
    array_bytes = (folder/'arrays.npz').read_bytes(); meta_bytes = (folder/'BANK.json').read_bytes()
    second = bank.save_bank(config, 'deploy', 'scorer_fit', scenes, rows, {'sources': ['expert_0']}, read_workers=8)
    assert first == second
    assert array_bytes == (folder/'arrays.npz').read_bytes()
    assert meta_bytes == (folder/'BANK.json').read_bytes()
    assert read_json(folder/'ASSEMBLY_PROGRESS.json')['scenes_verified'] == len(scenes)
    with np.load(folder/'arrays.npz') as data:
        assert not data['component_valid'][..., -1].any()
        assert np.isnan(data['components'][..., -1]).all()
    broken = rows.copy(); broken[0] = [(candidates[0], replace(rows[0][0][1], metric_context_hash='wrong'))]
    with pytest.raises(Exception, match='stale'):
        bank.save_bank(config, 'deploy', 'scorer_fit', scenes, broken, {'sources': ['expert_0']}, read_workers=8)
