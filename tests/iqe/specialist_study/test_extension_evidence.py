from types import SimpleNamespace

import numpy as np

from iqe.io import atomic_json, read_json
from iqe.specialist_study import diagnostics, experiments, recipes


def test_extension_training_diagnostic_cannot_choose_redundant_pool_member(tmp_path, monkeypatch):
    config = SimpleNamespace(root=tmp_path)
    pilot = {'cell': 'E11', 'step': 3000, 'endpoint': 'ema',
             'validation': {'selection_key': [.1, .2]}}
    extension = {'cell': 'EXT_interaction', 'step': 5000, 'endpoint': 'raw',
                 'validation': {'selection_key': [.01, .9]}}
    atomic_json(tmp_path/'BASE_BINDING.json', {'checkpoint': 'test-only'})
    atomic_json(tmp_path/'EXPERT_RECIPE_SELECTION.json', {'selected': pilot})
    atomic_json(tmp_path/'experts/EXT_interaction/result.json', {'status': 'COMPLETE'})
    monkeypatch.setattr(recipes, 'load', lambda *_: {
        'minimum_mean_oracle_gain': 1e-8, 'minimum_scene_oracle_gain': .01})
    monkeypatch.setattr(experiments, 'rows_for_cell', lambda *_: [extension])
    calls = []

    def diagnostic(_config, cell, device):
        calls.append((cell, device))
        # Even apparently excellent training-fit evidence must not override
        # the fixed heldout columns used for candidate admission.
        return {'H_train_direct_PDMS_gain': 1.0}

    monkeypatch.setattr(diagnostics, 'specialist_learning', diagnostic)
    monkeypatch.setattr(experiments, 'stage_val_columns', lambda *_: (
        [SimpleNamespace(scene_id='val1'), SimpleNamespace(scene_id='val2')],
        np.array([[.7, .8, .71], [.7, .8, .71]]), np.ones((2, 3, 2))))
    result = experiments.freeze_pool(config, 'cpu')
    assert calls == [('EXT_interaction', 'cpu')]
    assert [row['cell'] for row in result['experts']] == ['E11']
    assert result['selection_role'] == 'stage_val'
    assert read_json(tmp_path/'SERVING_CANDIDATE_POOL.json') == result
