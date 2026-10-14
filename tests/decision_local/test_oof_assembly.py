import pytest
from tools.decision_local.assemble_oof_diagnostic import merge_heldout


def test_each_log_is_predicted_only_by_opposite_fold():
    population = [{'token': 'a', 'log': 'log0'}, {'token': 'b', 'log': 'log1'}]
    models = {fold: [{**row, 'marker': fold} for row in population] for fold in (0, 1)}
    rows = merge_heldout(models, {'log0': 0, 'log1': 1}, population)
    assert [row['marker'] for row in rows] == [1, 0]
    assert [row['model_training_fold'] for row in rows] == [1, 0]


def test_missing_or_duplicate_scenes_are_not_silently_dropped():
    population = [{'token': 'a', 'log': 'log0'}, {'token': 'b', 'log': 'log1'}]
    with pytest.raises(ValueError, match='identical diagnostic scenes'):
        merge_heldout({0: population[:1], 1: population}, {'log0': 0, 'log1': 1}, population)
    with pytest.raises(ValueError, match='identical diagnostic scenes'):
        merge_heldout({0: population+[population[0]], 1: population}, {'log0': 0, 'log1': 1}, population)
