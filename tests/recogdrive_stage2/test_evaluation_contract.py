import random

import pytest

from tools.recogdrive_stage2.evaluate import scene_seed, validate_population


def population():
    return [dict(token=f'{i:016x}', log=f'log_{i % 136}') for i in range(12146)]


def test_scoring_population_rejects_duplicate_and_foreign_log():
    rows = population()
    validate_population(rows, list(reversed(rows)))
    with pytest.raises(ValueError, match='identities differ'):
        validate_population(rows + [rows[0]], rows)
    wrong = [dict(r) for r in rows]
    wrong[0]['log'] = 'foreign'
    with pytest.raises(ValueError, match='identities differ'):
        validate_population(rows, wrong)


def test_partial_population_cannot_be_named_complete_navtest():
    rows = population()[:4]
    with pytest.raises(ValueError, match='complete canonical'):
        validate_population(rows, rows)


def test_scene_randomness_is_independent_of_export_order_and_global_rng():
    rows = population()[:30]
    expected = {r['token']: scene_seed(r['token'], 42) for r in rows}
    random.seed(891); random.shuffle(rows)
    assert expected == {r['token']: scene_seed(r['token'], 42) for r in rows}
    assert len(set(expected.values())) == len(rows)
    assert all(scene_seed(t, 43) != seed for t, seed in expected.items())
