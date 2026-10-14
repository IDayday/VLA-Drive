"""Interim testing must not admit partial, qualification or unmatched models."""
import copy
import pytest

from scripts.analysis.constrained_flow_noise.interim_navtest import validate_boundary
from starVLA.rl.constrained_flow_noise.contracts import SFT_SHA256


def example():
    boundary = dict(phase='ready', raw_exposure=25398, next_inner=0, world=8)
    identity = dict(qualification_only=False, variant='C_SDE', SFT_sha256=SFT_SHA256,
                    training_seed=42, evaluation_seed=2101, budget_sha256='budget',
                    population_identity='population')
    return boundary, identity


def test_matched_native_boundary():
    boundary, identity = example()
    validate_boundary(boundary, identity, 'C_SDE', dict(identity_sha256='population'), 'budget')


@pytest.mark.parametrize('section,key,value', [
    ('boundary', 'raw_exposure', 12760), ('boundary', 'next_inner', 1),
    ('boundary', 'phase', 'updating'), ('identity', 'qualification_only', True),
    ('identity', 'variant', 'N_FLOW_NOISE'), ('identity', 'evaluation_seed', 42),
    ('identity', 'budget_sha256', 'changed'), ('identity', 'population_identity', 'DEV'),
])
def test_reject_unmatched_observation(section, key, value):
    boundary, identity = example()
    objects = dict(boundary=copy.deepcopy(boundary), identity=copy.deepcopy(identity))
    objects[section][key] = value
    with pytest.raises(ValueError):
        validate_boundary(objects['boundary'], objects['identity'], 'C_SDE',
                          dict(identity_sha256='population'), 'budget')
