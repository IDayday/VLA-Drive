import copy
import pytest
from scripts.analysis.fm_transfer.rescore_saved_navtest import validate_input_manifest
from starVLA.rl.simwam_recipe.contracts import fingerprint


def example():
    pop = dict(navtest_tokens=['fixture'], identity_sha256='population')
    manifest = dict(cohort='NAVTEST', seed=2101, qualification_only=False, scenes=12146,
        tokens_sha256=fingerprint(pop['navtest_tokens']), modes=['E_NATIVE', 'E_SHIFT5'],
        population_identity='population', weight_sha256='fixed-endpoint',
        final_lock=dict(status='LOCKED_BEFORE_FINAL_NAVTEST', formal_budget_complete=True,
                        weight_sha256=['fixed-endpoint']))
    return manifest, pop


def test_fixed_input_accepted():
    manifest, pop = example(); validate_input_manifest(manifest, pop)


@pytest.mark.parametrize('key,value', [('cohort', 'DEV'), ('seed', 43), ('qualification_only', True),
    ('scenes', 12145), ('tokens_sha256', 'changed'), ('modes', ['E_NATIVE']),
    ('population_identity', 'changed'), ('weight_sha256', 'best-intermediate')])
def test_wrong_input_refused(key, value):
    manifest, pop = example(); manifest[key] = value
    with pytest.raises(ValueError): validate_input_manifest(manifest, pop)


def test_endpoint_not_complete_refused():
    manifest, pop = example(); manifest['final_lock']['formal_budget_complete'] = False
    with pytest.raises(ValueError): validate_input_manifest(manifest, pop)
