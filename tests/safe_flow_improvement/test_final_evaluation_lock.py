import json

import pytest

from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.safe_flow_improvement.contracts import fingerprint


def test_final_evaluation_requires_prior_selection_lock(tmp_path):
    path = tmp_path/'lock.json'
    record = dict(status='NOT_SELECTED_YET'); record['sha256'] = fingerprint(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='locked before'):
        locked_contract(path, 'SFT', {})


def test_final_evaluation_rejects_seed_expansion(tmp_path):
    population = dict(identity_sha256='p', navtest_tokens=[str(i) for i in range(12146)])
    record = dict(status='LOCKED_BEFORE_FINAL_NAVTEST', population_identity='p', inference_seeds=[2101, 2102],
                  training_seeds=[42], navtest_tokens_sha256=fingerprint(population['navtest_tokens']))
    record['sha256'] = fingerprint(record)
    path = tmp_path/'lock.json'; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='seed contract'):
        locked_contract(path, 'SFT', population)


def test_one_sample_has_no_within_scene_diversity_estimate():
    import numpy as np
    from scripts.analysis.flow_safe_improvement.diagnose_policy_projection import diversity
    assert diversity(np.zeros((1, 8, 3))) == {}


def test_final_selection_cannot_skip_unfinished_registered_window(tmp_path):
    from scripts.analysis.flow_safe_improvement.lock_final_evaluation import checked_summary
    record = dict(all_registered_DEV_windows_complete=False, fixed_budget_DEV_endpoints_complete=False,
                  training_seeds=[42], inference_seeds=[2101], evaluations=[], missing=['ODE_NFT_round3'])
    record['sha256'] = fingerprint(record)
    path = tmp_path/'progress.json'; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='All 24'):
        checked_summary(path)
