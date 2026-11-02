"""The user10k minimum must not reuse scene-clock milestones as evaluation steps."""
import copy

import pytest

from tools.qwen_world_safety_exploration.integrated_campaign import (
    USER_10K_POLICY,validate_runtime_budget,evaluation_registry)


def requested_run():
    return dict(target_updates=10000,gates=[250,625,2000,5000,10000],
        retained_native_handoff_gates=[250,625],evaluation_gates=[2000,5000,10000],
        historical_evaluation_gates=[250],runtime_budget_policy=USER_10K_POLICY,
        minimum_committed_updates=10000,score_based_stopping_before_minimum=False,
        post10k_long_training_review_required=True)


def test_actual10k_minimum_and_only_requested_eval_milestones():
    run=requested_run();original=copy.deepcopy(run)
    assert validate_runtime_budget(run)is run
    view=evaluation_registry(dict(runs=[run]))
    assert view['runs'][0]['gates']==[250,2000,5000,10000]
    assert run==original and 625 in run['gates']
    for key,value in [('target_updates',6250),('minimum_committed_updates',1250),
                      ('score_based_stopping_before_minimum',True),
                      ('evaluation_gates',[250,625,1250,2000,5000,10000])]:
        bad=copy.deepcopy(run);bad[key]=value
        with pytest.raises(ValueError):validate_runtime_budget(bad)
