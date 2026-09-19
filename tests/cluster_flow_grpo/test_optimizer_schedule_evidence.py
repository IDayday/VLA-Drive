import pytest
from scripts.cluster_flow_grpo.optimizer_evidence import update_learning_rate


def test_adam_oracle_uses_performed_update_lr_not_saved_next_lr():
    cfg={'optimizer':{'schedule':{'type':'warmup_cosine','total_updates':100,'warmup_updates':10,'min_lr_ratio':.1}}}
    assert update_learning_rate({'update':1,'lr_used':[.1*.1]}, {'initial_lr':.1,'lr':.1*.2},0,cfg)==.1*.1


def test_scheduler_evidence_missing_or_tampered_is_rejected():
    cfg={'optimizer':{'schedule':{'type':'warmup_cosine','total_updates':100,'warmup_updates':10,'min_lr_ratio':.1}}}
    for row in ({'update':1},{'update':1,'lr_used':[.02]}):
        with pytest.raises(ValueError):update_learning_rate(row,{'initial_lr':.1,'lr':.02},0,cfg)


def test_old_constant_lr_oracle_remains_compatible():
    assert update_learning_rate({}, {'lr':.01},0,{})==.01
    assert update_learning_rate({'lr_used':[.01]}, {'lr':.01},0,{})==.01
