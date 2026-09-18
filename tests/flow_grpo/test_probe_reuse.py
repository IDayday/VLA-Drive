import pytest
import torch
from starVLA.rl.flow_grpo.metrics import Metrics
from starVLA.rl.flow_grpo.probe_reuse import defer_probe, decorate_probe_row


def ratios(prefix, values):
    metrics = Metrics()
    metrics.ratios(prefix, torch.tensor(values), .02)
    return metrics.result()


def test_full_probe_is_recovered_from_next_actual_current_ratios():
    # The actual trainer uses these functions on its gathered Metrics, not an
    # independently invented training simulator. CUDA update identity is separate.
    values = [.94, 1., 1.01, 1.07]
    row = {'update':2, 'inner_epoch':1, 'policy_version':0,
           **ratios('pre_update_ratio',values), **ratios('post_update_probe_ratio',[.93,1.,1.02,1.1])}
    decorate_probe_row(row, True, False)
    previous = row['previous_update_probe']
    assert previous['policy_update'] == 1 and previous['policy_version'] == 0
    assert previous['metrics'] == ratios('post_update_probe_ratio', values)
    assert row['post_update_probe_status'] == 'COMPLETE'


def test_deferred_does_not_fabricate_values_and_final_invocation_always_probes():
    assert defer_probe(True,0,2,1,2)
    assert not defer_probe(True,0,2,1,1)
    assert not defer_probe(True,1,2,2,2)
    assert not defer_probe(False,0,2,1,2)
    row = {'update':1, 'inner_epoch':0, 'policy_version':0}
    decorate_probe_row(row,True,True)
    assert row['post_update_probe_status']=='DEFERRED_TO_NEXT_INNER_FORWARD'
    assert 'post_update_probe_ratio_count' not in row
    assert 'previous_update_probe' not in row
    with pytest.raises(ValueError,match='already measured'):
        decorate_probe_row({**row,'post_update_probe_ratio_count':1},True,True)
    with pytest.raises(ValueError,match='final-boundary'):
        decorate_probe_row(row,True,False)


def test_resumed_inner_epoch_uses_checkpoint_update_without_extra_state():
    row = {'update':8,'inner_epoch':1,'policy_version':3,
           **ratios('pre_update_ratio',[1.,1.03]), **ratios('post_update_probe_ratio',[1.,1.04])}
    decorate_probe_row(row,True,False)
    assert row['previous_update_probe']['policy_update']==7
    assert row['previous_update_probe']['metrics']['post_update_probe_ratio_count']==2
    unchanged = {'custom':'value'}
    assert decorate_probe_row(unchanged,False,False)=={'custom':'value'}
