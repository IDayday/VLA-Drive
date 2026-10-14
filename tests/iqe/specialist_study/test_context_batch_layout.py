import pytest
from iqe.io import digest, ContractError
from iqe.specialist_study.context_batch_layout import original_batch_map, within_tolerance


def plan(batches):
    p = {'batch_composition_unchanged': True, 'cache_batch': 4,
         'tasks': [{'task_id': '0', 'batches': [{'ids': b} for b in batches]}]}
    p['identity'] = digest(p)
    return p


def test_original_tail_and_order_retained():
    batches = original_batch_map(plan([['a', 'b', 'c', 'd'], ['e', 'f']]))
    assert batches['f'] == ('e', 'f')
    assert batches['b'] == ('a', 'b', 'c', 'd')
    assert 'unknown' not in batches


def test_bad_manifest_and_duplicate_are_rejected():
    p = plan([['a', 'b']]); p['tasks'][0]['batches'][0]['ids'].reverse()
    with pytest.raises(ContractError): original_batch_map(p)
    with pytest.raises(ContractError): original_batch_map(plan([['a'], ['a']]))


def test_tolerance_not_relaxed_for_batch_shape():
    e = {'raw_max_abs': 1e-7, 'world_max_abs': 0., 'action_max_abs': 0.}
    assert within_tolerance(e)
    assert not within_tolerance(e | {'world_max_abs': .00002})
    assert not within_tolerance(e | {'action_max_abs': float('nan')})
