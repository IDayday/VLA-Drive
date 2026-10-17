import importlib
from types import SimpleNamespace
from unittest.mock import patch
import pytest
m=importlib.import_module('tools.st_world_policy.prepare_causal_history')

def args(**kw):
    return SimpleNamespace(**dict(action='build',pin_selection=None,scene_list='/fixed',target_times=[0.],mode='registered',max_scenes=64,**kw))

@pytest.mark.parametrize('field,value', [('target_times',[0.,.5]),('max_scenes',101592),('scene_list',None),('pin_selection','x'),('mode','smoke')])
def test_scope_rejects(field,value):
    a=args();setattr(a,field,value)
    with pytest.raises(ValueError):m.validate(a)

def test_no_ledger_admission():
    with patch.dict(m.os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(m.os,'sched_getaffinity',return_value={6,7}),patch.object(m.storage,'_ACTIVE_GUARD',None):
        with pytest.raises(ValueError,match='admission'):m.validate(args())

def test_original_parser_current_only():
    a=m.parser().parse_args(['build','--student-root','/s','--output','/o','--target-times','0'])
    assert a.target_times==[0.] and a.action=='build'

def test_budgeted_read_and_immutable_roundtrip(tmp_path):
    limits=dict(m.LIMITS,read_mib_per_second=100000)
    b=m.bounded.PreparationBudget(tmp_path,limits)
    m.bounded.write_json(tmp_path/'a.json',{'x':1},b)
    with m.bounded.BudgetedReader(tmp_path/'a.json',b) as f:assert b'"x": 1' in f.read()
    m.bounded.write_json(tmp_path/'a.json',{'x':1},b)
    with pytest.raises(ValueError,match='conflict'):m.bounded.write_json(tmp_path/'a.json',{'x':2},b)
    assert b.snapshot()['source_bytes_read']>0
