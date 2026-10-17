"""Independent arithmetic/source negatives; no model, real ledger or payload."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import pytest

SOURCE=Path(__file__).resolve().parents[2]/'full_future_metadata_preparation_v1/run_build_once.py'
spec=importlib.util.spec_from_file_location('independent_terminal_accounting',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('base,child',[(None,100),({},100),({'rchar':True},100),
                                     ({'rchar':-1},100),({'rchar':1.0},100),
                                     ({'rchar':10},True),({'rchar':10},-1),
                                     ({'rchar':10},None)])
def test_invalid_baseline_or_child_cannot_get_discount(base,child):
    a=m.terminal_read_accounting(base,{'rchar':20},{'rchar':200},child,1000,final_child_available=True)
    assert a['status']=='INCOMPLETE_CONSERVATIVE'
    assert a['charged_parent_child_rchar_bytes']==1000
    assert m.charge_after_reap(a,{'rchar':207})==1007

@pytest.mark.parametrize('base,parent_before,child,between,tail',[
    (0,0,0,0,0),(1234567,0,1000,0,90),(700,20,0,13,17),
    (100,200,300,400,500)])
def test_each_real_source_of_parent_and_child_activity_is_charged_once(base,parent_before,child,between,tail):
    pre=base+parent_before;post=pre+child+between
    a=m.terminal_read_accounting({'rchar':base},{'rchar':pre},{'rchar':post},child,10000,final_child_available=True)
    assert a['status']=='COMPLETE'
    assert a['parent_activity_during_wait_rchar_bytes']==between
    assert m.charge_after_reap(a,{'rchar':post+tail})==parent_before+child+between+tail
    # Repeated observation is a pure calculation, not a second charge mutation.
    assert m.charge_after_reap(a,{'rchar':post+tail})==parent_before+child+between+tail

def test_missing_post_reap_sample_fails_closed_instead_of_using_last_active():
    a=m.terminal_read_accounting({'rchar':0},{'rchar':10},None,20,1000,final_child_available=True)
    assert a['status']=='INCOMPLETE_CONSERVATIVE' and a['charged_parent_child_rchar_bytes']==1000
    with pytest.raises(ValueError):m.charge_after_reap(a,{'rchar':50})

def test_control_limits_child_reader_and_publication_functions_are_unchanged():
    relative='reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py'
    old=subprocess.check_output(['git','show','5b956b05^:'+relative],text=True)
    functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(old),functions(SOURCE.read_text())
    for name in before:
        if name not in ('monitor','run'):assert before[name]==after[name],name
    assert set(after)-set(before)=={'terminal_read_accounting','charge_after_reap'}
