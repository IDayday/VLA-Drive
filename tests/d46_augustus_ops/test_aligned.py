"""Corrected identities, complete-only review and event isolation; CPU only."""
import importlib.util
from pathlib import Path
import json
import pytest
from tools.d46_augustus_ops import evaluations as q
from tools.d46_augustus_ops.aligned_owner import loss_review,stop_own


@pytest.mark.parametrize('exp',['AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED'])
def test_aligned_native_uses_own_frozen_source_and_exposure_clock(tmp_path,exp):
    path=Path(__file__).parents[1]/'d44_native_rl/test_augustus_evaluations.py'
    spec=importlib.util.spec_from_file_location('aligned_queue_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r=m.route(tmp_path,exp);u=q.EARLY[exp][0];cp,_=m.checkpoint(tmp_path,r,u)
    binding=q.inspect_checkpoint(tmp_path,r,cp)
    assert binding['source'].endswith('augustus_aligned_science_v2')
    assert binding['scene_budget_clock']==u*(128 if 'FULL'in exp else 64)/2
    state=q.initial_state();q.discover(tmp_path,state,{exp:r})
    assert len(state['cases'])==1


def test_only_completed_aggregate_large_drop_can_stop_own_gang(tmp_path):
    row=dict(experiment='AUG4_LORA_ALIGNED',stages={'formal':{'nodes':[{'job':str(tmp_path/'own')}]}})
    (tmp_path/'own').mkdir();summary=tmp_path/'summary.json'
    summary.write_text(json.dumps(dict(status='COMPLETE',scenes=12146,invalid_scenes=0,PDMS=88.1)))
    case=dict(status='RUNNING',binding={'experiment':row['experiment'],'checkpoint':'cp'},summary=str(summary))
    assert loss_review(row,{'cases':{'x':case}})is None
    case['status']='COMPLETE';case['binding']['experiment']='other'
    assert loss_review(row,{'cases':{'x':case}})is None
    case['binding']['experiment']=row['experiment'];review=loss_review(row,{'cases':{'x':case}})
    assert review['NAVTEST_SELECTED']and 'token'not in review
    stop_own(row,review)
    assert (tmp_path/'own/STOP_REQUESTED').exists()
