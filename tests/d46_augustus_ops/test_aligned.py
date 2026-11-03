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
    summary.write_text(json.dumps(dict(status='COMPLETE',scenes=12146,invalid_scenes=0,PDMS=86.1)))
    case=dict(status='RUNNING',binding={'experiment':row['experiment'],'checkpoint':'cp'},summary=str(summary))
    assert loss_review(row,{'cases':{'x':case}})is None
    case['status']='COMPLETE';case['binding']['experiment']='other'
    assert loss_review(row,{'cases':{'x':case}})is None
    case['binding']['experiment']=row['experiment'];review=loss_review(row,{'cases':{'x':case}})
    assert review['NAVTEST_SELECTED']and 'token'not in review
    stop_own(row,review)
    assert (tmp_path/'own/STOP_REQUESTED').exists()


def test_small_early_drop_gets_more_data_and_old_bad_point_cannot_stop_recovery(tmp_path):
    row={'experiment':'AUG4_LORA_ALIGNED'};cases={}
    def add(key,clock,score):
        path=tmp_path/(key+'.json');path.write_text(json.dumps(dict(status='COMPLETE',scenes=12146,invalid_scenes=0,PDMS=score)))
        cases[key]=dict(status='COMPLETE',binding=dict(experiment=row['experiment'],checkpoint=key,scene_budget_clock=clock),summary=str(path))
    add('early',6400,88.1)
    assert loss_review(row,{'cases':cases})is None
    add('second',12800,88.2)
    assert loss_review(row,{'cases':cases})is not None
    add('recovered',25600,90.0)
    assert loss_review(row,{'cases':cases})is None


def test_evaluation_budget_clock_uses_valid_exposures_not_padding(tmp_path):
    path=Path(__file__).parents[1]/'d44_native_rl/test_augustus_evaluations.py'
    spec=importlib.util.spec_from_file_location('valid_clock_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r=m.route(tmp_path,'AUG3_FULL_ALIGNED');cp,_=m.checkpoint(tmp_path,r,1614)
    tx=json.loads((cp/'transaction.json').read_text())
    for rank in tx['ranks']:rank['valid_scene_exposures']=103288
    m.write(cp/'transaction.json',tx);files=json.loads((cp/'files.json').read_text());files['transaction.json']=m.file_sha(cp/'transaction.json');m.write(cp/'files.json',files)
    seal=json.loads((cp/'COMPLETE').read_text());seal['files_sha256']=m.file_sha(cp/'files.json');m.write(cp/'COMPLETE',seal)
    assert q.inspect_checkpoint(tmp_path,r,cp)['scene_budget_clock']==103288
