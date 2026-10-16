import json
from pathlib import Path
import pytest
from scripts.analysis.fm_transfer.dispatch_screen import completed_sft
from scripts.analysis.fm_transfer.milestone_controller import jobs,reference_for
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def test_RL_observations_use_own_complete_SFT(tmp_path):
    queue=list(jobs(tmp_path,include_rl=True))
    assert len(queue)==36 and len({j['output'] for j in queue})==36
    assert {j['cohort'] for j in queue}=={'DEV','TRAIN_DIAGNOSTIC'}
    for job in queue:
        if job['stage']!='RL':continue
        assert job['update'] in (5000,10000,15000,20000)
        assert Path(job['checkpoint']).name.endswith('_inner0')
        endpoint=20000 if job['variant']=='A_DDP_TIME' else 100000
        assert reference_for(tmp_path,job)==str(tmp_path/'evaluations'/f"{job['variant']}_SFT_update{endpoint:06d}_{job['cohort']}")
    # Original SFT-only usage cannot accidentally dispatch any RL evaluation.
    assert all(j['stage']=='SFT' for j in jobs(tmp_path))


@pytest.mark.parametrize('updates,qualification,phase,valid',[
    (20000,False,'READY',True),(5000,False,'READY',False),
    (20000,True,'READY',False),(20000,False,'COLLECTING',False)])
def test_screen_and_RL_refuse_partial_or_qualification_parents(tmp_path,updates,qualification,phase,valid):
    identity=dict(qualification=qualification,config=dict(max_updates=20000))
    atomic_json(tmp_path/'identity.json',identity)
    atomic_json(tmp_path/'files.json',{'identity.json':file_sha(tmp_path/'identity.json')})
    atomic_json(tmp_path/'COMPLETE',dict(phase=phase,updates=updates,files_sha256=file_sha(tmp_path/'files.json')))
    if valid:assert completed_sft(tmp_path)[0]==identity
    else:
        with pytest.raises(ValueError):completed_sft(tmp_path)
