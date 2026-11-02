"""A completed pending gate must be visible to the asynchronous evaluator."""
import hashlib
import json
from types import SimpleNamespace

from tools.qwen_world_safety_exploration.integrated_campaign import (
    checkpoint_at_gate,commit_completed_stage)


def test_commit_pending250_persists_before_gate_validation(tmp_path):
    arm='INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'
    root=tmp_path/'run';cp=root/'checkpoints/update000250_inner2';cp.mkdir(parents=True)
    owner_state=tmp_path/'STATE.json'
    run=dict(id=arm,root=str(root),owner_state=str(owner_state))
    files=cp/'files.json';files.write_text('{}')
    (cp/'COMPLETE').write_text(json.dumps(dict(updates=250,next_inner=2,world=8,
        files_sha256=hashlib.sha256(files.read_bytes()).hexdigest())))
    (cp/'identity.json').write_text(json.dumps(dict(purpose='formal',world=8,
        recipe=dict(research=dict(arm=arm)))))
    for i in range(8):(cp/f'rank{i}.pt').touch()
    state=dict(runs={arm:dict(status='TRAINING',stages={'250':{}})})
    owner_state.write_text(json.dumps(state))
    assert checkpoint_at_gate(run,250)[0]is None
    def update(key,**values):
        state['runs'][key].update(values);owner_state.write_text(json.dumps(state))
    result=commit_completed_stage(SimpleNamespace(update=update),run,250,
                                  state['runs'][arm]['stages']['250'],cp)
    material,_=checkpoint_at_gate(run,250)
    assert result==str(cp)and material['next_inner']==2
    assert material['actual_collection_budget']['candidate_queries']==64512
    assert state['runs'][arm]['actual_collection_budget']['scene_exposures']==4032
