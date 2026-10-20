from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from tools.action_effect.training import run_training,resolve_checkpoint


def config():
    return {'training':{'learning_rate':1e-3,'weight_decay':.01,'warmup_updates':2,
        'batch_scenes':4,'microbatch_scenes':2,'gradient_clip':1.,'validation_every':3,'max_updates':6}}


def fit(root,model,stop=None,resume=False,fail_result=False):
    seen=[]
    def step(m,indices,n):
        seen.append(list(indices));x=torch.tensor(indices).float()[:,None]
        # Normal training randomness remains enabled.
        y=m(x+torch.randn_like(x)*.01)
        loss=(y-1).square().mean();loss.backward();return {'fit':float(loss)}
    result=run_training(model,output=root,identity={'source_commit':'fixture','input':'fixed'},dataset_size=13,
        step_fn=step,cfg=config(),device='cpu',seed=42,max_updates=6,stop_after=stop,resume=resume)
    return result,seen


def test_pause_resume_matches_continuous_and_terminal_no_step(tmp_path):
    torch.manual_seed(2);initial=torch.nn.Linear(1,1)
    full=deepcopy(initial);a,sa=fit(tmp_path/'full',full)
    split=deepcopy(initial);b,sb=fit(tmp_path/'split',split,stop=3)
    assert b['status']=='PAUSED'
    # Construction/global random operations before resume must be overwritten.
    torch.randn(11);random.random();np.random.rand()
    new=torch.nn.Linear(1,1);c,sc=fit(tmp_path/'split',new,resume=True)
    assert sa==sb+sc and c['status']=='COMPLETE'
    assert all(torch.equal(x,y) for x,y in zip(full.parameters(),new.parameters()))
    c1=resolve_checkpoint(tmp_path/'full',a['run_identity']);c2=resolve_checkpoint(tmp_path/'split',c['run_identity'])
    assert torch.equal(c1['rng']['cpu'],c2['rng']['cpu'])
    for name in c1['optimizer']['state'][0]:
        assert torch.equal(c1['optimizer']['state'][0][name],c2['optimizer']['state'][0][name])
    (tmp_path/'split'/'RESULT.json').unlink()
    before=torch.get_rng_state().clone();d,sd=fit(tmp_path/'split',new,resume=True)
    assert d['segment_updates']==0 and sd==[] and d['termination_reason']=='MAX_UPDATES'
    assert torch.equal(before,torch.get_rng_state())


def test_latest_complete_beats_result_and_partial(tmp_path):
    m=torch.nn.Linear(1,1);a,_=fit(tmp_path,m,stop=3)
    (tmp_path/'step_999999.pt').write_bytes(b'partial')
    (tmp_path/'RESULT.json').write_text('{"optimizer_step":0}')
    c=resolve_checkpoint(tmp_path,a['run_identity']);assert c['optimizer_step']==3
    (tmp_path/'step_000003.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='No complete'):resolve_checkpoint(tmp_path,a['run_identity'])


def test_role_or_source_change_rejects_resume(tmp_path):
    m=torch.nn.Linear(1,1);fit(tmp_path,m,stop=3)
    with pytest.raises(ValueError,match='Immutable'):
        run_training(m,output=tmp_path,identity={'source_commit':'different'},dataset_size=13,
            step_fn=lambda *_:None,cfg=config(),device='cpu',seed=42,max_updates=6,resume=True)
