"""The new clock must survive stage restarts and report both material budgets."""
import copy
import io
from pathlib import Path

import pytest
import torch
import yaml

from starVLA.rl.qwen_world_safety_exploration import integrated64_recipe as R
from starVLA.rl.qwen_world_safety_exploration import integrated64_schedule as S


def test_schedule_plateau_and_actual_two_group_adam_restore():
    p=[torch.nn.Parameter(torch.ones(1)),torch.nn.Parameter(torch.ones(1))]
    def make(params):
        o=torch.optim.AdamW([dict(params=[params[0]],lr=5e-5),dict(params=[params[1]],lr=5e-6)])
        return o,S.IntegratedCosineLR(o)
    opt,sched=make(p)
    assert S.multiplier(0)==.001
    assert S.multiplier(125)==S.multiplier(625)==1.
    assert S.multiplier(2500)==S.multiplier(12500)==.1
    assert 1>S.multiplier(1250)>S.multiplier(2499)>.1
    for u in range(1,2502):
        for x in p:x.grad=torch.ones_like(x)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
        if u in (124,125,126,625,626,2499,2500):
            data=io.BytesIO();torch.save(dict(opt=opt.state_dict(),scheduler=sched.state_dict()),data);data.seek(0)
            saved=torch.load(data,weights_only=False)
            pp=[torch.nn.Parameter(x.detach().clone())for x in p];oo,ss=make(pp)
            oo.load_state_dict(saved['opt']);ss.load_state_dict(saved['scheduler'])
            assert [g['lr']for g in oo.param_groups]==[v*S.multiplier(u)for v in (5e-5,5e-6)]
            for x in pp:x.grad=torch.ones_like(x)
            oo.step();ss.step()
            assert ss.last_epoch==u+1
            assert [g['lr']for g in oo.param_groups]==[v*S.multiplier(u+1)for v in (5e-5,5e-6)]
    bad=copy.deepcopy(sched.state_dict());bad['integrated64_protocol']['keep_until_updates']=500
    with pytest.raises(ValueError):sched.load_state_dict(bad)


def test_recipe_retains_safety_scope_and_exposes_pending_budget():
    base=yaml.safe_load(Path('/mnt/project/qwen-world-safety-exploration-20261031/standard_training_20261101/configs/SAFETY_G8_CONTROL.yaml').read_text())
    original=copy.deepcopy(base);cfg=R.build_recipe(base)
    assert base==original and cfg['adaptation']==base['adaptation']
    assert cfg['grpo']['sample']['group_size']==16
    assert R.CONFIG['normalization_candidates']==1024
    assert R.RETENTION_CONFIG['admission_start_update']==250
    b=R.budget(6250)
    assert b['scene_exposures']==100032 and b['candidate_queries']==1600512
    assert b['next_inner']==2 and b['scene_surplus_at_pending_boundary']==32
    assert b['nominal_scene_reference_updates']==50000
    assert b['nominal_candidate_reference_updates']==100000
    assert not b['strict_equal_nominal_source_budget']
    for value in (-1,True,12501):
        with pytest.raises(ValueError):R.budget(value)
    bad=copy.deepcopy(cfg);bad['learning_rate']*=2
    with pytest.raises(ValueError):R.validate_integrated_recipe(bad)
