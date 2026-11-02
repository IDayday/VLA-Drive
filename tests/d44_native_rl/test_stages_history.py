import copy
import pytest
import torch
from starVLA.rl.d44_native_rl.stages import stage_at
from starVLA.rl.d44_native_rl import history


def nominal(token,score,update=0,run='own'):
    return dict(token=token,normalized_clean_action=torch.zeros(8,3).tolist(),
        version=dict(run=run,update=update,collection=update//2+1),
        record=dict(status='OK',metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,
            time_to_collision_within_bound=1.,comfort=1.,ego_progress=score,driving_direction_compliance=1.)))


def test_registered_stages_do_not_change_modules_or_optimizer_roles():
    stages=[stage_at(e)for e in(0,1,4,8,12)]
    assert [s.name for s in stages]==['I','II','III','IV','V']
    assert[(s.global_scenes,s.group)for s in stages]==[(32,8)]+[(64,16)]*4
    assert stages[0].film_lr==0 and not stages[0].post
    assert stages[3].film_lr==2e-6 and stages[3].historical
    with pytest.raises(ValueError):stage_at(0,(1,1,2,3))


def test_own_nominal_teacher_exact_ancestry_and_capacity():
    h=history.initial('own');legal={'a','b','c'}
    history.update(h,[nominal('a',.95),nominal('b',.96)],legal,capacity=1)
    assert len(h['teachers'])==1 and 'b'in h['teachers']
    assert history.historical_teacher(h,'b',.90,dict(run='own',update=2))['score']==.96
    assert history.historical_teacher(h,'b',.95,dict(run='own',update=2))is None
    with pytest.raises(ValueError):history.historical_teacher(h,'b',.90,dict(run='own',update=0))
    with pytest.raises(ValueError):history.update(h,[nominal('c',.97,run='another_algorithm')],legal)
    with pytest.raises(ValueError):history.update(h,[nominal('NAVTEST',.97)],legal)


def test_mixture_distinct_padding_and_deterministic_resume():
    h=history.initial('own');legal={str(i)for i in range(200)}
    history.update(h,[nominal(str(i),.4 if i%2 else .96)for i in range(100)],legal)
    history.update(h,[nominal('0',.5,2)],legal)
    base=[dict(token=str(i),padding=i==199)for i in range(136,200)]
    first=history.select(h,base,collection=50,enabled=True)
    resumed=history.select(copy.deepcopy(h),base,collection=50,enabled=True)
    assert first==resumed and len({x['token']for x in first})==64
    assert first[-1]['token']=='199'and first[-1]['padding']
    assert any(x['source']!='uniform'for x in first)
    assert history.select(h,base,collection=50,enabled=False)==[dict(x,source='uniform')for x in base]


def test_inactive_preinstalled_film_has_no_weight_decay_or_adam_state():
    text=torch.nn.Parameter(torch.ones(2));film=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([dict(params=[text],lr=1e-3),dict(params=[film],lr=0.)],weight_decay=.01)
    original=film.detach().clone()
    for _ in range(3):
        optimizer.zero_grad();(text.sum()+film.sum()).backward();film.grad=None;optimizer.step()
    assert torch.equal(film,original)and film not in optimizer.state
    before=optimizer.state[text]['step'].clone();optimizer.param_groups[1]['lr']=2e-6
    optimizer.zero_grad();(text.sum()+film.sum()).backward();optimizer.step()
    assert optimizer.state[text]['step']==before+1 and optimizer.state[film]['step']==1
