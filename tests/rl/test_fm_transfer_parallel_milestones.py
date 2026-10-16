import pytest
from scripts.analysis.fm_transfer.milestone_controller import dispatch_plan,jobs,capacity_available


def test_independent_cohorts_can_use_two_hosts_without_repeating_rollouts(tmp_path):
    ready=[j for j in jobs(tmp_path,True) if j['variant']=='B_SOURCE_ACTION_DIT' and j['stage']=='SFT' and j['update']==25000]
    assert len(ready)==2
    plan=dispatch_plan(ready,[],['DEV-host','TRAIN-host'])
    assert [p[0] for p in plan]==ready
    assert [p[1] for p in plan]==['DEV-host','TRAIN-host']
    assert {p[0]['cohort'] for p in plan}=={'DEV','TRAIN_DIAGNOSTIC'}
    assert len(list(jobs(tmp_path,True)))==36


def test_active_job_reserves_its_recorded_host_and_keeps_current_workers():
    ready=[dict(output='next-DEV'),dict(output='next-TRAIN')]
    active=[dict(output='running-old-milestone',host='host-A')]
    assert dispatch_plan(ready,active,['host-A','host-B'])==[(ready[0],'host-B')]
    assert dispatch_plan(ready,active,['host-A'])==[]
    assert dispatch_plan(ready,[],['host-A'])==[(ready[0],'host-A')]


def test_duplicate_host_or_milestone_never_gets_two_dispatches():
    with pytest.raises(ValueError):dispatch_plan([dict(output='x')],[],['host','host'])
    with pytest.raises(ValueError):dispatch_plan([dict(output='x')],[dict(output='x',host='host')],['host'])
    with pytest.raises(ValueError):dispatch_plan([dict(output='x'),dict(output='x')],[],['a','b'])


def test_repeated_capacity_and_physical_identity_precede_milestone_admission():
    uuids=[f'gpu{i}' for i in range(8)]
    first=dict(gpus=[dict(uuid=u,used=0,free=70000) for u in uuids],owners=[])
    assert capacity_available(first,first,uuids)
    import copy
    second=copy.deepcopy(first);second['gpus'][0]['free']=20000
    assert not capacity_available(first,second,uuids)
    second=copy.deepcopy(first);second['gpus'][0]['used']=50000
    assert not capacity_available(first,second,uuids)
    second['owners']=[dict(uuid=uuids[0],visible=True)]
    assert capacity_available(first,second,uuids)
    second['gpus'][0]['uuid']='aliased-or-changed'
    with pytest.raises(ValueError):capacity_available(first,second,uuids)
