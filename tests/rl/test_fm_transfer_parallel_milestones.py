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


def test_local_transport_keeps_argv_and_neutral_module_search_directory(monkeypatch):
    from scripts.analysis.fm_transfer import milestone_controller as m
    calls = []
    monkeypatch.setattr(m, 'worker_allowed', lambda host: calls.append(('authorized', host)))
    monkeypatch.setattr(m, 'require_dispatch', lambda: 'training-vla-zt')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda args, **kw: calls.append((args, kw)) or 'ok')
    args = ['env', 'PYTHONPATH=/frozen/source', '/python', '-m', 'scripts.analysis.fm_transfer.dispatch_evaluation']
    assert m.remote('training-vla-zt', args) == 'ok'
    assert calls == [('authorized', 'training-vla-zt'),
                     (args, dict(cwd='/', text=True, timeout=60))]


def test_other_host_keeps_existing_ssh_transport(monkeypatch):
    import shlex
    from scripts.analysis.fm_transfer import milestone_controller as m
    calls = []
    monkeypatch.setattr(m, 'worker_allowed', lambda host: None)
    monkeypatch.setattr(m, 'require_dispatch', lambda: 'training-vla-zt')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda args, **kw: calls.append((args, kw)) or 'ok')
    args = ['/python', '-c', 'print("actual argv")']
    assert m.remote('training-vlawm-tsinghua-zt', args) == 'ok'
    assert calls == [(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
                      'training-vlawm-tsinghua-zt', shlex.join(args)], dict(text=True, timeout=60))]


def test_local_transport_does_not_bypass_host_withdrawal(monkeypatch):
    from scripts.analysis.fm_transfer import milestone_controller as m
    def withdrawn(host):
        raise PermissionError('withdrawn')
    monkeypatch.setattr(m, 'worker_allowed', withdrawn)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: pytest.fail('Must not execute'))
    with pytest.raises(PermissionError): m.remote('training-vla-zt', ['/python'])


def inventory(host, free=70000, visible=True):
    uuids=[f'{host}-gpu{i}' for i in range(8)]
    return dict(gpus=[dict(uuid=u,used=1000,free=free) for u in uuids],
                owners=[dict(uuid=u,visible=visible) for u in uuids])


def test_busy_preferred_host_does_not_hide_idle_fallback():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    ready=[dict(output='DEV'),dict(output='TRAIN')]
    samples={h:inventory(h,free=19000 if h=='busy' else 70000) for h in ('busy','free1','free2')}
    calls=[]
    def probe(h):calls.append(h);return samples[h]
    expected={h:[g['uuid'] for g in s['gpus']] for h,s in samples.items()}
    plan,seen=capacity_dispatch_plan(ready,[],list(samples),expected,probe,lambda _:None)
    assert plan==[(ready[0],'free1'),(ready[1],'free2')]
    assert calls==['busy','busy','free1','free1','free2','free2']
    assert not seen['busy']['available']


def test_second_observation_can_reject_a_host_without_losing_the_job():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    ready=[dict(output='DEV')];calls=[]
    expected={h:[g['uuid'] for g in inventory(h)['gpus']] for h in ('changed','fallback')}
    def probe(h):
        calls.append(h)
        return inventory(h,visible=not(h=='changed' and calls.count(h)==2))
    plan,_=capacity_dispatch_plan(ready,[],list(expected),expected,probe,lambda _:None)
    assert plan==[(ready[0],'fallback')]


def test_active_host_is_never_probed_or_reassigned_and_no_ready_jobs_are_idle():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    ready=[dict(output='next')];active=[dict(output='running',host='active')]
    expected={h:[g['uuid'] for g in inventory(h)['gpus']] for h in ('active','free')}
    calls=[]
    def probe(h):calls.append(h);return inventory(h)
    plan,_=capacity_dispatch_plan(ready,active,list(expected),expected,probe,lambda _:None)
    assert plan==[(ready[0],'free')] and calls==['free','free']
    assert capacity_dispatch_plan([],active,list(expected),expected,lambda _:pytest.fail('No ready jobs'),lambda _:None)==([], {})


def test_all_hosts_unavailable_preserves_pending_jobs():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    ready=[dict(output='DEV')];expected={'busy':[g['uuid'] for g in inventory('busy')['gpus']]}
    plan,seen=capacity_dispatch_plan(ready,[],['busy'],expected,
        lambda h:inventory(h,free=1000),lambda _:None)
    assert plan==[] and not seen['busy']['available'] and ready==[dict(output='DEV')]


def test_probe_error_is_not_free_capacity_and_duplicates_fail_before_probe():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    def fail(_):raise TimeoutError('SSH observation timeout')
    with pytest.raises(TimeoutError):
        capacity_dispatch_plan([dict(output='x')],[],['host'],{'host':[]},fail,lambda _:None)
    with pytest.raises(ValueError):
        capacity_dispatch_plan([dict(output='x')],[],['host','host'],{},
            lambda _:pytest.fail('Duplicate host must fail first'),lambda _:None)


def test_changed_physical_gpu_identity_is_fatal_before_dispatch():
    from scripts.analysis.fm_transfer.milestone_controller import capacity_dispatch_plan
    expected={'host':[g['uuid'] for g in inventory('old')['gpus']]}
    with pytest.raises(ValueError):
        capacity_dispatch_plan([dict(output='x')],[],['host'],expected,inventory,lambda _:None)
