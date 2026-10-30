import json
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen_world_research.campaign_owner import CampaignOwner, ArmHeldForResourceLease, atomic_json


def test_research_stop_releases_only_selected_job_and_retains_any_inner_checkpoint(tmp_path):
    owner = CampaignOwner.__new__(CampaignOwner)
    owner.root = tmp_path
    owner.owner_root = tmp_path / 'owners/CAMPAIGN'
    owner.mutex = threading.RLock()
    owner.capacity_changed = threading.Condition(owner.mutex)
    owner.capacity_epoch = 0
    owner.runs = {}
    run = tmp_path / 'native'
    checkpoint = run / 'checkpoints/update028123_inner2'
    atomic_json(checkpoint / 'COMPLETE', dict(updates=28123, next_inner=2, phase='INNER_COMMITTED'))
    atomic_json(run / 'latest.json', dict(checkpoint=str(checkpoint)))
    job = tmp_path / 'job'
    job.mkdir()
    sibling = tmp_path / 'sibling'
    sibling.mkdir()
    evidence = tmp_path / 'evidence.json'
    atomic_json(evidence, dict(complete_test_budgets=[1000, 5000, 10000, 20000]))
    decision = dict(arm='NFT_FULL', disposition='STOP_FUTILITY', reason='flat four completed budgets', evidence=[str(evidence)])
    atomic_json(owner.owner_root / 'arm_decisions/NFT_FULL.json', decision)
    atomic_json(tmp_path / 'RUNS.json', dict(runs=[dict(id='NFT_FULL', root=str(run), job=str(job)),
                                                    dict(id='PPO_FULL', job=str(sibling))]))
    owner.state = dict(runs={'NFT_FULL': {}, 'PPO_FULL': dict(status='TRAINING')}, events=[])
    calls = []

    def wait_job(path):
        assert (Path(path) / 'STOP_REQUESTED').exists()
        calls.append(('wait_job', path))
        return dict(returncode=0)

    owner.dispatcher = SimpleNamespace(wait_job=wait_job, wait_transport_exit=lambda p: calls.append(('exit', p)))
    owner.stop_arm('NFT_FULL', owner.arm_stop_decision('NFT_FULL'))
    assert owner.state['runs']['NFT_FULL']['status'] == 'STOPPED_FUTILITY'
    assert owner.state['runs']['NFT_FULL']['updates'] == 28123
    assert owner.state['runs']['NFT_FULL']['checkpoint'] == str(checkpoint.resolve())
    assert owner.state['runs']['PPO_FULL']['status'] == 'TRAINING'
    assert not (sibling / 'STOP_REQUESTED').exists()
    assert calls == [('wait_job', str(job)), ('exit', str(job))]
    assert json.loads((tmp_path / 'RUNS.json').read_text())['runs'][1]['job'] == str(sibling)


def test_unsubstantiated_research_stop_is_rejected(tmp_path):
    owner = CampaignOwner.__new__(CampaignOwner)
    owner.owner_root = tmp_path
    atomic_json(tmp_path / 'arm_decisions/NFT_FULL.json', dict(arm='NFT_FULL', disposition='STOP_FUTILITY'))
    with pytest.raises(ValueError, match='Incomplete evidence'):
        owner.arm_stop_decision('NFT_FULL')


def test_optional_safety_probe_failure_does_not_stop_native_training():
    owner = CampaignOwner.__new__(CampaignOwner)
    owner.mutex = threading.RLock()
    probe = Future()
    probe.set_exception(RuntimeError('optional probe requires review'))
    owner.evaluation_futures = {('PPO_FULL', 12000, 'NAVTEST'): probe}
    owner.check_arm_evaluations('PPO_FULL', wait_all=True)
    owner.evaluation_futures[('PPO_FULL', 10000, 'NAVTEST')] = probe
    with pytest.raises(RuntimeError, match='optional probe requires review'):
        owner.check_arm_evaluations('PPO_FULL')


def test_expiring_temporary_lease_retains_native_partial_checkpoint(tmp_path):
    owner=CampaignOwner.__new__(CampaignOwner)
    owner.root=tmp_path;owner.owner_root=tmp_path/'owners/CAMPAIGN'
    owner.mutex=threading.RLock();owner.capacity_changed=threading.Condition(owner.mutex)
    owner.capacity_epoch=0;owner.resource_retries=2
    run=tmp_path/'native';job=tmp_path/'job';cp=run/'checkpoints/update000042_inner2'
    atomic_json(cp/'COMPLETE',dict(updates=42,next_inner=2,phase='INNER_COMMITTED'))
    atomic_json(run/'latest.json',dict(checkpoint=str(cp)))
    owner.runs={'X':dict(id='X',root=str(run),job=str(job))}
    atomic_json(tmp_path/'RUNS.json',dict(runs=list(owner.runs.values())))
    owner.state=dict(runs={'X':dict(stages={'1000':dict(train_job=str(job))})},events=[])
    owner.dispatcher=SimpleNamespace(wait_job=lambda _:dict(returncode=0,temporary=True,
        expiry_guard_requested=True,end_unix=100,guard_unix=90),wait_transport_exit=lambda _:None)
    with pytest.raises(ArmHeldForResourceLease):owner.train_to('X',1000)
    assert owner.state['runs']['X']['status']=='HELD_RESOURCE_LEASE'
    assert owner.state['runs']['X']['updates']==42
    assert owner.state['runs']['X']['checkpoint']==str(cp.resolve())


def test_pilot_budget_is_not_recorded_as_full50k_completion(tmp_path):
    owner=CampaignOwner.__new__(CampaignOwner)
    owner.root=tmp_path;owner.owner_root=tmp_path/'owners/CAMPAIGN';owner.mutex=threading.RLock()
    owner.runs={'X':dict(id='X',data_policy='uniform_full',research_budget_updates=1000)}
    owner.state=dict(runs={'X':dict(updates=1000,checkpoint='sealed_1k',stages={
        '1000':dict(train_completed=True,checkpoint='sealed_1k')})},events=[])
    atomic_json(tmp_path/'RUNS.json',dict(runs=list(owner.runs.values())))
    owner.check_arm_evaluations=lambda *a,**kw:None
    boundaries=[];owner.queue_boundary=lambda a,u,c:boundaries.append(u)
    assert owner.run_arm('X')=='PILOT_COMPLETE_REVIEW_REQUIRED'
    assert boundaries==[1000]
    assert owner.state['runs']['X']['updates']==1000
