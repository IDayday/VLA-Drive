"""Pure CPU owner persistence/admission tests; no probe, SSH or CUDA child."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen_world_abcd_research import eval_owner as E, job_worker as W
from starVLA.rl.qwen_world_compare.contracts import fingerprint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_evaluation_guard_respects_three_namespace_owner_and_holds_spare(tmp_path, monkeypatch):
    roots = {key:tmp_path/key for key in E.LEASES}
    for path in roots.values():path.mkdir()
    monkeypatch.setattr(E, 'LEASES', roots)
    for name, key in [('PRIMARY_LEASE_ROOT','primary'), ('SHARED_LEASE_ROOT','shared'),
                      ('RESEARCH_LEASE_ROOT','research'), ('EVALUATION_LEASE_ROOT','evaluation')]:
        monkeypatch.setattr(W, name, roots[key])
    gpu = dict(index=0, uuid='mockUUID', total_MiB=81920, free_MiB=81920, utilization=0)
    sample = dict(gpus=[gpu], owners=[])
    monkeypatch.setattr(W, '_admission_sample', lambda:({0:gpu}, []))
    args = SimpleNamespace(share_min_free_mib=32768, minimum_free_mib=None, lease_slot='evaluation')
    grant = dict(permanent_hosts=['mockhost'])
    guard = []
    for name in ('primary', 'shared', 'research'):
        stream=(roots[name]/'mockUUID.lock').open('a')
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB);guard.append(stream)
    assert not E.usable_gpu(gpu, sample, minimum=32768, reserved={})
    held=[]
    with pytest.raises(W.ResourceRefused, match='legacy namespaces'):
        W._gpu_admission(args, 'mockhost', grant, [0], ['mockUUID'], {}, held)
    assert held == []
    guard[1].close()  # Two real training slots + one spare legacy slot.
    assert E.usable_gpu(gpu, sample, minimum=32768, reserved={})
    try:
        W._gpu_admission(args, 'mockhost', grant, [0], ['mockUUID'], {}, held)
        assert len(held)==2  # actual spare legacy + evaluation locks, both retained
        assert not E.lease_free(roots['shared']/'mockUUID.lock')
        assert not E.lease_free(roots['evaluation']/'mockUUID.lock')
    finally:
        for stream in held+guard:stream.close()
    unknown=dict(sample, owners=[dict(uuid='mockUUID', visible=False, used_MiB=0)])
    assert not E.usable_gpu(gpu, unknown, minimum=32768, reserved={})


def test_reentry_keeps_frozen_source_world_and_only_prechild75_requeues(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'ROOT', tmp_path)
    roots = {key:tmp_path/'leases'/key for key in E.LEASES}
    monkeypatch.setattr(E, 'LEASES', roots)
    source=tmp_path/'frozen_science';parent=tmp_path/'parent'
    write(parent/'export.json', {'IL_parent':{'identity':{'config':{'data_root':str(tmp_path/'data')}}}})
    run=dict(id='ARCH_D_JAT_ACTION', source=str(source), parent_export=str(parent))
    owner=E.EvaluationOwner.__new__(E.EvaluationOwner)
    owner.directory=tmp_path/'owners/eval';owner.directory.mkdir(parents=True)
    owner.state_path=owner.directory/'STATE.json'
    owner.claims_path=tmp_path/'resources/EVALUATION_CLAIMS.json';owner.claims_path.parent.mkdir()
    owner.max_shards=1;owner.max_workers=32;owner.minimum=32768
    owner.registry=dict(runs=[run], parent_selection='selected.json')
    gate=dict(export=dict(status='COMPLETE', path='/sealed/export'))
    owner.state=dict(runs={run['id']:{'gates':{'1000':gate}}}, evaluations={})
    for cohort in E.COUNTS:
        owner.state['evaluations'][cohort]=dict(arm=run['id'], label='D_u001000', updates=1000,
            cohort=cohort, world=1, ranks={}, evaluation=str(tmp_path/cohort))
    calls=[]
    def launch(host, frozen, command, name, **kwargs):
        calls.append((host,frozen,command,name,kwargs))
        directory=tmp_path/'jobs'/name
        write(directory/'RUNNING.json', dict(status='RUNNING', command=command))
        return directory
    monkeypatch.setattr(E,'launch',launch)
    gpus=[dict(index=i,uuid=f'mock{i}',total_MiB=81920,free_MiB=81920,utilization=0) for i in range(2)]
    samples=[('mockhost',dict(gpus=gpus, owners=[]))]
    owner.dispatch_ranks(samples)
    assert len(calls)==2 and all(call[1]==str(source) for call in calls)
    assert all(call[4]['worker_source']==Path(E.__file__).resolve().parents[2] for call in calls)
    assert all(call[2][call[2].index('--world')+1]=='1' for call in calls)
    # Recreate process memory using only the persisted state; no duplicate work.
    owner.state=E.read(owner.state_path)
    owner.dispatch_ranks(samples);assert len(calls)==2
    node=owner.state['evaluations']['DEV']['ranks']['0']
    old_job=node['job']
    write(Path(old_job)/'RESULT.json', dict(status='REFUSED_RESOURCE',returncode=75,child_not_started=True))
    assert not owner.finished(node) and 'job' not in node
    owner.dispatch_ranks(samples)
    assert len(calls)==3 and node['attempt']==2 and node['refused_jobs']==[old_job]
    write(Path(node['job'])/'RESULT.json', dict(status='FAILED',returncode=1,child_not_started=False))
    assert not owner.finished(node) and node['status']=='FAILED_REVIEW_REQUIRED'
    owner.dispatch_ranks(samples);assert len(calls)==3
    assert not E.retryable(dict(status='REFUSED_RESOURCE',returncode=75,child_not_started=False))


def test_only_full_sealed_summary_can_publish(tmp_path):
    path=tmp_path/'SUMMARY.json'
    value=dict(status='COMPLETE',architecture_id='D',cohort='DEV',qualification=False,
               scenes=1696,scene_denominator=1696,invalid_scenes=0)
    value['sha256']=fingerprint(value);write(path,value)
    assert E.complete_summary(path,architecture='D',cohort='DEV')==value
    value.pop('sha256');value['scenes']=1695;value['sha256']=fingerprint(value);write(path,value)
    with pytest.raises(ValueError,match='full-population'):
        E.complete_summary(path,architecture='D',cohort='DEV')
