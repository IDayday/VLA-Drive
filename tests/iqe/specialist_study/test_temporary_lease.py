import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import scene
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.specialist_study.temporary_lease import authorization, can_admit, record_boundary, start_ticks
from iqe.specialist_study.rich_precompute import batch


def lease_fixture(tmp_path):
    lease={'authorization_id':'test_lease','scope':['existing_iqe_specialist_scorer_study'],
           'hosts':['test-host'],'starts_unix':100.,'expires_unix':30100.,
           'stop_admission_before_expiry_seconds':1800,'graceful_stop_before_expiry_seconds':1200}
    path=tmp_path/'AUTHORIZATION.json';atomic_json(path,lease)
    entry={'authorization_path':str(path),'authorization_sha256':file_hash(path),
           'new_task_dispatch_allowed':True,'hosts':lease['hosts'],
           'starts_unix':100.,'expires_unix':30100.}
    policy={'temporary_task_authorizations':{'test_lease':entry}}
    env={'IQE_TEMPORARY_LEASE_ID':'test_lease','IQE_TEMPORARY_LEASE_EXPIRES_UNIX':'30100'}
    return lease,policy,env


def test_temporary_lease_is_scope_host_hash_and_absolute_time_bound(tmp_path):
    lease,policy,env=lease_fixture(tmp_path)
    assert authorization(policy,'test-host-worker-0','precompute-rich-candidates',env,200,False)==lease
    for command,host,now in [('train-scorer-cell','test-host',200),('precompute-rich-candidates','other-host',200),
                             ('precompute-rich-candidates','test-host',30100),('precompute-rich-candidates','test-host',99)]:
        with pytest.raises(Exception):authorization(policy,host,command,env,now,False)
    policy['temporary_task_authorizations']['test_lease']['authorization_sha256']='wrong'
    with pytest.raises(Exception,match='checksum'):authorization(policy,'test-host','precompute-rich-candidates',env,200,False)


def test_live_process_group_guard_and_markers_required(tmp_path):
    lease,policy,env=lease_fixture(tmp_path)
    with pytest.raises(Exception):authorization(policy,'test-host','precompute-rich-candidates',env,200)
    path=tmp_path/'guard.json';atomic_json(path,{'lease_id':'test_lease',
        'lease_hash':policy['temporary_task_authorizations']['test_lease']['authorization_sha256'],
        'host':'test-host','pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'pgid':os.getpgrp()})
    env['IQE_TEMPORARY_GUARD_RECEIPT']=str(path)
    assert authorization(policy,'test-host','precompute-rich-candidates',env,200)==lease
    env['IQE_TEMPORARY_LEASE_EXPIRES_UNIX']='30101'
    with pytest.raises(Exception,match='expiry'):authorization(policy,'test-host','precompute-rich-candidates',env,200)


def test_admission_stops_before_record_drain(tmp_path,monkeypatch):
    lease,_,_=lease_fixture(tmp_path)
    assert can_admit(lease,28299) and not can_admit(lease,28300)
    assert record_boundary(lease,28899) and not record_boundary(lease,28900)
    stop=tmp_path/'STOP';stop.write_text('stop')
    monkeypatch.setenv('IQE_TEMPORARY_STOP_FILE',str(stop))
    assert not record_boundary(lease,200)


def test_wrapper_initial_environment_contains_lease_and_rejects_conflicts(tmp_path):
    import subprocess
    import sys
    from iqe.specialist_study.temporary_lease import process_environment
    lease,_,_=lease_fixture(tmp_path)
    inherited=dict(os.environ, IQE_TEMPORARY_GUARD_RECEIPT='previous_guard', IQE_TEMPORARY_STOP_FILE='previous_stop')
    env=process_environment(lease,inherited)
    assert 'IQE_TEMPORARY_GUARD_RECEIPT' not in env and 'IQE_TEMPORARY_STOP_FILE' not in env
    code="from pathlib import Path; import os; print(Path('/proc/self/environ').read_bytes().split(b'\\0').count(b'IQE_TEMPORARY_LEASE_ID=test_lease'))"
    assert subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip()=='1'
    with pytest.raises(Exception,match='different temporary lease'):
        process_environment(lease,dict(env,IQE_TEMPORARY_LEASE_ID='other'))


def test_rich_canonical_batch_reuses_and_rejects_corruption(tmp_path):
    config=SimpleNamespace(root=tmp_path,identity='fixture')
    rows=[scene(i) for i in range(4)]
    atomic_json(tmp_path/'BASE_BINDING.json',{'contract_hash':'fixture_contract','bound_at_utc':'fixture_time'})
    atomic_json(tmp_path/'OFFLINE_BASE_CONTRACT.json',{'trajectory':{'horizon':8,'dt':.5,
                'raw_representation':'normalized_xy_sincos','frame':'ego_relative'}})
    cp=tmp_path/'checkpoint';cp.write_bytes(b'test fixture')
    item={'expert_id':'E11_raw_1000','model_hash':'fixture_weights','checkpoint':str(cp),'checksum':file_hash(cp)}
    plan={'hash':'fixture_plan','scene_ids':[s.scene_id for s in rows]}
    class Model:
        calls=0
        def eval(self):return self
        def __call__(self,x):
            self.calls+=1
            return SimpleNamespace(physical=torch.zeros(len(x),8,3))
    model=Model();cache=SimpleNamespace(batch=lambda sub,device:sub)
    a,reused=batch(config,plan,item,rows,0,model,'cpu',cache)
    b,reused2=batch(config,plan,item,rows,0,model,'cpu',cache)
    assert a==b and not reused and reused2 and model.calls==1
    with pytest.raises(Exception,match='layout'):batch(config,plan,item,rows[::-1],0,model,'cpu',cache)
    (tmp_path/'candidates'/item['expert_id']/item['model_hash']/(rows[0].scene_id+'.npz')).write_bytes(b'corrupt')
    with pytest.raises(Exception,match='corrupt'):batch(config,plan,item,rows,0,model,'cpu',cache)
