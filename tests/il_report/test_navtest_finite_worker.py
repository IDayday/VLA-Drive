import json
import hashlib
from unittest.mock import patch

import pytest

from tools.qwen_world_compare.job_worker import shared_training_host,ResourceRefused,_gpu_admission


def test_explicit_finite_sharing_honors_guard_and_immutable_grant(tmp_path):
    path=tmp_path/'grant.json'
    grant=dict(permanent_hosts=['permanent'],unbounded_temporary_hosts=['unbounded'],
        temporary_hosts=['finite','not_explicit'],shared_training_hosts=['unbounded','finite'],
        starts_unix=100,expires_unix=1000,checkpoint_guard_seconds=100,record=str(path))
    path.write_text(json.dumps(grant))
    for at,allowed in [(99,False),(100,True),(899,True),(900,False),(1000,False)]:
        with patch('tools.qwen_world_compare.job_worker.time.time',return_value=at):
            assert shared_training_host('finite',grant) is allowed
            assert shared_training_host('not_explicit',grant) is False
            assert shared_training_host('unbounded',grant) is True
    changed=dict(grant,shared_training_hosts=['finite'])
    with pytest.raises(PermissionError):shared_training_host('finite',changed)


def test_unauthorized_sharing_is_retryable_before_any_lease_or_cuda(tmp_path):
    from types import SimpleNamespace
    from tools.qwen_world_compare import job_worker as worker
    args=SimpleNamespace(share_min_free_mib=49152)
    devices={0:dict(uuid='GPU-test')}
    with patch.object(worker,'_admission_sample',return_value=(devices,[])),\
            patch.object(worker,'shared_training_host',return_value=False):
        with pytest.raises(ResourceRefused):
            _gpu_admission(args,'withdrawn',{},[0],['GPU-test'],{},[])


def test_only_exact_sealed_performance_diagnostic_is_a_known_cohost(tmp_path):
    from tools.qwen_world_compare import job_worker as worker
    root=tmp_path/'diagnostics';output=root/'IL70_FULL_NAVTRAIN';export=root/'IL70_export'
    output.mkdir(parents=True);export.mkdir()
    (output/'identity.json').write_text(json.dumps(dict(qualification=False,cohort='TRAIN_DIAGNOSTIC',IL_updates=70000,model_sha256='exact')))
    payload=json.dumps(dict(model_sha256='exact')).encode();(export/'export.json').write_bytes(payload)
    (export/'COMPLETE').write_text(json.dumps(dict(export_sha256=hashlib.sha256(payload).hexdigest())))
    args=['python','-m','tools.qwen_world_safety_exploration.evaluate','--parent-checkpoint',
        '/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT','--output',str(output),
        '--export',str(export),'--cohort','TRAIN_DIAGNOSTIC','--manifest',str(root.parent/'data/NAVTRAIN_SCENES.json')]
    with patch.object(worker,'PERFORMANCE_DIAGNOSTIC',root):
        assert worker.known_owner_args(args)
        for flag,wrong in [('--parent-checkpoint','foreign'),('--output',str(root/'other')),
                           ('--export','foreign'),('--cohort','NAVTEST'),('--manifest','foreign')]:
            changed=args.copy();changed[changed.index(flag)+1]=wrong
            assert not worker.known_owner_args(changed)
        (export/'COMPLETE').write_text(json.dumps(dict(export_sha256='changed')))
        assert not worker.known_owner_args(args)
