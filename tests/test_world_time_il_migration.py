import json
import os
from pathlib import Path
import pytest
from tools.qwen_world_postcot.job_worker import shared_training_host
from tools.qwen_world_postcot import fresh_loss_owner as owner


def test_unbounded_host_is_not_silently_permanent_or_shared(tmp_path):
    grant=dict(permanent_hosts=['fixed'],unbounded_temporary_hosts=['temporary'],record=str(tmp_path/'grant.json'))
    assert shared_training_host('fixed',grant)
    assert not shared_training_host('temporary',grant)
    grant['shared_training_hosts']=['temporary'];Path(grant['record']).write_text(json.dumps(grant))
    assert shared_training_host('temporary',grant)
    assert grant['permanent_hosts']==['fixed']


def test_shared_host_requires_unchanged_immutable_record(tmp_path):
    record=tmp_path/'grant.json';record.write_text(json.dumps(dict(shared_training_hosts=[])))
    grant=dict(permanent_hosts=[],unbounded_temporary_hosts=['temp'],shared_training_hosts=['temp'],record=str(record))
    with pytest.raises(PermissionError):shared_training_host('temp',grant)


def test_adopted_test_waits_for_real_owner_exit(monkeypatch,tmp_path):
    done=tmp_path/'COMPLETE.json';done.write_text(json.dumps(dict(status='COMPLETE',all_owned_workers_exited=True)))
    monkeypatch.setattr(owner,'process_alive',lambda receipt:True)
    handle=owner.AdoptedProcess({},done,('COMPLETE',),require_workers_exited=True)
    assert handle.poll()is None
    monkeypatch.setattr(owner,'process_alive',lambda receipt:False)
    assert handle.poll()==0


def test_adopted_test_rejects_missing_worker_exit_proof(monkeypatch,tmp_path):
    done=tmp_path/'COMPLETE.json';done.write_text(json.dumps(dict(status='COMPLETE')))
    monkeypatch.setattr(owner,'process_alive',lambda receipt:False)
    assert owner.AdoptedProcess({},done,('COMPLETE',),require_workers_exited=True).poll()==1


def test_adopted_arm_rejects_unrelated_live_process(monkeypatch,tmp_path):
    arm=tmp_path/'TIME_RESIDUAL';arm.mkdir();source=tmp_path/'source';source.mkdir()
    receipt=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(source),arm='TIME_RESIDUAL')
    (arm/'OWNER.json').write_text(json.dumps(receipt));monkeypatch.setattr(owner,'ROOT',tmp_path);monkeypatch.setattr(owner,'SOURCE',source)
    with pytest.raises(ValueError,match='unrelated'):owner.existing_arm('TIME_RESIDUAL')
