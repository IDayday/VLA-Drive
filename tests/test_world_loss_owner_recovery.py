import json
import os
from pathlib import Path
import pytest
from tools.qwen_world_postcot import loss_designs_owner as m


def setup(tmp_path,monkeypatch):
    root=tmp_path/'run';source=tmp_path/'source';root.mkdir();source.mkdir();f=source/'scientific.py';f.write_text('UNCHANGED=True\n')
    owner=dict(pid=4294967294,start_ticks=1);(root/'OWNER.json').write_text(json.dumps(owner))
    failure=root/'failure_evidence.json';failure.write_text(json.dumps(dict(state=dict(jobs={},arms=dict(CONTROL='COMPLETE'),source=str(source)))))
    receipt=dict(schema='world_loss_owner_UTF8_recovery_v1',formal_updates_before_recovery=0,
        scientific_source_files={'scientific.py':m.file_sha(f)},previous_owner=owner,previous_failure=str(failure),previous_failure_sha256=m.file_sha(failure))
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'SOURCE',source);monkeypatch.setattr(m.c,'poll_jobs',lambda state:None)
    monkeypatch.setenv('PYTHONUTF8','1')
    return root,source,receipt


def save(root,receipt):
    receipt.pop('sha256',None);receipt['sha256']=m.fingerprint(receipt)
    path=root/'receipt.json';path.write_text(json.dumps(receipt));return path


def test_explicit_recovery_uses_new_delta_evidence_path_and_fixed_hosts(tmp_path,monkeypatch):
    root,source,receipt=setup(tmp_path,monkeypatch);state=m.recover_owner(save(root,receipt))
    assert state['fixed_hosts_only'] is True
    assert state['qualification_outputs']['DELTA'].endswith('/qualification/DELTA_UTF8_attempt002')
    assert state['arms']['CONTROL']=='COMPLETE'


def test_reject_scientific_source_change(tmp_path,monkeypatch):
    root,source,receipt=setup(tmp_path,monkeypatch);path=save(root,receipt);(source/'scientific.py').write_text('CHANGED=True\n')
    with pytest.raises(ValueError,match='Scientific source changed'):m.recover_owner(path)


def test_reject_live_original_owner(tmp_path,monkeypatch):
    root,source,receipt=setup(tmp_path,monkeypatch)
    owner=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    (root/'OWNER.json').write_text(json.dumps(owner));receipt['previous_owner']=owner
    with pytest.raises(RuntimeError,match='still alive'):m.recover_owner(save(root,receipt))


def test_missing_utf8_flag_never_retries_same_bad_environment(tmp_path,monkeypatch):
    root,source,receipt=setup(tmp_path,monkeypatch);monkeypatch.delenv('PYTHONUTF8')
    with pytest.raises(ValueError,match='UTF8'):m.recover_owner(save(root,receipt))


def test_unrelated_failure_not_silently_accepted(tmp_path,monkeypatch):
    root,source,receipt=setup(tmp_path,monkeypatch);failure=Path(receipt['previous_failure'])
    state=dict(source=str(source),jobs={'actor_failure':dict(status='FAILED')})
    failure.write_text(json.dumps(dict(state=state)));receipt['previous_failure_sha256']=m.file_sha(failure)
    with pytest.raises(ValueError,match='Unexpected failure'):m.recover_owner(save(root,receipt))
