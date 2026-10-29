import json
import os
from pathlib import Path
import pytest
from tools.qwen_world_postcot import optimized_il_owner as m


def receipt(tmp_path,monkeypatch):
    root=tmp_path/'root';source=tmp_path/'source';checkpoint=root/'checkpoint'
    root.mkdir();source.mkdir();checkpoint.mkdir();(checkpoint/'files.json').write_text('{}')
    scientific=source/'scientific.py';scientific.write_text('SAME=True\n')
    owner=dict(pid=4294967294,start_ticks=1);(root/'OWNER.json').write_text(json.dumps(owner))
    failure=root/'saved_failure.json';failure.write_text(json.dumps(dict(state=dict(jobs={},stage='FULL_IL'))))
    row=dict(schema='world_optimized_IL_authorization_recovery_v1',previous_owner=owner,
        scientific_source_files={'scientific.py':m.file_sha(scientific)},checkpoint=str(checkpoint),
        checkpoint_files_sha256=m.file_sha(checkpoint/'files.json'),previous_failure=str(failure),previous_failure_sha256=m.file_sha(failure))
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'SOURCE',source);monkeypatch.setattr(m.c,'poll_jobs',lambda state:None)
    return root,source,row


def seal(root,row):
    row.pop('sha256',None);row['sha256']=m.fingerprint(row);p=root/'receipt.json';p.write_text(json.dumps(row));return p


def test_reviewed_recovery_keeps_full_il_stage(tmp_path,monkeypatch):
    root,source,row=receipt(tmp_path,monkeypatch);state=m.recover_owner(seal(root,row))
    assert state['stage']=='FULL_IL'


def test_training_source_change_is_rejected(tmp_path,monkeypatch):
    root,source,row=receipt(tmp_path,monkeypatch);p=seal(root,row);(source/'scientific.py').write_text('CHANGED=True\n')
    with pytest.raises(ValueError,match='Scientific source'):m.recover_owner(p)


def test_live_owner_is_never_duplicated(tmp_path,monkeypatch):
    root,source,row=receipt(tmp_path,monkeypatch)
    row['previous_owner']=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    (root/'OWNER.json').write_text(json.dumps(row['previous_owner']))
    with pytest.raises(RuntimeError,match='still alive'):m.recover_owner(seal(root,row))
