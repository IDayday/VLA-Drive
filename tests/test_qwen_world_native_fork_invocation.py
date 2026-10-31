import copy
import json

import pytest

from starVLA.rl.simwam_recipe import invocation as inv


def material(tmp_path,monkeypatch):
    monkeypatch.setattr(inv,'gather',lambda x:[x])
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'files.json').write_text('{}')
    (parent/'identity.json').write_text('{}')
    sha=inv.file_sha(parent/'files.json')
    (parent/'COMPLETE').write_text(json.dumps(dict(schema=2,world=8,phase='READY',updates=10000,next_inner=0,files_sha256=sha)))
    f=dict(schema='world_Safety10k_exact_native_LR_fork_v1',checkpoint=str(parent),expected_updates=10000,
           initial_global_updates=10000,optimizer_state_inherited=True,RNG_state_inherited=True,
           stream_cursor_inherited=True,weights_only=False,phase_reset=False,
           checkpoint_files_sha256=sha,checkpoint_identity_sha256=inv.file_sha(parent/'identity.json'))
    request={k:f[k] for k in ('schema','checkpoint','expected_updates','checkpoint_files_sha256','checkpoint_identity_sha256')}
    return dict(sha256='child',native_fork_provenance=f,recipe=dict(research=dict(native_fork=request))),dict(updates=10000,collections=2500,scene_cursor=20000)


def row(u):
    return dict(updates=u,collection=(u-1)//4+1,inner_completed=(u-1)%4+1,
                behavior=dict(update=((u-1)//4)*4,run='child'))


def test_native_fork_prefix_is_sealed_and_child_history_remains_contiguous(tmp_path,monkeypatch):
    identity,state=material(tmp_path,monkeypatch);root=tmp_path/'child'
    first=inv.Invocation(root,0,state,identity)
    assert first.record['journal_origin_update']==10000
    assert not (root/'metrics_rank0.jsonl').exists()  # No invented parent history.
    with pytest.raises(ValueError,match='effective learning history'):
        inv.Invocation(tmp_path/'missing_child_history',0,dict(state,updates=10001),identity)
    journal=root/'metrics_rank0.jsonl'
    journal.write_text(''.join(json.dumps(row(u))+'\n' for u in (10001,10002)))
    state['updates']=10002;state['collections']=2501;state['scene_cursor']=20008
    inv.Invocation(root,0,state,identity)
    journal.write_text(json.dumps(row(10002))+'\n')
    with pytest.raises(ValueError,match='gap/duplicate'):inv.Invocation(root,0,state,identity)
    wrong=row(10001);wrong['behavior']['run']='parent'
    journal.write_text(json.dumps(wrong)+'\n'+json.dumps(row(10002))+'\n')
    with pytest.raises(ValueError,match='another scientific run'):inv.Invocation(root,0,state,identity)


def test_original_resume_guard_and_corrupt_native_fork_parent_are_rejected(tmp_path,monkeypatch):
    identity,state=material(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='effective learning history'):
        inv.Invocation(tmp_path/'ordinary',0,state,dict(sha256='ordinary'))
    bad=copy.deepcopy(identity);bad['native_fork_provenance']['weights_only']=True
    with pytest.raises(ValueError,match='provenance'):inv.Invocation(tmp_path/'bad',0,state,bad)
    (tmp_path/'parent/files.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checkpoint changed'):inv.Invocation(tmp_path/'changed',0,state,identity)
