"""Independent synthetic negatives; never open actual campaign artifacts."""
import fcntl
import hashlib
import json
from pathlib import Path
import zlib
import pytest
from tools.st_world_policy import finalize_future_metadata as final
from tools.st_world_policy import prepare_future_metadata as f
from tests.st_world.test_finalize_future_metadata import make_case, repin
from tests.st_world.test_metadata_storage_guard import write

@pytest.mark.parametrize('change',['wrong_role','successful_return','cleanup_unknown','automatic_retry','teacher_bytes'])
def test_terminal_fail_closed_after_coherent_repin(tmp_path,monkeypatch,change):
    case=make_case(tmp_path,monkeypatch);x=json.loads(case.terminal.read_text())
    if change=='wrong_role': x['role']='dev'
    if change=='successful_return': x['resources']['returncode']=0
    if change=='cleanup_unknown': x['resources']['cleanup']={}
    if change=='automatic_retry': x['automatic_retry']=True
    if change=='teacher_bytes': x['new_teacher_tensor_bytes']=1
    write(case.terminal,x);case.reg['source_terminal']['sha256']=f.file_sha256(case.terminal);repin(case)
    with pytest.raises(ValueError,match='failure/cleanup'):final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()

@pytest.mark.parametrize('change',['double_member','duplicate_key','nonfinite_constant'])
def test_replay_defensive_json_and_concatenation(tmp_path,change):
    b=f.PreparationBudget(tmp_path,dict(max_seconds=20,max_rss_bytes=2<<30,max_metadata_bytes=1<<20))
    c=final.CompressedScenes([{'token':'x'}],b,1<<20,128)
    c.add({'token':'x','v':1})
    if change=='double_member':
        blob,n,h=c.values['x'];c.values['x']=(blob+zlib.compress(b'{}'),n,h)
    else:
        raw=b'{"token":"x","v":1,"v":2}' if change=='duplicate_key' else b'{"token":"x","v":NaN}'
        c.values['x']=(zlib.compress(raw),len(raw),hashlib.sha256(raw).digest())
    with pytest.raises(ValueError):list(c())


def test_original_writer_lock_excludes_finalizer_without_mutation(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch);old={p.name:p.read_bytes() for p in case.source.iterdir() if p.is_file()}
    with (case.source/'BUILD.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()
    assert old=={p.name:p.read_bytes() for p in case.source.iterdir() if p.is_file()}


def test_validly_signed_wrong_physical_time_cannot_complete(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch);path=f.shard_path(case.source,case.plan['selected_logs'][0]);s=json.loads(path.read_text())
    s['scenes'][0]['frames'][1]['requested_s']=3.0;s.pop('identity');write(path,f.signed(s))
    case.reg['source_shards']=final.shard_census(case.plan,case.source)[0];repin(case)
    with pytest.raises(ValueError,match='time layout'):final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()


def test_census_change_before_final_commit_does_not_publish_complete(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch);original=final.shard_census;calls=[]
    def changed(*args):
        census,rows=original(*args);calls.append(True)
        if len(calls)==2:census=dict(census,identity='0'*64)
        return census,rows
    monkeypatch.setattr(final,'shard_census',changed)
    with pytest.raises(ValueError,match='before completion'):final.run(case.args)
    assert (case.output/'TEACHER_REQUESTS.json').exists()
    assert not (case.output/final.COMPLETE_NAME).exists()


@pytest.mark.parametrize('change',['request_sha','request_identity','scene_count','manifest_header','wrong_filename'])
def test_completion_consumer_rejects_mismatched_verified_inputs(tmp_path,monkeypatch,change):
    case=make_case(tmp_path,monkeypatch);done=final.run(case.args);m=json.loads((case.output/'TEACHER_REQUESTS.json').read_text())
    kw=dict(request_sha256=f.file_sha256(case.output/'TEACHER_REQUESTS.json'),request_identity=m['identity'],scene_count=2,manifest_header={k:v for k,v in m.items() if k!='scenes'})
    name=final.COMPLETE_NAME
    if change=='request_sha':kw['request_sha256']='0'*64
    if change=='request_identity':kw['request_identity']='0'*64
    if change=='scene_count':kw['scene_count']=1
    if change=='manifest_header':kw['manifest_header']['selection_identity']='0'*64
    if change=='wrong_filename':name='COMPLETE.json'
    with pytest.raises(ValueError):final.validate_completion(done,case.output/name,**kw)


def test_terminal_modified_after_rows_is_rechecked_before_complete(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch);original=f.validate_target_manifest;changed=[]
    def mutate(manifest):
        result=original(manifest)
        if not changed:case.terminal.write_bytes(b'{}');changed.append(True)
        return result
    monkeypatch.setattr(f,'validate_target_manifest',mutate)
    with pytest.raises(ValueError,match='checksum'):final.run(case.args)
    assert not (case.output/final.COMPLETE_NAME).exists()
