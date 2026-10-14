import pytest
from iqe.io import file_hash
from iqe.specialist_study.ram_stage import copy_verified


def test_stage_is_original_bytes_idempotent_and_rejects_corruption(tmp_path):
    source=tmp_path/'source';source.write_bytes(bytes(range(256))*8192)
    target=tmp_path/'ram'/'record';checksum=file_hash(source)
    copy_verified(source,target,checksum)
    assert source.read_bytes()==target.read_bytes()
    version=target.stat().st_mtime_ns
    copy_verified(source,target,checksum)
    assert target.stat().st_mtime_ns==version
    target.write_bytes(b'changed')
    with pytest.raises(Exception,match='corrupted'):copy_verified(source,target,checksum)
    target.unlink()
    with pytest.raises(Exception,match='source changed'):copy_verified(source,target,'wrong')
    assert not target.exists() and not list(target.parent.glob('.*'))
