import fcntl
from tools.qwen_world_postcot.fresh_loss_owner import held_shared_uuids


def test_shared_lease_filter_does_not_acquire_or_replace_live_owner(tmp_path):
    busy=tmp_path/'GPU-busy.lock';free=tmp_path/'GPU-free.lock';free.touch()
    with busy.open('a')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert held_shared_uuids(tmp_path)=={'GPU-busy'}
    assert held_shared_uuids(tmp_path)==set()


def test_no_lease_directory_is_empty_not_an_error(tmp_path):
    assert held_shared_uuids(tmp_path/'absent')==set()
