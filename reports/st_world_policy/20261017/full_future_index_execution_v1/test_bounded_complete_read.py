"""Small CPU-only completion receipt races; no real index or live admission."""
import hashlib
import importlib.util
import os
from pathlib import Path
import time

import pytest

spec = importlib.util.spec_from_file_location('bounded_index_receipt', Path(__file__).with_name('run_index_once.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_growth_during_read_consumes_at_most_limit_plus_one(tmp_path, monkeypatch):
    p = tmp_path / 'COMPLETE.json'
    p.write_text('{}')
    original = m.os.read
    counts = []
    def growing(fd, limit):
        if not counts:
            with p.open('ab') as stream:
                stream.write(b' ' * (2 * m.MiB))
        raw = original(fd, limit)
        counts.append((limit, len(raw)))
        return raw
    monkeypatch.setattr(m.os, 'read', growing)
    with pytest.raises(ValueError, match='exceeded bound'):
        m.read_small_actual(p)
    assert sum(n for _, n in counts) == m.MiB + 1
    assert sum(n for _, n in counts) < p.stat().st_size
    assert max(n for n, _ in counts) == m.MiB + 1


@pytest.mark.parametrize('replacement', ['fifo', 'symlink', 'larger_file'])
def test_swap_after_lstat_cannot_block_or_open_unbounded_data(tmp_path, monkeypatch, replacement):
    p = tmp_path / 'COMPLETE.json'
    p.write_text('{}')
    target = tmp_path / 'target.json'
    target.write_text('{}')
    original = m.os.open
    flags_seen = []
    def swapping(path, flags, *args, **kwargs):
        if Path(path) == p:
            flags_seen.append(flags)
            p.unlink()
            if replacement == 'fifo':os.mkfifo(p)
            elif replacement == 'symlink':p.symlink_to(target)
            else:p.write_bytes(b'x' * (2 * m.MiB))
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(m.os, 'open', swapping)
    monkeypatch.setattr(m.os, 'read', lambda *_: pytest.fail('Changed descriptor must be rejected before read'))
    start = time.monotonic()
    with pytest.raises((ValueError, OSError)):
        m.read_small_actual(p)
    assert time.monotonic() - start < 1
    assert len(flags_seen) == 1
    assert flags_seen[0] & os.O_NOFOLLOW
    assert flags_seen[0] & os.O_NONBLOCK


@pytest.mark.parametrize('change', ['path_inode', 'fd_mtime', 'parent_symlink'])
def test_post_read_fd_path_and_parent_changes_are_rejected(tmp_path, monkeypatch, change):
    directory = tmp_path / 'dir'
    directory.mkdir()
    p = directory / 'COMPLETE.json'
    p.write_text('{}')
    original = m.os.read
    called = []
    def changing(fd, size):
        raw = original(fd, size)
        if not called:
            called.append(True)
            if change == 'path_inode':
                other = directory / 'replacement'
                other.write_text('{}')
                other.replace(p)
            elif change == 'fd_mtime':
                st = os.fstat(fd)
                os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
            else:
                moved = tmp_path / 'moved'
                directory.rename(moved)
                directory.symlink_to(moved, target_is_directory=True)
        return raw
    monkeypatch.setattr(m.os, 'read', changing)
    with pytest.raises(ValueError, match='changed or exceeded bound'):
        m.read_small_actual(p)


@pytest.mark.parametrize('change', ['oversize', 'hardlink', 'parent_symlink'])
def test_initial_invalid_files_are_rejected_before_read(tmp_path, monkeypatch, change):
    p = tmp_path / 'COMPLETE.json'
    p.write_text('{}')
    if change == 'oversize':p.write_bytes(b'x' * (m.MiB + 1))
    elif change == 'hardlink':os.link(p, tmp_path / 'alias')
    else:
        alias = tmp_path / 'alias'
        alias.symlink_to(tmp_path, target_is_directory=True)
        p = alias / 'COMPLETE.json'
    monkeypatch.setattr(m.os, 'read', lambda *_: pytest.fail('Invalid receipt must not be read'))
    with pytest.raises(ValueError, match='Small non-aliased COMPLETE'):
        m.read_small_actual(p)


def test_exact_limit_valid_json_uses_descriptor_and_hashes_all_bytes(tmp_path, monkeypatch):
    p = tmp_path / 'COMPLETE.json'
    payload = b'{}' + b' ' * (m.MiB - 2)
    p.write_bytes(payload)
    monkeypatch.setattr(Path, 'read_bytes', lambda *_: pytest.fail('Unbounded path read forbidden'))
    value, pin = m.read_small_actual(p)
    assert value == {}
    assert pin == {'path': str(p), 'sha256': hashlib.sha256(payload).hexdigest()}


def test_short_regular_reads_are_reassembled_without_relaxing_cap(tmp_path, monkeypatch):
    p = tmp_path / 'COMPLETE.json'
    payload = b'{"complete": true}'
    p.write_bytes(payload)
    original = m.os.read
    calls = []
    def short(fd, size):
        raw = original(fd, min(size, 3))
        calls.append(len(raw))
        return raw
    monkeypatch.setattr(m.os, 'read', short)
    value, pin = m.read_small_actual(p)
    assert value == {'complete': True}
    assert sum(calls) == len(payload) and len(calls) > 1
    assert pin['sha256'] == hashlib.sha256(payload).hexdigest()
