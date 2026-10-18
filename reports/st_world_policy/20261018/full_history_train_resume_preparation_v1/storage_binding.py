"""Reuse the reviewed two-source W bridge with the fixed metadata-v3 runtime."""
import hashlib
import os
from pathlib import Path
import stat
import types

RUNTIME = Path('/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018')
RUNTIME_COMMIT = 'eae8922d72b3881c0c162e96b8dd67e7c15a41c7'
METADATA_SHA = '65644ac00b6a395bf2e1eb498e9f221a3cdebee63a027857513b1f50118151a6'
HISTORY32 = Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/temporal_history32_preparation_v1/run_history32_once.py')
HISTORY32_SHA = '7bf6934f0efcf0abc21c5f8be5cfa08e6b210ecde1aa3f648bc8a91534df8a22'


def source_bytes(path, sha):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Canonical source required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1 << 20:
            raise ValueError('Small regular source required')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read((1 << 20) + 1)
        after = os.fstat(fd)
        version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if version(before) != version(after) or version(after) != version(path.stat()) or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Pinned source changed')
        return raw
    finally:
        os.close(fd)


def inherited_binding():
    """No execution of the old entry; retain its original bridge implementation."""
    module = types.ModuleType('full_history_inherited_history32')
    module.__file__ = str(HISTORY32)
    exec(compile(source_bytes(HISTORY32, HISTORY32_SHA), str(HISTORY32), 'exec'), module.__dict__)
    module.RUNTIME = RUNTIME
    module.PINS = dict(module.PINS, **{'tools/st_world_policy/metadata_storage.py': METADATA_SHA})
    module.W_COMPAT = dict(module.W_COMPAT,
        current_validator=dict(module.W_COMPAT['current_validator'],
                               path=str(RUNTIME / 'tools/st_world_policy/spatial_w_storage.py')))
    return module


def install():
    module = inherited_binding()
    receipt = module.install_current_w_compatibility(module.W_COMPAT)
    return module, receipt
