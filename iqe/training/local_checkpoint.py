"""Verified local staging of immutable checkpoints; shared originals stay intact."""
from pathlib import Path
import os
import shutil
import tempfile
from ..io import file_hash, lock
from ..contracts import require


def stage_checkpoint(source, checksum, root):
    source = Path(source)
    if root is None:
        return source
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root / (checksum + ".pt")
    with lock(str(target) + ".lock"):
        if target.exists():
            require(file_hash(target) == checksum, "corrupt local checkpoint staging")
            return target
        require(shutil.disk_usage(root).free >= source.stat().st_size + 1024**3,
                "insufficient local disk for verified checkpoint staging")
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint_", dir=root)
        try:
            with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst, length=8*1024*1024)
                dst.flush(); os.fsync(dst.fileno())
            require(file_hash(temporary) == checksum, "staged checkpoint checksum mismatch")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return target
