"""Local, atomic publications. Failed attempts are evidence, never garbage."""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def publication_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Outside the renamed directory; process death releases flock automatically.
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def preserve_attempt(path):
    path = Path(path)
    target = path.with_name(
        f"{path.name}.attempt-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    )
    os.rename(path, target)
    return target
