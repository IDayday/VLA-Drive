import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def check_official(source, revision):
    source = Path(source)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                     cwd=source, text=True).splitlines()
    changed = [line for line in status if not (line.startswith('?? ') and
               '/__pycache__/' in line and line.endswith('.pyc'))]
    # Python bytecode generated during a read-only check is not source. No .py
    # additions or tracked edits are allowed; prevent further bytecode writes.
    sys.dont_write_bytecode = True
    if actual != revision or changed:
        raise ValueError('Official source must be clean and match the registered revision')
    return source
