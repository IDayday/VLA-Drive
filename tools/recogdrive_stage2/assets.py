import hashlib
import json
import subprocess
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
    if actual != revision or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source).strip():
        raise ValueError('Official source must be clean and match the registered revision')
    return source
