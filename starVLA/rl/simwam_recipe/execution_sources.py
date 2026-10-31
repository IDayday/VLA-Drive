"""Bind imported project modules to a sealed snapshot or its pinned base tree."""
import hashlib
import json
from pathlib import Path


OVERLAY_SCHEMA = 'immutable_frozen_runtime_git_delta_overlay_v1'


def snapshot_roots(snapshot):
    root = Path(snapshot).resolve()
    roots = [root]
    metadata_path = root/'FROZEN_SOURCE.json'
    if not metadata_path.is_file():
        return tuple(roots)
    metadata = json.loads(metadata_path.read_text())
    while metadata.get('schema') == OVERLAY_SCHEMA:
        body = dict(metadata);seal = body.pop('sha256', None)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if seal != digest:
            raise ValueError('Execution overlay metadata seal changed')
        base = Path(metadata['base']['path']).resolve()
        if base in roots or len(roots) >= 16:
            raise ValueError('Cyclic or excessive frozen snapshot lineage')
        raw = (base/'FROZEN_SOURCE.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['base']['frozen_source_sha256']:
            raise ValueError('Pinned frozen base metadata changed')
        roots.append(base);metadata = json.loads(raw)
    return tuple(roots)


def checked_module_path(name, filename, roots):
    path = Path(filename).resolve();root = roots[0]
    if path.is_relative_to(root):
        return path
    # Preserve module aliases while requiring the exact file selected by the
    # current logical tree; a base file shadowed by an overlay is not accepted.
    for base in roots[1:]:
        if path.is_relative_to(base):
            logical = root/path.relative_to(base)
            if logical.is_file() and logical.resolve() == path:
                return path
    raise ValueError('Imported project module from outside the execution snapshot: '+str(path))
