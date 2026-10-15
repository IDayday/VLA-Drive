"""Load the recorded S implementation and its explicit action codec.

Training-wrapper provenance and the unchanged model-runtime revision are
separate identities. Old full-data anchors retain their original defaults.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def activate(anchor):
    helper = _load('_anchor_runtime_helpers', ROOT/'tools/decision_local/common.py')
    legacy = Path(anchor['source_root'])
    expected = anchor.get('runtime_source_sha', anchor['training_source_sha'])
    if helper.source_sha(legacy) != expected:
        raise ValueError('Pinned S model runtime changed')
    sys.path.insert(0, str(legacy))
    import starVLA
    if Path(starVLA.__file__).resolve().parent != (legacy/'starVLA').resolve():
        raise RuntimeError('Pinned model runtime was shadowed')
    if anchor.get('training_role') == 'A_train':
        codec = helper.read(anchor['action_codec'])
        if (codec['identity'] != anchor['action_codec_identity'] or
                helper.digest(anchor['action_codec']) != anchor['action_codec_sha256']):
            raise ValueError('A-only action normalization changed')
        if codec['source_audit_identity'] != anchor['source_audit_identity']:
            raise ValueError('Codec belongs to another base population')
        _load('_anchor_action_codec', ROOT/'starVLA/dataloader/a_only_codec.py').install_legacy_codec(codec)
    elif anchor.get('action_codec') is not None:
        raise ValueError('A new codec requires an explicitly registered A-only anchor')
    return legacy


def selected_index(anchor, split, role=None):
    helper = _load('_anchor_selection_helpers', ROOT/'tools/decision_local/common.py')
    rows = helper.read(Path(anchor['data'][split]['current_root'])/'index.json')
    if role is None:
        if anchor.get('training_role') == 'A_train' and split == 'train':
            raise ValueError('Choose the explicit A/B role; do not encode all training logs implicitly')
        return rows
    if split != 'train' or role not in ('A_train', 'A_calibration', 'B_predictor', 'B_refiner'):
        raise ValueError('A/B role is only valid within the registered training population')
    record = anchor.get('role_indices', {}).get(role)
    if not record or helper.digest(record['path']) != record['sha256']:
        raise ValueError('Unregistered or changed role index')
    selected = helper.read(record['path'])
    lookup = {r['token']: r for r in rows}
    if any(lookup.get(r['token']) != r for r in selected):
        raise ValueError('Role index is outside the original current-input manifest')
    return selected
