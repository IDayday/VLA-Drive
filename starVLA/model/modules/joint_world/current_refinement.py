"""Strict optional current-head restore, bound to the exact public foundation."""
import hashlib
from pathlib import Path
import torch


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def head_code_digest():
    root = Path(__file__).resolve().parents[1] / 'structured_world'
    return {name: file_digest(root / name) for name in ('rehab.py', 'agent_heads.py')}


def restore_current_head(head, checkpoint, foundation_sha256, public_origin):
    """Labels are never opened. Reject a head trained on different upstream features."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = saved['identity']
    if identity.get('kind') != 'public_frozen_feature_current_refinement_v1':
        raise ValueError('Not a supported current-head refinement checkpoint')
    if identity['foundation_sha256'] != foundation_sha256:
        raise ValueError('Current refinement parent foundation differs')
    if identity['public_origin'] != public_origin or public_origin.get('private_driving_weights_loaded', True):
        raise ValueError('Current refinement public origin differs')
    if identity.get('head_source_files') != head_code_digest():
        raise ValueError('Current head implementation changed')
    if not identity.get('future_labels_erased') or identity.get('Navtest_consulted', True):
        raise ValueError('Current refinement supervision contract differs')
    head.load_state_dict(saved['head'], strict=True)
    return {'checkpoint_sha256': file_digest(checkpoint), 'foundation_sha256': foundation_sha256,
            'epoch': saved['epoch'], 'step': saved['step'], 'selection': identity['selection'],
            'head_source_files': identity['head_source_files']}
