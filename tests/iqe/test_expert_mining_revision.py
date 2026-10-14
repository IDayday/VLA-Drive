from types import SimpleNamespace
import pytest
from iqe.internal_baseline.bank import failure_and_target_audit, COMPONENTS
from iqe.internal_baseline.expert_data_revision import artifact, ARTIFACTS
from iqe.io import atomic_json, file_hash


def label(value):
    return {'score_valid': True, 'total_score_01': value,
            'named_components': {c: 1. for c in COMPONENTS},
            'component_valid_masks': {c: True for c in COMPONENTS}}


def test_expansion_only_changes_failure_threshold_not_target_gain():
    base, gt = label(.82), label(.90)
    assert not failure_and_target_audit(base, gt)['eligible']
    assert failure_and_target_audit(base, gt, failure_threshold=.85)['eligible']
    assert not failure_and_target_audit(base, label(.86), failure_threshold=.85)['eligible']
    assert not failure_and_target_audit(label(.70), label(.79), failure_threshold=.85)['eligible']
    assert not failure_and_target_audit(label(.85), label(.95), failure_threshold=.85)['eligible']


def test_selected_revision_preserves_legacy_and_rejects_changed_files(tmp_path):
    config = SimpleNamespace(root=tmp_path)
    atomic_json(tmp_path/'BASE_BINDING.json', {'checkpoint': 'same'})
    atomic_json(tmp_path/'DATA_PARTITION.json', {'roles': 'same'})
    atomic_json(tmp_path/'EXPERT_MANIFEST.json', {'hard': ['old']})
    legacy = file_hash(tmp_path/'EXPERT_MANIFEST.json')
    assert artifact(config, 'EXPERT_MANIFEST.json') == tmp_path/'EXPERT_MANIFEST.json'
    folder = tmp_path/'expert_data_revisions'/'threshold85'
    for name in ARTIFACTS:
        atomic_json(folder/name, {'hard': ['new']})
    atomic_json(tmp_path/'SELECTED_EXPERT_DATA.json', {
        'directory': str(folder), 'base_binding_hash': file_hash(tmp_path/'BASE_BINDING.json'),
        'partition_hash': file_hash(tmp_path/'DATA_PARTITION.json'),
        'artifacts': {name: file_hash(folder/name) for name in ARTIFACTS}})
    assert artifact(config, 'EXPERT_MANIFEST.json') == folder/'EXPERT_MANIFEST.json'
    assert file_hash(tmp_path/'EXPERT_MANIFEST.json') == legacy
    atomic_json(folder/'EXPERT_MANIFEST.json', {'hard': ['tampered']})
    with pytest.raises(ValueError, match='checksum'):
        artifact(config, 'EXPERT_MANIFEST.json')
