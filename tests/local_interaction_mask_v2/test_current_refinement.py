import copy
import pytest
import torch
from starVLA.model.modules.joint_world.current_refinement import head_code_digest, restore_current_head
from starVLA.model.modules.joint_world.local_planner import PublicLocalPolicy
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads


def test_refinement_restore_rejects_wrong_features_source_and_missing_weights(tmp_path):
    head = ReferenceAgentHeads(16, slots=4, classes=3, steps=2, dim=8)
    origin = {'private_driving_weights_loaded': False, 'public_revision': 'test'}
    identity = dict(kind='public_frozen_feature_current_refinement_v1', foundation_sha256='parent',
                    public_origin=origin, head_source_files=head_code_digest(), future_labels_erased=True,
                    Navtest_consulted=False, selection='holdout current F1 including epoch0')
    saved = dict(identity=identity, head=head.state_dict(), epoch=2, step=3)
    path = tmp_path / 'head.pt'; torch.save(saved, path)
    restored = ReferenceAgentHeads(16, slots=4, classes=3, steps=2, dim=8)
    metadata = restore_current_head(restored, path, 'parent', origin)
    x = torch.randn(2, 4, 16)
    for key, value in head(x).items():
        torch.testing.assert_close(restored(x)[key], value, atol=0, rtol=0)
    assert len(metadata['checkpoint_sha256']) == 64
    with pytest.raises(ValueError, match='parent foundation'):
        restore_current_head(restored, path, 'other_parent', origin)
    changed = copy.deepcopy(saved); changed['identity']['head_source_files']['rehab.py'] = 'changed'
    torch.save(changed, path)
    with pytest.raises(ValueError, match='implementation'):
        restore_current_head(restored, path, 'parent', origin)
    changed = copy.deepcopy(saved); del changed['head']['classifier.weight']; torch.save(changed, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        restore_current_head(restored, path, 'parent', origin)


def test_online_policy_requires_same_current_head_and_foundation():
    world = torch.nn.Linear(1, 1).requires_grad_(False)
    world.foundation_sha256 = 'foundation'
    world.perception_refinement = {'checkpoint_sha256': 'head'}
    identity = {'foundation_sha256': 'foundation', 'perception_override': {'checkpoint_sha256': 'head'}}
    PublicLocalPolicy(world, None, identity)
    with pytest.raises(ValueError, match='current head'):
        PublicLocalPolicy(world, None, {'foundation_sha256': 'foundation'})
    with pytest.raises(ValueError, match='foundation'):
        PublicLocalPolicy(world, None, dict(identity, foundation_sha256='other'))
