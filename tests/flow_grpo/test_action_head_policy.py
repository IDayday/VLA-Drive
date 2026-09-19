import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.action_head_policy import (
    freeze_for_action_head, assert_frozen_prefix, FrozenFeatureStore,
    install_frozen_features, CachedKeyedDataset,
)
from starVLA.rl.flow_grpo.observation import prepare_policy_observation


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(3, 4)
        self.action_input_model = nn.Linear(2, 3)
        self.action_model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        self.calls = 0
        self.config = OmegaConf.create({'datasets': {'video_data': {'load_2d_data': False},
            'gs_data': {'load_3d_data': False}, 'reward_data': {'load_reward_data': False}},
            'framework': {'qwenvl': {'sft_feature_output': 'normalized'}}})

    def encode_policy_features(self, examples, feature_output=None):
        self.calls += 1
        state = torch.tensor(np.array([x['state'] for x in examples])).float()
        raw = self.qwen_vl_interface(self.action_input_model(state))
        return raw, {'action': torch.zeros((len(examples), 1), dtype=torch.long)}


def sample():
    return {'token': 'abcdef01', 'state': np.ones((1, 2), np.float32), 'lang': 'straight',
            'image': ('current-image',), 'action': np.ones((8, 4), np.float32)}


def test_object_contract_preserves_source_freezes_and_rejects_cross_boundary_alias():
    p = SmallPolicy().eval(); p.action_model[0].bias.requires_grad_(False)
    original = {id(x) for x in p.parameters() if x.requires_grad}
    head = {id(x) for x in p.action_model.parameters()}
    freeze_for_action_head(p)
    assert {id(x) for x in p.parameters() if x.requires_grad} == original & head
    assert_frozen_prefix(p)
    p.other = p.action_model[0]
    with pytest.raises(ValueError, match='shares'): freeze_for_action_head(p)


def test_cached_prefix_preserves_trainable_projection_gradients_and_original_labels(tmp_path):
    p = SmallPolicy().eval(); freeze_for_action_head(p)
    ref = copy.deepcopy(p); s = sample()
    expected, _ = p.encode_policy_features([s])
    store = FrozenFeatureStore(tmp_path, {'checkpoint': 'F'})
    install_frozen_features(p, store)
    dataset = CachedKeyedDataset({('abcdef01', 42): s}, store, p.encode_policy_features)
    cached = dataset[('abcdef01', 42)]
    assert cached['image'] == () and np.array_equal(cached['action'], s['action'])
    obs = prepare_policy_observation([cached])
    assert 'action' not in obs.examples()[0]
    raw, _ = p.encode_policy_features(obs.examples())
    assert torch.equal(expected, raw) and not raw.requires_grad
    p.action_model(raw).sum().backward()
    assert all(x.grad is not None for x in p.action_model.parameters())
    assert all(x.grad is None for x in p.qwen_vl_interface.parameters())
    calls = p.calls
    p.encode_policy_features([cached]); assert p.calls == calls
    ref.requires_grad_(False); install_frozen_features(ref, store)
    with torch.no_grad(): p.action_model[0].weight.add_(1)
    assert not torch.equal(p.action_model(raw), ref.action_model(raw))


@pytest.mark.parametrize('change', ['weights', 'processor', 'data', 'precision'])
def test_cache_identity_conflicts_fail_closed(tmp_path, change):
    FrozenFeatureStore(tmp_path, {change: 'old'})
    with pytest.raises(ValueError, match='conflict'): FrozenFeatureStore(tmp_path, {change: 'new'})


def test_cache_transactions_corruption_and_complete_reuse(tmp_path):
    s = sample(); store = FrozenFeatureStore(tmp_path, {'source': 'unit'})
    path, seal = store.paths(s['token']); path.parent.mkdir(); path.write_bytes(b'interrupted')
    store.put(s['token'], torch.ones(1, 1, 4), s)
    assert list(path.parent.glob('*.interrupted-*'))
    marker = store.finalize([s['token']]); assert store.finalize([s['token']]) == marker
    reader = FrozenFeatureStore(tmp_path, store.identity, readonly=True)
    assert reader.get(s['token'])['raw'].shape == (1, 1, 4)
    path.write_bytes(b'corrupt')
    reader.memory.clear()
    with pytest.raises(ValueError, match='content'): reader.get(s['token'])


def test_trainable_encoder_and_changed_observation_refused(tmp_path):
    p = SmallPolicy().eval(); store = FrozenFeatureStore(tmp_path, {'source': 'unit'})
    with pytest.raises(ValueError, match='fully frozen'): install_frozen_features(p, store)
    freeze_for_action_head(p); install_frozen_features(p, store)
    s = sample(); p.encode_policy_features([s]); s['lang'] = 'changed'
    with pytest.raises(ValueError, match='differs'): p.encode_policy_features([s])


def test_old_full_parameter_release_cannot_authorize_action_head(tmp_path, monkeypatch):
    from tests.flow_grpo.test_full_epoch import fixture, save
    from starVLA.rl.flow_grpo.config import config_hash
    from starVLA.rl.flow_grpo.acceptance import enforce_training_budget
    cfg, context, record, _, _ = fixture(tmp_path, monkeypatch)
    cfg['trainable_policy'] = 'action_head'
    context['config_sha256'] = config_hash(cfg)
    save(cfg, record)
    with pytest.raises(ValueError, match='own cached-prefix CUDA proof'):
        enforce_training_budget(cfg, context)
