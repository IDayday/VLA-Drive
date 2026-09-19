from dataclasses import dataclass
import numpy as np
import pytest
from scripts.analysis.action_head_cache_probe import restore_current_images
from starVLA.rl.flow_grpo.observation import prepare_policy_observation


@dataclass
class Chain:
    observation: object
    chain: object
    old_elementwise_logprob: object


@pytest.mark.parametrize('bad', [None, 'token', 'lang', 'state', 'image'])
def test_oracle_hydration_keeps_behavior_and_excludes_labels(bad):
    saved = {'token': 'a', 'lang': 'go', 'state': np.ones((1, 3)), 'image': ()}
    chain = Chain(prepare_policy_observation([saved]), object(), object())
    raw = dict(saved, image=('current1', 'current2', 'current3'), action='FUTURE_LABEL')
    if bad:
        raw[bad] = {'token': 'b', 'lang': 'different', 'state': np.zeros((1, 3)), 'image': ()}[bad]
        with pytest.raises(ValueError): restore_current_images(chain, raw)
    else:
        restored = restore_current_images(chain, raw)
        assert restored.chain is chain.chain
        assert restored.old_elementwise_logprob is chain.old_elementwise_logprob
        assert 'action' not in restored.observation.examples()[0]
        assert restored.observation.images == (raw['image'],)
