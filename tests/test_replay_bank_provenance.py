import numpy as np
import pytest
from navsim.agents.EpisodeDrive.scorer_replay import validate_train_tokens, group_identity, validate_group


def test_trainval_only_and_labels_bound_to_full_group():
    with pytest.raises(ValueError):
        validate_train_tokens(['test'], {'split': 'navtest', 'tokens': ['test']})
    with pytest.raises(ValueError):
        validate_train_tokens(['test'], {'split': 'trainval', 'tokens': ['train']})
    proposals = np.zeros((64, 8, 3), dtype=np.float32)
    identity = group_identity('train', proposals, 'evaluator', 'checkpoint', 'scene')
    validate_group(identity, 'train', proposals)
    proposals[63, 7, 0] += .01
    with pytest.raises(RuntimeError, match='Stale'):
        validate_group(identity, 'train', proposals)
    with pytest.raises(ValueError, match='complete'):
        group_identity('train', proposals[:1], 'evaluator', 'checkpoint', 'scene')
