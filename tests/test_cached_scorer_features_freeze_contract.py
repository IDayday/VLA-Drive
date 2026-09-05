import torch
import pytest
from navsim.agents.EpisodeDrive.scorer_replay import FrozenScorerCacheContract


def test_upstream_unfreeze_and_weight_change_invalidate_cache():
    module = torch.nn.Linear(4, 4).requires_grad_(False)
    contract = FrozenScorerCacheContract({'vision': module})
    contract.validate()
    module.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match='unfrozen'):
        contract.validate()
    module.requires_grad_(False)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match='changed'):
        contract.validate()
