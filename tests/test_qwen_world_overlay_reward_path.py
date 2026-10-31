import os
import sys
from types import ModuleType

import pytest

from starVLA.rl.safe_flow_improvement import v1_reward


@pytest.mark.parametrize('different_version', [False, True])
def test_v1_module_guard_normalizes_linked_frozen_scorer_tree(monkeypatch, tmp_path, different_version):
    base = tmp_path/'base/navsim_v1.1/navsim'
    (base/'navsim').mkdir(parents=True)
    overlay = tmp_path/'overlay';overlay.mkdir()
    (overlay/'navsim_v1.1').symlink_to(base.parent, target_is_directory=True)
    module = ModuleType('navsim')
    module.__file__ = str(tmp_path/'v2/navsim/__init__.py' if different_version else base/'navsim/__init__.py')
    monkeypatch.setitem(sys.modules, 'navsim', module)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(v1_reward, 'protocol_identity', lambda repo: {'sha256': 'canonical'})
    spec = dict(repo=str(overlay), maps_root='/maps', protocol_sha256='canonical')
    if different_version:
        with pytest.raises(RuntimeError, match='v1/v2 module collision'):
            v1_reward.initialize(spec)
    else:
        v1_reward.initialize(spec)
