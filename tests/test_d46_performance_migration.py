import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from starVLA.rl.d46_fpo_engineering import migration
from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('native_fixture', Path(__file__).with_name('test_d46_world16_execution_upgrade.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root, cp, source, old, _ = helper.fixture(tmp_path, monkeypatch, arm='D3_FPO')
    # A tiny CPU-only structural fixture. Production prepare still requires
    # the real D46 parent constant and refuses all qualification identities.
    old['qualification'] = False
    old.pop('sha256')
    old['sha256'] = fingerprint(old)
    atomic_json(cp/'identity.json', old)
    atomic_json(root/'identity.json', old)
    tx = json.loads((cp/'transaction.json').read_text())
    for rank in range(16):
        own = torch.load(cp/f'rank{rank}.pt', weights_only=False)
        own['identity_sha256'] = old['sha256']
        migration.checkpoint.atomic_torch(cp/f'rank{rank}.pt', own)
        tx['ranks'][rank]['run'] = old['sha256']
    atomic_json(cp/'transaction.json', tx)
    helper.reseal(cp)
    from starVLA.rl.d44_native_rl import parent
    monkeypatch.setattr(parent, 'D46_MODEL_SHA', old['parent_model_sha256'])
    monkeypatch.setattr(migration, 'inventory', lambda: {'new.py': 'a'*64})
    evidence = tmp_path/'evidence.json'
    atomic_json(evidence, dict(status='COMPLETE', world=16, source_checkpoint=str(cp), rows=[
        dict(mode='exact', native_roundtrip_exact=True, final_parameters_bitexact_to_baseline=True)]))
    return cp, source, old, evidence


def test_native_copy_preserves_optimizer_rng_state_and_seals_identity(tmp_path, monkeypatch):
    cp, source, old, evidence = fixture(tmp_path, monkeypatch)
    result = migration.prepare(cp, source, tmp_path/'new', profile='exact', evidence=evidence)
    target = Path(result['checkpoint'])
    assert file_sha(cp/'optimizer.pt') == file_sha(target/'optimizer.pt')
    new = json.loads((target/'identity.json').read_text())
    assert new['training_run_id'] == old['sha256']
    from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
    for rank in range(16):
        before = torch.load(cp/f'rank{rank}.pt', weights_only=False)
        after = torch.load(target/f'rank{rank}.pt', weights_only=False)
        assert after.pop('identity_sha256') == new['sha256']
        before.pop('identity_sha256')
        assert tensor_digest(before) == tensor_digest(after)
    migration.validate(tmp_path/'new', new)
    changed = copy.deepcopy(new)
    changed['recipe']['lr'] = 1.
    changed.pop('sha256'); changed['sha256'] = fingerprint(changed)
    with pytest.raises(ValueError):
        migration.validate(tmp_path/'new', changed)


def test_qualification_weights_and_unmeasured_profile_are_rejected(tmp_path, monkeypatch):
    cp, source, old, evidence = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='qualification'):
        migration.prepare(cp, source, tmp_path/'fast', profile='fast', evidence=evidence)
    old['qualification'] = True; old.pop('sha256'); old['sha256'] = fingerprint(old)
    atomic_json(cp/'identity.json', old)
    with pytest.raises(ValueError, match='formal'):
        migration.prepare(cp, source, tmp_path/'bad', profile='exact', evidence=evidence)


def test_legacy_pending_is_finished_before_fast_layout_activation(monkeypatch):
    from starVLA.rl.d46_fpo_engineering.runtime import PerformanceExtension
    from types import SimpleNamespace
    extension = PerformanceExtension()
    extension.saved = {'performance_execution': {'origin_updates': 1921}}
    extension.verified = False; extension.state = {'updates': 1921}; extension.caches = []
    activated = []
    extension.activate = lambda: activated.append(True)
    actor = SimpleNamespace()
    extension.configure_actor(actor)
    assert not activated and not actor.execution_fuse_fm_mc
    extension.delegate = SimpleNamespace(before_collection=lambda *args: None)
    extension.before_collection({}, None, 0, 16)
    assert activated == [True]
    extension.state = {'updates': 1923}
    extension.configure_actor(actor)
    assert len(activated) == 2
