from dataclasses import replace
import numpy as np
import pytest

from conftest import scene
from iqe.data.manifests import save_scenes
from iqe.internal_baseline.config import Config
from iqe.internal_baseline.partition import partition_fit, training_ids
from iqe.internal_baseline.scorer_scope import amend_scope, scorer_training_ids, scope, dependencies
from iqe.io import atomic_json, file_hash, read_json


def setup(tmp_path):
    cfg = Config("scope-unit", str(tmp_path), "unit", "unit", "unit", fit_limit="all_fit",
                 research_protocol="two_candidate_v1", scorer_dropout=0.)
    rows = [scene(i, group=f"log{i//2}") for i in range(20)]
    rows += [scene(100, role="stage_val"), scene(101, role="selector_cal"), scene(102, role="dev_report")]
    chosen = partition_fit(cfg, rows) + rows[-3:]
    save_scenes(cfg.root/"scenes.json", chosen)
    atomic_json(cfg.root/"BASE_BINDING.json", {"immutable_test_binding": True})
    return cfg, chosen


def test_explicit_full_fit_scope_includes_de_and_ds_but_not_evaluation_roles(tmp_path):
    cfg, rows = setup(tmp_path)
    original = scorer_training_ids(cfg, rows)
    de = training_ids(cfg, rows, "expert_fit")
    assert original == training_ids(cfg, rows, "scorer_fit") and not original & de
    old_partition = file_hash(cfg.root/"DATA_PARTITION.json")
    old_binding = file_hash(cfg.root/"BASE_BINDING.json")
    amended = amend_scope(cfg, "explicit user full fit choice")
    assert scorer_training_ids(cfg, rows) == de | original
    assert len(scorer_training_ids(cfg, rows)) == 20
    assert all(s.scene_id not in scorer_training_ids(cfg, rows) for s in rows if s.split_role != "incremental_fit")
    assert training_ids(cfg, rows, "expert_fit") == de
    assert file_hash(cfg.root/"DATA_PARTITION.json") == old_partition
    assert file_hash(cfg.root/"BASE_BINDING.json") == old_binding
    assert dependencies(cfg)["scorer_fit_scope"] == amended["amendment_hash"]
    assert amend_scope(cfg, "same identity is idempotent") == amended


@pytest.mark.parametrize("filename", ["BASE_BINDING.json", "DATA_PARTITION.json", "scenes.json"])
def test_scope_rejects_changed_dependencies(tmp_path, filename):
    cfg, _ = setup(tmp_path)
    amend_scope(cfg, "test")
    path = cfg.root/filename
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="dependency changed"):
        scope(cfg)


def test_scope_cannot_be_added_after_optimizer_start(tmp_path):
    cfg, _ = setup(tmp_path)
    atomic_json(cfg.root/"expert/status.json", {"step": 1})
    with pytest.raises(ValueError, match="after head training"):
        amend_scope(cfg, "late change")


def test_full_fit_label_injection_is_rejected(tmp_path):
    cfg, rows = setup(tmp_path)
    amend_scope(cfg, "test")
    changed = [replace(s, split_role="stage_val") if s.scene_id == "s0" else s for s in rows]
    with pytest.raises(ValueError, match="subrole scene/role mismatch"):
        scorer_training_ids(cfg, changed)


def test_actual_scorer_entrypoint_rejects_ds_only_bank_under_all_fit_recipe(tmp_path, monkeypatch):
    from iqe.internal_baseline import training
    cfg, rows = setup(tmp_path)
    amend_scope(cfg, "test")
    ds = sorted(training_ids(cfg, rows, "scorer_fit"))
    monkeypatch.setattr(training, "prepare_head_runtime", lambda c: None)
    monkeypatch.setattr(training, "load_bank", lambda c: ({"rows": [{"scene_id": i} for i in ds]},
                                                       {"scores": np.ones((len(ds), 2))}))
    with pytest.raises(ValueError, match="complete fit population"):
        training.train_scorer(cfg, "cpu")


def test_actual_scorer_entrypoint_delivers_both_fit_subroles_to_both_streams(tmp_path, monkeypatch):
    from iqe.internal_baseline import training, cache_execution
    cfg, rows = setup(tmp_path)
    amendment = amend_scope(cfg, "test full legal training population")
    ids = [s.scene_id for s in rows]
    scores = np.tile([.9, .3], (len(rows), 1))
    bank = {"rows": [{"scene_id": s.scene_id, "role": s.split_role} for s in rows]}
    atomic_json(cfg.root/"PAIR_BANK.json", bank)
    atomic_json(cfg.root/"SCORER_INITIALIZATION.json", {"test": True})
    monkeypatch.setattr(training, "prepare_head_runtime", lambda c: None)
    monkeypatch.setattr(training, "load_bank", lambda c: (bank, {"scores": scores}))
    cached = []
    monkeypatch.setattr(cache_execution, "training_cache",
                        lambda c, scenes, **kw: cached.extend(s.scene_id for s in scenes))
    monkeypatch.setattr(training, "make_scorer", lambda c: None)
    def inspect_train(module, loss, fetch, sampler, recipe, deps, output, **kwargs):
        allowed = scorer_training_ids(cfg, rows)
        assert set(sampler.ids) == set(cached) == allowed
        assert not set(ids[-3:]) & allowed
        assert deps["scorer_fit_scope"] == amendment["amendment_hash"]
        assert np.allclose(sampler.q, np.ones(len(allowed))/len(allowed))
        sampler.step = cfg.pointwise_updates
        batch = sampler.peek(128)
        de = training_ids(cfg, rows, "expert_fit")
        ds = training_ids(cfg, rows, "scorer_fit")
        for stream in ("natural", "comparison"):
            chosen = {e["scene_id"] for e in batch if e[stream]}
            assert chosen & de and chosen & ds
            assert chosen <= allowed
        atomic_json(output/"result.json", {"status": "TEST_ENTRYPOINT_VALIDATED"})
    monkeypatch.setattr(training, "train", inspect_train)
    assert training.train_scorer(cfg, "cpu")["status"] == "TEST_ENTRYPOINT_VALIDATED"
