"""Budget-only revision: preserve Base/data identities and wire actual training."""
from dataclasses import asdict, replace
import pytest
from iqe.internal_baseline.config import Config
from iqe.internal_baseline.budget import (
    FILE, amend_training_budget, training_budget, training_dependencies, bind_training_budget,
)
from iqe.internal_baseline.training import training_config
from iqe.internal_baseline.sampling import TwoStreamSampler
from iqe.io import atomic_json, file_hash, read_json


TARGET = dict(expert_updates=10000, scorer_updates=20000, pointwise_updates=5000)


def fixture_config(tmp_path):
    c = Config(experiment_id="budget-fixture", output_root=str(tmp_path), contract="fixture",
               base_run="fixture", scene_manifest="fixture", research_protocol="two_candidate_v1", scorer_dropout=0.)
    atomic_json(c.root/"BASE_BINDING.json", {"checkpoint_hash": "synthetic-fixture-only"})
    atomic_json(c.root/"DATA_PARTITION.json", {"D_E": ["e"], "D_S": ["s"]})
    atomic_json(c.root/"resolved_config.json", asdict(c))
    return c


def test_amendment_leaves_base_config_and_partition_immutable_and_wires_budgets(tmp_path):
    c = fixture_config(tmp_path)
    hashes = {name: file_hash(c.root/name) for name in ("BASE_BINDING.json", "DATA_PARTITION.json", "resolved_config.json")}
    identity = c.identity
    first = amend_training_budget(c, TARGET, "explicit user longer training")
    assert amend_training_budget(c, TARGET, "idempotent invocation") == first
    assert {k: first[k] for k in TARGET} == TARGET
    assert c.identity == identity and c.expert_updates == 1000 and c.scorer_updates == 2000
    assert hashes == {name: file_hash(c.root/name) for name in hashes}
    assert training_config(c, "expert")["max_optimizer_steps"] == 10000
    assert training_config(c, "scorer")["max_optimizer_steps"] == 20000
    assert training_config(c, "expert")["warmup_steps"] == 50
    assert training_config(c, "scorer")["warmup_steps"] == 100
    assert training_config(c, "scorer")["schedule"] == "warmup_constant"
    assert training_dependencies(c) == {"training_budget_amendment": file_hash(c.root/FILE)}
    assert bind_training_budget(c, "expert")["expert_updates"] == 10000
    assert read_json(c.root/"expert/TRAINING_BUDGET.json")["budget"] == first


def test_phase_switch_uses_amended_boundary_and_one_sampler(tmp_path):
    c = fixture_config(tmp_path); amend_training_budget(c, TARGET, "longer budget")
    budget = training_budget(c)
    s = TwoStreamSampler(["a", "b"], [.95, .02], natural_seed=1, comparison_seed=2,
                         pointwise_updates=budget["pointwise_updates"], bank_hash="fixture-bank")
    s.step = 4999
    assert sum(e["natural"] for e in s.peek(128)) == 128
    s.consume(128)
    assert s.step == 5000
    assert sum(e["natural"] for e in s.peek(128)) == 64
    assert sum(e["comparison"] for e in s.peek(128)) == 64


@pytest.mark.parametrize("change", [dict(lr=.2), dict(expert_updates=True), dict(expert_updates=10000.),
                                    dict(expert_updates=10001), dict(scorer_updates=20001), dict(pointwise_updates=4999)])
def test_budget_cannot_modify_recipe_use_wrong_types_or_exceed_bounds(tmp_path, change):
    c = fixture_config(tmp_path)
    with pytest.raises(ValueError):
        amend_training_budget(c, TARGET | change, "invalid")
    assert not (c.root/FILE).exists()


@pytest.mark.parametrize("name", ["BASE_BINDING.json", "DATA_PARTITION.json"])
def test_changed_dependency_refuses_amendment_reuse(tmp_path, name):
    c = fixture_config(tmp_path); amend_training_budget(c, TARGET, "longer")
    atomic_json(c.root/name, {"changed": True})
    with pytest.raises(ValueError, match="binding mismatch"):
        training_budget(c)


def test_changed_recipe_or_amendment_identity_is_rejected(tmp_path):
    c = fixture_config(tmp_path); amend_training_budget(c, TARGET, "longer")
    with pytest.raises(ValueError, match="config changed"):
        training_budget(replace(c, training_rng_seed=999))
    data = read_json(c.root/FILE); data["retained_recipe"]["lr"] = .9
    atomic_json(c.root/FILE, data)
    with pytest.raises(ValueError, match="recipe"):
        training_budget(c)


@pytest.mark.parametrize("started_artifact", ["status.json", "TRAINING_BUDGET.json", "step_000250.pt.COMPLETE.json"])
def test_new_budget_rejected_once_training_lineage_begins(tmp_path, started_artifact):
    c = fixture_config(tmp_path); atomic_json(c.root/"expert"/started_artifact, {"started": True})
    with pytest.raises(ValueError, match="started"):
        amend_training_budget(c, TARGET, "too late")


@pytest.mark.parametrize("change", [dict(expert_updates=True), dict(scorer_updates=2000.), dict(pointwise_updates=True)])
def test_fresh_two_candidate_config_refuses_bool_or_float_budget(tmp_path, change):
    c = fixture_config(tmp_path)
    with pytest.raises(ValueError):
        replace(c, **change)


def test_legacy_budget_remains_unchanged(tmp_path):
    c = replace(fixture_config(tmp_path), research_protocol="internal_legacy_v1")
    assert training_budget(c)["scorer_updates"] == 2000
    with pytest.raises(ValueError):
        replace(c, **TARGET)
    with pytest.raises(ValueError):
        amend_training_budget(c, TARGET, "not legacy")


def test_fresh_two_candidate_config_supports_explicit_larger_recipe(tmp_path):
    c = replace(fixture_config(tmp_path), **TARGET)
    assert training_config(c, "expert")["max_optimizer_steps"] == 10000
    assert training_config(c, "scorer")["max_optimizer_steps"] == 20000
