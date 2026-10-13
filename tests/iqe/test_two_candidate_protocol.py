"""Two-candidate protocol behavior on explicitly synthetic, score-free fixtures."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from conftest import scene
from iqe.data.manifests import save_scenes
from iqe.data.splits import source_components, validate_isolation
from iqe.internal_baseline.bank import failure_and_target_audit
from iqe.internal_baseline.config import Config
from iqe.internal_baseline.partition import (
    finalize_partition, partition_fit, subrole_ids, training_ids,
)
from iqe.internal_baseline.sampling import ExpertSampler
from iqe.internal_baseline.scorer import construct
from iqe.internal_baseline.training import training_config, train_expert
from iqe.io import atomic_json, file_hash, read_json
from iqe.query_base import source_trajectory_contract
from iqe.scoring.reference import COMPONENTS
from iqe.training.trainer import scheduler_factory


def config(tmp_path, **changes):
    values = dict(experiment_id="unit-two-candidate", output_root=str(tmp_path),
                  contract="unit-contract", base_run="unit-base-run", scene_manifest="unit-scenes",
                  research_protocol="two_candidate_v1", scorer_dropout=0.)
    values.update(changes)
    return Config(**values)


def grouped_scenes():
    # Two windows per log; both remain in one role, even if budget sampling later
    # retains only one. Cross-log source-parent linkage is transitive as well.
    rows = [scene(i*2+j, group=f"source-{i}", s0_seen="yes") for i in range(10) for j in range(2)]
    rows += [replace(scene(100, group="child-1"), parent_group_id="source-0"),
             replace(scene(101, group="child-2"), parent_group_id="child-1")]
    rows += [scene(200, role="stage_val", group="val", s0_seen="yes"),
             scene(201, role="selector_cal", group="cal", s0_seen="unknown"),
             scene(202, role="dev_report", group="dev", s0_seen="no")]
    return rows


def test_partition_whole_logs_and_transitive_parent_union_are_disjoint(tmp_path):
    cfg, rows = config(tmp_path), grouped_scenes()
    validate_isolation(rows)
    selected = partition_fit(cfg, rows)
    p = read_json(cfg.root/"DATA_PARTITION.json")
    assert p["group_split"] == [8, 2]
    assert p["selected_before_scores"] is True
    assert p["grouping"] == "source_log_and_source_parent_union"
    de, ds = subrole_ids(cfg, "expert_fit"), subrole_ids(cfg, "scorer_fit")
    assert not de & ds
    assert de | ds == {s.scene_id for s in selected}
    roots = source_components(rows)
    assert roots["s0"] == roots["s1"] == roots["s100"] == roots["s101"]
    for component in {roots[s.scene_id] for s in selected}:
        ids = {s.scene_id for s in selected if roots[s.scene_id] == component}
        assert ids <= de or ids <= ds
    for field in ("source_logs", "source_groups", "assigned_components"):
        assert not set(p["roles"]["expert_fit"][field]) & set(p["roles"]["scorer_fit"][field])
    assert all(s.split_role == "incremental_fit" for s in selected)


def test_partition_is_independent_of_input_order_and_idempotent(tmp_path):
    cfg, rows = config(tmp_path), grouped_scenes()
    first = partition_fit(cfg, rows)
    before = file_hash(cfg.root/"DATA_PARTITION.json")
    second = partition_fit(cfg, list(reversed(rows)))
    assert [s.scene_id for s in first] == [s.scene_id for s in second]
    assert file_hash(cfg.root/"DATA_PARTITION.json") == before
    assert not (cfg.root/"scores").exists()


def test_budget_sampling_never_transfers_remaining_log_windows_to_other_role(tmp_path):
    rows = grouped_scenes()
    bounded = config(tmp_path/"bounded", expert_fit_limit=3, scorer_fit_limit=2)
    all_fit = config(tmp_path/"all", fit_limit="all_fit", expert_fit_limit=3, scorer_fit_limit=2)
    selected, whole = partition_fit(bounded, rows), partition_fit(all_fit, rows)
    small, full = read_json(bounded.root/"DATA_PARTITION.json"), read_json(all_fit.root/"DATA_PARTITION.json")
    assert len(selected) == 5 and len(whole) == 22
    assert small["all_assignments"] == full["all_assignments"]
    for role in ("expert_fit", "scorer_fit"):
        assert subrole_ids(bounded, role) <= subrole_ids(all_fit, role)
        assert small["roles"][role]["available_scenes"] == full["roles"][role]["available_scenes"]


def test_single_parent_component_cannot_be_split_to_create_two_roles(tmp_path):
    rows = [scene(0, group="parent"), replace(scene(1, group="child"), parent_group_id="parent")]
    with pytest.raises(ValueError, match="at least two independent"):
        partition_fit(config(tmp_path), rows)


def test_duplicate_observation_across_subroles_is_rejected(tmp_path):
    cfg, rows = config(tmp_path/"discover"), grouped_scenes()
    partition_fit(cfg, rows)
    de, ds = subrole_ids(cfg, "expert_fit"), subrole_ids(cfg, "scorer_fit")
    one, other = sorted(de)[0], sorted(ds)[0]
    changed = [replace(s, observation_hash="same-observation") if s.scene_id in {one, other} else s for s in rows]
    with pytest.raises(ValueError, match="observation leakage"):
        partition_fit(config(tmp_path/"reject"), changed)


def test_partition_receipt_tampering_and_heldout_role_injection_are_rejected(tmp_path):
    cfg, rows = config(tmp_path), grouped_scenes()
    partition_fit(cfg, rows)
    de = subrole_ids(cfg, "expert_fit")
    changed = [replace(s, split_role="stage_val") if s.scene_id == sorted(de)[0] else s for s in rows]
    with pytest.raises(ValueError, match="subrole scene/role mismatch"):
        training_ids(cfg, changed, "expert_fit")
    p = read_json(cfg.root/"DATA_PARTITION.json")
    p["roles"]["expert_fit"]["scene_ids"].append("not-in-original-partition")
    atomic_json(cfg.root/"DATA_PARTITION.json", p)
    with pytest.raises(ValueError, match="partition changed"):
        subrole_ids(cfg, "expert_fit")


def test_partition_exposure_does_not_claim_base_unseen(tmp_path):
    cfg, rows = config(tmp_path), grouped_scenes()
    chosen = partition_fit(cfg, rows)
    finalize_partition(cfg, chosen+[s for s in rows if s.split_role != "incremental_fit"])
    evidence = read_json(cfg.root/"DATA_PARTITION_EXPOSURE.json")
    assert evidence["roles"]["scorer_fit"]["base_seen"] == {"yes": len(subrole_ids(cfg, "scorer_fit"))}
    assert evidence["roles"]["stage_val"]["base_seen"] == {"yes": 1}
    assert evidence["roles"]["dev_report"]["base_seen"] == {"no": 1}
    assert not evidence["final_test_accessed"] and not evidence["navtest_accessed"]


@pytest.mark.parametrize("bad_bucket", ["hard", "ordinary"])
def test_actual_expert_training_rejects_scorer_fit_before_cache_or_model_load(tmp_path, monkeypatch, bad_bucket):
    cfg, rows = config(tmp_path), grouped_scenes()
    selected = partition_fit(cfg, rows)
    save_scenes(cfg.root/"scenes.json", selected)
    de, ds = sorted(subrole_ids(cfg, "expert_fit")), sorted(subrole_ids(cfg, "scorer_fit"))
    manifest = {"hard": [de[0]], "ordinary": [de[1]]}
    manifest[bad_bucket] = [ds[0]]
    atomic_json(cfg.root/"EXPERT_MANIFEST.json", manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("data leakage must fail before model/cache construction")
    monkeypatch.setattr("iqe.internal_baseline.training.load_head", forbidden)
    monkeypatch.setattr("iqe.internal_baseline.cache_execution.training_cache", forbidden)
    with pytest.raises(ValueError, match="D_S entered expert training"):
        train_expert(cfg, device="cpu")


def label(score=.9, nc=1., dac=1., *, valid=True, missing=()):
    components = {c: 1. for c in COMPONENTS}
    components[COMPONENTS[0]], components[COMPONENTS[1]] = nc, dac
    masks = {c: c not in missing for c in COMPONENTS}
    for c in missing:
        components[c] = None
    return {"score_valid": valid, "total_score_01": score if valid else None,
            "named_components": components, "component_valid_masks": masks}


@pytest.mark.parametrize("base,expected_failure", [
    (label(.8), False), (label(.8-1e-6), True), (label(.9, nc=.5), True),
    (label(.9, dac=0), True), (label(.9, nc=1-5e-9), False),
    (label(.9, nc=1-2e-8), True),
])
def test_base_failure_quality_and_named_safety_thresholds(base, expected_failure):
    audit = failure_and_target_audit(base)
    assert audit == {"failure": expected_failure, "eligible": False, "reasons": []}


@pytest.mark.parametrize("component", COMPONENTS[:2])
def test_missing_base_protected_component_is_unavailable_not_failure_or_full(component):
    audit = failure_and_target_audit(label(.1, missing=(component,)), label(1.))
    assert not audit["failure"] and not audit["eligible"]
    assert audit["reasons"] == ["BASE_LABEL_OR_PROTECTED_COMPONENT_UNAVAILABLE"]


def test_invalid_base_score_is_not_valid_zero():
    audit = failure_and_target_audit(label(valid=False), label(1.))
    assert not audit["failure"] and not audit["eligible"]
    assert audit["reasons"] == ["BASE_LABEL_OR_PROTECTED_COMPONENT_UNAVAILABLE"]


def test_gt_quality_and_gain_boundaries_are_inclusive():
    # IEEE representation of 0.8-0.75 must not spuriously fail the exact boundary.
    good = failure_and_target_audit(label(.75), label(.8))
    assert good == {"failure": True, "eligible": True, "reasons": []}
    low = failure_and_target_audit(label(.70), label(.8-1e-6))
    assert not low["eligible"] and "GT_PDMS_BELOW_0_8" in low["reasons"]
    weak = failure_and_target_audit(label(.750001), label(.8))
    assert not weak["eligible"] and "GT_GAIN_BELOW_0_05" in weak["reasons"]


@pytest.mark.parametrize("target", [label(.95, nc=.5), label(.95, dac=0),
                                    label(.95, missing=(COMPONENTS[0],)), label(.95, missing=(COMPONENTS[1],))])
def test_gt_protected_safety_must_be_valid_and_full(target):
    audit = failure_and_target_audit(label(.5), target)
    assert audit["failure"] and not audit["eligible"]
    assert "GT_NC_DAC_NOT_FULL" in audit["reasons"]


def test_gt_invalid_score_and_nonfailure_base_do_not_enter_hard_set():
    unavailable = failure_and_target_audit(label(.1), label(valid=False))
    assert not unavailable["eligible"] and "TARGET_SCORE_UNAVAILABLE" in unavailable["reasons"]
    ordinary = failure_and_target_audit(label(.85), label(.99))
    assert not ordinary["failure"] and not ordinary["eligible"]
    assert ordinary["reasons"] == ["NOT_BASE_FAILURE"]


def test_expert_80_20_cumulative_exposure_varied_batch_sizes_and_resume():
    sampler = ExpertSampler(["h0", "h1"], ["n0"], 19, "unit-manifest", hard_fraction=.8)
    counts, consumed = Counter(), 0
    for size in [1, 3, 17, 128, 127, 4, 1000]:
        before = deepcopy(sampler.state_dict())
        batch = sampler.peek(size)
        assert sampler.state_dict() == before  # Prefetch has not consumed RNG or IDs.
        assert sampler.peek(size) == batch
        counts.update(e["bucket"] for e in batch)
        assert all("weight" not in e and "loss_weight" not in e for e in batch)
        sampler.consume(size); consumed += size
        assert counts["hard"] == round(consumed*.8)
        assert counts["ordinary"] == consumed-round(consumed*.8)
    resume = ExpertSampler(["h0", "h1"], ["n0"], 19, "unit-manifest", hard_fraction=.8)
    expected = sampler.peek(128)  # Pending work must be rebuilt, not skipped.
    resume.load_state_dict(deepcopy(sampler.state_dict()))
    assert resume.peek(128) == expected
    resume.consume(128); sampler.consume(128)
    assert resume.state_dict() == sampler.state_dict()


def test_expert_fraction_and_manifest_are_resume_identity(tmp_path):
    sampler = ExpertSampler(["h"], ["n"], 5, "manifest", hard_fraction=.8)
    sampler.peek(128); sampler.consume(128)
    for fraction, manifest in ((.9, "manifest"), (.8, "other-manifest")):
        changed = ExpertSampler(["h"], ["n"], 5, manifest, hard_fraction=fraction)
        with pytest.raises(ValueError, match="manifest/sampler changed"):
            changed.load_state_dict(sampler.state_dict())
    for fraction in (0., 1., -.1):
        with pytest.raises(ValueError, match="fractions"):
            ExpertSampler(["h"], ["n"], 5, "m", hard_fraction=fraction)
    with pytest.raises(ValueError, match="both contain audited GT"):
        ExpertSampler(["h"], [], 5, "m", hard_fraction=.8)


@pytest.mark.parametrize("kind,warmup,updates", [("expert", 50, 1000), ("scorer", 100, 2000)])
def test_two_candidate_actual_scheduler_warmup_then_constant(tmp_path, kind, warmup, updates):
    cfg = training_config(config(tmp_path), kind)
    assert cfg["schedule"] == "warmup_constant"
    assert cfg["lr"] == 1e-4 and cfg["weight_decay"] == .01 and cfg["global_batch_size"] == 128
    assert cfg["warmup_steps"] == warmup and cfg["max_optimizer_steps"] == updates
    assert cfg["validation_every_steps"] == 250
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.AdamW([parameter], lr=cfg["lr"])
    scheduler = scheduler_factory(opt, cfg)
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4/warmup)
    for step in range(1, updates+1):
        opt.step(); scheduler.step()
        expected = 1e-4*min((step+1)/warmup, 1.)
        assert scheduler.get_last_lr()[0] == pytest.approx(expected)
    # Increasing the budget does not restart warmup or rewrite a cosine horizon.
    extended = dict(cfg, max_optimizer_steps=3000)
    restored_opt = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.]))], lr=cfg["lr"])
    restored_scheduler = scheduler_factory(restored_opt, extended)
    restored_opt.load_state_dict(deepcopy(opt.state_dict()))
    restored_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
    restored_opt.step(); restored_scheduler.step()
    assert restored_scheduler.last_epoch == updates+1
    assert restored_scheduler.get_last_lr()[0] == cfg["lr"]


def test_legacy_schedule_is_preserved(tmp_path):
    cfg = config(tmp_path, research_protocol="internal_legacy_v1", scorer_dropout=.1)
    assert training_config(cfg, "expert")["schedule"] == "warmup_cosine"


def test_scorer_recipe_constructs_zero_dropout_and_random_initialization_receipt(tmp_path):
    cfg = config(tmp_path)
    atomic_json(cfg.root/"BASE_BINDING.json", {"checkpoint": "unit-base.pt", "checkpoint_hash": "unit-sha",
                                               "head_hash": "unit-head"})
    model = construct(cfg, source_trajectory_contract(), 8, 8)
    drops = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    attentions = [m.dropout for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)]
    assert drops and attentions and set(drops+attentions) == {0.}
    assert len(model.decoder.layers) == 2
    assert model.decoder.layers[0].self_attn.num_heads == 4
    assert model.decoder.layers[0].self_attn.embed_dim == 256
    receipt = read_json(cfg.root/"SCORER_INITIALIZATION.json")
    assert not receipt["external_scorer_weights_loaded"] and receipt["loaded_scorer_weights"] == []
    assert receipt["init_seed"] == cfg.module_init_seed
    assert receipt["allowed_resume_lineage"] == cfg.experiment_id
    assert receipt["parameter_count"] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("bad", [
    {"research_protocol": "unknown"}, {"scorer_dropout": .1}, {"scorer_dropout": -.1},
    {"diagnostic_updates": 301}, {"expert_fit_limit": 16001}, {"scorer_fit_limit": 4001},
    {"expert_fit_limit": 0}, {"scorer_fit_limit": 0}, {"score_context": "external_encoder"},
    {"expert_updates": 3001}, {"scorer_updates": 1999}, {"pointwise_updates": 0},
    {"validation_interval": 200}, {"cache_batch": 0}, {"scorer_micro_batch": 0},
])
def test_two_candidate_invalid_or_changed_recipe_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        config(tmp_path, **bad)


def test_external_scorer_and_final_test_config_fields_are_not_supported(tmp_path):
    with pytest.raises(TypeError):
        config(tmp_path, external_scorer_checkpoint="forbidden.pt")
    with pytest.raises(TypeError):
        config(tmp_path, final_test="forbidden")
