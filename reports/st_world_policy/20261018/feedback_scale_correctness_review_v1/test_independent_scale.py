"""Synthetic checks only: no primary compact input, payload or model is opened."""
import importlib.util
import json
import math
import os
from pathlib import Path
import struct

import pytest

SOURCE = Path(os.environ["FEEDBACK_SCALE_SOURCE"])
SPEC = importlib.util.spec_from_file_location("independent_feedback_scale", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def f32(x):
    return struct.unpack("!f", struct.pack("!f", x))[0]


def fixture(groups, version=0, *, event=(0, 0, 0), d=(), c=(), t=(), ep=None, delta=0.):
    active = [i for i, g in enumerate(groups) if any(x != 0 for x in g)]
    if ep is None:
        ep = active
    return {
        "centered_A_float32": groups,
        "scene_log_pairs": [["scene-%d" % i, "log-%d" % (i // 2)] for i in range(len(groups))],
        "scene_presentations": len(groups), "candidate_presentations": 4*len(groups),
        "support_scene_slots": {"C1": active, "D": list(d), "C": list(c), "T": list(t), "EP_only": list(ep)},
        "nonzero_advantage_scene_presentations": {"C1": len(active), "D": len(d), "C": len(c), "T": len(t)},
        "EP_only_scene_presentations": len(ep), "event_candidate_counts_D_C_T": list(event),
        "reward_value_decomposition": {"max_absolute_rounding_residual": delta},
        "canonical_score_reconstruction_max_error": 0., "policy_version": version,
    }


@pytest.mark.parametrize("scenes", [32, 24])
def test_actual_population_and_sample_std_denominator(scenes):
    row = MOD.describe_update(fixture([[-.25, -.125, .125, .25]] * scenes))
    n = 4*scenes
    assert row["N"] == n
    assert row["sum_A"] == row["mean_A"] == 0.
    assert row["energy_A"] == row["M2"] == scenes*.15625
    assert row["std_correction1"] == math.sqrt(scenes*.15625/(n-1))
    assert row["gain"] == 1/(row["std_correction1"]+1e-10)
    assert row["zero_groups"] == 0 and row["nonzero_groups"] == scenes


def test_zero_ulp_and_constant_nonzero_diagnostics():
    row = MOD.describe_update(fixture([[0.]*4]*32))
    assert row["allzero"] and row["std_correction1"] == 0
    assert row["gain"] is row["rounding_bound_fraction"] is None
    assert not row["_strata"]["safe_EP_only_update"]
    one = MOD.describe_values([0., 0., 0., -2.**-24] + [0.]*4)
    assert one["sum_A"] == -2.**-24 and one["mean_A"] == -2.**-27
    assert math.isclose(one["std_correction1"], (2.**-24)/math.sqrt(8), rel_tol=1e-15)
    bad = MOD.describe_update(fixture([[.125]*4], ep=[]))
    assert bad["constant_nonzero_groups"] == 1
    assert {"constant_nonzero_group", "nonzero_A_but_std_zero", "std_zero"} <= set(bad["flags"])
    assert bad["gain"] is None
    tiny = MOD.describe_values([f32(-1e-12), f32(1e-12)])
    assert "positive_std_le_epsilon" in tiny["flags"]
    assert tiny["gain"] == 1/(tiny["std_correction1"]+1e-10)


def test_float32_roundtrip_and_invalid_compact_invariants():
    with pytest.raises(ValueError, match="float32"):
        MOD.describe_values([.1, -.1])
    assert "nonfinite_input_or_statistics" in MOD.describe_values([float("nan"), 0.])["flags"]
    assert "N_below_two" in MOD.describe_values([])["flags"]
    for change in ("duplicate_scene", "wrong_count", "wrong_active", "bad_slot", "missing_candidate"):
        a = fixture([[-.5, 0., 0., .5]]*2)
        if change == "duplicate_scene": a["scene_log_pairs"][1] = a["scene_log_pairs"][0]
        if change == "wrong_count": a["candidate_presentations"] += 1
        if change == "wrong_active": a["support_scene_slots"]["C1"] = []
        if change == "bad_slot": a["support_scene_slots"]["D"] = [2]
        if change == "missing_candidate": a["centered_A_float32"][0].pop()
        with pytest.raises(ValueError): MOD.describe_update(a)


def test_event_presence_discriminative_support_and_window_boundaries():
    zero = MOD.describe_update(fixture([[0.]*4]*2, 1023, event=(8, 0, 0)))
    ep = MOD.describe_update(fixture([[-.5, 0., 0., .5], [0.]*4], 1024))
    road = MOD.describe_update(fixture([[-.5, 0., 0., .5], [0.]*4], 3072, event=(1, 0, 0), d=(0,), ep=[]))
    assert zero["_strata"]["D_event_present_update"]
    assert not zero["_strata"]["D_metric_discriminative_update"]
    assert ep["_strata"]["safe_EP_only_update"]
    assert road["_strata"]["DCT_event_present_update"] and road["_strata"]["DCT_metric_discriminative_update"]
    assert not road["_strata"]["safe_EP_only_update"]
    report = MOD.summarize([road, zero, ep])
    allpop = report["strata"]["all"]
    assert allpop["updates"] == 3 and allpop["scene_presentations"] == 6
    assert allpop["unique_scenes"] == 2 and allpop["unique_logs"] == 1
    assert allpop["nonzero_scene_presentations"] == 2 and allpop["nonzero_unique_scenes"] == 1
    windows = report["fixed_policy_version_windows"]
    assert [windows[k]["all"]["updates"] for k in ("[0,1024)", "[1024,2048)", "[2048,3072)", "[3072,4096)")] == [1, 1, 0, 1]
    assert report["strata"]["C_event_present_update"]["distributions"]["gain"]["quantiles"]["0.5"] is None
    assert [x[0] for x in report["update_table"]["rows"]] == [1023, 1024, 3072]
    changed = MOD.describe_update(fixture([[0.]*4]*2, 3))
    changed["_scenes"][0][1] = "foreign-log"
    with pytest.raises(ValueError, match="Scene/log identity"):
        MOD.population_summary([ep, changed])


def test_precision_energy_upper_bound_is_not_causal_attribution():
    a = MOD.describe_update(fixture([[-.5, 0., 0., .5]], delta=.5))
    assert a["rounding_energy_bound"] == 1.
    assert a["rounding_bound_fraction"] == 2.
    assert "rounding_bound_can_cover_all_energy" in a["flags"]
    assert a["energy_A"] == .5  # A conservative bound exceeding this does not prove dominance.
    b = MOD.describe_update(fixture([[-.25, 0., 0., .25]], 1, delta=0.))
    result = MOD.population_summary([a, b])
    # The ratio of summed energies differs from equal-update average fractions (1.0).
    assert result["sum_energy_A"] == .625
    assert result["sum_rounding_energy_bound"] == 1.
    assert result["ratio_of_rounding_bound_sum_to_energy_sum"] == 1.6
    assert result["distributions"]["rounding_bound_fraction"]["mean"] == 1.


def test_resealed_duplicate_boundary_cannot_pass_complete_join(tmp_path):
    fixture_file = SOURCE.parents[2] / "tests/st_world/test_feedback_scale.py"
    spec = importlib.util.spec_from_file_location("author_scale_fixture", fixture_file)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    f = author.PrimaryFixture(tmp_path)
    folder = f.root / "shard_01"
    path = folder / (f.items[1]["ledger"] + ".json")
    doc = json.loads(path.read_bytes())
    doc.pop("content_sha256")
    doc["analysis"]["policy_version"] = 0  # Duplicate 0, missing 1; all hashes will be valid.
    raw = author.write(path, doc, True)
    complete_path = folder / "SHARD_COMPLETE.json"
    complete = json.loads(complete_path.read_bytes())
    complete.pop("content_sha256")
    complete["receipts"][0]["sha256"] = MOD.digest(raw)
    complete_raw = author.write(complete_path, complete, True)
    full = dict(f.full)
    full["shard_receipt_sha256"][1] = MOD.digest(complete_raw)
    author.write(f.root / "RESULT.json", full, True)
    before = os.sched_getaffinity(0)
    try:
        with pytest.raises(ValueError, match="boundary coverage incomplete"):
            MOD.execute(f.plan_path, tmp_path / "derived", min(before), production=False)
    finally:
        os.sched_setaffinity(0, before)
    assert (tmp_path / "derived/ERROR.json").is_file()
    assert not (tmp_path / "derived/RESULT.json").exists()
