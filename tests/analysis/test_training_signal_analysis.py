import copy
import gzip
import json

from scripts.analysis.analyze_training_signal import analyze


def test_fresh_groups_counted_once_and_progress_only_is_lower_bound(tmp_path):
    rank = dict(scene_count=2, advantage_nonzero_group_fraction=.5,
                reward_group_std_quantiles={"0.0": 0, "1.0": .002},
                reward_group_span_quantiles={"0.0": 0, "1.0": .008})
    for key in ["no_at_fault_collisions", "drivable_area_compliance",
                "driving_direction_compliance", "traffic_light_compliance",
                "time_to_collision_within_bound", "lane_keeping", "history_comfort"]:
        rank[f"reward_component/{key}"] = 1
    other = copy.deepcopy(rank)
    other["reward_component/time_to_collision_within_bound"] = .5
    first = dict(update=1, inner_epoch=0, scene_count=4, ranks=[rank, other],
                 loss_coefficients={"reference": .04}, lr=[1e-6])
    second = dict(first, update=2, inner_epoch=1)
    source = tmp_path / "training.jsonl"
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    out = tmp_path / "report"
    report = analyze(source, out, 2)
    for window in report["windows"]:
        values = window["per_scene_distribution"]
        assert values["count"] == 4  # Not 8 after the second inner update.
        assert values["zero_std_count"] == 2
        assert values["active_scenes"] == 2
        assert values["active_progress_only_lower_bound"] == 1
        assert values["active_progress_only_fraction_lower_bound"] == .5
        assert values["std_quantiles"][.5] == .001
    with gzip.open(out / "per_update.csv.gz", "rt") as f:
        assert len(f.readlines()) == 3
