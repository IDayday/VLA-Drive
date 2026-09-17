"""Real cached adjacent frames verify official two-frame/denominator semantics."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.reward import OfficialEvaluator, build_cache_index
from starVLA.rl.flow_grpo.evaluation import original_protocol_scores


def test_real_adjacent_frames_and_missing_comfort_weight():
    cfg, _ = resolve_config("configs/flow_grpo/action_only_frozen_visual.yaml")
    index = build_cache_index(cfg["paths"]["metric_cache"])
    evaluator = OfficialEvaluator(Path("navsim").resolve(), index)
    groups = defaultdict(list)
    for token, path in index.items():
        groups[Path(path).parents[2].name].append(token)
    pair = None
    # Select solely by log/time adjacency, never by reward.
    for log in sorted(groups):
        rows = sorted(
            [(evaluator.load_cache(t).timepoint.time_s, t) for t in groups[log]]
        )
        for a, b in zip(rows, rows[1:]):
            if 0 < b[0] - a[0] <= 0.55:
                pair = [a[1], b[1]]
                break
        if pair:
            break
    assert pair is not None, "no adjacent pair in available real caches"
    trajectories = [evaluator.load_cache(t).human_trajectory.poses for t in pair]
    frame = original_protocol_scores(
        pair, trajectories, Path("navsim").resolve(), cfg["paths"]["metric_cache"]
    )
    assert frame.attrs["aggregation"]["adjacent_pairs"] == 1
    assert frame.two_frame_extended_comfort.notna().any()
    assert len(frame) == 2 and np.isfinite(frame.score).all()
    single = original_protocol_scores(
        pair[:1],
        trajectories[:1],
        Path("navsim").resolve(),
        cfg["paths"]["metric_cache"],
    )
    assert single.attrs["aggregation"]["adjacent_pairs"] == 0
    assert single.two_frame_extended_comfort.isna().all()
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        WeightedMetricIndex,
    )

    weights = json.loads(single.iloc[0].effective_metric_weights)
    assert weights[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] == 0
