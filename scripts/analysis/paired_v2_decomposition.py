"""Explain saved v2 results without altering scores, rewards or trajectories.

The counterfactual is an ordered arithmetic decomposition, not causal evidence
and not a replacement benchmark. Every saved official score must reconstruct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from navsim.planning.script.run_pdm_score_one_stage import compute_final_scores

WEIGHTED = ["ego_progress", "time_to_collision_within_bound", "lane_keeping",
            "history_comfort", "two_frame_extended_comfort"]
MULTIPLICATIVE = ["no_at_fault_collisions", "drivable_area_compliance",
                  "driving_direction_compliance", "traffic_light_compliance"]


def score(frame, comfort=None):
    frame = frame.copy()
    if comfort is not None:
        frame["two_frame_extended_comfort"] = comfort
    frame["weighted_metrics"] = [np.nan_to_num(x) for x in frame[WEIGHTED].to_numpy()]
    frame["weighted_metrics_array"] = [np.asarray(json.loads(x), dtype=float)
                                       for x in frame.effective_metric_weights]
    frame["multiplicative_metrics_prod"] = frame[MULTIPLICATIVE].prod(axis=1)
    return compute_final_scores(frame)["score"]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = {"scope":"fixed dev seed42; no new model inference",
              "method":"official finalizer reconstruction; current other metrics + SFT comfort counterfactual",
              "variants":{},"inputs":{}}
    for variant in ("frozen_visual", "unfrozen_visual"):
        root = Path(a.run)/(variant+"_evaluation")
        locations = json.loads((root/"locations.json").read_text())
        frames, values = {}, {}
        for label in ("sft", "step100", "step200"):
            directory = Path(locations[label+"/rl_dev/42"])
            report = json.loads((directory/"evaluation.json").read_text())
            if not (directory/"COMPLETE").is_file() or report["status"] != "COMPLETE":
                raise ValueError("incomplete source evaluation")
            path = directory/"original_protocol_scores.csv"
            frame = pd.read_csv(path, float_precision="round_trip").set_index("token").sort_index()
            if (len(frame) != report["scene_count"] or frame.index.duplicated().any() or
                not frame.valid.all() or not np.isfinite(frame[MULTIPLICATIVE+WEIGHTED[:-1]].to_numpy()).all()):
                raise ValueError("invalid/incomplete saved scores")
            calculated = score(frame)
            error = float(np.abs(calculated-frame.score).max())
            if error > 1e-12:
                raise ValueError("official finalizer did not reconstruct saved scores")
            single = score(frame, np.full(len(frame), np.nan))
            values[label] = {"epdms":float(calculated.mean()),
                "single_scene_v2_reward_protocol_on_dev_ode":float(single.mean()),
                "raw_v2_intermediate_pdm_score_not_v1":float(frame.pdm_score.mean()),
                "reconstruction_max_error":error, "scenes":len(frame),
                "means":{c:float(frame[c].mean()) for c in WEIGHTED+MULTIPLICATIVE}}
            result["inputs"][variant+"/"+label] = {"csv":str(path),
                "csv_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "evaluation_identity":report["identity"]}
            frames[label] = frame
        baseline, current = frames["sft"], frames["step200"]
        if not baseline.index.equals(current.index) or not (baseline.log_name == current.log_name).all():
            raise ValueError("different paired token/log sets")
        if not baseline.two_frame_extended_comfort.isna().equals(current.two_frame_extended_comfort.isna()):
            raise ValueError("changed adjacent-frame coverage")
        counterfactual = score(current, baseline.two_frame_extended_comfort.values)
        values["step200_decomposition"] = {
            "epdms_delta":float(current.score.mean()-baseline.score.mean()),
            "comfort_change_holding_current_other_metrics":float(current.score.mean()-counterfactual.mean()),
            "other_metrics_with_sft_comfort":float(counterfactual.mean()-baseline.score.mean()),
            "warning":"ordered arithmetic decomposition; not causal proof or replacement benchmark"}
        result["variants"][variant] = values
    output = Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result["variants"],indent=2))


if __name__ == "__main__":
    main()
