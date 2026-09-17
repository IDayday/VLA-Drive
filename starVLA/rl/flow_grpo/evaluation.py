"""Original ODE + original one-stage NAVSIM aggregation, fixed validation split."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
import pandas as pd
from .config import split_tokens
from .data import KeyedDataset
from .reward import OfficialEvaluator, build_cache_index


def original_protocol_scores(tokens, trajectories, devkit, cache_root):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.script.run_pdm_score_one_stage import (
        infer_start_adjacent_mapping,
        create_scene_aggregators,
        compute_final_scores,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses

    evaluator = OfficialEvaluator(devkit, build_cache_index(cache_root))
    rows = []
    for token, poses in zip(tokens, trajectories):
        cache = evaluator.load_cache(token)
        trajectory = Trajectory(
            np.asarray(poses, dtype=np.float64),
            TrajectorySampling(num_poses=8, interval_length=0.5),
        )
        row, states = pdm_score(
            cache,
            trajectory,
            evaluator.sampling,
            evaluator.simulator,
            evaluator.scorer,
            evaluator.traffic,
        )
        row["valid"] = True
        row["token"] = token
        row["log_name"] = cache.log_name
        row["frame_type"] = cache.scene_type
        row["start_time"] = cache.timepoint.time_s
        end = relative_to_absolute_poses(
            cache.ego_state.rear_axle, [StateSE2(*poses[-1])]
        )[0]
        row["endpoint_x"] = end.x
        row["endpoint_y"] = end.y
        row["start_point_x"] = cache.ego_state.rear_axle.x
        row["start_point_y"] = cache.ego_state.rear_axle.y
        row["ego_simulated_states"] = [states]
        rows.append(row)
    frame = pd.concat(rows, ignore_index=True)
    adjacent = infer_start_adjacent_mapping(frame)
    if adjacent:
        frame = create_scene_aggregators(adjacent, frame, evaluator.sampling)
    else:
        # Original aggregator pd.concat([]) fails for a set with no adjacent pairs.
        # Exact no-neighbor semantics: unavailable comfort, official finalizer.
        frame["two_frame_extended_comfort"] = np.nan
        frame = frame.drop(columns=["ego_simulated_states"])
    return compute_final_scores(frame)


def evaluate(cfg, sft, checkpoint, output):
    from infer import VLAAgent, deal_action_1225, set_inference_seed

    if not checkpoint or not output:
        raise ValueError("evaluate requires --checkpoint and --output-dir")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _, tokens = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    os.environ["BASE_VLM"] = cfg["paths"]["base_vlm"]
    os.environ["VLM_ATTN_IMPLEMENTATION"] = sft.framework.qwenvl.attn_implementation
    convention = cfg["checkpoint_contract"]["policy_feature_output"]
    agent = VLAAgent(
        str(checkpoint),
        device="cuda",
        qwen_forward_mode="optimized" if convention == "normalized" else "legacy",
    )
    set_inference_seed(cfg["runtime"]["seed"])
    actions = []
    physical = []
    start = time.monotonic()
    predictions = root / "predictions" / "train"
    predictions.mkdir(parents=True)
    for token in tokens:
        sample = dataset[(token, cfg["runtime"]["seed"])]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = agent.predict([sample])
        action = output["normalized_actions"]
        trajectory = deal_action_1225(
            action, act_norm=int(sft.datasets.vla_data.act_norm)
        )[0]
        if not np.isfinite(trajectory).all():
            raise FloatingPointError(f"nonfinite inference: {token}")
        np.save(predictions / f"{token}.npy", trajectory)
        actions.append(action[0])
        physical.append(trajectory)
    frame = original_protocol_scores(
        tokens, physical, Path("navsim").resolve(), cfg["paths"]["metric_cache"]
    )
    frame.to_csv(root / "original_protocol_scores.csv", index=False)
    np.savez(
        root / "trajectories.npz",
        tokens=np.array(tokens),
        normalized=np.array(actions),
        physical=np.array(physical),
    )
    report = dict(
        checkpoint=str(checkpoint),
        protocol="original 10-step ODE, one candidate, no selection/smoothing; official NAVSIM v2 one-stage finalization",
        split="validation held out from navtrain; never navtest",
        seed=cfg["runtime"]["seed"],
        scene_count=len(tokens),
        epdms=float(frame["score"].mean()),
        valid=int(frame["valid"].sum()),
        seconds=time.monotonic() - start,
    )
    (root / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report
