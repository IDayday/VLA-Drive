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
from .contracts import digest
from .loading import file_sha, weight_path

METRIC_PROTOCOL = "navsim_v2_official_one_stage"
EVALUATION_SEEDS = (42, 43, 44, 45, 46)


def scene_noise_seed(seed, token):
    return int(digest(["single_candidate_ode_v1", int(seed), str(token)])[:8], 16)


def evaluation_identity(cfg, sft, checkpoint, tokens_file, seed, split):
    from .audit import source_fingerprints
    from .reproducibility import (
        processor_identity,
        dependency_versions,
        numerical_profile,
    )

    return {
        "checkpoint_sha256": file_sha(weight_path(checkpoint)),
        "tokens_sha256": file_sha(tokens_file),
        "seed": int(seed),
        "split": split,
        "metric_protocol": METRIC_PROTOCOL,
        "source_sha256": digest(source_fingerprints()),
        "processor": processor_identity(cfg["paths"]["base_vlm"]),
        "checkpoint_config_sha256": file_sha(Path(checkpoint) / "config.yaml"),
        "dependencies": dependency_versions(),
        "numerics": numerical_profile(cfg, sft),
    }


def validate_evaluation_tokens(cfg, split, tokens):
    if (
        split not in {"rl_dev", "navtest"}
        or not tokens
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError(
            "explicit rl_dev/navtest split and unique nonempty tokens required"
        )
    if split == "rl_dev":
        _, allowed = split_tokens(cfg)
    else:
        allowed = json.loads(Path(cfg["paths"]["test_list"]).read_text())
    if not set(tokens) <= set(allowed):
        raise ValueError("evaluation tokens are outside selected split")
    return set(tokens) == set(allowed)


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
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        WeightedMetricIndex,
    )

    weights = []
    for _, row in frame.iterrows():
        values = np.array(row["weighted_metrics_array"], copy=True)
        if pd.isna(row["two_frame_extended_comfort"]):
            values[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0
        weights.append(json.dumps(values.tolist()))
    frame["effective_metric_weights"] = weights
    final = compute_final_scores(frame)
    final.attrs["aggregation"] = {
        "adjacent_pairs": len(adjacent),
        "adjacent_mapping": adjacent,
        "two_frame_available": int(final.two_frame_extended_comfort.notna().sum()),
        "two_frame_coverage": float(final.two_frame_extended_comfort.notna().mean()),
        "missing_metric_semantics": "official finalizer zeroes unavailable two-frame metric AND its denominator weight",
        "protocol": METRIC_PROTOCOL,
        "single_scene_training_reward": "navsim_v2_one_stage_single_scene; no adjacent two-frame comfort",
    }
    return final


def evaluate(
    cfg,
    sft,
    checkpoint,
    output,
    *,
    split,
    tokens_file,
    data_root,
    metric_cache,
    seed,
    metric_protocol,
):
    from infer import VLAAgent, deal_action_1225, set_inference_seed

    if (
        not all([checkpoint, output, tokens_file, data_root, metric_cache])
        or metric_protocol != METRIC_PROTOCOL
    ):
        raise ValueError(
            "evaluate requires explicit checkpoint, output, split, tokens, data root, metric cache, seed and locked protocol"
        )
    tokens = json.loads(Path(tokens_file).read_text())
    complete = validate_evaluation_tokens(cfg, split, tokens)
    cache_index = build_cache_index(metric_cache)
    missing = set(tokens) - set(cache_index)
    if missing:
        raise FileNotFoundError(
            f"evaluation requires every selected metric cache: {len(missing)} missing"
        )
    from .reproducibility import configure_numerics

    configure_numerics()
    identity = evaluation_identity(cfg, sft, checkpoint, tokens_file, seed, split)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    # Content identity, including same-path replacement, not an old cache path label.
    cache_files = {token: file_sha(cache_index[token]) for token in sorted(tokens)}
    (root / "metric_cache_identity.json").write_text(json.dumps(cache_files, indent=2))
    from copy import deepcopy

    eval_sft = deepcopy(sft)
    eval_sft.datasets.vla_data.datalist_path = str(tokens_file)
    eval_sft.datasets.vla_data.data_root = str(data_root)
    dataset = KeyedDataset(eval_sft, split="test" if split == "navtest" else "train")
    os.environ["BASE_VLM"] = cfg["paths"]["base_vlm"]
    os.environ["VLM_ATTN_IMPLEMENTATION"] = sft.framework.qwenvl.attn_implementation
    convention = cfg["checkpoint_contract"]["policy_feature_output"]
    agent = VLAAgent(
        str(checkpoint),
        device="cuda",
        qwen_forward_mode="optimized" if convention == "normalized" else "legacy",
    )
    actions = []
    physical = []
    start = time.monotonic()
    predictions = root / "predictions" / split
    predictions.mkdir(parents=True)
    for token in tokens:
        token_seed = scene_noise_seed(seed, token)
        sample = dataset[(token, token_seed)]
        set_inference_seed(token_seed)
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
        tokens, physical, Path("navsim").resolve(), metric_cache
    )
    frame.to_csv(root / "original_protocol_scores.csv", index=False)
    np.savez(
        root / "trajectories.npz",
        tokens=np.array(tokens),
        normalized=np.array(actions),
        physical=np.array(physical),
    )
    report = dict(
        identity=identity,
        metric_cache_identity=digest(cache_files),
        checkpoint=str(checkpoint),
        protocol="original 10-step ODE, one candidate, no selection/smoothing; official NAVSIM v2 one-stage finalization",
        split=split,
        complete_split=complete,
        tokens_sha256=file_sha(tokens_file),
        seed=int(seed),
        noise_schedule="sha256(single_candidate_ode_v1, seed, token); independent of iteration order and sharding",
        metric_protocol=metric_protocol,
        metric_name="navtest_v2_EPDMS"
        if split == "navtest" and complete
        else "rl_dev_v2_EPDMS"
        if split == "rl_dev" and complete
        else f"partial_{split}_v2_EPDMS",
        aggregation=frame.attrs["aggregation"],
        scene_count=len(tokens),
        epdms=float(frame["score"].mean()),
        valid=int(frame["valid"].sum()),
        seconds=time.monotonic() - start,
    )
    (root / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


def paired_scores(baseline, trained, bootstrap_seed=42, bootstrap_samples=2000):
    """Paired scene deltas; cluster bootstrap resamples logs, retaining all frames."""
    fields = ["token", "log_name", "score"]
    left, right = baseline[fields], trained[fields]
    if (
        set(left.token) != set(right.token)
        or left.token.duplicated().any()
        or right.token.duplicated().any()
    ):
        raise ValueError("paired scores require identical complete unique token sets")
    merged = left.merge(
        right, on="token", suffixes=("_sft", "_rl"), validate="one_to_one"
    )
    if (
        not (merged.log_name_sft == merged.log_name_rl).all()
        or not np.isfinite(merged[["score_sft", "score_rl"]].to_numpy()).all()
    ):
        raise ValueError("invalid paired logs/scores")
    merged["delta"] = merged.score_rl - merged.score_sft
    grouped = merged.groupby("log_name_sft").delta.agg(["sum", "count"])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(grouped), size=(bootstrap_samples, len(grouped)))
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    summary = {
        "scenes": len(merged),
        "logs": len(grouped),
        "mean_paired_delta": float(merged.delta.mean()),
        "log_bootstrap_95ci": np.quantile(estimates, [0.025, 0.975]).tolist()
        if len(grouped) > 1
        else None,
        "zero_to_nonzero": int(((merged.score_sft == 0) & (merged.score_rl > 0)).sum()),
        "nonzero_to_zero": int(((merged.score_sft > 0) & (merged.score_rl == 0)).sum()),
        "high_score_degraded": int(
            ((merged.score_sft >= 0.9) & (merged.delta < 0)).sum()
        ),
        "high_score_threshold": 0.9,
        "bootstrap_unit": "log",
        "bootstrap_seed": bootstrap_seed,
        "causal_scope": "each initialization relative to its own SFT; different SFT step counts do not isolate visual training causality",
    }
    return merged, summary
