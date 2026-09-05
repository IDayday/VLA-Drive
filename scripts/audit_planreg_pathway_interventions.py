#!/usr/bin/env python3
"""Small held-out task audit for each PlanReg scene pathway.

This is an intervention/sensitivity audit, not a replacement for the immutable
full Navtest result.  It reuses each image/VLM forward once, changes one scene
input at a time, and scores the resulting 64 proposals with the official PDM
target implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from planreg_audit_runtime import (
    build_navtest_samples,
    collate_samples,
    load_formal_training_agent,
    select_representative_tokens,
    sha256_file,
)


CONDITIONS = (
    "baseline",
    "semantic_context_off",
    "nonthumbnail_tiles_off",
    "planning_slots_collapsed",
    "navigation_command_off",
    "ego_velocity_off",
    "all_ego_status_off",
)


def _clone_inputs(value: Dict[str, object]) -> Dict[str, object]:
    return {
        key: item.clone() if isinstance(item, torch.Tensor) else item
        for key, item in value.items()
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@torch.no_grad()
def _run_action_conditions(agent, action_inputs, thumbnail_registers):
    outputs = {"baseline": agent.action_head(_clone_inputs(action_inputs))}

    original_gate = agent.action_head.semantic_gate.detach().clone()
    try:
        agent.action_head.semantic_gate.fill_(-30.0)
        outputs["semantic_context_off"] = agent.action_head(
            _clone_inputs(action_inputs)
        )
    finally:
        agent.action_head.semantic_gate.copy_(original_gate)

    modified = _clone_inputs(action_inputs)
    modified["planning_registers"] = thumbnail_registers
    outputs["nonthumbnail_tiles_off"] = agent.action_head(modified)

    modified = _clone_inputs(action_inputs)
    planning = modified["planning_registers"]
    modified["planning_registers"] = planning.mean(dim=1, keepdim=True).expand_as(
        planning
    )
    outputs["planning_slots_collapsed"] = agent.action_head(modified)

    for name, indices in (
        ("navigation_command_off", slice(0, 4)),
        ("ego_velocity_off", slice(4, 6)),
        ("all_ego_status_off", slice(None)),
    ):
        modified = _clone_inputs(action_inputs)
        modified["status_feature"][:, indices] = 0
        outputs[name] = agent.action_head(modified)
    return outputs


@torch.no_grad()
def run(args) -> dict:
    os.environ["NAVSIM_TRAIN_METRIC_CACHE"] = str(args.metric_cache.resolve())
    os.environ.setdefault("DRIVEVLA_SCORE_RAY", "0")
    os.environ.setdefault("DRIVEVLA_SCORE_PROCESSES", str(args.score_processes))
    os.environ.setdefault("DRIVEVLA_SCORE_START_METHOD", "forkserver")
    device = torch.device(args.device)
    _, agent, checkpoint = load_formal_training_agent(
        args.resolved_config,
        args.checkpoint,
        device=device,
        compute_dtype="float32",
    )
    agent.remove_training_only_world_model()
    agent.world_model_enabled = False
    agent.future_mode = "disabled"
    agent.eval()

    tokens, token_metadata = select_representative_tokens(
        args.candidate_bank, args.metric_cache, args.scene_count
    )
    samples = build_navtest_samples(
        agent,
        tokens,
        token_metadata,
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
    )
    with np.load(args.candidate_bank, allow_pickle=False) as payload:
        bank_tokens = [str(value) for value in payload["tokens"]]
        bank_rows = {token: index for index, token in enumerate(bank_tokens)}
        bank_proposals = np.asarray(payload["proposals"], dtype=np.float32)
        bank_predicted = np.asarray(payload["predicted_pdms"], dtype=np.float32)
        official_names = [str(value) for value in payload["official_component_names"]]

    proposals: Dict[str, list] = {name: [] for name in CONDITIONS}
    predicted: Dict[str, list] = {name: [] for name in CONDITIONS}
    selected_indices: Dict[str, list] = {name: [] for name in CONDITIONS}
    targets = []
    baseline_bank_proposal_diff = []
    baseline_bank_score_diff = []

    for token in tokens:
        features, target = collate_samples([samples[token]])
        captured_inputs = {}
        captured_adapter = {}

        def action_pre_hook(_module, inputs):
            captured_inputs.update(_clone_inputs(inputs[0]))

        def adapter_hook(_module, _inputs, output):
            captured_adapter["per_tile_registers"] = output.per_tile_registers.detach()

        action_handle = agent.action_head.register_forward_pre_hook(action_pre_hook)
        adapter_handle = agent.backbone.planning_register_adapter.register_forward_hook(
            adapter_hook
        )
        try:
            direct_baseline = agent.forward(features)
        finally:
            action_handle.remove()
            adapter_handle.remove()
        tile_metadata = features.get("tile_metadata")
        if tile_metadata is None:
            # The normal image-path path creates metadata inside the backbone;
            # recover it from the adapter's one-thumbnail invariant by using the
            # already aggregated output relation.  For this audit, preload the
            # image explicitly when metadata was not part of the dataset cache.
            from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image

            path_tensor = features["image_path_tensor"][0]
            path = "".join(chr(int(v)) for v in path_tensor.tolist() if int(v))
            pixels, metadata = load_image(path, return_tile_metadata=True)
            features["pixel_values"] = pixels.unsqueeze(0)
            features["tile_metadata"] = metadata.unsqueeze(0)
            captured_inputs.clear()
            captured_adapter.clear()
            action_handle = agent.action_head.register_forward_pre_hook(action_pre_hook)
            adapter_handle = agent.backbone.planning_register_adapter.register_forward_hook(
                adapter_hook
            )
            try:
                direct_baseline = agent.forward(features)
            finally:
                action_handle.remove()
                adapter_handle.remove()
            tile_metadata = metadata.unsqueeze(0)
        metadata = tile_metadata[0].to(device=device)
        thumbnail_indices = (metadata[:, 4] > 0.5).nonzero(as_tuple=False).flatten()
        if thumbnail_indices.numel() != 1:
            raise RuntimeError(f"Expected one thumbnail for {token}")
        thumbnail_registers = captured_adapter["per_tile_registers"][
            int(thumbnail_indices.item())
        ][None]
        condition_outputs = _run_action_conditions(
            agent, captured_inputs, thumbnail_registers
        )
        # Ensure the baseline action-only replay exactly matches the end-to-end
        # forward before interpreting any intervention.
        replay_diff = (
            condition_outputs["baseline"]["proposals"]
            - direct_baseline["proposals"]
        ).abs().max()
        if float(replay_diff) > 1e-6:
            raise RuntimeError(f"Action replay mismatch for {token}: {float(replay_diff)}")
        row = bank_rows[token]
        baseline_bank_proposal_diff.append(
            float(
                np.max(
                    np.abs(
                        direct_baseline["proposals"][0].float().cpu().numpy()
                        - bank_proposals[row]
                    )
                )
            )
        )
        baseline_bank_score_diff.append(
            float(
                np.max(
                    np.abs(
                        direct_baseline["pred_pdms"][0].float().cpu().numpy()
                        - bank_predicted[row]
                    )
                )
            )
        )
        for name, output in condition_outputs.items():
            proposals[name].append(output["proposals"][0].detach().float().cpu())
            predicted[name].append(output["pred_pdms"][0].detach().float().cpu())
            selected_indices[name].append(int(output["pdm_score"][0].argmax()))
        targets.append(target)
        torch.cuda.empty_cache()

    _, target_batch = collate_samples([samples[token] for token in tokens])
    condition_reports = {}
    baseline_proposals = torch.stack(proposals["baseline"])
    baseline_predicted = torch.stack(predicted["baseline"])
    baseline_selected = np.asarray(selected_indices["baseline"], dtype=np.int64)
    official_by_condition = {}
    for name in CONDITIONS:
        proposal_tensor = torch.stack(proposals[name]).to(device)
        final_scores, best_scores, target_scores, *_ = agent.compute_score(
            target_batch, proposal_tensor, test=False
        )
        official = target_scores.detach().float().cpu().numpy()
        official_by_condition[name] = official
        rows = np.arange(len(tokens))
        chosen = np.asarray(selected_indices[name], dtype=np.int64)
        selected = official[rows, chosen, -1]
        oracle = official[..., -1].max(axis=1)
        report = {
            "selected_pdms": _summary(selected),
            "offline_oracle_at_64": _summary(oracle),
            "scorer_regret": _summary(oracle - selected),
            "candidate_mean_pdms": float(official[..., -1].mean()),
            "selected_index_change_fraction_vs_baseline": float(
                (chosen != baseline_selected).mean()
            ),
            "proposal_rms_change_vs_baseline": float(
                (proposal_tensor.cpu() - baseline_proposals).square().mean().sqrt()
            ),
            "predicted_score_rms_change_vs_baseline": float(
                (torch.stack(predicted[name]) - baseline_predicted)
                .square()
                .mean()
                .sqrt()
            ),
            "selected_indices": chosen.tolist(),
        }
        condition_reports[name] = report

    baseline_official = official_by_condition["baseline"]
    baseline_selected_score = baseline_official[
        np.arange(len(tokens)), baseline_selected, -1
    ]
    for name in CONDITIONS[1:]:
        chosen = np.asarray(selected_indices[name], dtype=np.int64)
        official = official_by_condition[name]
        selected = official[np.arange(len(tokens)), chosen, -1]
        condition_reports[name]["paired_selected_pdms_delta_vs_baseline"] = _summary(
            selected - baseline_selected_score
        )
        component_delta = (
            official[np.arange(len(tokens)), chosen, :-1]
            - baseline_official[np.arange(len(tokens)), baseline_selected, :-1]
        ).mean(axis=0)
        condition_reports[name]["paired_selected_component_delta_vs_baseline"] = {
            component: float(delta)
            for component, delta in zip(official_names[:-1], component_delta)
        }

    if getattr(agent, "_score_process_pool", None) is not None:
        agent._score_process_pool.shutdown(wait=True)
        agent._score_process_pool = None
    result = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "candidate_bank": str(args.candidate_bank.resolve()),
        "candidate_bank_sha256": sha256_file(args.candidate_bank),
        "split": "navtest",
        "scene_count": len(tokens),
        "sampling": "four task strata with distinct logs; exploratory, not a full-split estimate",
        "tokens": tokens,
        "inference_uses_future_inputs": False,
        "baseline_candidate_bank_max_abs_proposal_diff": float(
            max(baseline_bank_proposal_diff)
        ),
        "baseline_candidate_bank_max_abs_predicted_score_diff": float(
            max(baseline_bank_score_diff)
        ),
        "conditions": condition_reports,
        "interpretation_caveat": (
            "Interventions measure sensitivity and can be out-of-distribution; "
            "they do not by themselves prove causal task semantics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=12)
    parser.add_argument("--score-processes", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
