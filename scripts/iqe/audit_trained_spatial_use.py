"""Read-only sensitivity audit of the already selected S11; no optimization.

Uses fixed prefixes of fit/validation, never test/development or score-based
sampling. This diagnoses wiring and numerical dependence, not generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from iqe.contracts import require
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, file_hash, read_json
from iqe.robust_scorer.config import load
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.data import ReusedContext
from iqe.specialist_study.metrics import select
from iqe.specialist_study.models import StudyScorer, load_bridge
from iqe.specialist_study.scorer_training import fetch_context


def delta(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "exactly_equal": bool(torch.equal(a, b))}


@torch.no_grad()
def audit(config, limit):
    source = config.source()
    directory = source.root / "scorers/S11"
    # The training constructor writes a path-specific resolved-recipe receipt.
    # Rebuild the exact architecture read-only, checking its full initial hash,
    # rather than writing any artifact back to the original campaign.
    initialization = read_json(directory / "SCORER_INITIALIZATION.json")
    recipe = read_json(source.root / "resolved_recipes/scorer_2x2.json")
    require(file_hash(recipe["source"]) == recipe["checksum"], "original recipe changed")
    require(recipe["resolved"]["dropout"] == 0., "original dropout contract")
    scene = scenes_by_role(source)["scorer_fit"][0]
    x = ReusedContext(source, capacity=1).read(scene)
    visual = VerifiedSpatialReader(source, capacity=1).read(scene)
    require((source.root / "BASE_PLANNER.pt").exists(), "read-only audit requires existing planner")
    contract = load_bridge(source).trajectory_contract
    views = len(read_json(source.root / "OFFLINE_BASE_CONTRACT.json")["input"]["views"])
    with initialization_rng(source.seed):
        model = StudyScorer(x["world"].shape[-1], x["action"].shape[-1], contract,
                            spatial=True, visual_dim=visual["visual"].shape[-1], views=views)
    require(module_hash(model) == initialization["initial_state_hash"], "initialization identity changed")
    initial = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("spatial_reader.")}
    result = read_json(directory / "result.json")
    checkpoint = directory / "best.pt"
    require(result["status"] == "COMPLETE" and result["optimizer_steps"] == 6000,
            "original S11 is incomplete")
    require(file_hash(checkpoint) == result["best_checkpoint_hash"], "selected S11 hash changed")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    before = module_hash(model)
    updates = {k: delta(v, initial[k]) for k, v in model.state_dict().items() if k in initial}
    with (directory / "steps.jsonl").open() as stream:
        first = json.loads(next(stream))
    gradients = {k: v for k, v in first["gradients"].items() if k.startswith("spatial_reader.")}
    require(gradients and all(v["has_grad"] and v["finite"] and v["norm"] > 0 for v in gradients.values()),
            "spatial parameter initial training gradients were absent/nonfinite/zero")
    captured = {}
    handle = model.spatial_reader.register_forward_hook(
        lambda _m, _args, output: captured.update(spatial=output.detach().clone()))
    rows = []
    try:
        for role in ("scorer_fit", "stage_val"):
            metadata, bank = load_bank(source, "deploy", role)
            scenes = scenes_by_role(source)[role]
            require(metadata["scene_ids"] == [s.scene_id for s in scenes], "bank order mismatch")
            chosen = scenes[:min(limit, len(scenes))]
            cache = ReusedContext(source, capacity=4)
            spatial = VerifiedSpatialReader(source, capacity=4)
            for at in range(0, len(chosen), 2):
                group = chosen[at:at+2]
                context = fetch_context(source, group, "cpu", cache, spatial)
                trajectories = torch.from_numpy(bank["trajectories"][at:at+len(group)])
                values = model(context, trajectories)["values"].clone()
                features = captured["spatial"].clone()
                valid = bank["valid"][at:at+len(group)]
                choices = select(values.numpy(), valid, expert_ids=metadata["provenance"]["sources"])
                variants = {"zero_V": context | {"visual": torch.zeros_like(context["visual"])}}
                wrong = [scenes[(i+1) % len(scenes)] for i in range(at, at+len(group))]
                variants["shuffle_V"] = context | spatial.batch(wrong, "cpu")
                for name, changed in variants.items():
                    prediction = model(changed, trajectories)["values"]
                    selection = select(prediction.numpy(), valid, expert_ids=metadata["provenance"]["sources"])
                    rows.append({"role": role, "scene_ids": [s.scene_id for s in group], "intervention": name,
                                 "visual_input": delta(context["visual"], changed["visual"]),
                                 "spatial_output": delta(features, captured["spatial"]),
                                 "predictions": delta(values, prediction),
                                 "changed_choices": int(np.sum(choices != selection)),
                                 "candidate_count": int(values.numel()),
                                 "visual_shape": list(context["visual"].shape),
                                 "predictions_before": values.tolist(), "predictions_after": prediction.tolist()})
    finally:
        handle.remove()
    require(module_hash(model) == before, "read-only audit changed parameters/buffers")
    return {"schema": "trained_spatial_readonly_sensitivity_v1", "cell": "S11",
            "checkpoint": str(checkpoint), "checkpoint_hash": file_hash(checkpoint),
            "initial_state_hash": initialization["initial_state_hash"], "state_hash": before,
            "device": "cpu", "threads": torch.get_num_threads(), "samples_per_role": limit,
            "selection_rule": "fixed manifest prefix, chosen without model scores",
            "optimizer_steps": 0, "used_to_select_or_train_models": False,
            "initial_attempt": "Training constructor rejected a cross-worktree resolved-recipe path; no source artifact was overwritten. Audit now constructs read-only and verifies the full original initialization hash.",
            "spatial_parameter_updates": updates, "first_update_spatial_gradients": gradients,
            "rows": rows, "parameters_unchanged": True,
            "limitation": "Small CPU sensitivity diagnostic; neither a deployment latency nor a full-population quality result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenes-per-role", type=int, default=16)
    args = parser.parse_args()
    require(1 <= args.scenes_per_role <= 32, "bounded sensitivity audit, maximum32 per role")
    torch.set_num_threads(2)
    config = load(args.config)
    output = config.root / "qualification/trained_S11_spatial_sensitivity.json"
    require(not output.exists(), "audit already exists; retain existing evidence")
    result = audit(config, args.scenes_per_role)
    result["audit_source_hash"] = file_hash(Path(__file__))
    atomic_json(output, result, immutable=True)
    print(json.dumps({"status": "COMPLETE", "output": str(output),
                      "rows": len(result["rows"]), "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
