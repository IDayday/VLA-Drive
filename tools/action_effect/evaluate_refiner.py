"""Separate bounded EGR prediction and official CPU scoring; never simulate in training.

Prediction runs the small cached-context refiner only. Official scoring is a
separate --phase score invocation with its own durable global call/retry budget.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import (artifact, digest, load_config, lock, read_json, verify_artifact,
                     verify_identity, write_json)
from .context_data import batch_context
from .training import load_private
from .train_refiner import prepare_dataset
from starVLA.model.framework.effect_guided_refiner import EffectGuidedRefiner
from starVLA.model.modules.action_effect.effect_model import EffectModelConfig
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec
from starVLA.model.modules.action_effect.refinement_policy import CoefficientRefiner, RefinerConfig, reconstruct_torch


def load_refiner(checkpoint, *, expected_base_binding_hash, expected_data_artifact=None):
    checkpoint = Path(checkpoint)
    identity = read_json(checkpoint.parent / "IDENTITY.json")
    receipt = read_json(checkpoint.with_suffix(".COMPLETE.json"))
    if (identity.get("schema") != "action_effect_small_training_v1" or identity.get("arm") not in ("A0", "A1", "ScoreRefiner")
            or identity["base_binding_hash"] != expected_base_binding_hash
            or (expected_data_artifact is not None and identity["data_artifact"] != expected_data_artifact)
            or receipt["checkpoint"] != artifact(checkpoint)
            or verify_artifact(receipt["checkpoint"]) != checkpoint.resolve()
            or receipt["run_identity"] != digest(identity)):
        raise ValueError("Refiner checkpoint/receipt/Base/data identity mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["run_identity"] != receipt["run_identity"] or state["optimizer_step"] != receipt["optimizer_step"]:
        raise ValueError("Refiner checkpoint completion metadata mismatch")
    cfg = dict(identity["model_config"])
    cfg["model"] = EffectModelConfig(**cfg["model"])
    b = dict(cfg["basis"]); b["windows"] = tuple(tuple(x) for x in b["windows"])
    cfg["basis"] = BasisSpec(**b)
    cfg["coefficient_scales"] = tuple(cfg["coefficient_scales"])
    cfg["active_coefficients"] = tuple(cfg["active_coefficients"])
    model = EffectGuidedRefiner(CoefficientRefiner(RefinerConfig(**cfg)))
    load_private(model, state["model"]); model.requires_grad_(False).eval()
    model.provenance = {"checkpoint": artifact(checkpoint), "identity": artifact(checkpoint.parent / "IDENTITY.json"),
                        "optimizer_step": state["optimizer_step"], "run_identity": state["run_identity"], "arm": identity["arm"]}
    return model


def predict(cfg, *, dataset_path, context_manifest, checkpoint, output, role, max_samples,
            device, campaign_binding=None, resume=False):
    if role not in ("fit", "validation") or type(max_samples) is not int or not 1 <= max_samples <= cfg["cohort_size"]:
        raise ValueError("Explicit bounded fit/validation prediction population required")
    dataset = prepare_dataset(cfg, dataset_path, context_manifest, campaign_binding)
    selected = [r for r in dataset.entries if r["role"] == role][:max_samples]
    if not selected: raise ValueError("Requested role has no registered scenes")
    model = load_refiner(checkpoint, expected_base_binding_hash=dataset.reader.binding["base_checkpoint_binding_hash"],
                         expected_data_artifact=artifact(dataset_path)).to(device)
    identity = {"schema": "action_effect_refiner_prediction_registration_v1", "arm": model.provenance["arm"],
        "checkpoint": model.provenance["checkpoint"], "model_provenance": model.provenance,
        "repair_dataset": artifact(dataset_path), "context_manifest": artifact(context_manifest),
        "campaign_binding": dataset.reader.binding["identity"], "g0_registration": dataset.manifest["registration"],
        "role": role, "requested_max_samples": max_samples, "scene_ids": [r["scene_id"] for r in selected],
        "base_checkpoint": dataset.manifest["base_checkpoint"], "score_protocol_hash": dataset.manifest["score_protocol_hash"],
        "numeric_runtime_hash": dataset.manifest["numeric_runtime_hash"], "effect_teacher_deployed": False,
        "baseline_definition": "same immutable S0 physical trajectory and same reference context as G0"}
    identity["identity"] = digest(identity)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with lock(output / "PREDICT.lock"):
        if (output / "PREDICTIONS.json").exists() and not resume:
            raise ValueError("Prior predictions exist; explicit --resume required")
        write_json(output / "PREDICTION_REGISTRATION.json", identity)
        rows, started = [], time.perf_counter()
        with torch.no_grad():
            for row in selected:
                current = dataset.reader[row["scene_id"]]
                context = batch_context([current], device)
                base = current["physical"][None].to(device).float()
                proposal = model.deploy(context, base)
                physical = proposal["trajectory"][0].cpu().double().numpy()
                proposed = reconstruct_torch(base, proposal["coefficients"].to(base.dtype), model.refiner.config.basis)[0].cpu().double().numpy()
                record = {"scene_id": row["scene_id"], "source_log": row["source_log"], "role": row["role"],
                    "observation_hash": row["observation_hash"], "base_physical": row["base_physical"],
                    "physical": physical.tolist(), "proposed_physical": proposed.tolist(),
                    "coefficients": proposal["coefficients"][0].cpu().double().tolist(),
                    "geometry_accepted": bool(proposal["geometry_accepted"][0]),
                    "geometry_screen": proposal["geometry_screen"][0],
                    "base_metrics": row["base_metrics"], "base_feedback": row["base_feedback"],
                    "base_canonical_receipt": row["base_canonical_receipt"],
                    "target_kind": row["target_kind"], "repair_status": row["repair_status"],
                    "teacher_forward_calls": 0, "real_feedback_used_for_prediction": False}
                record["identity"] = digest(record); rows.append(record)
        result = {"schema": "action_effect_refiner_predictions_v1", "status": "COMPLETE", "registration": artifact(output/"PREDICTION_REGISTRATION.json"),
                  "registration_identity": identity["identity"], "requested_scenes": len(selected), "entries": rows,
                  "elapsed_seconds": time.perf_counter()-started, "device": device}
        # Timing belongs to an invocation record, not the immutable predictions identity.
        elapsed = result.pop("elapsed_seconds")
        result["identity"] = digest(result)
        if (output/"PREDICTIONS.json").exists():
            old = verify_identity(read_json(output/"PREDICTIONS.json"))
            if old != result: raise ValueError("Same prediction identity produced different trajectories")
        else: write_json(output/"PREDICTIONS.json", result)
        write_json(output / f"PREDICT_INVOCATION_{time.time_ns()}.json", {"status": "COMPLETE", "elapsed_seconds": elapsed,
                  "prediction_artifact": artifact(output/"PREDICTIONS.json"), "device": device})
        return {"status": "COMPLETE", "requested_scenes": len(selected), "predictions": artifact(output/"PREDICTIONS.json"),
                "elapsed_seconds": elapsed}


def _summary(rows, requested):
    paired = [r for r in rows if r["status"] == "VALID_SCORED" and r["base_metrics"] is not None]
    valid = [r for r in rows if r["status"] == "VALID_SCORED"]
    transient = [r for r in rows if r["status"] == "PARTIAL"]
    invalid = [r for r in rows if r["status"] not in ("VALID_SCORED", "PARTIAL")]
    out = {"requested_scenes": requested, "valid_refined_scenes": len(valid), "valid_paired_scenes": len(paired),
           "transient_failures": len(transient), "invalid_predictions_or_contexts": len(invalid),
           "unprocessed_scenes": requested-len(rows),
           "PDMS": None, "PDMS_valid_subset": None, "paired_PDMS_delta": None}
    if valid:
        out["PDMS_valid_subset"] = float(np.mean([r["metrics"]["score"] for r in valid]))
        if len(valid) == requested: out["PDMS"] = out["PDMS_valid_subset"]
    if paired:
        before = np.asarray([r["base_metrics"]["score"] for r in paired]); after = np.asarray([r["metrics"]["score"] for r in paired])
        out.update(paired_Base_PDMS=float(before.mean()), paired_PDMS_delta=float((after-before).mean()),
            base_zero=int((before<=1e-8).sum()), refined_zero=int((after<=1e-8).sum()),
            zero_repaired=int(((before<=1e-8)&(after>1e-8)).sum()),
            high_to_zero=int(((before>=.8)&(after<=1e-8)).sum()),
            mean_change_m=float(np.mean([r["distance_m"] for r in paired])))
        out["new_safety_failures"] = {name: sum(r["base_metrics"][name]>=1.-1e-8 and r["metrics"][name]<1.-1e-8 for r in paired)
            for name in ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")}
        out["paired_component_means"] = {name: {"base": float(np.mean([r["base_metrics"][name] for r in paired])),
                                               "refined": float(np.mean([r["metrics"][name] for r in paired]))}
            for name in paired[0]["base_metrics"]}
    return out


def score(cfg, *, predictions_path, output, max_simulation_budget, max_seconds=3600., resume=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("Official scoring requires CUDA hidden and all CPU BLAS limits1")
    if (type(max_simulation_budget) is not int or not 1 <= max_simulation_budget <= cfg["g0"]["max_simulation_budget"]
            or not np.isfinite(max_seconds) or not 0 < max_seconds <= 14400):
        raise ValueError("Explicit bounded simulation/time budget required")
    from .simulate_interventions import (CallLedger, BudgetExhausted, score_candidate, verified_feedback,
                                         worker_engine, write_gzip)
    from .feedback_schema import canonical_record
    from starVLA.model.modules.action_effect.trajectory_constraints import tracking_screen, geometric_screen, ConstraintSpec
    from starVLA.model.modules.st_world.consequence import ConsequenceSpec, ConsequenceRequest, trajectory_sha256
    predictions = verify_identity(read_json(predictions_path))
    reg = verify_identity(read_json(verify_artifact(predictions["registration"])))
    if (predictions["schema"] != "action_effect_refiner_predictions_v1" or predictions["status"] != "COMPLETE"
            or predictions["registration_identity"] != reg["identity"]
            or [r["scene_id"] for r in predictions["entries"]] != reg["scene_ids"]
            or predictions["requested_scenes"] != len(reg["scene_ids"])):
        raise ValueError("Incomplete or mismatched prediction population")
    for pin in (reg["checkpoint"], reg["repair_dataset"], reg["context_manifest"]): verify_artifact(pin)
    source = verify_identity(read_json(verify_artifact(reg["g0_registration"])))
    population = {item["context"]["scene_id"]: item for item in source["population"]}
    if (reg["base_checkpoint"] != source["base_checkpoint"] or reg["score_protocol_hash"] != source["spec"]["score_protocol_hash"]
            or reg["numeric_runtime_hash"] != source["numeric_runtime_hash"]):
        raise ValueError("Prediction/G0 simulation identity mismatch")
    spec = ConsequenceSpec(**source["spec"])
    os.environ["NUPLAN_MAPS_ROOT"] = spec.maps_root; os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "action_effect_refiner_evaluation_v1", "predictions": artifact(predictions_path),
                "max_calls": max_simulation_budget, "retry_budget": cfg["g0"]["retry_budget"], "spec": asdict(spec)}
    identity["identity"] = digest(identity)
    write_json(output/"EVALUATION_REGISTRATION.json", identity)
    ledger = CallLedger(output/"calls", identity=identity["identity"], max_calls=max_simulation_budget,
                        retry_budget=cfg["g0"]["retry_budget"])
    rows, deadline = [], time.monotonic()+max_seconds
    with lock(output/"EVALUATE.lock"):
        if (output/"RESULT.json").exists() and not resume: raise ValueError("Prior evaluation exists; use --resume")
        for row in predictions["entries"]:
            verify_identity(row)
            sid = row["scene_id"]
            if sid not in population or row["role"] != reg["role"]: raise ValueError("Prediction role/scene mismatch")
            path = output/"scenes"/sid/"RESULT.json"
            if path.exists():
                previous = verify_identity(read_json(path))
                if previous["prediction_identity"] != row["identity"]: raise ValueError("Stale scene result")
                if "feedback" in previous: verified_feedback(previous)
                rows.append(previous); continue
            if time.monotonic() >= deadline: break
            try:
                before_feedback = None
                if row["base_metrics"] is not None and row["base_feedback"] is None:
                    raise ValueError("A comparable Base score requires its independent official feedback receipt")
                if row["base_feedback"] is not None:
                    before_feedback = verified_feedback({"feedback": row["base_feedback"], "canonical_receipt": row["base_canonical_receipt"]})
                    if row["base_metrics"] != canonical_record(before_feedback).get("metrics"):
                        raise ValueError("Reported Base metrics differ from verified native feedback")
                poses = np.asarray(row["physical"], dtype=np.float64)
                request = ConsequenceRequest(spec.campaign_id, sid, row["source_log"], reg["arm"],
                    "EGR_"+reg["checkpoint"]["sha256"][:24], None, source["numeric_runtime_hash"],
                    trajectory_sha256(poses), population[sid]["observation"]["metric_cache"]["sha256"], spec.score_protocol_hash,
                    population[sid]["observation"]["metric_cache"]["path"], poses.tolist(),
                    generator_kind="engineering_control", generator_source_hash=digest([reg["checkpoint"], row["identity"]]))
                engine = worker_engine(spec)
                native_context = engine.load_context(request)
                native_velocity = native_context.ego_state.dynamic_car_state.rear_axle_velocity_2d.array
                native_geometry = geometric_screen(poses, BasisSpec(), ConstraintSpec(), initial_velocity_xy=native_velocity)
                reuse_base = (before_feedback is not None and before_feedback["valid"]
                              and np.array_equal(poses, np.asarray(row["base_physical"], dtype=np.float64)))
                if reuse_base:
                    native = canonical_record(before_feedback)
                    feedback_pin, receipt_pin = row["base_feedback"], row["base_canonical_receipt"]
                else:
                    native, wrapped, attempt = score_candidate(engine, ledger,
                        output/"candidate_cache"/digest(request.identity()), request,
                        observation=population[sid]["observation"], deadline=deadline)
                    feedback_path = output/"scenes"/sid/"FEEDBACK.json.gz"; write_gzip(feedback_path, wrapped)
                    feedback_pin, receipt_pin = artifact(feedback_path), artifact(attempt/"COMPLETE.json")
                result = {"scene_id": sid, "source_log": row["source_log"], "status": native["status"],
                          "prediction_identity": row["identity"], "base_metrics": row["base_metrics"], "metrics": native["metrics"],
                          "feedback": feedback_pin, "canonical_receipt": receipt_pin,
                          "exact_Base_score_reused": bool(reuse_base), "native_initial_velocity_geometry": native_geometry,
                          "distance_m": float(np.linalg.norm(poses[:,:2]-np.asarray(row["base_physical"])[:,:2],axis=-1).mean())}
                if native["valid"]:
                    result["tracking"] = tracking_screen(poses, native["executed_states"], BasisSpec(), ConstraintSpec())
                if not native["valid"] and native["status"] != "INVALID_PREDICTION":
                    raise ValueError("Unresolved official dependency: "+native["status"])
                result["identity"] = digest(result); write_json(path, result); rows.append(result)
            except BudgetExhausted as exc:
                rows.append({"scene_id": sid, "status": "PARTIAL", "error_type": type(exc).__name__, "error": str(exc)})
                break
        summary = _summary(rows, predictions["requested_scenes"])
        complete = len(rows) == predictions["requested_scenes"] and not summary["transient_failures"]
        result = {"schema": "action_effect_refiner_evaluation_result_v1", "status": "COMPLETE" if complete else "PARTIAL",
                  "registration_identity": identity["identity"], "summary": summary, "entries": rows,
                  "calls": read_json(output/"calls"/"COUNTER.json")}
        write_json(output/f"INVOCATION_{time.time_ns()}.json", result)
        write_json(output/"RESULT.json", result, immutable=False)
        if complete:
            write_json(output/"COMPLETE.json", {"registration_identity": identity["identity"], "result": artifact(output/"RESULT.json"),
                "scenes": [artifact(output/"scenes"/r["scene_id"]/"RESULT.json") for r in rows]})
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--phase", required=True, choices=("predict", "score"))
    p.add_argument("--repair-dataset"); p.add_argument("--context-manifest"); p.add_argument("--campaign-binding")
    p.add_argument("--checkpoint"); p.add_argument("--predictions"); p.add_argument("--output", required=True)
    p.add_argument("--role", choices=("fit", "validation"), default="validation"); p.add_argument("--max-samples", type=int)
    p.add_argument("--max-simulation-budget", type=int); p.add_argument("--max-seconds", type=float, default=3600.)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.dry_run:
            print(json.dumps({"status": "NOT_RUN", "phase": a.phase, "config_hash": digest(cfg), "simulation_calls": 0})); return 0
        if a.phase == "predict":
            if not all((a.repair_dataset,a.context_manifest,a.checkpoint,a.max_samples)):
                raise ValueError("Prediction needs explicit dataset/context/checkpoint/sample budget")
            result = predict(cfg, dataset_path=a.repair_dataset, context_manifest=a.context_manifest,
                checkpoint=a.checkpoint, output=a.output, role=a.role, max_samples=a.max_samples,
                device=a.device, campaign_binding=a.campaign_binding, resume=a.resume)
        else:
            if not a.predictions or a.max_simulation_budget is None: raise ValueError("Scoring needs predictions and a call budget")
            result = score(cfg, predictions_path=a.predictions, output=a.output,
                           max_simulation_budget=a.max_simulation_budget, max_seconds=a.max_seconds, resume=a.resume)
        print(json.dumps(result)); return 0 if result["status"] == "COMPLETE" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})); return 1


if __name__ == "__main__": raise SystemExit(main())
