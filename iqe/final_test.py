"""Explicit locked final evaluation, with no writes to fit/mining/calibration artifacts."""
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .contracts import CandidateRecord, require
from .data.manifests import load_scenes
from .data.sources import load_observation
from .data.candidate_bank import save_trajectory
from .scoring.reference import ReferenceBackend
from .io import atomic_json, file_hash, digest, read_json


def evaluate_final(model, rule, metadata, manifest, output, maximum, metric_config):
    require(metadata["mode"] == "full" and metadata["gate_results"]["passed"], "final_test requires accepted locked full bundle")
    scenes = load_scenes(manifest)
    require(scenes and all(s.split_role == "final_test" for s in scenes), "explicit final_test-only manifest required")
    seen = metadata["data_role_groups"]
    require(all(s.source_group_id not in seen and s.source_log_id not in seen and (s.parent_group_id is None or s.parent_group_id not in seen) for s in scenes), "final_test source overlaps prior training/selection/dev")
    backend = ReferenceBackend(metric_config["navsim_root"], metric_config["python"], {k: metric_config[k] for k in ("proposal_sampling", "scorer")}, eval_seed=metric_config["eval_seed"])
    require(backend.source_hash == metadata["contract"]["metric"]["python_source_hash"] and digest(backend.config) == metadata["contract"]["metric"]["config_hash"], "final metric protocol mismatch")
    output = Path(output) / metadata["bundle_hash"] / digest({"manifest": file_hash(manifest), "maximum": maximum})
    output.mkdir(parents=True, exist_ok=True)
    access = output / "access_log" / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    atomic_json(access, {"bundle_hash": metadata["bundle_hash"], "manifest_hash": file_hash(manifest), "purpose": "locked_final_evaluation"}, immutable=True)
    if (output/'RESULT.json').exists():
        result=read_json(output/'RESULT.json')
        if result['status']=='COMPLETE':return result
    scores = []
    for s in sorted(scenes, key=lambda s: s.scene_id)[:maximum]:
        with torch.no_grad():
            observation = load_observation(s, metadata["contract"])
            if metadata["selector_type"] == "scene_router":
                trajectory, ids = model.forward_prerouted([observation], rule)
            else:
                trajectory, ids, _ = model.predict([observation], rule)
        ref = output / "trajectories" / (s.scene_id + ".npz")
        th = save_trajectory(ref, trajectory[0].cpu().numpy())
        c = CandidateRecord(1, s.scene_id, ids[0], metadata["weights"][ids[0] + ".pt"], digest(metadata["contract"]), th, str(ref),
            metadata["contract"]["trajectory"]["raw_representation"], "ego_relative_rear_axle", trajectory.shape[-2], metadata["contract"]["trajectory"]["dt"],
            bool(torch.isfinite(trajectory).all()), metadata["contract"]["iqe_code_hash"], metadata["config_hash"], "final_test")
        score = backend.score(c, s)
        atomic_json(output / "scores" / (s.scene_id + ".json"), asdict(score))
        scores.append(score)
    valid = [s.total_score_01 for s in scores if s.score_valid]
    result = {"status": "COMPLETE" if len(valid) == len(scores) else "SCORING_ERRORS", "scenes": len(scores), "valid": len(valid),
              "mean_score_01": float(np.mean(valid)) if valid else None, "output": str(output), "bundle_hash": metadata["bundle_hash"], "training_writeback": False}
    atomic_json(output / "RESULT.json", result)
    require(len(valid) == len(scores), "final scoring errors retained; no failed label silently converted to zero")
    return result
