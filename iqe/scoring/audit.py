"""Real GT/perturbation probes test protocol correctness, never scientific gain."""
from dataclasses import asdict
import numpy as np
from ..contracts import CandidateRecord, require
from ..data.sources import load_target
from ..data.candidate_bank import save_trajectory
from ..io import digest, atomic_json


def verify_reference(pipeline):
    scene = pipeline.scenes(["stage_val"])[0]
    target = pipeline.trajectory_contract.physical(load_target(scene, pipeline.trajectory_contract)).numpy()
    candidates = []
    folder = pipeline.root / "protocol_audit"
    for i in range(2):
        trajectory = target.copy()
        trajectory[:,1] += i * .5
        ref = folder / f"probe_{i}.npz"
        th = save_trajectory(ref, trajectory)
        candidates.append(CandidateRecord(1, scene.scene_id, f"probe_{i}", scene.target_id, pipeline.contract["framework_contract_hash"], th, str(ref),
            pipeline.trajectory_contract.raw_representation, "ego_relative_rear_axle", pipeline.trajectory_contract.horizon, pipeline.trajectory_contract.dt,
            True, pipeline.contract["iqe_code_hash"], pipeline.config["config_hash"], "protocol_probe"))
    reference = pipeline.backend().score(candidates[0], scene)
    original = pipeline.backend().score(candidates[0], scene, original_entry_source=pipeline.contract["source_root"])
    require(reference.score_valid and original.score_valid, f"real official scoring failed: {reference.error_reason}; {original.error_reason}")
    require(asdict(reference) == asdict(original), "reference differs from original formal S0 single-trajectory entry")
    report = pipeline.backend().validate_candidate_set_invariance(candidates, scene)
    report.update(original_entry_equivalence=True, scene_id=scene.scene_id, role=scene.split_role, score_scale="zero_one",
        real_metric_context=scene.metric_context_ref, original_reference_score=asdict(reference), scientific_gain="UNTESTED", probes="GT and lateral perturbation; neither is a trained IQE expert")
    atomic_json(folder / "SCORE_CONTEXT_EVIDENCE.json", report, immutable=True)
    return report
