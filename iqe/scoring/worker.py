"""One fresh official metric_cache/simulator/scorer/observation per candidate."""
from __future__ import annotations
from dataclasses import asdict
import lzma
import hashlib
import pickle
from pathlib import Path
import random
import sys
import numpy as np
from ..io import atomic_json, read_json, digest
from ..contracts import strict_record, CandidateRecord, SceneRecord, ScoreRecord, require


def main(request, response):
    r = read_json(request)
    sys.path.insert(0, r["navsim_root"])
    from navsim.evaluate.pdm_score import pdm_score, get_trajectory_as_array
    import navsim.evaluate.pdm_score as imported
    # Python `import a.b.c as x` may resolve a package attribute; inspect the callable's module instead.
    import importlib
    imported = importlib.import_module("navsim.evaluate.pdm_score")
    require(Path(imported.__file__).resolve().is_relative_to(Path(r["navsim_root"]).resolve()), "wrong NAVSIM import")
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from .reference import COMPONENTS
    candidate, scene = strict_record(CandidateRecord, r["candidate"]), strict_record(SceneRecord, r["scene"])
    random.seed(r["seed"])
    np.random.seed(r["seed"])
    with lzma.open(scene.metric_context_ref, "rb") as f:
        cache = pickle.load(f)
    sampling = TrajectorySampling(**r["config"]["proposal_sampling"])
    reference = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
    reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
    try:
        with np.load(candidate.trajectory_ref) as f:
            poses = f["trajectory"].astype(np.float64)
        require(poses.shape == (candidate.horizon, 3) and np.isfinite(poses).all(), "invalid physical trajectory")
        future = TrajectorySampling(num_poses=candidate.horizon, interval_length=candidate.dt)
        if r.get("original_entry_source"):
            require(asdict(PDMScorerConfig()) == r["config"]["scorer"] and candidate.horizon == 8 and candidate.dt == .5, "original single entry contract mismatch")
            sys.path.insert(1, r["original_entry_source"])
            from tools.local_interaction_mask_v2.score_async import score_chunk
            from ..io import file_hash
            result = score_chunk([{"token": scene.scene_id, "log": scene.source_log_id, "variant": "iqe_equivalence_probe",
                                  "proposal_path": candidate.trajectory_ref, "proposal_sha256": file_hash(candidate.trajectory_ref), "cache_path": scene.metric_context_ref}])[0]
            require(result["status"] == "ok", f"original S0 evaluator failed: {result}")
            total = result["score"]
            components = {c: float(result[c]) for c in COMPONENTS}
        else:
            result = pdm_score(cache, Trajectory(poses, future), sampling, PDMSimulator(sampling),
                               PDMScorer(sampling, PDMScorerConfig(**r["config"]["scorer"])))
            total = result.score
            components = {c: float(getattr(result, c)) for c in COMPONENTS}
        valid = {c: np.isfinite(v).item() and 0 <= v <= 1 for c, v in components.items()}
        require(all(valid.values()), "official component label invalid")
        score = ScoreRecord(1, candidate.key, r["protocol_hash"], scene.metric_context_hash, reference_hash,
                            float(total), components, valid, "official_single_reference", "reference-v1",
                            r["eval_seed"], r["repetition"], True, None)
    except Exception as e:
        score = ScoreRecord(1, candidate.key, r["protocol_hash"], scene.metric_context_hash, reference_hash, None,
                            {c: None for c in COMPONENTS}, {c: False for c in COMPONENTS},
                            "official_single_reference", "reference-v1", r["eval_seed"], r["repetition"], False, repr(e))
    atomic_json(response, asdict(score))


if __name__ == "__main__":
    main(*sys.argv[1:])
