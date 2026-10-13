"""Interfaces for a later own-lineage round; not invoked by the first-round CLI."""
from pathlib import Path
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, file_hash, digest


def comparison_population(scene_ids, expert_ids, rewards, valid, tie_epsilon=.01):
    """Rebuild comparisons from ALL old/new columns; no expert classifier rows."""
    rewards, valid = np.asarray(rewards), np.asarray(valid, dtype=bool)
    require(rewards.shape == valid.shape == (len(scene_ids), len(expert_ids)), "current candidate pool shape")
    require(len(set(expert_ids)) == len(expert_ids) and expert_ids[0] == "expert_0", "stable Base-first registry required")
    entries = []
    for row, scene in enumerate(scene_ids):
        for i in range(len(expert_ids)):
            for j in range(i+1, len(expert_ids)):
                if not (valid[row, i] and valid[row, j] and np.isfinite(rewards[row, [i,j]]).all()): continue
                gap = float(rewards[row,j]-rewards[row,i])
                if abs(gap) <= tie_epsilon: continue
                entries.append({"scene_id": scene, "left_id": expert_ids[i], "right_id": expert_ids[j],
                                "left_index": i, "right_index": j, "signed_gap": gap, "weight": abs(gap)})
    total = sum(x["weight"] for x in entries)
    for entry in entries: entry["probability"] = entry["weight"]/total
    return {"candidate_ids": list(expert_ids), "entries": entries, "total_gap_weight": total,
            "population_hash": digest([list(scene_ids),list(expert_ids),entries]),
            "status": "PRESENT" if total else "SCORER_COMPARISON_SIGNAL_ABSENT"}


def warm_start_own_parent(module, parent_root, *, declared_parent_experiment_id, base_checkpoint_hash, feature_contract_hash):
    """Only a completed own experiment with the same frozen encoder may parent round2.

    The first-round training entry never calls this function. It loads weights
    only; optimizer/scheduler and the refreshed comparison distribution are new.
    """
    root = Path(parent_root)
    binding = read_json(root/"BASE_BINDING.json")
    init = read_json(root/"SCORER_INITIALIZATION.json")
    result = read_json(root/"scorer/result.json")
    require(binding["experiment_id"] == init["experiment_id"] == declared_parent_experiment_id, "own parent experiment lineage mismatch")
    require(init["external_scorer_weights_loaded"] is False and init["loaded_scorer_weights"] == [], "external scorer lineage prohibited")
    require(binding["checkpoint_hash"] == base_checkpoint_hash and binding["contract_hash"] == feature_contract_hash, "shared Base/feature contract changed")
    require(result["status"] == "COMPLETE" and file_hash(root/"scorer/best.pt") == result["best_checkpoint_hash"], "parent scorer incomplete/corrupt")
    module.load_state_dict(torch.load(root/"scorer/best.pt", weights_only=True, map_location="cpu"), strict=True)
    return {"parent_experiment_id": declared_parent_experiment_id, "parent_checkpoint": str(root/"scorer/best.pt"),
            "parent_checkpoint_hash": result["best_checkpoint_hash"], "base_checkpoint_hash": base_checkpoint_hash,
            "external_scorer_weights_loaded": False, "optimizer_reused": False}
