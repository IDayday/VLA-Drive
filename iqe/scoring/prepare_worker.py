"""Construct actual official reference caches with the pinned source's original helper."""
import sys
import lzma
import pickle
import hashlib
import numpy as np
from pathlib import Path
from ..io import read_json, atomic_json
from ..contracts import require


def main(request, response):
    r = read_json(request)
    sys.path.insert(0, r["navsim_root"])
    sys.path.insert(1, r["source_root"])
    from tools.local_interaction_mask_v2.prepare_metric_cache import initialize, cache_log
    initialize(r["navsim_root"], r["args"]["map_root"])
    rows = cache_log((r["log"], r["tokens"], r["args"]))
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    for row in rows:
        if row["status"] != "ok":
            continue
        try:
            with lzma.open(row["cache_path"], "rb") as stream:
                cache = pickle.load(stream)
            require(cache.__class__.__module__ == "navsim.planning.metric_caching.metric_cache" and hasattr(cache,"trajectory"), "cache lacks locked official reference")
            reference = get_trajectory_as_array(cache.trajectory, TrajectorySampling(num_poses=40, interval_length=.1), cache.ego_state.time_point)
            require(np.isfinite(reference).all(), "official reference nonfinite")
            row["official_reference_hash"] = hashlib.sha256(reference.tobytes()).hexdigest()
        except Exception as error:
            row.update(status="failed", error=repr(error))
    atomic_json(response, rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
