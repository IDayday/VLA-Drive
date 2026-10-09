from __future__ import annotations
import numpy as np
from ..contracts import require
from ..io import digest


def retrieve(query_scene, query_vector, library, vectors, limit=16, feature_version="", seed=42):
    require(len(library) == len(vectors) and limit > 0, "retrieval dimensions/budget")
    for s in library:
        s.eligible_input(external_targets=True)
    vectors = np.asarray(vectors, dtype=np.float64)
    require(vectors.ndim == 2 and np.isfinite(vectors).all(), "retrieval features")
    mean, scale = vectors.mean(0), vectors.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    distances = (((vectors - mean) / scale - (np.asarray(query_vector) - mean) / scale)**2).sum(-1)
    allowed = [i for i, s in enumerate(library) if s.source_group_id != query_scene.source_group_id
               and s.source_log_id != query_scene.source_log_id and s.observation_hash != query_scene.observation_hash]
    chosen = sorted(allowed, key=lambda i: (float(distances[i]), library[i].scene_id))[:limit]
    return [library[i] for i in chosen], {"quality": "UNVERIFIED", "distance": "squared Euclidean, fit-only standardized pooled scene+ego",
        "feature_version": feature_version, "seed": seed, "limit": limit, "mean": mean.tolist(), "scale": scale.tolist(),
        "library_hash": digest([s.scene_id for s in library]),
        "candidates": [{"scene_id": library[i].scene_id, "distance": float(distances[i])} for i in chosen]}
