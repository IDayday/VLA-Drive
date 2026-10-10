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


class RetrievalIndex:
    """Fixed fit-only statistics and matrix; exact deterministic chunked queries."""
    def __init__(self, library, vectors, feature_version='', seed=42):
        require(len(library) == len(vectors) and len(library) > 0, 'retrieval library')
        for scene in library:
            scene.eligible_input(external_targets=True)
        self.library = list(library)
        self.vectors = np.asarray(vectors, dtype=np.float64)
        require(self.vectors.ndim == 2 and np.isfinite(self.vectors).all(), 'retrieval features')
        self.mean = self.vectors.mean(0)
        self.scale = self.vectors.std(0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.)
        self.matrix = (self.vectors - self.mean) / self.scale
        self.feature_version, self.seed = feature_version, seed
        self.groups = np.array([s.source_group_id for s in library])
        self.logs = np.array([s.source_log_id for s in library])
        self.observations = np.array([s.observation_hash for s in library])
        self.identity = {'library_hash': digest([s.to_dict() if hasattr(s, 'to_dict') else __import__('dataclasses').asdict(s) for s in library]),
                         'statistics_hash': digest({'mean': self.mean.tolist(), 'scale': self.scale.tolist()}),
                         'matrix_hash': __import__('hashlib').sha256(self.matrix.tobytes()).hexdigest(),
                         'feature_version': feature_version, 'seed': seed, 'count': len(library)}

    def query(self, scene, vector, limit=16):
        require(limit > 0, 'retrieval result budget')
        vector = np.asarray(vector, dtype=np.float64)
        require(vector.shape == self.mean.shape and np.isfinite(vector).all(), 'retrieval query')
        # Same subtraction/sum ordering as reference retrieve; no approximate index.
        distances = ((self.matrix - (vector-self.mean)/self.scale)**2).sum(-1)
        return self._select(scene, distances, limit)

    def _select(self, scene, distances, limit):
        allowed = np.flatnonzero((self.groups != scene.source_group_id) &
            (self.logs != scene.source_log_id) & (self.observations != scene.observation_hash))
        if len(allowed) > limit:
            boundary = np.partition(distances[allowed], limit-1)[limit-1]
            allowed = allowed[distances[allowed] <= boundary]
        chosen = sorted(allowed, key=lambda i: (float(distances[i]), self.library[i].scene_id))[:limit]
        report = {'quality': 'UNVERIFIED', 'distance': 'squared Euclidean, fit-only standardized pooled scene+ego',
            'feature_version': self.feature_version, 'seed': self.seed, 'limit': limit,
            'index_identity': self.identity,
            'candidates': [{'scene_id': self.library[i].scene_id, 'distance': float(distances[i])} for i in chosen]}
        return [self.library[i] for i in chosen], report

    def query_many(self, scenes, vectors, limit=16, chunk_size=8, library_chunk_size=2048):
        require(len(scenes) == len(vectors) and chunk_size > 0 and library_chunk_size > 0, 'retrieval query budget')
        for start in range(0, len(scenes), chunk_size):
            query = np.asarray(vectors[start:start+chunk_size],dtype=np.float64)
            require(query.ndim == 2 and query.shape[1] == len(self.mean) and np.isfinite(query).all(), 'retrieval queries')
            query = (query-self.mean)/self.scale
            distances = np.empty((len(query),len(self.library)),dtype=np.float64)
            for at in range(0,len(self.library),library_chunk_size):
                end = min(at+library_chunk_size,len(self.library))
                # Identical elementwise arithmetic and reduction as the reference;
                # bounded Q x library-chunk x D temporary, without GEMM tie drift.
                distances[:,at:end] = ((self.matrix[None,at:end,:]-query[:,None,:])**2).sum(-1)
            for scene,distance in zip(scenes[start:start+chunk_size],distances):
                yield self._select(scene,distance,limit)
