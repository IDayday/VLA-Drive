"""Slow official single-output reference, isolated per candidate in a subprocess.

The official reference is metric_cache.trajectory, never GT or a candidate-set
maximum. NAVSIM module loading is isolated from the VLA/Agent environment.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from ..contracts import CandidateRecord, ScoreRecord, strict_record, require
from ..io import atomic_json, digest, file_hash, read_json

COMPONENTS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def python_source_hash(root):
    root = Path(root)
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py")) if "__pycache__" not in str(p)})


class ReferenceBackend:
    def __init__(self, navsim_root, python, scoring_config, *, timeout=300, eval_seed=42):
        self.root = str(Path(navsim_root).resolve())
        self.python = python
        self.config = scoring_config
        self.timeout, self.eval_seed = timeout, eval_seed
        self.source_hash = python_source_hash(Path(self.root) / "navsim")
        self.protocol = {"metric": "NAVSIM_v1.1_PDMS", "source_hash": self.source_hash,
                         "config": scoring_config, "backend": "official_single_reference",
                         "score_scale": "zero_one", "traffic": "official_nonreactive", "reference": "metric_cache.trajectory"}
        self.protocol_hash = digest(self.protocol)

    def score(self, candidate, scene, *, repetition=0, original_entry_source=None):
        require(candidate.scene_id == scene.scene_id, "score ID join mismatch")
        require(file_hash(scene.metric_context_ref) == scene.metric_context_hash, "metric context changed")
        require(file_hash(candidate.trajectory_ref) == candidate.trajectory_hash, "candidate trajectory changed")
        # Stable paired seed: unrelated candidate identity/order/chunk size is excluded.
        seed = int(digest({"scene_id": scene.scene_id, "seed": self.eval_seed, "repetition": repetition})[:8], 16)
        payload = {"candidate": asdict(candidate), "scene": asdict(scene), "protocol_hash": self.protocol_hash,
                   "navsim_root": self.root, "source_hash": self.source_hash, "config": self.config,
                   "seed": seed, "eval_seed": self.eval_seed, "repetition": repetition}
        if original_entry_source:
            payload["original_entry_source"] = str(original_entry_source)
        with tempfile.TemporaryDirectory(prefix="iqe-reference-") as tmp:
            request, response = Path(tmp) / "request.json", Path(tmp) / "response.json"
            atomic_json(request, payload)
            env = dict(os.environ)
            repo = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = repo + os.pathsep + self.root + os.pathsep + str(Path(repo) / "nuplan-devkit")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "1"
            env["CUDA_VISIBLE_DEVICES"] = ""
            try:
                run = subprocess.run([self.python, "-m", "iqe.scoring.worker", str(request), str(response)],
                                     env=env, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                return ScoreRecord(1, candidate.key, self.protocol_hash, scene.metric_context_hash, "unavailable", None,
                    {c: None for c in COMPONENTS}, {c: False for c in COMPONENTS}, "official_single_reference", "reference-v1",
                    self.eval_seed, repetition, False, f"official scoring timeout after {self.timeout}s")
            if run.returncode != 0 or not response.exists():
                return ScoreRecord(1, candidate.key, self.protocol_hash, scene.metric_context_hash, "unavailable", None,
                                   {c: None for c in COMPONENTS}, {c: False for c in COMPONENTS},
                                   "official_single_reference", "reference-v1", self.eval_seed, repetition, False,
                                   "official backend failed: " + run.stderr[-2000:])
            return strict_record(ScoreRecord, read_json(response))

    def validate_candidate_set_invariance(self, candidates, scene, tolerance=1e-12):
        require(len(candidates) >= 2, "invariance audit needs at least two trajectories")
        base = [self.score(c, scene) for c in candidates]
        require(all(s.score_valid for s in base), "invalid real scoring audit")
        expected = {c.key: asdict(s) for c, s in zip(candidates, base)}
        layouts = [list(reversed(candidates)), candidates[:1], candidates + [candidates[0]], candidates[1:]]
        calls = 0
        for layout in layouts:
            for chunk in (1, 2, len(layout)):
                for at in range(0, len(layout), chunk):
                    for c in layout[at:at + chunk]:
                        s = self.score(c, scene)
                        require(asdict(s) == expected[c.key], "candidate-set/chunk/repetition changed an official label")
                        calls += 1
        return {"passed": True, "independent_real_scoring_calls": calls + len(base),
                "protocol_hash": self.protocol_hash, "tolerance": tolerance}
