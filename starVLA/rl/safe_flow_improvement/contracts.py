"""Scientific identities and population boundaries, independent of GPU routing."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_COMMIT = "25f9148eb75ddf171313720e758a280b82c24f7f"
SFT_SHA256 = "9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4"
V1_PROTOCOL = "navsim_v1.1_independent_reference_candidate"
SAFETY_METRICS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")
V1_COMPONENTS = SAFETY_METRICS + ("ego_progress", "comfort", "driving_direction_compliance")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(namespace, seed, token, *keys):
    """Scene-bound independent streams; shard/host/rank never enter this key."""
    return int(fingerprint([namespace, int(seed), str(token), *keys])[:15], 16)


def validate_probability(values, size=None, name="probability"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or (size is not None and len(values) != size):
        raise ValueError(f"{name}: invalid shape")
    if not np.isfinite(values).all() or (values < 0).any() or abs(values.sum() - 1) > 1e-12:
        raise ValueError(f"{name}: nonnegative finite normalized mass required")
    return values


@dataclass(frozen=True)
class PopulationContract:
    train: frozenset
    dev: frozenset
    navtest: frozenset
    token_logs: dict

    def __post_init__(self):
        if self.train & self.dev or self.train & self.navtest or self.dev & self.navtest:
            raise ValueError("population token overlap")
        populations = [self.train, self.dev, self.navtest]
        logs = [{self.token_logs[token] for token in population} for population in populations]
        if any(logs[i] & logs[j] for i in range(3) for j in range(i)):
            raise ValueError("population log overlap")

    def require_training(self, tokens, purpose):
        if not tokens or not set(tokens) <= self.train:
            raise ValueError(f"{purpose} may use only registered RL_TRAIN")


@dataclass(frozen=True)
class WindowIdentity:
    stage: str
    variant: str
    training_seed: int
    iteration: int
    sft_sha256: str
    behavior_sha256: str
    split_sha256: str
    bank_sha256: str
    protocol_sha256: str
    objective_sha256: str
    numerics_sha256: str

    @property
    def sha256(self):
        return fingerprint(asdict(self))


def require_same_window(expected, restored):
    if asdict(expected) != asdict(restored):
        raise ValueError("resume changes model/data/behavior/bank/objective/numerical identity")
