"""Typed candidate records and exact trajectory deduplication before weighting."""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .contracts import V1_COMPONENTS, V1_PROTOCOL


@dataclass
class Candidate:
    token: str
    log_name: str
    source: str
    behavior_sha256: str
    normalized_action: np.ndarray
    requested_trajectory: np.ndarray
    simulated_states: np.ndarray
    score: float
    metrics: dict
    protocol_sha256: str
    noise_keys: dict
    source_parameters: dict = field(default_factory=dict)
    status: str = "valid"
    is_reference: bool = False
    old_logprob: object = None
    protocol: str = V1_PROTOCOL

    def validate(self):
        if self.protocol != V1_PROTOCOL or not self.protocol_sha256:
            raise ValueError("candidate requires a pinned canonical v1.1 score")
        if self.status != "valid":
            raise ValueError(f"candidate {self.token} is {self.status}; update must not hide it")
        if not self.token or not self.log_name or not self.behavior_sha256:
            raise ValueError("missing candidate identity")
        for name, shape in (("normalized_action", (8, 4)), ("requested_trajectory", (8, 3))):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}")
        states = np.asarray(self.simulated_states)
        if states.ndim != 2 or states.shape[0] != 41 or not np.isfinite(states).all():
            raise ValueError("must retain the distinct official 0..4s simulated states")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("invalid raw official score")
        for metric in V1_COMPONENTS:
            if metric not in self.metrics or not np.isfinite(self.metrics[metric]) or not 0 <= self.metrics[metric] <= 1:
                raise ValueError(f"missing/invalid official {metric}")
        if self.source not in ("reference_ode", "ode", "legacy_sde", "latent_search", "behavior_search", "gt_probe", "pdm_probe"):
            raise ValueError("unregistered candidate source")
        if self.source != "legacy_sde" and self.old_logprob is not None:
            raise ValueError("ODE/external trajectories do not have SDE old-policy logprob")
        if self.is_reference != (self.source == 'reference_ode'):
            raise ValueError('Reference identity must be explicit and separate from the current behavior')
        if self.source in ("gt_probe", "pdm_probe"):
            raise ValueError("privileged probes are separate from the main policy banks")

    @property
    def trajectory_key(self):
        poses = np.array(self.requested_trajectory, dtype="<f8", order="C")
        poses[poses == 0] = 0  # canonicalize signed zero, not near-duplicates
        return hashlib.sha256(poses.tobytes()).hexdigest()

    def require_nft_behavior(self, sha256):
        self.validate()
        if self.source != "ode" or self.behavior_sha256 != sha256 or self.is_reference:
            raise ValueError("NFT accepts only actual samples of its fixed ODE behavior")


def deduplicate(candidates):
    if not candidates:
        raise ValueError("empty candidate bank scene")
    first = candidates[0]
    unique, groups, key_to_index = [], [], {}
    for i, candidate in enumerate(candidates):
        candidate.validate()
        if (candidate.token, candidate.log_name, candidate.protocol_sha256) != (first.token, first.log_name, first.protocol_sha256):
            raise ValueError("deduplication crossed scene/protocol boundaries")
        key = candidate.trajectory_key
        if key not in key_to_index:
            key_to_index[key] = len(unique)
            unique.append(candidate)
            groups.append([])
        j = key_to_index[key]
        original = unique[j]
        if original.score != candidate.score or original.metrics != candidate.metrics or not np.array_equal(original.simulated_states, candidate.simulated_states):
            raise ValueError("identical requests have different canonical outcomes")
        groups[j].append(i)
    reference_count = sum(candidate.is_reference for candidate in candidates)
    if reference_count != 4:
        raise ValueError("exactly four predeclared SFT reference draws required")
    # Includes all logical bank slots, including the four references. Elites
    # retained between CEM iterations are not appended again as new slots.
    empirical = np.asarray([len(group) / len(candidates) for group in groups], dtype=np.float64)
    q_ref = np.asarray([sum(candidates[i].is_reference for i in group) / reference_count for group in groups], dtype=np.float64)
    return unique, q_ref, empirical, groups
