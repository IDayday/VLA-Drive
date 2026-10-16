"""Generated trajectories, independent truth feedback and bounded reuse.

No environment/GT object is accepted by the rollout function. Canonical CPU
scoring happens outside this module and joins by immutable rollout/context IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Tuple

import torch

from .policy_snapshot import FrozenPolicySnapshot, PolicyIdentity, policy_eval_mode, tensor_hash
from .fpo_objective import CFMSamples, FPOConfig, sample_cfm


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected SHA256 content identity")


@dataclass(frozen=True)
class RolloutBatch:
    scene_ids: Tuple[str, ...]
    observation_hashes: Tuple[str, ...]
    condition_hashes: Tuple[str, ...]
    policy: PolicyIdentity
    initial_noise: torch.Tensor  # [scene,4,8,4], float32 as original inference
    raw_actions: torch.Tensor
    rollout_id: str
    flow_steps: int = 10
    raw_representation: str = "normalized_xy_sin_cos"

    def content_identity(self):
        return _sha({"schema": "fpo_rollout_v1", "scene_ids": self.scene_ids,
                     "observation_hashes": self.observation_hashes, "condition_hashes": self.condition_hashes,
                     "policy": self.policy.__dict__, "initial_noise": tensor_hash(self.initial_noise),
                     "raw_actions": tensor_hash(self.raw_actions), "flow_steps": self.flow_steps,
                     "raw_representation": self.raw_representation})

    def validate(self):
        if self.policy.role != "old" or self.flow_steps != 10 or self.raw_representation != "normalized_xy_sin_cos":
            raise ValueError("Rollout must use the frozen old ten-step original action policy")
        if (not self.scene_ids or any(not isinstance(x, str) or not x for x in self.scene_ids)
            or len(self.observation_hashes) != len(self.scene_ids) or len(self.condition_hashes) != len(self.scene_ids)):
            raise ValueError("Rollout scene/observation/condition identity mismatch")
        for value in (*self.observation_hashes, *self.condition_hashes):
            _validate_hash(value)
        expected = (len(self.scene_ids), 4, 8, 4)
        for value in (self.initial_noise, self.raw_actions):
            if value.shape != expected or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError("Rollout requires finite detached float32 four-candidate raw actions")
        if self.rollout_id != self.content_identity():
            raise ValueError("Rollout content changed or identity corrupted")

    def state_dict(self):
        self.validate()
        return {"schema": "fpo_rollout_v1", "scene_ids": self.scene_ids,
                "observation_hashes": self.observation_hashes, "condition_hashes": self.condition_hashes,
                "policy": dict(self.policy.__dict__), "initial_noise": self.initial_noise.cpu().clone(),
                "raw_actions": self.raw_actions.cpu().clone(), "rollout_id": self.rollout_id,
                "flow_steps": self.flow_steps, "raw_representation": self.raw_representation}

    @classmethod
    def from_state_dict(cls, state, *, device="cpu"):
        expected = {"schema", "scene_ids", "observation_hashes", "condition_hashes", "policy",
                    "initial_noise", "raw_actions", "rollout_id", "flow_steps", "raw_representation"}
        if set(state) != expected or state["schema"] != "fpo_rollout_v1":
            raise ValueError("Invalid rollout checkpoint schema")
        result = cls(tuple(state["scene_ids"]), tuple(state["observation_hashes"]), tuple(state["condition_hashes"]),
                     PolicyIdentity(**state["policy"]), state["initial_noise"].to(device), state["raw_actions"].to(device),
                     state["rollout_id"], state["flow_steps"], state["raw_representation"])
        result.validate()
        return result


@torch.no_grad()
def sample_rollout_group(snapshot: FrozenPolicySnapshot, condition: torch.Tensor, *, scene_ids,
                         observation_hashes, noise_generator):
    """Four true ten-Euler-step trajectories; one shared H_A projection.

    Caller encodes current observations ONCE and passes only frozen conditions.
    There is no GT bridge and no robot [-1,1] action clipping. Autocast/precision
    is inherited from the caller's registered original inference runtime.
    """
    if not isinstance(snapshot, FrozenPolicySnapshot) or snapshot.identity.role != "old":
        raise ValueError("Rollout requires an explicit frozen old-policy snapshot")
    if any(p.requires_grad for p in snapshot.parameters()) or any(m.training for m in snapshot.modules()):
        raise ValueError("Old rollout policy must remain frozen and eval")
    policy = snapshot.policy
    policy._check_condition(condition)
    if len(scene_ids) != len(condition) or len(observation_hashes) != len(condition):
        raise ValueError("Scene IDs do not match the frozen context batch")
    initial = torch.randn((len(condition), 4, policy.horizon, policy.channels), device=condition.device,
                          dtype=torch.float32, generator=noise_generator)
    with policy_eval_mode(snapshot):
        projected = policy.project_condition(condition)
        memory = projected[:, None].expand(-1, 4, -1, -1).reshape(-1, *projected.shape[1:])
        state = initial.reshape(-1, policy.horizon, policy.channels).clone()
        for index in range(10):
            # Match original int((i/10)*num_timestep_buckets) exactly, including
            # Python scalar rounding. Uniform CFM uses the original tensor cast.
            buckets = torch.full((len(state),), int((index / 10.) * policy.action_head.num_timestep_buckets),
                                 device=state.device, dtype=torch.long)
            velocity = policy.action_head.forward_onestep(memory, state, buckets)[0]
            state = state + .1 * velocity
            if not torch.isfinite(state).all():
                raise ValueError(f"Nonfinite actual generated trajectory at Euler step {index+1}")
    raw = state.reshape_as(initial).detach().clone()
    if raw.dtype != torch.float32:
        raise ValueError("Original float32 Euler residual stream changed dtype")
    fields = dict(scene_ids=tuple(scene_ids), observation_hashes=tuple(observation_hashes),
                  condition_hashes=tuple(tensor_hash(x) for x in condition), policy=snapshot.identity,
                  initial_noise=initial.detach().clone(), raw_actions=raw)
    draft = RolloutBatch(**fields, rollout_id="")
    result = RolloutBatch(**fields, rollout_id=draft.content_identity())
    result.validate()
    return result


@dataclass(frozen=True)
class FeedbackBatch:
    rollout_id: str
    score_protocol_hash: str
    context_hashes: Tuple[str, ...]
    reward_kind: str  # C1-PDMS, C2-D, C3-C, C4-T. C5 requires a separate registration.
    rewards: torch.Tensor
    valid: torch.Tensor
    statuses: Tuple[str, ...]

    def validate(self, rollout: RolloutBatch, *, require_complete=True):
        rollout.validate()
        if self.rollout_id != rollout.rollout_id:
            raise ValueError("Feedback joined to the wrong actually generated trajectories")
        _validate_hash(self.score_protocol_hash)
        if len(self.context_hashes) != len(rollout.scene_ids):
            raise ValueError("Missing scene metric context identity")
        for value in self.context_hashes:
            _validate_hash(value)
        expected = rollout.raw_actions.shape[:2]
        if self.rewards.shape != expected or self.valid.shape != expected or self.valid.dtype != torch.bool:
            raise ValueError("Feedback candidate count/valid mask mismatch")
        if self.rewards.requires_grad or self.valid.requires_grad:
            raise ValueError("Official simulator feedback must be detached")
        if len(self.statuses) != self.rewards.numel():
            raise ValueError("Every candidate needs an explicit scoring status")
        for valid, status in zip(self.valid.flatten().tolist(), self.statuses):
            if bool(valid) != (status == "valid_scored"):
                raise ValueError("Scoring failure cannot masquerade as valid zero reward")
            if status not in ("valid_scored", "transient_scoring_failure", "invalid_prediction", "contract_mismatch"):
                raise ValueError("Unknown scoring status")
        if not torch.isfinite(self.rewards[self.valid]).all():
            raise ValueError("Nonfinite valid official reward")
        bounds = {"C1-PDMS": (0., 1.), "C2-D": (-1., 0.), "C3-C": (-1., 0.), "C4-T": (-1., 0.)}
        if self.reward_kind not in bounds:
            raise ValueError("Unknown/unregistered reward definition")
        lower, upper = bounds[self.reward_kind]
        if ((self.rewards[self.valid] < lower) | (self.rewards[self.valid] > upper)).any():
            raise ValueError("Official reward unit/range mismatch")
        allowed = {"C2-D": (-1., 0.), "C3-C": (-1., -.5, 0.), "C4-T": (-1., 0.)}
        if self.reward_kind in allowed and any(float(x) not in allowed[self.reward_kind]
                                               for x in self.rewards[self.valid].flatten().tolist()):
            raise ValueError("Canonical event/severity reward lost its discrete meaning")
        if require_complete and not self.valid.all():
            raise ValueError("Incomplete feedback requires retry; no policy update")

    def state_dict(self):
        return {"schema": "fpo_feedback_v1", "rollout_id": self.rollout_id,
                "score_protocol_hash": self.score_protocol_hash, "context_hashes": self.context_hashes,
                "reward_kind": self.reward_kind, "rewards": self.rewards.detach().cpu().clone(),
                "valid": self.valid.cpu().clone(), "statuses": self.statuses}

    @classmethod
    def from_state_dict(cls, state, rollout, *, device="cpu"):
        if set(state) != {"schema", "rollout_id", "score_protocol_hash", "context_hashes", "reward_kind",
                          "rewards", "valid", "statuses"} or state["schema"] != "fpo_feedback_v1":
            raise ValueError("Invalid feedback checkpoint schema")
        result = cls(state["rollout_id"], state["score_protocol_hash"], tuple(state["context_hashes"]),
                     state["reward_kind"], state["rewards"].to(device), state["valid"].to(device), tuple(state["statuses"]))
        result.validate(rollout, require_complete=False)
        return result


@dataclass(frozen=True)
class CoupledFPOBatch:
    """One saved old-CFM/noise/time identity, reused unchanged by current policy."""
    rollout_id: str
    policy: PolicyIdentity
    config: FPOConfig
    samples: CFMSamples
    old_cfm: torch.Tensor
    comparison_id: str

    def content_identity(self):
        return _sha({"schema": "fpo_comparison_v1", "rollout_id": self.rollout_id,
                     "policy": self.policy.__dict__, "config": self.config.to_dict(),
                     "noise": tensor_hash(self.samples.noise), "times": tensor_hash(self.samples.times),
                     "old_cfm": tensor_hash(self.old_cfm)})

    def validate(self, rollout):
        rollout.validate()
        if rollout.rollout_id != self.rollout_id or rollout.policy != self.policy:
            raise ValueError("CFM comparison is not bound to these generated actions/old policy")
        actions = rollout.raw_actions.flatten(0,1)
        self.samples.validate(actions)
        if (self.old_cfm.shape != (len(actions), self.config.mc_samples) or self.old_cfm.requires_grad
            or not self.old_cfm.is_floating_point() or not torch.isfinite(self.old_cfm).all()):
            raise ValueError("Invalid frozen CFM reference population")
        if self.comparison_id != self.content_identity():
            raise ValueError("CFM noise/time/old loss coupling content changed")

    def current_losses(self, current, rollout, condition, *, mc_chunk=1, checkpoint_velocity=False):
        self.validate(rollout)
        if tuple(tensor_hash(x) for x in condition) != rollout.condition_hashes:
            raise ValueError("Current CFM planner context differs from scored old rollout")
        return current(condition.repeat_interleave(4,0), rollout.raw_actions.flatten(0,1), self.samples,
                       mc_chunk=mc_chunk, checkpoint_velocity=checkpoint_velocity)

    def state_dict(self):
        return {"schema": "fpo_comparison_v1", "rollout_id": self.rollout_id,
                "policy": dict(self.policy.__dict__), "config": self.config.to_dict(),
                "samples": self.samples.state_dict(), "old_cfm": self.old_cfm.cpu().clone(),
                "comparison_id": self.comparison_id}

    @classmethod
    def from_state_dict(cls, state, rollout, *, device="cpu"):
        if set(state) != {"schema", "rollout_id", "policy", "config", "samples", "old_cfm", "comparison_id"} or state["schema"] != "fpo_comparison_v1":
            raise ValueError("Invalid coupled FPO checkpoint schema")
        result = cls(state["rollout_id"], PolicyIdentity(**state["policy"]), FPOConfig.from_dict(state["config"]),
                     CFMSamples.from_state_dict(state["samples"], device=device), state["old_cfm"].to(device), state["comparison_id"])
        result.validate(rollout)
        return result


@torch.no_grad()
def prepare_fpo_batch(snapshot, rollout, condition, *, config, streams, mc_chunk=1):
    """Fresh MC draws independent of generation noise; persist before updating."""
    rollout.validate()
    if snapshot.identity != rollout.policy:
        raise ValueError("Wrong old policy for saved rollout CFM loss")
    if tuple(tensor_hash(x) for x in condition) != rollout.condition_hashes:
        raise ValueError("Old CFM context differs from the scored trajectory context")
    actions = rollout.raw_actions.flatten(0,1)
    samples = sample_cfm(actions, config, noise_generator=streams["policy_noise"], time_generator=streams["policy_time"])
    old_cfm = snapshot(condition.repeat_interleave(4,0), actions, samples, mc_chunk=mc_chunk).detach().clone()
    fields = dict(rollout_id=rollout.rollout_id, policy=snapshot.identity, config=config, samples=samples, old_cfm=old_cfm)
    draft = CoupledFPOBatch(**fields, comparison_id="")
    result = CoupledFPOBatch(**fields, comparison_id=draft.content_identity())
    result.validate(rollout)
    return result


class FeedbackReuseGuard:
    """Consume only at a successfully saved optimizer boundary.

    The trainer separately checkpoints pending RolloutBatch, FeedbackBatch,
    CFMSamples, old-CFM values and the old snapshot. Prefetch does not consume.
    Config is identity-bound; resuming cannot silently extend the reuse budget.
    """
    def __init__(self, *, score_protocol_hash, max_reuses=1, max_update_age=1, max_records=128):
        _validate_hash(score_protocol_hash)
        if any(type(x) is not int or x < 1 for x in (max_reuses, max_update_age, max_records)):
            raise ValueError("Bounded feedback limits must be positive integers")
        if max_reuses > 16 or max_update_age > 16 or max_records > 4096:
            raise ValueError("Feedback budget exceeds this campaign's bounded contract")
        self.limits = dict(max_reuses=max_reuses, max_update_age=max_update_age, max_records=max_records,
                           score_protocol_hash=score_protocol_hash)
        self.consumed = {}
        self.retired_before = 0

    def check(self, rollout, feedback, *, active_old: PolicyIdentity, optimizer_step: int, expected_context_hashes):
        feedback.validate(rollout)
        if feedback.score_protocol_hash != self.limits["score_protocol_hash"] or tuple(expected_context_hashes) != feedback.context_hashes:
            raise ValueError("Feedback protocol/context differs from the registered canonical scoring inputs")
        if active_old != rollout.policy:
            raise ValueError("Stale policy version/hash; regenerate on-policy trajectories")
        age = optimizer_step - rollout.policy.optimizer_step
        if (type(optimizer_step) is not int or age < 0 or age >= self.limits["max_update_age"]
            or rollout.policy.optimizer_step < self.retired_before):
            raise ValueError("Rollout exceeds registered optimizer staleness budget")
        used = self.consumed.get(rollout.rollout_id, [])
        if len(used) >= self.limits["max_reuses"] or optimizer_step in used:
            raise ValueError("Feedback already consumed at this update or reuse limit")
        if rollout.rollout_id not in self.consumed and len(self.consumed) >= self.limits["max_records"]:
            raise ValueError("Bounded feedback ledger full; retire obsolete policy records explicitly")

    def commit_consumed(self, rollout, feedback, *, active_old, optimizer_step, expected_context_hashes):
        self.check(rollout, feedback, active_old=active_old, optimizer_step=optimizer_step,
                   expected_context_hashes=expected_context_hashes)
        self.consumed.setdefault(rollout.rollout_id, []).append(optimizer_step)

    def retire_before(self, minimum_optimizer_step):
        if type(minimum_optimizer_step) is not int or minimum_optimizer_step < 0:
            raise ValueError("Invalid explicitly retired update boundary")
        if minimum_optimizer_step < self.retired_before:
            raise ValueError("Cannot unretire an old feedback population")
        self.retired_before = minimum_optimizer_step
        self.consumed = {key: steps for key, steps in self.consumed.items() if max(steps) >= minimum_optimizer_step}

    def state_dict(self):
        return {"schema": "fpo_reuse_v1", "limits": dict(self.limits),
                "consumed": {key: list(steps) for key, steps in self.consumed.items()},
                "retired_before": self.retired_before}

    def load_state_dict(self, state):
        if set(state) != {"schema", "limits", "consumed", "retired_before"} or state["schema"] != "fpo_reuse_v1" or state["limits"] != self.limits:
            raise ValueError("Feedback reuse checkpoint contract mismatch")
        if type(state["retired_before"]) is not int or state["retired_before"] < 0:
            raise ValueError("Invalid retired feedback boundary")
        if len(state["consumed"]) > self.limits["max_records"]:
            raise ValueError("Feedback ledger exceeds bounded capacity")
        for key, steps in state["consumed"].items():
            _validate_hash(key)
            if (not isinstance(steps, list) or not steps or len(steps) > self.limits["max_reuses"]
                or len(set(steps)) != len(steps) or any(type(x) is not int or x < 0 for x in steps)):
                raise ValueError("Invalid consumed feedback cursor")
        self.consumed = {key: list(steps) for key, steps in state["consumed"].items()}
        self.retired_before = state["retired_before"]
