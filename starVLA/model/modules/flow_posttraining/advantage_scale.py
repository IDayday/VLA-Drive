"""One explicitly registered adaptation of complete-rollout FPO advantages.

The original FP32 scene centering is preserved. A detached, FP64 two-pass sample
standard deviation scales exactly one complete optimizer rollout, before any
microbatching. This changes cross-update policy/BC weighting; it is not a
lossless optimizer change or an assurance of safety. No RNG is consumed.

Every rank must call this function in the same process group. Input preparation
errors are voted on before statistics collectives. Backend/process failure still
uses the caller's process-group timeout; this helper cannot recover a dead rank.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import distributed as dist

from .fpo_objective import group_centered_advantages
from .policy_snapshot import PolicyIdentity, tensor_hash


METHOD = "scene_center_rollout_std_v1"
EPSILON = 1e-10
_HASH_FIELDS = ("local_identity_sha256", "rewards_sha256", "valid_sha256",
                "raw_advantages_sha256", "normalized_advantages_sha256")


class RolloutNormalizationError(ValueError):
    """A rejected complete rollout; no partial feedback or fallback scale."""


def normalization_contract():
    """Return a new JSON value, suitable for the execution/checkpoint identity."""
    return {"method": METHOD, "group_size": 4,
            "centering": "legacy_fp32_scene_sum_divide_count",
            "population": "one_complete_global_old_policy_optimizer_rollout",
            "include_zero_advantages": True, "require_complete": True,
            "statistics_dtype": "float64", "statistics": "two_pass_centered_M2",
            "correction": 1, "epsilon": EPSILON, "epsilon_placement": "after_sqrt",
            "numerator_global_recentering": False, "output_dtype": "float32",
            "detached": True}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def receipt_sha256(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _hash_check(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Identity requires a lowercase SHA256 content hash")


def _identity(value, required):
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("Incomplete rollout normalization identity")
    # JSON roundtrip also gives receipts ownership independent of mutable input.
    encoded = _json_bytes(dict(value))
    if len(encoded) > 65536:
        raise ValueError("Normalization identity exceeds its 64 KiB metadata bound")
    return json.loads(encoded)


def _vote(error, *, phase, device, active):
    """The only failure path before a subsequent statistics/gather collective."""
    failed = torch.tensor(int(error is not None), dtype=torch.int64, device=device)
    if active:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if not failed.item():
        return
    message = "" if error is None else f"{type(error).__name__}: {error}"
    if active:
        # Fixed-size UTF-8 bytes avoid pickle/object collectives and unbounded
        # diagnostics. Error text truncation has no effect on the failure vote.
        raw = message.encode("utf-8", errors="replace")[:1023]
        payload = torch.zeros(1024, dtype=torch.uint8, device=device)
        if raw:
            payload[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=device)
        gathered = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, payload)
        messages = []
        for rank, item in enumerate(gathered):
            text = bytes(item.cpu().tolist()).split(b"\0", 1)[0].decode("utf-8", errors="replace")
            if text:
                messages.append(f"rank {rank}: {text}")
        message = "; ".join(messages)
    raise RolloutNormalizationError(f"{METHOD} rejected in {phase}: {message}")


def _gather(tensor, active):
    if not active:
        return [tensor]
    values = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(values, tensor)
    return values


def normalize_rollout_advantages(rewards, valid, *, collective_device,
                                  expected_global_count, common_identity,
                                  local_identity, distributed=True):
    """Return ``(FP32 detached [scene,4] advantages, JSON receipt)``.

    ``common_identity`` binds old_policy (PolicyIdentity fields), numeric and
    score-protocol SHA256s, C1-PDMS reward_kind, and global_scene_count. It must be
    identical across ranks. ``local_identity`` binds rollout_id and feedback_hash.
    Additional JSON dependencies are allowed and hashed. All feedback must be
    valid and finite [0,1]. Empty local [0,4] shards contribute neutral moments;
    the existing trainer may independently prohibit empty backward ranks.

    The all-rank common receipt includes every local input/output hash. Repeating
    the same rollout with the same topology reproduces that identity without
    consuming sampler or RNG state. No scale from another rollout is reused.
    """
    initialized = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if initialized else 1
    rank = dist.get_rank() if initialized else 0
    # Use a known working group device for error votes even if the supplied
    # collective_device is malformed on one rank. NCCL callers already set the
    # rank's current CUDA device before creating their process group.
    backend = str(dist.get_backend()) if initialized else None
    active = initialized and (world > 1 or distributed is not False)
    vote_device = (torch.device("cuda", torch.cuda.current_device())
                   if backend == "nccl" else torch.device("cpu"))
    error = None
    try:
        if type(distributed) is not bool:
            raise ValueError("distributed must be bool")
        if initialized and world > 1 and not distributed:
            raise ValueError("Cannot disable normalization collectives in a multi-rank group")
        if initialized and backend not in ("gloo", "nccl"):
            raise ValueError("Only the qualified Gloo and NCCL backends are supported")
        requested_device = torch.device(collective_device)
        if requested_device.type not in ("cpu", "cuda"):
            raise ValueError("collective_device must be CPU or CUDA")
        if initialized:
            if requested_device.type != vote_device.type:
                raise ValueError("collective_device does not match process-group backend")
            if (requested_device.type == "cuda" and requested_device.index is not None
                    and requested_device.index != vote_device.index):
                raise ValueError("NCCL collective_device must be the rank's current CUDA device")
        else:
            # Statistics can be CPU in an ordinary single-process call. Input
            # tensors and returned advantages keep their original device.
            torch.empty(0, device=requested_device)
            vote_device = requested_device
        if type(expected_global_count) is not int or not 0 <= expected_global_count < 2**63:
            raise ValueError("expected_global_count must be an int64-range nonnegative integer")
        common = _identity(common_identity, {"old_policy", "numeric_execution_hash",
                           "score_protocol_hash", "reward_kind", "global_scene_count"})
        policy = PolicyIdentity(**common["old_policy"])
        if policy.role != "old":
            raise ValueError("Normalization requires the actual old policy, not the reference")
        for key in ("numeric_execution_hash", "score_protocol_hash"):
            _hash_check(common[key])
        if common["reward_kind"] != "C1-PDMS":
            raise ValueError("Only the registered C1-PDMS scale adaptation is implemented")
        if (type(common["global_scene_count"]) is not int
                or common["global_scene_count"] < 0
                or 4 * common["global_scene_count"] != expected_global_count):
            raise ValueError("Global scene identity disagrees with complete G4 candidate count")
        local = _identity(local_identity, {"rollout_id", "feedback_hash"})
        for key in ("rollout_id", "feedback_hash"):
            _hash_check(local[key])
        if not isinstance(rewards, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise ValueError("Reward/valid inputs must be tensors")
        if (rewards.dtype != torch.float32 or rewards.ndim != 2 or rewards.shape[1] != 4
                or valid.shape != rewards.shape or valid.dtype != torch.bool):
            raise ValueError("Complete feedback requires FP32 rewards and bool mask [scene,4]")
        if rewards.requires_grad or valid.requires_grad:
            raise ValueError("Official rewards/validity must be detached")
        if rewards.device != valid.device or rewards.layout != torch.strided or valid.layout != torch.strided:
            raise ValueError("Reward/mask device or dense-layout mismatch")
        if not valid.all():
            raise ValueError("Incomplete official feedback: retry before policy update")
        if not torch.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
            raise ValueError("C1-PDMS rewards must be finite and in [0,1]")
        raw = group_centered_advantages(rewards, valid, require_complete=True)
        raw64 = raw.to(device=vote_device, dtype=torch.float64)
        local_count = raw.numel()
        counts = torch.tensor([local_count, int(torch.count_nonzero(raw))],
                              dtype=torch.int64, device=vote_device)
        summed = raw64.sum()
        common_identity_hash = receipt_sha256(common)
        # Expected population is part of the first fixed-size identity gather.
        identity_packet = torch.tensor([expected_global_count, *bytes.fromhex(common_identity_hash)],
                                       dtype=torch.int64, device=vote_device)
        record = {"rank": rank, "local_count": local_count,
                  "local_identity_sha256": receipt_sha256(local),
                  "rewards_sha256": tensor_hash(rewards), "valid_sha256": tensor_hash(valid),
                  "raw_advantages_sha256": tensor_hash(raw)}
    except Exception as exc:
        error = exc
    _vote(error, phase="local preparation", device=vote_device, active=active)

    identities = _gather(identity_packet, active)
    error = None
    if any(not torch.equal(item, identities[0]) for item in identities):
        error = ValueError("Ranks disagree on common identity or expected global count")
    _vote(error, phase="common identity", device=vote_device, active=active)
    if active:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    count, nonzero_count = (int(x) for x in counts.cpu().tolist())
    error = None
    try:
        if count < 2 or count != expected_global_count:
            raise ValueError("Complete global rollout count must be >=2 and equal its identity")
        if not torch.isfinite(summed):
            raise ValueError("Nonfinite global advantage sum")
        mean = summed / count
        centered_m2 = (raw64 - mean).square().sum()
        if not torch.isfinite(centered_m2):
            raise ValueError("Nonfinite local second moment")
    except Exception as exc:
        error = exc
    _vote(error, phase="count and second-moment preparation", device=vote_device, active=active)
    if active:
        dist.all_reduce(centered_m2, op=dist.ReduceOp.SUM)
    error = None
    try:
        if not torch.isfinite(centered_m2) or centered_m2 < 0:
            raise ValueError("Invalid global second central moment")
        std = (centered_m2 / (count - 1)).sqrt()
        if nonzero_count and std == 0:
            raise ValueError("Nonzero advantages have zero rollout variance")
        divisor = std + EPSILON
        # The variance is centered globally; the numerator deliberately is not.
        normalized = (raw.double() / divisor.to(raw.device)).float().detach()
        if not torch.isfinite(normalized).all():
            raise ValueError("Nonfinite normalized advantages")
        if not torch.equal(normalized[raw == 0], raw[raw == 0]):
            raise ValueError("Rollout scaling changed an exact zero advantage")
        record["normalized_advantages_sha256"] = tensor_hash(normalized)
        packet = torch.tensor([local_count, *[b for field in _HASH_FIELDS
                                              for b in bytes.fromhex(record[field])]],
                              dtype=torch.int64, device=vote_device)
    except Exception as exc:
        error = exc
    _vote(error, phase="normalization and receipt preparation", device=vote_device, active=active)
    packets = _gather(packet, active)
    rank_records = []
    for owner, item in enumerate(packets):
        values = item.cpu().tolist()
        entry = {"rank": owner, "local_count": int(values[0])}
        for index, field in enumerate(_HASH_FIELDS):
            start = 1 + index * 32
            entry[field] = bytes(values[start:start + 32]).hex()
        rank_records.append(entry)
    common_receipt = {"definition": normalization_contract(), "identity": common,
                      "identity_sha256": common_identity_hash,
                      "world_size": world if active else 1,
                      "expected_global_count": expected_global_count,
                      "count": count, "nonzero_count": nonzero_count,
                      "sum": float(summed), "mean": float(mean),
                      "centered_m2": float(centered_m2), "std": float(std),
                      "gain": float(divisor.reciprocal()), "all_zero": nonzero_count == 0,
                      "rank_records": rank_records}
    result = {"schema": "fpo_rollout_scale_receipt_v1", "method": METHOD,
              "common": common_receipt, "common_sha256": receipt_sha256(common_receipt),
              "local": {**record, "identity": local}}
    return normalized, result
