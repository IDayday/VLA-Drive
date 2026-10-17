"""No-update per-objective diagnostics on a declared shared parameter subset.

Uses the existing differentiable losses, never reconstructs them from scalars.
The one-shot coefficient rule is a project proposal, not GradNorm training.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import math
import random

import numpy as np
import torch

OBJECTIVES = ("ego_fm", "current", "future", "delta")
RULE = "current_anchored_min_loss_and_shared_gradient_ratio_v1"


def tensor_hash(value):
    value = value.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for chunk in value.split(4 * 1024 * 1024):
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_identity(model):
    """Full bytes, including frozen parameters/buffers; no saved model copy."""
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": tensor_hash(value)} for name, value in itertools.chain(model.named_parameters(), model.named_buffers())}


@contextmanager
def preserved_rng(generators):
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "cpu": torch.get_rng_state(),
              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
              "generators": {name: value.get_state() for name, value in generators.items()}}
    try:
        yield
    finally:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["cpu"])
        if states["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["cuda"])
        if set(states["generators"]) != set(generators):
            raise ValueError("Calibration generator inventory changed")
        for name, value in generators.items():
            value.set_state(states["generators"][name])


def shared_groups(model):
    """Predetermined subset: full last language block and the learned W queries.

    This is not the whole LLM gradient. Layer number is resolved from the real
    module layout, then all exact names/shapes are committed in the result.
    """
    language = model.base.qwen_vl_interface.model.model.language_model
    layers = language.layers
    if len(layers) < 1:
        raise ValueError("No real language blocks")
    names = {id(p): name for name, p in model.named_parameters()}
    result = {"last_language_block": [names[id(p)] for p in layers[-1].parameters()],
              "world_queries": [names[id(model.base.foresight_queries)]]}
    return result


def validate_groups(model, groups):
    named = dict(model.named_parameters())
    if set(groups) != {"last_language_block", "world_queries"}:
        raise ValueError("Fixed language/W groups required")
    chosen = list(itertools.chain.from_iterable(groups.values()))
    if not chosen or any(not names for names in groups.values()) or len(set(chosen)) != len(chosen):
        raise ValueError("Empty or overlapping shared parameter groups")
    if any(name not in named or not named[name].requires_grad for name in chosen):
        raise ValueError("Unknown or frozen shared parameters")
    if len({id(named[name]) for name in chosen}) != len(chosen):
        raise ValueError("Shared parameter alias")
    return chosen, [named[name] for name in chosen]


def _dot(left, right):
    total = 0.
    for x, y in zip(left, right):
        # FP64 reporting reductions only. Forward/backward/accumulation dtype
        # stays native; no giant flattened gradient or FP64 model is created.
        for a, b in zip(x.reshape(-1).split(262144), y.reshape(-1).split(262144)):
            total += float((a.double() * b.double()).sum().item())
    return total


def gradient_summary(gradients, names, groups, missing):
    summary = {}
    for group, members in {**groups, "declared_shared_union": names}.items():
        positions = [names.index(name) for name in members]
        vectors = {task: [gradients[task][i] for i in positions] for task in OBJECTIVES}
        norms = {task: math.sqrt(max(_dot(value, value), 0.)) for task, value in vectors.items()}
        cosines = {}
        for a, b in itertools.combinations(OBJECTIVES, 2):
            divisor = norms[a] * norms[b]
            cosines[a + ":" + b] = None if divisor == 0 else _dot(vectors[a], vectors[b]) / divisor
        summary[group] = {"norms": norms, "cosines": cosines,
            "missing_gradient_parameters": {task: [name for name in members if name in missing[task]]
                                            for task in OBJECTIVES}}
    return summary


def propose_weights(losses, summary, counts):
    """An immutable proposal, never mutates config or chooses on dev results.

    Each new term is bounded by current in both scalar contribution and the
    norm on this fixed shared subset. No clipping, epsilon or fallback hides
    zero signal. This does NOT guarantee a bound on the summed task gradient.
    """
    norms = summary["declared_shared_union"]["norms"]
    reasons = []
    for key in ("current", "future_a", "future_b", "delta", "ego_scenes"):
        if type(counts.get(key)) is not int or counts[key] <= 0:
            reasons.append("missing_positive_count:" + key)
    for key in ("current", "future", "delta"):
        if not math.isfinite(losses[key]) or losses[key] <= 0:
            reasons.append("nonpositive_or_nonfinite_loss:" + key)
        if not math.isfinite(norms[key]) or norms[key] <= 0:
            reasons.append("nonpositive_or_nonfinite_shared_gradient:" + key)
        for group in ("last_language_block", "world_queries"):
            if summary[group]["norms"][key] <= 0:
                reasons.append("zero_gradient_group:" + group + ":" + key)
    if reasons:
        return {"status": "UNCALIBRATABLE", "rule": RULE, "reasons": reasons,
                "weights": None, "applied_to_training": False}
    ratios = {task: {"loss_ratio": losses["current"] / losses[task],
                     "gradient_ratio": norms["current"] / norms[task]}
              for task in ("future", "delta")}
    weights = {"current": 1., **{task: min(value.values()) for task, value in ratios.items()}}
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise FloatingPointError("Invalid coefficient proposal")
    return {"status": "PROPOSAL_ONLY", "rule": RULE, "ratios": ratios, "weights": weights,
            "applied_to_training": False, "ego_fm_weight_unchanged": 1.,
            "statistical_optimality_claim": False}


def measure(model, forwards, *, groups, counts, generators, max_microbatches):
    """Sum original full-batch-normalized microbatch contributions once.

    Each forward returns all four differentiable scalar losses on one graph.
    Every task therefore sees exactly the same dropout/noise/patch/time draw.
    Per-task gradients accumulate in parameter dtype (BF16 for official
    execution), before FP64 reporting. No optimizer and no .backward() calls.
    Single-process measurement only: no claim of ZeRO2 communication parity.
    """
    if torch.distributed.is_initialized():
        raise ValueError("This no-update diagnostic is single process; no implicit distributed averaging")
    if type(max_microbatches) is not int or not 1 <= max_microbatches <= 64:
        raise ValueError("Explicit <=64 microbatch limit required")
    names, parameters = validate_groups(model, groups)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Calibration requires a separate clean model copy; existing gradients must not be cleared")
    initial = model_identity(model)
    modes = {name: module.training for name, module in model.named_modules()}
    gradients = {task: [torch.zeros_like(p) for p in parameters] for task in OBJECTIVES}
    missing = {task: set(names) for task in OBJECTIVES}
    losses = dict.fromkeys(OBJECTIVES, 0.)
    used = 0
    with preserved_rng(generators):
        try:
            for forward in forwards:
                if used == max_microbatches:
                    raise ValueError("Microbatch budget exceeded")
                output = forward()
                terms = output["losses"]
                if set(terms) != set(OBJECTIVES):
                    raise ValueError("Need actual unweighted ego/current/future/delta losses")
                for i, task in enumerate(OBJECTIVES):
                    loss = terms[task]
                    if (not isinstance(loss, torch.Tensor) or loss.ndim != 0 or
                            not loss.requires_grad or not torch.isfinite(loss)):
                        raise ValueError("Missing/nonfinite differentiable objective: " + task)
                    values = torch.autograd.grad(loss, parameters, retain_graph=i < len(OBJECTIVES) - 1,
                                                 allow_unused=True, create_graph=False)
                    for at, value in enumerate(values):
                        if value is not None:
                            if not torch.isfinite(value).all():
                                raise FloatingPointError("Nonfinite shared gradient: " + task + ":" + names[at])
                            gradients[task][at].add_(value.detach())
                            missing[task].discard(names[at])
                    losses[task] += float(loss.detach().float().item())
                del output, terms, values
                used += 1
            if used == 0:
                raise ValueError("Empty fixed training batch")
            summary = gradient_summary(gradients, names, groups, missing)
        finally:
            if model_identity(model) != initial or modes != {n: m.training for n, m in model.named_modules()}:
                raise RuntimeError("Calibration changed model parameters/buffers/modes")
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Calibration populated model .grad")
    return {"schema": "st_loss_gradient_measurement_v1", "losses": losses, "counts": counts,
            "microbatches": used, "gradient_groups": summary,
            "parameter_subset": {group: [{"name": name, "shape": list(dict(model.named_parameters())[name].shape),
                "dtype": str(dict(model.named_parameters())[name].dtype), "initial_sha256": initial[name]["sha256"]}
                for name in members] for group, members in groups.items()},
            "shared_subset_not_whole_LLM": True, "model_state_before_after_equal": True,
            "RNG_restored": True, "optimizer_steps": 0,
            "proposal": propose_weights(losses, summary, counts)}
