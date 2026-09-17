"""Exhaustive tensor diagnostics; historical allclose is reported, not relaxed."""
from collections import defaultdict
import math
import torch


def tensor_comparison(reference, actual, atol=2e-6, rtol=2e-3):
    if reference is None or actual is None:
        return {
            "reference_present": reference is not None,
            "actual_present": actual is not None,
            "allclose": reference is None and actual is None,
        }
    if reference.shape != actual.shape:
        return {
            "allclose": False,
            "shape_mismatch": [list(reference.shape), list(actual.shape)],
        }
    a, b = (
        reference.detach().cpu().double().flatten(),
        actual.detach().cpu().double().flatten(),
    )
    diff = (b - a).abs()
    an, bn, dn = float(a.norm()), float(b.norm()), float(diff.norm())
    # Near-zero classification is independent of the acceptance tolerance. In
    # particular a 1e-6 Adam update is NOT "zero" just because atol is 2e-6.
    near_zero = an <= 1e-12 * math.sqrt(a.numel())
    bad = diff > atol + rtol * a.abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "reference_present": True,
        "actual_present": True,
        "numel": a.numel(),
        "max_abs": float(diff.max()) if a.numel() else 0.0,
        "reference_norm": an,
        "actual_norm": bn,
        "difference_norm": dn,
        "near_zero_rms_threshold": 1e-12,
        "near_zero_reference": near_zero,
        "relative_l2": None if near_zero else dn / an,
        "cosine": None if near_zero or not bn else float(torch.dot(a, b) / (an * bn)),
        "norm_ratio": None if near_zero else bn / an,
        "nonidentical": int((a != b).sum()),
        "outside_tolerance": int(bad.sum()),
        "error_histogram": {
            str(limit): int((diff <= limit).sum())
            for limit in (0.0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4, 1e-3)
        },
        "first_bad_flat_indices": bad.nonzero().flatten()[:16].tolist(),
        "finite": finite,
        "allclose": finite and not bool(bad.any()),
        "atol": atol,
        "rtol": rtol,
    }


def module_group(name):
    if "qwen_proj" in name:
        return "projector"
    if name.startswith("qwen_vl_interface"):
        return "language"
    if name.startswith("action_input_model"):
        return "history"
    if name.startswith("action_model"):
        return "action"
    return name.split(".")[0]


def compare_named(reference, actual, atol=2e-6, rtol=2e-3):
    entries = {
        name: tensor_comparison(reference.get(name), actual.get(name), atol, rtol)
        for name in sorted(reference.keys() | actual.keys())
    }
    groups = defaultdict(list)
    for name, entry in entries.items():
        groups[module_group(name)].append(entry)
        groups["whole_model"].append(entry)
        if "/" in name:
            groups["optimizer_state/" + name.rsplit("/", 1)[-1]].append(entry)
    summaries = {}
    for name, rows in groups.items():
        a2 = sum(x.get("reference_norm", 0.0) ** 2 for x in rows)
        b2 = sum(x.get("actual_norm", 0.0) ** 2 for x in rows)
        d2 = sum(x.get("difference_norm", 0.0) ** 2 for x in rows)
        near_zero = a2 <= 1e-24 * sum(x.get("numel", 0) for x in rows)
        summaries[name] = {
            "tensors": len(rows),
            "failed_tensors": sum(not x["allclose"] for x in rows),
            "max_abs": max(x.get("max_abs", 0.0) for x in rows),
            "relative_l2": math.sqrt(d2 / a2) if a2 and not near_zero else None,
            "norm_ratio": math.sqrt(b2 / a2) if a2 and not near_zero else None,
            "cosine": (a2 + b2 - d2) / (2 * math.sqrt(a2 * b2))
            if a2 and b2 and not near_zero
            else None,
            "outside_tolerance": sum(x.get("outside_tolerance", 0) for x in rows),
            "near_zero_tensors": sum(x.get("near_zero_reference", False) for x in rows),
        }
    return {
        "status": "PASS" if all(x["allclose"] for x in entries.values()) else "FAIL",
        "parameters": entries,
        "modules": summaries,
    }
