"""Compare real saved training states or audit updates against the original SFT.

Only trusted, locally generated optimizer/RNG files are read with pickle enabled.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=False)


def walk(left, right, path="", differences=None, stats=None):
    differences = [] if differences is None else differences
    stats = defaultdict(int) if stats is None else stats
    if isinstance(left, torch.Tensor):
        stats["tensor_count"] += 1
        stats["numel"] += left.numel()
        if (
            not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            differences.append(dict(path=path, reason="shape/dtype/type"))
        elif not torch.equal(left, right):
            delta = 0.0
            changed = 0
            for a, b in zip(
                left.reshape(-1).split(1_000_000), right.reshape(-1).split(1_000_000)
            ):
                delta = max(delta, float((a.double() - b.double()).abs().max()))
                changed += int((a != b).sum())
            differences.append(dict(path=path, max_abs=delta, changed_elements=changed))
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            differences.append(dict(path=path, reason="numpy mismatch"))
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            differences.append(dict(path=path, reason="dictionary keys"))
        for key in left.keys() & right.keys():
            walk(left[key], right[key], f"{path}/{key}", differences, stats)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            differences.append(dict(path=path, reason="length"))
        for i, (a, b) in enumerate(zip(left, right)):
            walk(a, b, f"{path}/{i}", differences, stats)
    elif hasattr(left, "__dict__"):
        if type(left) is not type(right):
            differences.append(dict(path=path, reason="object type"))
        else:
            walk(vars(left), vars(right), path + "/__dict__", differences, stats)
    elif left != right:
        differences.append(
            dict(path=path, reason="value", left=str(left), right=str(right))
        )
    return differences, dict(stats)


def compare(left, right, output=None):
    results = {}
    files = sorted(
        p.relative_to(left)
        for p in left.rglob("*")
        if p.is_file() and p.suffix in {".pt", ".bin", ".pkl"}
    )
    for relative in files:
        a, b = load(left / relative), load(right / relative)
        differences, stats = walk(a, b)
        results[str(relative)] = dict(
            bitwise_equal=not differences, differences=differences, **stats
        )
        print(
            relative,
            results[str(relative)]["bitwise_equal"],
            len(differences),
            flush=True,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_suffix(".partial.json").write_text(
                json.dumps(results, indent=2)
            )
        del a, b
    return dict(
        kind="continuous_vs_resume",
        left=str(left),
        right=str(right),
        files=results,
        bitwise_equal=all(r["bitwise_equal"] for r in results.values()),
    )


def audit(checkpoint, baseline):
    manifest = json.loads((checkpoint / "sft_parameter_manifest.json").read_text())
    current = load(next(checkpoint.glob("*/mp_rank_00_model_states.pt")))["module"]
    original = load(baseline)
    modules = defaultdict(
        lambda: dict(
            parameters=0,
            changed_tensors=0,
            changed_elements=0,
            squared_delta=0.0,
            max_abs=0.0,
        )
    )
    frozen_changes, dormant_changes = [], []
    for entry in manifest["parameters"]:
        name = entry["name"]
        a = original[name]
        b = current["policy." + name]
        changed, squared, maximum = 0, 0.0, 0.0
        for ac, bc in zip(
            a.reshape(-1).split(1_000_000), b.reshape(-1).split(1_000_000)
        ):
            ac = ac.to(bc.dtype)
            changed += int((ac != bc).sum())
            delta = ac.float() - bc.float()
            squared += float(delta.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        root = name.split(".")[0]
        result = modules[root]
        result["parameters"] += 1
        result["changed_tensors"] += int(changed > 0)
        result["changed_elements"] += changed
        result["squared_delta"] += squared
        result["max_abs"] = max(result["max_abs"], maximum)
        if changed and not entry["requires_grad"]:
            frozen_changes.append(name)
        if changed and entry["category"] == "sft_dormant":
            dormant_changes.append(name)
    return dict(
        kind="full_parameter_update_audit",
        checkpoint=str(checkpoint),
        baseline=str(baseline),
        comparison="SFT values converted to runtime BF16 before comparison",
        frozen_unchanged=not frozen_changes,
        frozen_changes=frozen_changes,
        dormant_changes=dormant_changes,
        modules=dict(modules),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["compare", "audit"])
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = (
        compare(args.left, args.right, args.output)
        if args.command == "compare"
        else audit(args.left, args.right)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if result.get("frozen_unchanged") is False:
        raise AssertionError("frozen parameters changed")
    if result.get("bitwise_equal") is False:
        raise AssertionError(
            "continuous/resume states are not bitwise equal; see detailed report"
        )


if __name__ == "__main__":
    main()
