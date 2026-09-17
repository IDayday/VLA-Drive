"""Compare complete real training boundaries, including ZeRO shards and RNG.

Use after identical continuous and interrupted/resumed diagnostic jobs. Checks
all stored tensors exactly; records every difference instead of first failure.
It never regards file existence or a successful backward as resume evidence.
"""
from pathlib import Path
import argparse
import dataclasses
import json
import numpy as np
import torch
from starVLA.rl.flow_grpo.comparison import tensor_comparison
from starVLA.rl.flow_grpo.config import config_hash


def flatten(value, prefix=""):
    if dataclasses.is_dataclass(value):
        yield prefix + "/@type", type(value).__qualname__
        for field in dataclasses.fields(value):
            yield from flatten(getattr(value, field.name), prefix + "/" + field.name)
    elif isinstance(value, dict):
        yield prefix + "/@type", "dict"
        for key in sorted(value, key=str):
            yield from flatten(value[key], prefix + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        yield prefix + "/@type", type(value).__name__
        for index, child in enumerate(value):
            yield from flatten(child, prefix + "/" + str(index))
    elif isinstance(value, np.ndarray):
        yield prefix, torch.from_numpy(value.copy())
    else:
        yield prefix, value


def compare_values(left, right):
    result = {}
    a, b = dict(flatten(left)), dict(flatten(right))
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            result[key] = {"allclose": False, "reason": "missing key"}
        elif isinstance(a[key], torch.Tensor) and isinstance(b[key], torch.Tensor):
            row = tensor_comparison(a[key], b[key], atol=0.0, rtol=0.0)
            row["dtype_equal"] = a[key].dtype == b[key].dtype
            row["allclose"] = row["dtype_equal"] and torch.equal(a[key], b[key])
            if a[key].shape == b[key].shape:
                row["nonidentical"] = int((a[key] != b[key]).sum())
            result[key] = row
        else:
            equal = type(a[key]) is type(b[key]) and a[key] == b[key]
            result[key] = {"allclose": bool(equal)}
            if not equal:
                result[key].update(reference=repr(a[key]), actual=repr(b[key]))
    return result


def compare_boundaries(left, right, output):
    left, right, out = Path(left), Path(right), Path(output)
    if out.exists():
        raise FileExistsError(out)
    if not all((p / "COMPLETE").is_file() for p in (left, right)):
        raise ValueError("only complete boundaries may be compared")
    for key in (
        "update",
        "policy_version",
        "boundary",
        "next_inner_epoch",
        "world_size",
        "config_hash",
        "provenance",
    ):
        a, b = [
            json.loads((p / "trainer_state.json").read_text()).get(key)
            for p in (left, right)
        ]
        if a != b:
            raise ValueError("boundary identity mismatch: " + key)
    a, b = [json.loads((p / "rl_config.json").read_text()) for p in (left, right)]
    if config_hash(a) != config_hash(b):
        raise ValueError("configuration differs")
    files = [
        {
            str(x.relative_to(p))
            for x in p.rglob("*")
            if x.suffix in (".pt", ".bin", ".pkl")
        }
        for p in (left, right)
    ]
    if files[0] != files[1] or not files[0]:
        raise ValueError("state shard inventory differs or is empty")
    report = {
        "scope": "all stored model/optimizer/scheduler/rank RNG/pending tensors, exact zero tolerance",
        "files": {},
        "status": "PASS",
    }
    for name in sorted(files[0]):
        # Checkpoints are trusted local artifacts; NumPy RNG/pending dataclasses
        # need full pickle loading. Never use this utility on untrusted files.
        states = [
            torch.load(p / name, map_location="cpu", weights_only=False)
            for p in (left, right)
        ]
        rows = compare_values(*states)
        passed = all(x["allclose"] for x in rows.values())
        report["files"][name] = {"status": "PASS" if passed else "FAIL", "values": rows}
        if not passed:
            report["status"] = "FAIL"
        del states, rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise AssertionError("resume mismatch; exhaustive report retained")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_boundaries(args.continuous, args.resumed, args.output)
    print(json.dumps({"status": report["status"], "files": len(report["files"])}))


if __name__ == "__main__":
    main()
