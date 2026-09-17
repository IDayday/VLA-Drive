"""Correct an old report that compared LossScaler object identity instead of state.

All tensor comparison results remain unchanged. The original report is retained.
This is not a numerical tolerance adjustment or an exclusion of optimizer state.
"""
import argparse
import json
from pathlib import Path
import torch
from verify_artifacts import load, walk


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.report.read_text())
    original = args.report.with_name(
        args.report.stem + "_object_identity_original.json"
    )
    if original.exists():
        raise FileExistsError(original)
    original.write_text(args.report.read_text())
    corrected = []
    for name, result in report["files"].items():
        differences = result["differences"]
        match = [
            d
            for d in differences
            if d.get("path") == "/optimizer_state_dict/loss_scaler"
            and d.get("reason") == "value"
        ]
        if not match:
            continue
        left = load(Path(report["left"]) / name)["optimizer_state_dict"]["loss_scaler"]
        right = load(Path(report["right"]) / name)["optimizer_state_dict"][
            "loss_scaler"
        ]
        if (
            type(left) is not type(right)
            or type(left).__module__ != "deepspeed.runtime.fp16.loss_scaler"
        ):
            raise TypeError("unexpected object in historical comparison")
        checked, _ = walk(left, right, path="/optimizer_state_dict/loss_scaler")
        result["differences"] = [d for d in differences if d not in match] + checked
        result["bitwise_equal"] = not result["differences"]
        corrected.append(
            dict(
                file=name,
                type=type(left).__qualname__,
                state_fields=list(vars(left)),
                state_equal=not checked,
            )
        )
    report["comparison_correction"] = dict(
        original_report=str(original),
        checked=corrected,
        reason="compare DeepSpeed LossScaler serialized fields instead of Python object identity; tensor tests unchanged",
    )
    report["bitwise_equal"] = all(v["bitwise_equal"] for v in report["files"].values())
    args.report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"bitwise_equal": report["bitwise_equal"], "objects_rechecked": corrected},
            indent=2,
        )
    )
    if not report["bitwise_equal"]:
        raise AssertionError("actual state differences remain")


if __name__ == "__main__":
    main()
