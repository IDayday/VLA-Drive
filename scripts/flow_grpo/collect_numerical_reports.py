"""Publish exhaustive diagnostics without copying large model/optimizer tensors.

Original numerical reports remain byte-identical. Near-zero presentation is
recomputed from saved norms independently of the existing allclose tolerance.
No pass/fail classification or tolerance is changed.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import shutil
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.comparison import module_group


def restat(comparison):
    groups = defaultdict(list)
    for name, entry in comparison["parameters"].items():
        if "reference_norm" in entry:
            an, bn, dn = (
                entry["reference_norm"],
                entry["actual_norm"],
                entry["difference_norm"],
            )
            near = an <= 1e-12 * math.sqrt(entry["numel"])
            entry.update(
                near_zero_reference=near,
                near_zero_rms_threshold=1e-12,
                relative_l2=dn / an if not near else None,
                norm_ratio=bn / an if not near else None,
                cosine=(an * an + bn * bn - dn * dn) / (2 * an * bn)
                if not near and bn
                else None,
                cosine_computation="saved norm identity",
            )
        groups["whole_model"].append(entry)
        groups[module_group(name)].append(entry)
        if "/" in name:
            groups["optimizer_state/" + name.rsplit("/", 1)[-1]].append(entry)
    modules = {}
    for name, entries in groups.items():
        a2 = sum(e.get("reference_norm", 0.0) ** 2 for e in entries)
        b2 = sum(e.get("actual_norm", 0.0) ** 2 for e in entries)
        d2 = sum(e.get("difference_norm", 0.0) ** 2 for e in entries)
        near = a2 <= 1e-24 * sum(e.get("numel", 0) for e in entries)
        modules[name] = dict(
            tensors=len(entries),
            failed_tensors=sum(not e["allclose"] for e in entries),
            max_abs=max(e.get("max_abs", 0.0) for e in entries),
            relative_l2=math.sqrt(d2 / a2) if not near else None,
            norm_ratio=math.sqrt(b2 / a2) if not near else None,
            cosine=(a2 + b2 - d2) / (2 * math.sqrt(a2 * b2))
            if not near and b2
            else None,
            outside_tolerance=sum(e.get("outside_tolerance", 0) for e in entries),
            nonidentical=sum(e.get("nonidentical", 0) for e in entries),
            near_zero_tensors=sum(e.get("near_zero_reference", False) for e in entries),
        )
    comparison["modules"] = modules


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    original = source / "numerical_report.json"
    shutil.copy2(original, out / "original_numerical_report.json")
    report = json.loads(original.read_text())
    summary = {}
    for group, result in report["groups"].items():
        summary[group] = {}
        for kind, comparison in result["comparisons"].items():
            restat(comparison)
            summary[group][kind] = {
                "status": comparison["status"],
                "modules": comparison["modules"],
            }
    report[
        "presentation_note"
    ] = "Near-zero RMS <=1e-12, independent of historical atol/rtol; recomputed from saved norms. All allclose outcomes unchanged."
    (out / "numerical_report.json").write_text(json.dumps(report, indent=2))
    (out / "summary.json").write_text(
        json.dumps(
            {"groups": summary, **{k: v for k, v in report.items() if k != "groups"}},
            indent=2,
        )
    )
    for path in source.glob("source_environment*.json"):
        shutil.copy2(path, out / path.name)
    if (source / "executed_script.py").is_file():
        shutil.copy2(
            source / "executed_script.py", out / "executed_script_snapshot.txt"
        )
    (out / "raw_artifacts.json").write_text(
        json.dumps(
            {
                "root": str(source.resolve()),
                "original_report_sha256": file_sha(original),
                "executed_script_snapshot": "CAPTURED"
                if (source / "executed_script.py").is_file()
                else "NOT_CAPTURED; first exploratory run, cannot certify exact executable",
                "files": [
                    {"path": str(x.resolve()), "size": x.stat().st_size}
                    for x in sorted(source.glob("*.pt"))
                ],
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                g: {
                    k: {"status": v["status"], **v["modules"]["whole_model"]}
                    for k, v in items.items()
                }
                for g, items in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
