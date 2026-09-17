"""Reconstruct the first actual ZeRO-2 AdamW update from saved pre-clip gradients.

Uses the installed CUDA AdamW as an independent unsharded oracle, one tensor at
a time. Tests every moment/master/forward weight, without storing another model.
The FP32 bounds are fixed before execution (8 eps on masters; 4e-6 relative on
moments), independent of the historical BF16 candidate-layout tolerance.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.loading import weight_path


def comparison(expected, actual, *, master):
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    error = (expected - actual).abs()
    bound = (8 * torch.finfo(torch.float32).eps if master else 4e-6) * expected.abs() + 1e-30
    return {
        "pass": finite and not bool((error > bound).any()),
        "finite": finite, "max_abs": float(error.max()),
        "outside_bound": int((error > bound).sum()),
        "relative_l2": float(error.double().norm() / expected.double().norm()) if expected.count_nonzero() else None,
    }


def run(run_root, output):
    root, output = Path(run_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((root / "rl_config.json").read_text())
    boundary = root / "checkpoints/update_000001"
    if not (boundary / "COMPLETE").is_file():
        raise ValueError("first optimizer boundary is incomplete")
    state = torch.load(next(boundary.glob("*/mp_rank_00_model_states.pt")), map_location="cpu", weights_only=False, mmap=True)
    shards = [torch.load(p, map_location="cpu", weights_only=False, mmap=True)["optimizer_state_dict"] for p in sorted(boundary.glob("*/bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))]
    if not shards or any(s["zero_stage"] != 2 for s in shards):
        raise ValueError("expected real ZeRO-2 optimizer partitions")
    source = torch.load(weight_path(cfg["sft_checkpoint"]), map_location="cpu", weights_only=False, mmap=True)
    source = source.get("module", source)
    folder = root / "optimizer_gradients/update_000001/rank_0"
    metadata = json.loads((folder / "manifest.json").read_text())
    gradients = {row["name"]: row for row in metadata["parameters"]}
    row = json.loads((root / "training_rank0.jsonl").read_text().splitlines()[0])
    norm = torch.tensor(row["pre_clip_grad_norm"], device="cuda", dtype=torch.float32)
    limit = cfg["optimizer"]["max_grad_norm"]
    scale = 1 / torch.clamp((norm + 1e-6) / limit, min=1) if limit > 0 else torch.ones_like(norm)
    report = {"status": "PASS", "scope": "all real first-step pre-clip gradients, Adam moments, FP32 masters, BF16 forward weights", "master_bound_eps": 8, "moment_relative_bound": 4e-6, "clip_scale": float(scale), "parameters": {}}
    for group, shapes in enumerate(state["param_shapes"]):
        bases = [s["base_optimizer_state"] for s in shards]
        options = bases[0]["param_groups"][group]
        ids = [b["param_groups"][group]["params"][0] for b in bases]
        states = [b["state"][i] for b, i in zip(bases, ids)]
        if any(int(s["step"]) != 1 for s in states):
            raise ValueError("requires exactly one actual Adam update")
        actuals = {
            **{key: torch.cat([s[key] for s in states]) for key in ("exp_avg", "exp_avg_sq")},
            "master": torch.cat([s["single_partition_of_fp32_groups"][group] for s in shards]),
        }
        offset = 0
        for name, shape in shapes.items():
            count, key = math.prod(shape), name.removeprefix("policy.")
            info = gradients[key]
            if info["dtype"] != "torch.float32" or not info.get("file"):
                raise ValueError("missing actual pre-clip FP32 optimizer gradient")
            gradient = torch.load(folder / info["file"], map_location="cuda", weights_only=True).flatten()
            initial = source[key].to(device="cuda", dtype=torch.bfloat16).float().flatten()
            parameter = torch.nn.Parameter(initial.clone())
            parameter.grad = gradient * scale
            optimizer = torch.optim.AdamW([parameter], lr=options["lr"], betas=options["betas"], eps=options["eps"], weight_decay=options["weight_decay"], foreach=True)
            optimizer.step()
            expected = {"master": parameter.detach(), **{k: optimizer.state[parameter][k] for k in ("exp_avg", "exp_avg_sq")}}
            entries = {k: comparison(expected[k], value[offset:offset + count].cuda(), master=k == "master") for k, value in actuals.items()}
            master = actuals["master"][offset:offset + count]
            forward = state["module"][name].flatten()
            entries["forward_equals_actual_master_cast"] = torch.equal(forward, master.to(dtype=forward.dtype))
            entries["master_changed_elements"] = int((master != initial.cpu()).sum())
            entries["forward_changed_elements"] = int((forward != source[key].to(dtype=forward.dtype).flatten()).sum())
            report["parameters"][key] = entries
            if not entries["forward_equals_actual_master_cast"] or not all(entries[k]["pass"] for k in actuals):
                report["status"] = "FAIL"
            offset += count
            del gradient, parameter, optimizer, expected, initial
        alignment = 2 * len(shards)
        assert math.ceil(offset / alignment) == math.ceil(actuals["master"].numel() / alignment)
        del actuals
    if set(report["parameters"]) != set(gradients):
        raise ValueError("optimizer oracle did not cover every trainable tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    assert report["status"] == "PASS", "optimizer oracle failed; exhaustive evidence retained"
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    report = run(args.run, args.output)
    print(report["status"], len(report["parameters"]), "parameter tensors")


if __name__ == "__main__":
    main()
