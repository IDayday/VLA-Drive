"""Compare real ZeRO-2 Adam moments across rank/accumulation layouts.

Run one update with the same effective global scene batch, scene-key RNG, no
CPU optimizer offload, and max_grad_norm=1e6 so clipping cannot hide a factor
error. Production bounded training still uses its original clipping setting.
The 1% relative L2 gate tolerates BF16 reduction order, not factor-of-two errors.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch


def state(path):
    model_file = next(path.glob("*/mp_rank_00_model_states.pt"))
    metadata = torch.load(model_file, weights_only=False, map_location="cpu", mmap=True)
    paths = sorted(path.glob("*/bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    optimizers = [
        torch.load(p, weights_only=False, map_location="cpu", mmap=False)[
            "optimizer_state_dict"
        ]
        for p in paths
    ]
    assert all(o["zero_stage"] == 2 for o in optimizers)
    assert all(max(o["partition_count"]) == len(paths) for o in optimizers)
    shapes = metadata["param_shapes"]
    result = {}
    for group, mapping in enumerate(shapes):
        required = sum(math.prod(shape) for shape in mapping.values())
        for field in ["exp_avg", "exp_avg_sq"]:
            pieces = []
            for optimizer in optimizers:
                base = optimizer["base_optimizer_state"]
                ids = base["param_groups"][group]["params"]
                assert len(ids) == 1
                item = base["state"][ids[0]]
                assert (
                    int(item["step"]) == 1
                ), "scaling test requires exactly one optimizer update"
                pieces.append(item[field])
            merged = torch.cat(pieces)
            # Same ZeRO2 padding rule as the installed official zero_to_fp32.
            alignment = 2 * len(paths)
            assert math.ceil(required / alignment) == math.ceil(
                merged.numel() / alignment
            )
            offset = 0
            for name, shape in mapping.items():
                count = math.prod(shape)
                result[(name, field)] = merged[offset : offset + count]
                offset += count
    return result


def compare(left, right):
    a, b = state(left), state(right)
    assert a.keys() == b.keys()
    rows = defaultdict(
        lambda: {
            "reference_squared_norm": 0.0,
            "difference_squared_norm": 0.0,
            "max_abs": 0.0,
            "numel": 0,
        }
    )
    for name, field in a:
        root = name.removeprefix("policy.").split(".")[0] + "/" + field
        row = rows[root]
        x, y = a[(name, field)], b[(name, field)]
        for xc, yc in zip(x.split(1_000_000), y.split(1_000_000)):
            assert torch.isfinite(xc).all() and torch.isfinite(yc).all()
            delta = xc.double() - yc.double()
            row["reference_squared_norm"] += float(xc.double().square().sum())
            row["difference_squared_norm"] += float(delta.square().sum())
            row["max_abs"] = max(row["max_abs"], float(delta.abs().max()))
            row["numel"] += xc.numel()
    for row in rows.values():
        row["relative_l2"] = math.sqrt(
            row["difference_squared_norm"] / max(row["reference_squared_norm"], 1e-300)
        )
        row["pass"] = row["relative_l2"] <= 0.01
    return dict(
        left=str(left),
        right=str(right),
        scope="actual named Adam first/second moments before adaptive division",
        relative_l2_tolerance=0.01,
        reason="BF16 accumulation/all-reduce order; clipping must be inactive",
        passed=all(row["pass"] for row in rows.values()),
        modules=dict(rows),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for path in [args.left, args.right]:
        cfg = json.loads((path / "rl_config.json").read_text())
        assert cfg["runtime"]["noise_seed_schedule"] == "global_scene_v1"
        assert cfg["optimizer"]["max_grad_norm"] == 1e6
        assert not cfg["runtime"]["optimizer_offload"]
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result["passed"], "real rank/accumulation scaling mismatch"


if __name__ == "__main__":
    main()
