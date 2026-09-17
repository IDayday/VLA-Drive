"""Actual CUDA/ZeRO-2 precision regression; a small optimizer test, not DDP policy acceptance."""
import argparse
import json
import os
from pathlib import Path
import torch
from torch import nn
from starVLA.rl.flow_grpo.trainer import make_accelerator
from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncorrected", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = {
        "runtime": {"accumulation_steps": 4, "deepspeed_stage": 2, "optimizer_offload": False},
        "optimizer": {"max_grad_norm": 0.0},
    }
    accelerator = make_accelerator(cfg)
    model = nn.Linear(8, 1, bias=False)
    model.weight.data.fill_(0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0)
    engine, optimizer = accelerator.prepare(model, optimizer)
    if not args.uncorrected:
        install_fp32_partitions(engine)
    world, rank = accelerator.num_processes, accelerator.process_index
    # All contributions are BF16-exact. Sequential BF16 addition loses the three
    # small terms after 1.0, whereas FP32 retains them. Rank averaging is exact.
    values = [1.0, 1 / 512, 1 / 512, 1 / 512]
    expected = sum(values) / 4 * (world + 1) / 2
    rows = []
    for micro, value in enumerate(values):
        x = torch.full((1, 8), value * (rank + 1), device=accelerator.device, dtype=torch.bfloat16)
        engine.backward(engine(x).sum())
        buffers = engine.optimizer.averaged_gradients[0]
        if micro == 3:
            actual = torch.cat([t.flatten() for t in buffers]).float().cpu()
        rows.append({"microbatch": micro, "dtypes": [str(t.dtype) for t in buffers]})
        engine.step()
    base = engine.optimizer.optimizer
    moment = next(iter(base.state.values()))["exp_avg"].cpu()
    grad_pass = bool(torch.allclose(actual, torch.full_like(actual, expected), atol=0, rtol=0))
    moment_pass = bool(torch.allclose(moment, torch.full_like(moment, expected * 0.1), atol=1e-8, rtol=0))
    result = {
        "status": "PASS" if grad_pass and moment_pass else "FAIL",
        "scope": "small CUDA ZeRO-2 optimizer precision regression; not full policy evidence",
        "rank": rank, "world_size": world, "uncorrected": args.uncorrected,
        "expected_gradient": expected, "actual_gradient": actual.tolist(),
        "actual_first_moment": moment.tolist(), "microbatches": rows,
        "gradient_exact": grad_pass, "adam_moment_match": moment_pass,
        "deepspeed_step": engine.global_steps,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    if not args.uncorrected:
        assert result["status"] == "PASS", result
    else:
        assert result["status"] == "FAIL", "installed defect no longer reproduced; re-audit compatibility correction"
    accelerator.end_training()


if __name__ == "__main__":
    main()
