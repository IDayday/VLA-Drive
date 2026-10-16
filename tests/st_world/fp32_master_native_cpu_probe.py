"""Actual installed DS0.16.9 CPU/Gloo engine test, not a CUDA qualification."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import deepspeed

from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups
from tools.st_world_policy.fp32_master_init import (
    capture_fp32_master_initialization, restore_fp32_master_initialization)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = torch.nn.Linear(3, 5)
        self.new_readout = torch.nn.Linear(5, 3)
        self.frozen = torch.nn.Parameter(torch.tensor([.27182818]), requires_grad=False)

    def forward(self, x):
        return self.new_readout(self.parent(x).tanh())


def worker(rank, output):
    output = Path(output)
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="2", LOCAL_SIZE="2")
    dist.init_process_group("gloo", init_method="file://" + str(output / "native_rendezvous"),
                            rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(411)
        model = Tiny()
        groups, _ = bounded_parameter_groups(model.named_parameters(), maximum=25)
        snapshot = capture_fp32_master_initialization(model, groups, is_fresh_run=True, chunk_elements=3)
        expected = [g.values.clone() for g in snapshot.groups]
        config = {"train_batch_size": 4, "train_micro_batch_size_per_gpu": 2,
                  "gradient_accumulation_steps": 1, "bf16": {"enabled": True},
                  "zero_allow_untested_optimizer": True,
                  "zero_optimization": {"stage": 2, "overlap_comm": False, "contiguous_gradients": True,
                      "reduce_bucket_size": 100, "allgather_bucket_size": 100}, "steps_per_print": 999999}
        adam = torch.optim.AdamW(groups, lr=1e-5)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=groups, optimizer=adam, config=config)
        before = [p.detach().clone() for p in engine.optimizer.single_partition_of_fp32_groups]
        execution_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        audit = restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
        assert torch.equal(rng_before, torch.get_rng_state())
        for actual, source in zip(engine.optimizer.single_partition_of_fp32_groups, expected):
            assert torch.equal(actual.view(torch.int32), source.view(torch.int32))
        assert any(not torch.equal(p, q) for p, q in zip(before, expected))
        assert audit["ranks"][1]["local_padding_elements"] == 2
        for name, value in model.state_dict().items():
            assert torch.equal(value.view(torch.uint8), execution_before[name].view(torch.uint8))
        # One real engine backward/update, then native checkpoint and reload.
        inputs = torch.tensor([[.2, -.3, .1], [.5, .4, -.2]], dtype=torch.bfloat16)
        loss = engine(inputs).float().square().mean()
        engine.backward(loss)
        engine.step()
        assert engine.global_steps == 1
        engine.save_checkpoint(str(output / "native_checkpoints"), tag="after_one", client_state={"test": True})
        saved = [p.detach().clone() for p in engine.optimizer.single_partition_of_fp32_groups]
        moments = [{k: v.detach().clone() if torch.is_tensor(v) else v
                    for k, v in engine.optimizer.optimizer.state[p].items()}
                   for p in engine.optimizer.single_partition_of_fp32_groups]
        second = Tiny()
        groups2, _ = bounded_parameter_groups(second.named_parameters(), maximum=25)
        assert capture_fp32_master_initialization(second, groups2, is_fresh_run=False) is None
        resumed, _, _, _ = deepspeed.initialize(model=second, model_parameters=groups2,
            optimizer=torch.optim.AdamW(groups2, lr=1e-5), config=config)
        skip = restore_fp32_master_initialization(resumed, None, is_fresh_run=False)
        loaded, state = resumed.load_checkpoint(str(output / "native_checkpoints"), tag="after_one",
            load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
        assert loaded and state["test"] and resumed.global_steps == 1
        for actual, wanted, moment in zip(resumed.optimizer.single_partition_of_fp32_groups, saved, moments):
            assert torch.equal(actual.view(torch.int32), wanted.view(torch.int32))
            for key, value in moment.items():
                current = resumed.optimizer.optimizer.state[actual][key]
                assert torch.equal(current, value) if torch.is_tensor(value) else current == value
        (output / f"native_rank{rank}.json").write_text(json.dumps({
            "status": "PASS", "scope": "actual_DeepSpeed_CPU_Gloo_two_ranks",
            "fresh_audit": audit, "resume_audit": skip, "real_optimizer_steps": 1,
            "native_checkpoint_reload_exact": True, "loss": float(loss), "CUDA_tested": False}, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(sys.argv[1],), nprocs=2, join=True)
