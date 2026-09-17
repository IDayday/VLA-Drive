"""Real torchrun failure injection of production control and checkpoint phases."""
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import json
import os
import sys
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.distributed import (
    initialize_control_group,
    rank0_call,
    synchronized_call,
)
from starVLA.rl.flow_grpo.checkpoint import save_boundary


class Processor:
    def save_pretrained(self, path):
        Path(path).mkdir()


class Accelerator:
    device = torch.device("cpu")
    num_processes = 2

    def __init__(self, rank):
        self.process_index = rank
        self.is_main_process = rank == 0

    def unwrap_model(self, model):
        return model

    def save_state(self, path, **kwargs):
        # Represents a backend phase with its own collectives; not wrapped.
        dist.barrier()
        torch.save(
            {"rank": self.process_index}, Path(path) / f"backend{self.process_index}.pt"
        )


def main():
    case, out = sys.argv[1], Path(sys.argv[2])
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=8))
    initialize_control_group(8)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if case == "output_conflict":
            from starVLA.rl.flow_grpo.trainer import initialize_run_directory

            if rank == 0:
                (out / "training.jsonl").write_text("occupied")
            rank0_call(lambda: initialize_run_directory(out, {}, None), "cpu")
        elif case in {"checkpoint_mkdir", "rank0_write"}:
            if case == "checkpoint_mkdir" and rank == 0:
                (out / "checkpoints").write_text("a file prevents directory creation")
            policy = torch.nn.Linear(1, 1)
            policy.qwen_vl_interface = SimpleNamespace(processor=Processor())
            cfg = {
                "runtime": {"output_dir": str(out)},
                "algorithm": {"logprob_reduction": "flow_grpo_dimension_mean"},
            }
            if case == "rank0_write" and rank == 0:
                original = Path.write_text

                def injected(path, *args, **kwargs):
                    if path.name == "trainer_state.json":
                        raise OSError("injected rank0 metadata write failure")
                    return original(path, *args, **kwargs)

                Path.write_text = injected
            save_boundary(
                Accelerator(rank),
                SimpleNamespace(policy=policy),
                cfg,
                OmegaConf.create({"datasets": {"vla_data": {"act_norm": 1}}}),
                {},
                {},
                1,
                1,
                [],
            )
        elif case == "reward_failure":

            def scoring():
                if rank == 1:
                    raise ValueError("injected rank1 official scoring error")

            synchronized_call(scoring, "cpu")
        elif case == "rank_exit":
            if rank == 1:
                (out / "rank1.json").write_text(
                    json.dumps({"status": "FAIL", "error": "injected process exit 23"})
                )
                os._exit(23)
            dist.barrier()
        else:
            raise ValueError(case)
    except BaseException as exc:
        (out / f"rank{rank}.json").write_text(
            json.dumps({"status": "FAIL", "error": repr(exc)})
        )
        raise
    raise AssertionError("fault injection unexpectedly succeeded")


if __name__ == "__main__":
    main()
