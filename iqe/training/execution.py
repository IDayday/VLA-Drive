"""Scheduling optimizations: no changes to model math, precision or sample order.

Only deterministic, frozen operations may run ahead. Checkpoints always save the
consumer cursor; queued work is disposable and regenerated after restoration.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
import torch
import torch.distributed as dist
from ..contracts import require


def collective_check(error=None, *, device="cpu"):
    """One scalar collective on success; gather error strings only on failure."""
    if dist.is_initialized():
        flag = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if not bool(flag):
            return
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    require(not any(e is not None for e in errors), f"distributed data failure: {errors}")


def finite_gradients(parameters):
    """Preserve elementwise isfinite checks with one device-to-host decision."""
    checks = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return not checks or bool(torch.stack(checks).all())


@dataclass(frozen=True)
class BatchAddress:
    epoch: int
    offset: int
    entries: tuple
    mine: tuple
    valid_count: int


class EpochPlan:
    def __init__(self, size, global_batch, seed, rank, world, make_batches):
        require(size > 0 and global_batch > 0 and world > 0 and 0 <= rank < world,
                "invalid epoch plan")
        self.size, self.global_batch, self.seed = size, global_batch, seed
        self.rank, self.world, self.make_batches = rank, world, make_batches
        self.epoch, self.batches = None, None

    def at(self, epoch, offset):
        if epoch != self.epoch:
            self.batches = self.make_batches(self.size, self.global_batch, self.seed, epoch)
            self.epoch = epoch
        require(0 <= offset <= len(self.batches), "invalid consumed batch offset")
        if offset == len(self.batches):
            return self.at(epoch + 1, 0)
        entries = tuple(self.batches[offset])
        mine = list(entries[self.rank::self.world])
        valid = len(mine)
        mine.extend([entries[0]] * (math.ceil(len(entries) / self.world) - valid))
        return BatchAddress(epoch, offset, entries, tuple(mine), valid)

    def iter_from(self, epoch, offset, count):
        for _ in range(count):
            address = self.at(epoch, offset)
            yield address
            epoch, offset = address.epoch, address.offset + 1


class OrderedPrefetch:
    """Bounded producer; exceptions are raised by the consumer on every rank.

    `prepare` must not mutate any training RNG, parameter, optimizer or sampler.
    Depth counts outstanding batches, independently of the consumed cursor.
    """
    def __init__(self, addresses, prepare, depth=1):
        require(depth >= 0, "negative prefetch depth")
        self.addresses, self.prepare, self.depth = iter(addresses), prepare, depth
        self.executor = ThreadPoolExecutor(max_workers=1) if depth else None
        self.queue = deque()
        self.closed = False
        self._fill()

    def _fill(self):
        while len(self.queue) < self.depth:
            address = next(self.addresses, None)
            if address is None:
                break
            self.queue.append((address, self.executor.submit(self.prepare, address)))

    def pop(self):
        require(not self.closed, "prefetch already closed")
        if not self.depth:
            address = next(self.addresses)
            return address, self.prepare(address)
        address, pending = self.queue.popleft()
        value = pending.result()
        self._fill()
        return address, value

    def close(self):
        if not self.closed:
            self.closed = True
            if self.executor:
                for _, future in self.queue:
                    future.cancel()
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.queue.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrainingPrefetch:
    """Independent bounded CPU and frozen-device producers.

    CPU loading for N+1 overlaps frozen vision for N, which in turn overlaps
    trainable work for N-1. No generated cursor is ever used for checkpointing.
    """
    def __init__(self, addresses, count, load, transform=None, depth=1):
        self.cpu = OrderedPrefetch(addresses, load, depth)
        self.transform = transform
        self.prepared = OrderedPrefetch(range(count), self._transform, 1 if depth else 0) if transform else None

    def _transform(self, unused):
        address, value = self.cpu.pop()
        return address, self.transform(value)

    def pop(self):
        return self.prepared.pop()[1] if self.prepared else self.cpu.pop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.prepared:
            self.prepared.close()
        self.cpu.close()


def _signature(observations, instructions):
    # This is an in-memory, one-use handoff, never a reusable feature cache.
    return tuple((tuple(id(image) for image in obs["image"]), instruction)
                 for obs, instruction in zip(observations, instructions))


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, dict):
        for v in value.values():
            _record_stream(v, stream)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _record_stream(v, stream)


class FrozenVisionAhead:
    """Run the original frozen Qwen input builder on a producer CUDA stream.

    No inference_mode, dtype conversion, new attention kernel, token change,
    trainable embedding, GT or action head is introduced. The original builder
    is retained and all trainable work remains on the consumer stream.
    """
    def __init__(self, model, *, concurrent=True):
        self.model = model
        self.visual = model.qwen_vl_interface.model.model.visual
        require(not self.visual.training and all(not p.requires_grad for p in self.visual.parameters()),
                "vision prefetch requires a frozen eval visual tower")
        self.original = model._build_qwen_batch
        self.device = next(self.visual.parameters()).device
        self.stream = torch.cuda.Stream(device=self.device) if concurrent and self.device.type == "cuda" else None
        if self.stream:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.active = None
        model._build_qwen_batch = self.consume

    def instructions(self, observations):
        from starVLA.model.framework.DDPForesight import ROBOT_HISTORY_TOKEN
        return [e["lang"] + " " + ROBOT_HISTORY_TOKEN + "".join(self.model.foresight_tokens)
                + "".join(self.model.act_query_tokens) for e in observations]

    def prepare(self, observations, micro_batch):
        require(not self.visual.training, "frozen vision switched to train during prefetch")
        pieces = []
        start = time.perf_counter()
        def run():
            with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
                for at in range(0, len(observations), micro_batch):
                    obs = observations[at:at + micro_batch]
                    current = [{k: e[k] for k in ("image", "lang", "state")} for e in obs]
                    instructions = self.instructions(current)
                    value = self.original(current, instructions)
                    pieces.append((_signature(current, instructions), value))
        if self.stream:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                run()
                ready = torch.cuda.Event()
                ready.record(self.stream)
        else:
            run()
            ready = None
        return {"pieces": deque(pieces), "ready": ready, "prepare_wall_seconds": time.perf_counter() - start}

    @contextmanager
    def activate(self, prepared):
        require(self.active is None, "overlapping vision consumers")
        if prepared["ready"]:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(prepared["ready"])
            for _, value in prepared["pieces"]:
                _record_stream(value, stream)
        self.active = prepared["pieces"]
        try:
            yield
            require(not self.active, "prefetched microbatches not fully consumed")
        finally:
            self.active = None

    def consume(self, observations, instructions):
        require(self.active is not None and len(self.active) > 0, "missing staged vision batch")
        signature, result = self.active.popleft()
        require(signature == _signature(observations, instructions), "vision input/order/prompt mismatch")
        return result

    def close(self):
        if self.stream:
            self.stream.synchronize()
        self.model._build_qwen_batch = self.original
