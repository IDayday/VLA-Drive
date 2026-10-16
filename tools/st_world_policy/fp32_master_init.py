"""Preserve genuine FP32 initialization across DeepSpeed 0.16.9 ZeRO-2 setup.

The installed engine casts the module to BF16 before ZeRO clones FP32 master
partitions. Capture *before* ``deepspeed.initialize`` and restore immediately
after it, only for a fresh run. Native resume/continuation must skip both calls.

This is deliberately a narrow adaptation of that installed layout, not another
optimizer. It does not change BF16 execution tensors or the backend's gradient
dtype. Frozen parameters/buffers are outside this helper (canonical deployment
continues to obtain the frozen visual tower from the bound FP32 parent).

All ranks must call both functions. Validation failures are communicated before
any rank writes a master. An error during the write/verification aborts the new
run; never train with a partially restored engine.
"""
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
import time
from typing import Any, Callable

import torch
import torch.distributed as dist


SCHEMA = "st_fp32_master_initialization_v1"
SUPPORTED_DEEPSPEED = "0.16.9"
DEFAULT_CHUNK_ELEMENTS = 1 << 20
MAX_CHUNK_ELEMENTS = 1 << 24


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _topology() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _gather(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def _collective_local(function: Callable[[], Any], phase: str) -> Any:
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather(error)
    if any(errors):
        detail = "; ".join(f"rank {i}: {item}" for i, item in enumerate(errors) if item)
        raise ValueError(f"FP32 master {phase} rejected: {detail}")
    return result


def _agree(value: Any, phase: str) -> None:
    values = _gather(value)
    _require(all(item == values[0] for item in values), f"{phase} differs across ranks")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().view(torch.uint8).numpy().tobytes()


def _tensor_hash(tensor: torch.Tensor, chunk_elements: int) -> str:
    digest = hashlib.sha256()
    flat = tensor.detach().view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        digest.update(_bytes(flat[start:start + chunk_elements].cpu()))
    return digest.hexdigest()


def _header(digest: Any, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    encoded = _json_bytes({"name": name, "shape": shape, "dtype": str(dtype)})
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bit equality, including signed zero, for equal-dtype CPU tensors."""
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.view(torch.uint8), right.view(torch.uint8))


@dataclass(frozen=True)
class ParameterLayout:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass
class GroupSnapshot:
    parameters: tuple[ParameterLayout, ...]
    numel: int
    padded_numel: int
    partition_start: int
    partition_size: int
    padding: int
    values: torch.Tensor | None = field(repr=False)
    sha256: str

    def layout(self) -> dict[str, Any]:
        return {"parameters": [vars(item) for item in self.parameters],
                "numel": self.numel, "padded_numel": self.padded_numel,
                "partition_size": self.partition_size}


@dataclass
class FP32MasterInitializationSnapshot:
    """Ephemeral local CPU source; never use this as a checkpoint or resume."""
    groups: tuple[GroupSnapshot, ...]
    rank: int
    world_size: int
    source_sha256: str
    execution_sha256: str
    layout_sha256: str
    chunk_elements: int
    capture_seconds: float
    consumed: bool = False

    @property
    def identity(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "world_size": self.world_size,
                "source_sha256": self.source_sha256,
                "expected_bf16_execution_sha256": self.execution_sha256,
                "layout_sha256": self.layout_sha256}

    @property
    def retained_cpu_bytes(self) -> int:
        return sum(group.partition_size for group in self.groups) * 4


def capture_fp32_master_initialization(
    model: torch.nn.Module,
    parameter_groups: list[dict[str, Any]],
    *,
    is_fresh_run: bool,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> FP32MasterInitializationSnapshot | None:
    """Capture original named trainable FP32 fragments before engine conversion.

    Pass the unchanged groups returned by ``bounded_parameter_groups``. The
    full source and expected BF16 execution hashes are compared across ranks;
    only each rank's padded FP32 partition is retained, on CPU.
    """
    start_time = time.perf_counter()

    @torch.no_grad()
    def capture():
        _require(type(is_fresh_run) is bool, "is_fresh_run must be explicit bool")
        if not is_fresh_run:
            return None
        _require(type(chunk_elements) is int and 0 < chunk_elements <= MAX_CHUNK_ELEMENTS,
                 "Invalid bounded copy chunk")
        rank, world = _topology()
        named = {id(p): (name, p) for name, p in model.named_parameters() if p.requires_grad}
        _require(bool(named), "No trainable FP32 source parameters")
        _require(isinstance(parameter_groups, list) and bool(parameter_groups), "Missing parameter groups")
        seen, groups = set(), []
        source_hash, execution_hash = hashlib.sha256(), hashlib.sha256()
        for group in parameter_groups:
            params = group.get("params")
            _require(isinstance(params, (list, tuple)) and bool(params), "Groups must contain explicit parameter lists")
            layouts, size = [], 0
            for parameter in params:
                _require(id(parameter) in named, "Optimizer contains a frozen or unknown source parameter")
                _require(id(parameter) not in seen, "Duplicate optimizer parameter")
                name, _ = named[id(parameter)]
                _require(parameter.dtype == torch.float32, f"Source is not FP32: {name}")
                _require(parameter.is_contiguous() and parameter.numel() > 0,
                         f"Source must be nonempty and contiguous: {name}")
                _require(parameter.grad is None, f"Source already has a gradient: {name}")
                seen.add(id(parameter))
                layouts.append(ParameterLayout(name, tuple(parameter.shape), parameter.numel(), size))
                size += parameter.numel()
            alignment = 2 * world
            padded = ((size + alignment - 1) // alignment) * alignment
            part_size = padded // world
            part_start = rank * part_size
            values = torch.zeros(part_size, dtype=torch.float32, device="cpu")
            for parameter, item in zip(params, layouts):
                _header(source_hash, item.name, item.shape, torch.float32)
                _header(execution_hash, item.name, item.shape, torch.bfloat16)
                flat = parameter.detach().view(-1)
                for offset in range(0, item.numel, chunk_elements):
                    chunk = flat[offset:offset + chunk_elements].cpu()
                    _require(bool(torch.isfinite(chunk).all()), f"Nonfinite FP32 source: {item.name}")
                    source_hash.update(_bytes(chunk))
                    rounded = chunk.to(torch.bfloat16)
                    _require(bool(torch.isfinite(rounded).all()), f"Source overflows BF16: {item.name}")
                    execution_hash.update(_bytes(rounded))
                    lo = max(item.offset + offset, part_start)
                    hi = min(item.offset + offset + chunk.numel(), part_start + part_size)
                    if hi > lo:
                        values[lo - part_start:hi - part_start].copy_(
                            chunk[lo - item.offset - offset:hi - item.offset - offset])
            groups.append(GroupSnapshot(tuple(layouts), size, padded, part_start, part_size,
                min(part_size, max(0, part_start + part_size - size)), values,
                _tensor_hash(values, chunk_elements)))
        _require(seen == set(named), "Optimizer groups do not cover all trainable source parameters")
        layout_hash = hashlib.sha256(_json_bytes([group.layout() for group in groups])).hexdigest()
        return FP32MasterInitializationSnapshot(tuple(groups), rank, world, source_hash.hexdigest(),
            execution_hash.hexdigest(), layout_hash, chunk_elements, time.perf_counter() - start_time)

    snapshot = _collective_local(capture, "source capture")
    _agree(None if snapshot is None else snapshot.identity, "Fresh source/layout or resume decision")
    return snapshot


def _same_view(actual: torch.Tensor, source: torch.Tensor, offset: int, numel: int, label: str) -> None:
    _require(actual.dtype == source.dtype and actual.device == source.device and actual.is_contiguous()
             and actual.numel() == numel and actual.data_ptr() == source.data_ptr() + offset * source.element_size(),
             f"Wrong native storage view: {label}")


@torch.no_grad()
def _validate_engine(engine: Any, snapshot: FP32MasterInitializationSnapshot) -> dict[str, Any]:
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    _require(version("deepspeed") == SUPPORTED_DEEPSPEED, "Unsupported DeepSpeed version")
    _require(isinstance(snapshot, FP32MasterInitializationSnapshot) and not snapshot.consumed,
             "Missing or already consumed FP32 snapshot")
    _require((snapshot.rank, snapshot.world_size) == _topology(), "Capture/restore topology changed")
    _require(getattr(engine, "_st_fp32_master_initialization", None) is None, "Master initialization already applied")
    _require(engine.zero_optimization_stage() == 2 and engine.bfloat16_enabled(), "Requires BF16 ZeRO-2")
    for name in ("global_steps", "global_samples", "micro_steps", "skipped_steps"):
        _require(getattr(engine, name, None) == 0, f"Not a fresh engine: {name}")
    optimizer = engine.optimizer
    _require(type(optimizer) is DeepSpeedZeroOptimizer, "Unsupported optimizer wrapper")
    _require(optimizer.partition_gradients and optimizer.dtype == torch.bfloat16,
             "Requires native BF16 ZeRO-2 partitions")
    _require(not optimizer.cpu_offload and not optimizer.round_robin_gradients and not optimizer.has_moe_layers
             and not optimizer.fp16_master_weights_and_gradients and optimizer.mpu is None,
             "Offload, round-robin, MoE, low-precision master or MPU is unsupported")
    _require(optimizer.nccl_start_alignment_factor == 2, "Unsupported alignment")
    _require(optimizer.micro_step_id == -1 and not optimizer.averaged_gradients, "Optimizer already consumed gradients")
    inner = optimizer.optimizer
    implementation = (type(inner).__module__, type(inner).__name__)
    _require(implementation in {("torch.optim.adamw", "AdamW"),
                               ("deepspeed.ops.adam.fused_adam", "FusedAdam")}, "Requires existing AdamW implementation")
    if implementation[1] == "FusedAdam":
        _require(bool(inner.adam_w_mode), "FusedAdam must retain AdamW mode")
    _require(all(not item for item in inner.state.values()), "Optimizer moments/state already initialized; use native resume")
    group_fields = ("bit16_groups", "bit16_groups_flat", "round_robin_bit16_groups", "round_robin_bit16_indices",
        "single_partition_of_fp32_groups", "parallel_partitioned_bit16_groups", "partition_size",
        "groups_padding", "real_dp_process_group", "partition_count")
    _require(all(len(getattr(optimizer, name)) == len(snapshot.groups) for name in group_fields)
             and len(inner.param_groups) == len(snapshot.groups), "Optimizer group count changed")
    named = dict(engine.module.named_parameters())
    expected_names = [item.name for group in snapshot.groups for item in group.parameters]
    _require(set(expected_names) == {name for name, p in named.items() if p.requires_grad},
             "Trainable parameter membership changed")
    execution_hash = hashlib.sha256()
    changed_elements, max_rounding_error = 0, 0.0
    for index, group in enumerate(snapshot.groups):
        values = group.values
        _require(values is not None and values.device.type == "cpu" and values.dtype == torch.float32
                 and values.is_contiguous() and values.shape == (group.partition_size,)
                 and _tensor_hash(values, snapshot.chunk_elements) == group.sha256, "Captured partition corrupted")
        dp_group = optimizer.real_dp_process_group[index]
        if dist.is_initialized():
            _require(dist.get_rank(dp_group) == snapshot.rank and dist.get_world_size(dp_group) == snapshot.world_size,
                     "Optimizer data-parallel topology changed")
        else:
            _require(dp_group is None and snapshot.world_size == 1, "Distributed optimizer without initialized group")
        _require(optimizer.partition_count[index] == snapshot.world_size, "Partition count changed")
        parameters = optimizer.bit16_groups[index]
        _require([optimizer.param_names[p] for p in parameters] == [item.name for item in group.parameters],
                 "Parameter name/order changed")
        _require(len(optimizer.round_robin_bit16_groups[index]) == len(parameters)
                 and all(p is q for p, q in zip(parameters, optimizer.round_robin_bit16_groups[index]))
                 and optimizer.round_robin_bit16_indices[index] == list(range(len(parameters))), "Group reordered")
        flat = optimizer.bit16_groups_flat[index]
        master = optimizer.single_partition_of_fp32_groups[index]
        _require(flat.dtype == torch.bfloat16 and flat.is_contiguous() and flat.shape == (group.padded_numel,),
                 "Wrong native execution flat tensor")
        _require(master.dtype == torch.float32 and master.is_contiguous() and master.shape == (group.partition_size,)
                 and master.requires_grad and master.grad is None and master.device == flat.device,
                 "Wrong FP32 master tensor/gradient")
        owned = inner.param_groups[index]["params"]
        _require(len(owned) == 1 and owned[0] is master, "Optimizer no longer owns the native master")
        _require(master.untyped_storage().data_ptr() != flat.untyped_storage().data_ptr(),
                 "Master aliases execution storage")
        _require(optimizer.partition_size[index] == group.partition_size and optimizer.groups_padding[index] == group.padding,
                 "Native partition extent/padding differs")
        partitions = optimizer.parallel_partitioned_bit16_groups[index]
        _require(len(partitions) == snapshot.world_size, "Execution partition count differs")
        for rank, partition in enumerate(partitions):
            _same_view(partition, flat, rank * group.partition_size, group.partition_size, "execution partition")
        _require(not bool(torch.count_nonzero(flat[group.numel:]).item()), "Nonzero native alignment padding")
        for parameter, item in zip(parameters, group.parameters):
            _require(named[item.name] is parameter and tuple(parameter.shape) == item.shape and parameter.grad is None
                     and getattr(parameter, "grad_accum", None) is None, f"Parameter changed: {item.name}")
            _same_view(parameter, flat, item.offset, item.numel, item.name)
            _header(execution_hash, item.name, item.shape, torch.bfloat16)
            for offset in range(0, item.numel, snapshot.chunk_elements):
                execution_hash.update(_bytes(parameter.detach().view(-1)[offset:offset + snapshot.chunk_elements].cpu()))
            lo = max(item.offset, group.partition_start)
            hi = min(item.offset + item.numel, group.partition_start + group.partition_size)
            mapping = parameter._hp_mapping
            if hi <= lo:
                _require(mapping is None, f"Unexpected nonlocal master mapping: {item.name}")
                continue
            _require(mapping is not None and not mapping.use_offload and mapping.param_group_index == index,
                     f"Missing/foreign master mapping: {item.name}")
            _require((mapping.hp_fragment_address.start, mapping.hp_fragment_address.numel) ==
                     (lo - group.partition_start, hi - lo) and
                     (mapping.lp_fragment_address.start, mapping.lp_fragment_address.numel) == (lo - item.offset, hi - lo),
                     f"Wrong master fragment address: {item.name}")
            _same_view(mapping.hp_fragment, master, lo - group.partition_start, hi - lo, "master fragment")
            _same_view(mapping.lp_fragment, parameter, lo - item.offset, hi - lo, "execution fragment")
        for offset in range(0, group.partition_size, snapshot.chunk_elements):
            source = values[offset:offset + snapshot.chunk_elements]
            rounded = source.to(torch.bfloat16)
            expected = rounded.float()
            actual = master.detach()[offset:offset + snapshot.chunk_elements].cpu()
            execution = partitions[snapshot.rank][offset:offset + snapshot.chunk_elements].cpu()
            _require(_exact(actual, expected) and _exact(execution, rounded),
                     "Native initialization differs from captured BF16 roundtrip; refuse to overwrite")
            changed_elements += int(torch.count_nonzero(source.view(torch.int32) != actual.view(torch.int32)))
            max_rounding_error = max(max_rounding_error, float((source - actual).abs().max()))
    _require(execution_hash.hexdigest() == snapshot.execution_sha256, "Execution values differ from full captured source")
    return {"rank": snapshot.rank, "restored_elements": sum(g.partition_size for g in snapshot.groups),
            "source_elements": sum(g.numel for g in snapshot.groups),
            "changed_from_rounded_elements": changed_elements, "max_startup_rounding_error": max_rounding_error,
            "local_padding_elements": sum(g.padding for g in snapshot.groups),
            "retained_cpu_bytes": snapshot.retained_cpu_bytes,
            "capture_seconds": snapshot.capture_seconds,
            "execution_dtype": str(optimizer.dtype),
            "gradient_accumulation_dtype": str(optimizer.gradient_accumulation_dtype),
            "gradient_precision_changed": False, "optimizer_implementation": ".".join(implementation),
            "local_partition_sha256": [g.sha256 for g in snapshot.groups]}


def restore_fp32_master_initialization(
    engine: Any,
    snapshot: FP32MasterInitializationSnapshot | None,
    *,
    is_fresh_run: bool,
) -> dict[str, Any]:
    """Restore existing native FP32 masters once; never apply to native resume.

    BF16 execution values, optimizer groups, scheduler, data cursor and RNG are
    untouched. Returns a serializable audit; include it in initialization/run
    identity. On success local CPU snapshot tensors are released.
    """
    start_time = time.perf_counter()

    def validate():
        _require(type(is_fresh_run) is bool, "is_fresh_run must be explicit bool")
        if not is_fresh_run:
            _require(snapshot is None, "Native resume/continuation must not capture fresh initialization")
            return None
        return _validate_engine(engine, snapshot)

    local = _collective_local(validate, "pre-write validation")
    _agree(None if local is None else snapshot.identity, "Fresh restore or native resume decision")
    if local is None:
        return {"schema": SCHEMA, "status": "SKIPPED_NATIVE_RESUME", "mutated": False}
    validation_seconds = time.perf_counter() - start_time

    @torch.no_grad()
    def restore():
        for group, master in zip(snapshot.groups, engine.optimizer.single_partition_of_fp32_groups):
            for offset in range(0, group.partition_size, snapshot.chunk_elements):
                master[offset:offset + snapshot.chunk_elements].copy_(group.values[offset:offset + snapshot.chunk_elements])
            _require(_tensor_hash(master, snapshot.chunk_elements) == group.sha256, "Restored master bit verification failed")
        local["validation_seconds"] = validation_seconds
        local["restore_and_verify_seconds"] = time.perf_counter() - start_time - validation_seconds
        local["master_exact"] = True
        return local

    local = _collective_local(restore, "write/post-write verification (abort new run on failure)")
    audit = {**snapshot.identity, "status": "RESTORED_EXACT_FP32_SOURCE", "deepspeed_version": SUPPORTED_DEEPSPEED,
             "native_resume": False, "execution_unchanged": True, "initialization_only": True,
             "chunk_elements": snapshot.chunk_elements, "max_transfer_chunk_bytes": snapshot.chunk_elements * 4,
             "bounded_cpu_working_buffer_budget_bytes": snapshot.chunk_elements * 32,
             "extra_persistent_gpu_bytes": 0, "ranks": _gather(local),
             "layout": [g.layout() for g in snapshot.groups], "formal_qualification": "NOT_GRANTED_BY_HELPER"}
    engine._st_fp32_master_initialization = audit
    for group in snapshot.groups:
        group.values = None
    snapshot.consumed = True
    return audit
