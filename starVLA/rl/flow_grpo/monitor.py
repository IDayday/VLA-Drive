"""Gradient evidence captured before DeepSpeed consumes/partitions gradients."""

import torch


class GradientMonitor:
    def __init__(self, policy):
        self.handles = []
        self.values = {}
        self.names = set()
        for name, p in policy.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_hook(self._hook(name)))

    def _hook(self, name):
        def hook(gradient):
            group = name.split(".")[0]
            norm = gradient.detach().norm(dtype=torch.float32).square()
            self.values[group] = self.values.get(group, 0) + norm
            self.names.add(name)
            return gradient

        return hook

    def consume(self):
        values = {name: float(value.sqrt()) for name, value in self.values.items()}
        if any(not torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise FloatingPointError("nonfinite module gradient norm")
        result = {
            "local_backward_contribution_l2": values,
            "gradient_tensors": len(self.names),
            "gradient_hook_scope": "local sum of squared backward contributions; not optimizer gradient norm",
        }
        self.values.clear()
        self.names.clear()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()

    def assert_finite_collective(self, device):
        values = list(self.values.values())
        finite = (
            torch.stack([torch.isfinite(value) for value in values]).all()
            if values
            else torch.tensor(True, device=device)
        )
        flag = finite.to(dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        if not flag.item():
            raise FloatingPointError(
                "nonfinite gradients: all ranks stopped before optimizer step"
            )


class CheckedDeepSpeedBackward:
    """Accelerate 1.5.2 backward adapter, with a collective pre-step finite gate.

    Same engine.backward/engine.step API and accumulation ownership as the
    installed DeepSpeedEngineWrapper; no submodule-forward bypass.
    """

    def __init__(self, engine, monitor, device, boundary_probe=None):
        self.engine = engine
        self.monitor = monitor
        self.device = device
        self.boundary_probe = boundary_probe

    def backward(self, loss, **kwargs):
        self.engine.backward(loss, **kwargs)
        self.monitor.assert_finite_collective(self.device)
        if self.engine.is_gradient_accumulation_boundary():
            self.engine.flow_pre_step_dtype = dtype_inventory(
                self.engine.module.policy, self.engine
            )
            if self.boundary_probe is not None:
                self.boundary_probe(self.engine)
        self.engine.step()


class ParameterProbe:
    """Cheap per-update sampled deltas; full tensor deltas are audited offline."""

    def __init__(self, policy, samples_per_tensor=16):
        self.samples = {}
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                indices = torch.linspace(
                    0,
                    parameter.numel() - 1,
                    min(samples_per_tensor, parameter.numel()),
                    device=parameter.device,
                ).long()
                self.samples[name] = (parameter, indices)
        self.previous = self.snapshot()

    def snapshot(self):
        return {
            name: parameter.detach().reshape(-1)[indices].float().clone()
            for name, (parameter, indices) in self.samples.items()
        }

    def consume(self):
        current = self.snapshot()
        norms = {}
        for name, value in current.items():
            root = name.split(".")[0]
            norms[root] = (
                norms.get(root, 0) + (value - self.previous[name]).square().sum()
            )
        self.previous = current
        return {
            "module_parameter_probe_delta_l2": {
                name: float(value.sqrt()) for name, value in norms.items()
            },
            "parameter_probe_values": sum(value.numel() for value in current.values()),
        }


def optimizer_norm_metrics(engine, backend, explicit_pre_norm=None, clip=None):
    value = (
        engine.get_global_grad_norm()
        if backend and hasattr(engine, "get_global_grad_norm")
        else explicit_pre_norm
    )
    if value is None:
        return {
            "pre_clip_grad_norm": None,
            "clip_scale": None,
            "post_clip_grad_norm": None,
            "grad_norm_status": "unavailable from this backend",
        }
    norm = float(value)
    # Scale derived from the installed ZeRO-2 / torch clip formula. Post-clip norm
    # is NOT measured: DeepSpeed consumes gradients inside engine.step().
    scale = min(1.0, clip / (norm + 1e-6)) if clip and clip > 0 else 1.0
    return {
        "pre_clip_grad_norm": norm,
        "clip_scale": scale,
        "post_clip_grad_norm": None,
        "grad_norm_status": "pre: backend global norm; scale: backend formula; post: unavailable",
    }


def dtype_inventory(policy, engine=None):
    from collections import Counter

    result = {
        "parameters": dict(Counter(str(p.dtype) for p in policy.parameters())),
        "gradients": dict(
            Counter(
                str(p.grad.dtype) for p in policy.parameters() if p.grad is not None
            )
        ),
        "parameter_grad_accum": dict(
            Counter(
                str(p.grad_accum.dtype)
                for p in policy.parameters()
                if getattr(p, "grad_accum", None) is not None
            )
        ),
    }
    if engine is not None:
        optimizer = engine.optimizer
        result.update(
            communication_buffers=sorted(
                getattr(engine, "flow_communication_dtypes", set())
            ),
            accumulation_dtype=str(
                getattr(optimizer, "gradient_accumulation_dtype", "unavailable")
            ),
            communication_dtype=str(
                getattr(optimizer, "communication_data_type", "unavailable")
            ),
            partition_buffers=[
                str(t.dtype)
                for group in getattr(optimizer, "averaged_gradients", {}).values()
                # ZeRO-2 0.16.9 releases each group by assigning None after step.
                # Absence is an empty observation; pre-step inventory is separate.
                if group is not None
                for t in group
                if t is not None
            ],
            master_weights=[
                str(t.dtype)
                for t in getattr(optimizer, "single_partition_of_fp32_groups", [])
            ],
            optimizer_states=dict(
                Counter(
                    str(v.dtype)
                    for state in optimizer.optimizer.state.values()
                    for v in state.values()
                    if isinstance(v, torch.Tensor)
                )
            ),
        )
    return result


class CommunicationDtypeMonitor:
    """Read-only observation at installed DeepSpeed collective API boundaries.

    Bounded diagnostic runs only. No casts, autograd hooks, new collectives or
    changes to arguments/results. Scalar norm/control collectives are excluded.
    """

    def __init__(self, engine, module=None):
        if module is None:
            import deepspeed.comm as module
        self.module, self.originals = module, {}
        engine.flow_communication_dtypes = set()
        for name in ("all_reduce", "reduce", "reduce_scatter", "reduce_scatter_fn"):
            original = getattr(module, name, None)
            if original is None:
                continue
            self.originals[name] = original

            def observe(*args, _original=original, **kwargs):
                for value in [*args, *kwargs.values()]:
                    if (
                        isinstance(value, torch.Tensor)
                        and value.is_cuda
                        and value.is_floating_point()
                        and value.numel() > 1
                    ):
                        engine.flow_communication_dtypes.add(str(value.dtype))
                return _original(*args, **kwargs)

            setattr(module, name, observe)

    def close(self):
        for name, original in self.originals.items():
            setattr(self.module, name, original)


class ActivationDtypeMonitor:
    """Observe real tensors (autocast parameters and output dtypes can differ)."""

    def __init__(self, policy):
        self.values, self.handles = {}, []
        for path in (
            "qwen_vl_interface.model.visual",
            "qwen_vl_interface.model.model.language_model",
            "action_input_model",
            "action_model.qwen_proj",
            "action_model.action_encoder",
            "action_model.model",
        ):
            module = policy.get_submodule(path)
            self.handles.append(module.register_forward_hook(self._hook(path)))

    def _hook(self, name):
        def visit(value):
            if isinstance(value, torch.Tensor):
                return [str(value.dtype)]
            if isinstance(value, dict):
                return sum((visit(x) for x in value.values()), [])
            if isinstance(value, (list, tuple)):
                return sum((visit(x) for x in value), [])
            return []

        def hook(module, inputs, output):
            self.values[name] = {
                "input": sorted(set(visit(inputs))),
                "output": sorted(set(visit(output))),
                "parameter": sorted({str(p.dtype) for p in module.parameters()}),
            }

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def save_full_optimizer_gradients(engine, output):
    """Diagnostic only; installed DeepSpeed public API, all ranks participate.

    This reads gradients after actual ZeRO reduction/accumulation and before
    clipping/Adam. No cast of a model's final BF16 .grad is called FP32 accumulation.
    """
    from pathlib import Path
    import json
    from deepspeed.utils import safe_get_full_grad
    from .distributed import synchronized_call

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(output) / f"update_{engine.global_steps + 1:06d}" / f"rank_{rank}"
    synchronized_call(lambda: path.mkdir(parents=True, exist_ok=False), engine.device)
    metadata = []
    for index, (name, parameter) in enumerate(engine.module.policy.named_parameters()):
        if not parameter.requires_grad:
            continue
        # Contains collectives: fatal backend errors must escape to torchrun.
        gradient = safe_get_full_grad(parameter)
        entry = {
            "name": name,
            "shape": list(parameter.shape),
            "present": gradient is not None,
            "dtype": str(gradient.dtype) if gradient is not None else None,
            "file": f"{index:04d}.pt",
        }
        synchronized_call(
            lambda captured=gradient: torch.save(
                captured.cpu() if captured is not None else None, path / entry["file"]
            ),
            engine.device,
        )
        metadata.append(entry)
        del gradient
    synchronized_call(
        lambda: (path / "manifest.json").write_text(
            json.dumps(
                {
                    "scope": "actual globally reduced optimizer gradient BEFORE clip/Adam; safe_get_full_grad",
                    "parameters": metadata,
                },
                indent=2,
            )
        ),
        engine.device,
    )
