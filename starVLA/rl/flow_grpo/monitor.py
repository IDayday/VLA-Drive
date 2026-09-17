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
        result = {"module_grad_norm": values, "gradient_tensors": len(self.names)}
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

    def __init__(self, engine, monitor, device):
        self.engine = engine
        self.monitor = monitor
        self.device = device

    def backward(self, loss, **kwargs):
        self.engine.backward(loss, **kwargs)
        self.monitor.assert_finite_collective(self.device)
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
