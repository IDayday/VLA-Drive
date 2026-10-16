"""One-entry, process-local reuse of a permanently frozen reference evaluation.

Only the exact saved inputs are reused. This does not cache CURRENT velocity,
old-policy material, reward or trainable activations. The entry is disposable
and is deliberately absent from model/optimizer/checkpoint state.
"""
import torch


def _signature(value):
    if torch.is_tensor(value):
        # Keep tensor versions, layout and device, not merely its data pointer.
        # Inference-mode tensors have no version counter and are not admitted.
        return ('tensor', id(value), value.data_ptr(), value._version,
                tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                str(value.dtype), str(value.device), value.requires_grad)
    if isinstance(value, dict):
        return ('dict', tuple((k, _signature(v)) for k, v in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_signature(v) for v in value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value).__name__, value)
    raise TypeError('Unsupported frozen reference input: ' + type(value).__name__)


def _detach(value):
    if torch.is_tensor(value):
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError('Frozen reference reuse must not retain an autograd graph')
        return value.detach()
    if isinstance(value, dict):return {k: _detach(v) for k, v in value.items()}
    if isinstance(value, list):return [_detach(v) for v in value]
    if isinstance(value, tuple):return tuple(_detach(v) for v in value)
    return value


class FrozenReferenceReuse:
    """Reuse only inside unchanged input/reference versions; at most one entry."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.clear()

    def clear(self):
        self.key = None
        self.inputs = None
        self.value = None
        self.output_signature = None

    def get(self, reference, inputs, sampler_identity, produce):
        if torch.is_grad_enabled():
            raise ValueError('Reference reuse requires the original no_grad context')
        if any(m.training for m in reference.modules()):
            raise ValueError('Reference must remain in deterministic eval mode')
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('A trainable reference cannot be reused')
        parameters = tuple((n, _signature(p)) for n, p in reference.named_parameters())
        buffers = tuple((n, _signature(p)) for n, p in reference.named_buffers())
        key = (_signature(inputs), _signature(sampler_identity), parameters, buffers)
        if key == self.key:
            if _signature(self.value) != self.output_signature:
                raise ValueError('Cached reference output was mutated')
            self.hits += 1
            return self.value
        value = _detach(produce())
        self.key = key
        # Strong references prevent Python object/data-pointer recycling. There
        # is only one entry; a new collection releases the preceding input set.
        self.inputs = inputs
        self.value = value
        self.output_signature = _signature(value)
        self.misses += 1
        return value

