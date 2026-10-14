"""Frozen-model diagnostic at an intermediate layer, preserving token positions."""
from contextlib import contextmanager
import torch


@contextmanager
def permute_world_at_layer(model, layer_index, permutation):
    if model.training:
        raise ValueError('Intermediate-W intervention is a read-only diagnostic')
    layers = model.qwen_vl_interface.model.model.language_model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError('Intervention layer out of range')
    record = {'layer': layer_index, 'image_path_changed': False, 'sequence_length_changed': False,
              'position_encoding_changed': False, 'intervention': 'permute W output, then continue Qwen'}
    def apply(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError('Unsupported Qwen decoder layer output')
        positions = model._diagnostic_world_positions
        order = torch.as_tensor(permutation, device=hidden.device, dtype=torch.long)
        if order.shape != (len(hidden),) or sorted(order.tolist()) != list(range(len(hidden))):
            raise ValueError('A scene permutation, not a token permutation, is required')
        rows = torch.arange(len(hidden), device=hidden.device)
        world = hidden[rows[:, None], positions]
        replacement = hidden.clone()
        replacement[rows[:, None], positions] = world[order]
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement
    handle = layers[layer_index].register_forward_hook(apply)
    try:
        yield record
    finally:
        handle.remove()
