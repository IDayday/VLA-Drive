# IDayday/VLA-Drive f9449d55, visual_training.py; Apache-2.0
def encode_qwen_images(qwen_model: nn.Module, *, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, freeze_visual: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode images as ``[N,C]`` tokens with an explicit gradient policy."""
    assert pixel_values.ndim >= 2, 'pixel_values must have at least two dimensions'
    assert image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3, 'image_grid_thw must be [N,3]'
    gradient_context = torch.no_grad() if freeze_visual else nullcontext()
    with gradient_context:
        (image_parts, deepstack_embeds) = qwen_model.get_image_features(pixel_values, image_grid_thw)
        if not image_parts:
            raise RuntimeError('Qwen visual encoder returned no image features')
        image_embeds = torch.cat(tuple(image_parts), dim=0)
    assert image_embeds.ndim == 2, 'Qwen image embeddings must be [N,C]'
    if not isinstance(deepstack_embeds, (tuple, list)):
        raise TypeError('Qwen deepstack embeddings must be a list or tuple')
    deepstack = list(deepstack_embeds)
    for (index, value) in enumerate(deepstack):
        assert value.ndim == 2, f'deepstack[{index}] must be [N,C]'
    if not freeze_visual and torch.is_grad_enabled() and (not image_embeds.requires_grad):
        raise RuntimeError('Qwen visual fine-tuning is enabled, but image embeddings have no gradient graph')
    return (image_embeds, deepstack)
