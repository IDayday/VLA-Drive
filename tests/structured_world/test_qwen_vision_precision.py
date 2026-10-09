"""Real Qwen vision operators exercise the training/inference precision boundary."""
from unittest.mock import patch
import torch
from torch import nn
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cfg = Qwen3VLVisionConfig(depth=2, hidden_size=64, intermediate_size=128,
            num_heads=4, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=64, num_position_embeddings=16, deepstack_visual_indexes=[0])
        cfg._attn_implementation = 'sdpa'
        model = VLAStructuredFGTR.__new__(VLAStructuredFGTR)
        nn.Module.__init__(model)
        model.vision = Qwen3VLVisionModel(cfg).float().eval()
        model.vision.requires_grad_(False)
        pixels = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    def extract(self, examples, instructions):
        assert examples == ['current_images'] and instructions == ['navigation']
        with torch.no_grad():
            return self.vision(pixels, grid)
    return model, extract


def test_training_vision_compute_is_bf16_with_fp32_parameters():
    model, extract = fixture()
    observed = []
    hook = model.vision.blocks[0].attn.qkv.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype))
    rng = torch.random.get_rng_state().clone()
    with patch.object(DDPForesight, '_build_qwen_batch', extract):
        model._build_qwen_batch(['current_images'], ['navigation'])
    hook.remove()
    assert observed == [torch.bfloat16]
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_canonical_fp32_vision_is_unchanged_by_training_wrapper():
    model, extract = fixture()
    model.inference_fp32 = True
    observed = []
    hook = model.vision.blocks[0].attn.qkv.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype))
    original = extract(model, ['current_images'], ['navigation'])
    with patch.object(DDPForesight, '_build_qwen_batch', extract):
        corrected = model._build_qwen_batch(['current_images'], ['navigation'])
    hook.remove()
    assert observed == [torch.float32, torch.float32]
    for actual, expected in zip([corrected[0]]+corrected[1], [original[0]]+original[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
