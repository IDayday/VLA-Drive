"""Opt-in FP32 continuous-action projection, not enabled in active recipes.

FP32 parameters followed by an autocast BF16 Linear and `.float()` still give
BF16-quantized clean-x. Small physical exploration must account for that.
Training AND deployment AND the IL reference must share this numerical mode;
its parent-weight identity is unchanged but initial B1 outputs can differ.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .physical_noise import codec_jacobian


class FP32ActionOutput(nn.Linear):
    def forward(self,inputs):
        if self.weight.dtype!=torch.float32 or(self.bias is not None and self.bias.dtype!=torch.float32):
            raise ValueError('FP32 action head needs FP32 master parameters')
        with torch.autocast(inputs.device.type,enabled=False):
            return F.linear(inputs.float(),self.weight,self.bias)


def install_fp32_action_output(decision):
    old=decision.action_output
    if isinstance(old,FP32ActionOutput):return old
    if type(old)is not nn.Linear or old._forward_hooks or old._forward_pre_hooks:
        raise ValueError('Only an unhooked native Linear action projection can change numerical mode')
    # Preserve the SAME Parameter objects/Adam keys and state-dict names. No
    # random initialization, tensor cloning or optimizer reset is performed.
    new=FP32ActionOutput.__new__(FP32ActionOutput);nn.Module.__init__(new)
    new.in_features=old.in_features;new.out_features=old.out_features
    new.weight=old.weight;new.bias=old.bias;new.train(old.training)
    decision.action_output=new
    return new


def bf16_output_spacing_physical(normalized):
    value=torch.as_tensor(normalized,dtype=torch.float32)
    if value.shape[-2:]!=(8,3)or not torch.isfinite(value).all():raise ValueError('Finite normalized8x3 action required')
    quantized=value.to(torch.bfloat16)
    next_value=torch.nextafter(quantized,torch.full_like(quantized,float('inf')))
    return (next_value.float()-quantized.float()).double()*codec_jacobian()
