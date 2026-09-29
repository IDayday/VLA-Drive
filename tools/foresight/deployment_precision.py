"""Explicit provenance for timing points; BF16 training timing is not FP32 deployment."""
from collections import Counter
import torch


def describe(model, parameter_source, loading_path):
    return {
        'parameter_source': parameter_source,
        'loading_path': loading_path,
        'parameter_dtype_elements': dict(Counter({str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
            for dtype in {p.dtype for p in model.parameters()}})),
        'compute_policy': 'FP32, autocast disabled' if getattr(model, 'inference_fp32', False) else 'BF16 autocast; original FP32 action integration',
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'auxiliary_heads_removed': not any(hasattr(model, k) for k in ('dino_head', 'interaction_head', 'future_head')),
        'W_retained': int(model.foresight_config.num_queries),
    }
