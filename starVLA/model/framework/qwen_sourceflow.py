"""Optional registry compatibility import; native entries bypass eager discovery.

The legacy framework package discovers unrelated video modules and can initialize
NCCL during import under torchrun. The native model has no such dependency.
"""
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA

__all__ = ['QwenSourceFlowVLA']
