"""Run under torchrun: native imports must not construct a process group/model."""
import sys
import torch.distributed as dist
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
from starVLA.rl.qwen_sourceflow.rl_runtime import NativeRuntime


if __name__=='__main__':
    if dist.is_initialized() or 'starVLA.model.framework' in sys.modules:
        raise RuntimeError('Native imports have legacy framework/distributed side effects')
    print('PASS: native import has no process-group or legacy-registry side effects',flush=True)
