"""Opt-in native text/action precision; frozen visual encoding is untouched.

Switch only the two existing native entry points. Parameter identities, keys,
autograd and the functional C/P prefix lifecycle are preserved. A new numeric
mode is a new scientific recipe; never install into a running trainer.
"""
import torch
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import QwenFlowAdapter


class PrecisionQwenFlowAdapter(QwenFlowAdapter):
    def _precision_context(self):
        mode=self.rl_text_action_precision
        return torch.autocast(self.device.type,dtype=torch.float16,enabled=mode=='float16')

    def prepare_prefix(self,*args,**kwargs):
        with self._precision_context():return super().prepare_prefix(*args,**kwargs)

    def denoise_x0(self,*args,**kwargs):
        with self._precision_context():return super().denoise_x0(*args,**kwargs)


def install_text_action_precision(decision,mode):
    if mode not in ('float32','float16'):raise ValueError('Explicit float32 or float16 text/action mode required')
    if type(decision) not in (QwenFlowAdapter,PrecisionQwenFlowAdapter):raise TypeError('Only native D text/action execution')
    if any(p.dtype!=torch.float32 for p in decision.parameters()):raise ValueError('Native FP32 master parameters required')
    decision.__class__=PrecisionQwenFlowAdapter
    decision.rl_text_action_precision=mode
    return decision
