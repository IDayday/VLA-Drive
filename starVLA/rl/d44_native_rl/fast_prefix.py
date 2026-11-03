"""Opt-in D-prefix metadata hoisting, without changing deployment model code."""
from dataclasses import replace
import torch
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import DPrefix
from starVLA.model.modules.qwen_world_compare.contracts import ArchitecturePrefix


def fast_index_select(prefix,indices):
    """Original DPrefix layout/strides/gradients with one CPU metadata read.

    Non-Tensor/invalid shape/dtype falls through to the original implementation
    to preserve its boundary exception and scene_identity lookup order.
    """
    wrapped=isinstance(prefix,ArchitecturePrefix)
    base=prefix.tensors if wrapped else prefix
    if not isinstance(base,DPrefix)or(wrapped and prefix.architecture_id!='D'):raise TypeError('Fast selection requires actual DPrefix or its D ArchitecturePrefix')
    if not isinstance(indices,torch.Tensor)or indices.ndim!=1 or indices.dtype!=torch.long:
        return prefix.index_select(indices)
    values=indices.detach().cpu().tolist()if len(indices)else[]
    # Preserve the outer ArchitecturePrefix lookup before the inner DPrefix.
    outer_names=tuple(prefix.scene_identity[i]for i in values)if wrapped else()
    names=tuple(base.scene_identity[i]for i in values)if base.scene_identity else()
    same=bool(values)and all(i==values[0]for i in values)
    def select(tensor):
        if not values:return tensor.index_select(0,indices)
        if same:
            shape=list(tensor.shape);shape[0]=len(values)
            return tensor.narrow(0,values[0],1).expand(*shape)
        return tensor.index_select(0,indices)
    selected=replace(base,layer_kv=tuple((select(k),select(v))for k,v in base.layer_kv),
        valid=select(base.valid),next_positions=select(base.next_positions),
        reasoning=select(base.reasoning),scene_identity=names)
    return replace(prefix,scene_identity=outer_names,tensors=selected)if wrapped else selected
