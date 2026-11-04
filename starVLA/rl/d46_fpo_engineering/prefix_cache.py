"""Reuse detached C/P only within an explicit, unchanged-policy collection."""
from contextlib import contextmanager
from dataclasses import replace
import torch
from starVLA.model.modules.qwen_world_compare.contracts import ArchitecturePrefix


def repeat_scene(prefix, count):
    wrapped=isinstance(prefix,ArchitecturePrefix)
    base=prefix.tensors if wrapped else prefix
    if not len(base.valid)or len(set(base.scene_identity))!=1:
        raise ValueError('Collection cache requires exactly one scene')
    def repeat(tensor):
        shape=list(tensor.shape);shape[0]=count
        return tensor.narrow(0,0,1).expand(*shape)
    value=replace(base,layer_kv=tuple((repeat(k),repeat(v))for k,v in base.layer_kv),
        valid=repeat(base.valid),next_positions=repeat(base.next_positions),reasoning=repeat(base.reasoning),
        scene_identity=base.scene_identity[:1]*count)
    return replace(prefix,tensors=value,scene_identity=prefix.scene_identity[:1]*count)if wrapped else value


@contextmanager
def collection_prefix_cache(actor):
    """A gradient-bearing current prefix always goes through the original path.

    The caller must end this scope before optimizer.step. Parameter mutation is
    checked at exit; the cache stores no graph, is bounded by this collection,
    and is never serialized in a checkpoint.
    """
    original=actor._prepare;cache={};stats={'hits':0,'misses':0,'gradient_bypasses':0}
    parameters=[p for p in actor.parameters()if p.requires_grad]
    versions=[p._version for p in parameters]
    policy_version=actor._policy_version
    def prepare(condition,use_ref):
        if torch.is_grad_enabled()and not use_ref:
            stats['gradient_bypasses']+=1
            return original(condition,use_ref)
        if actor._policy_version!=policy_version:
            raise ValueError('Policy changed inside collection prefix cache')
        material=actor.materialize(condition)
        current=material.get('ref_current',material['current'])if use_ref else material['current']
        if len(current.scene_identity)!=1 or material.get('candidate_to_scene')is not None:
            return original(material,use_ref)
        encoded=current.encoded
        key=(use_ref,current.scene_identity,id(encoded.inputs_embeds),id(encoded.ego),
            encoded.inputs_embeds._version,encoded.ego._version)
        if key not in cache:
            with torch.no_grad():value=original(material,use_ref)
            base=value.tensors if isinstance(value,ArchitecturePrefix)else value
            if any(x.requires_grad for pair in base.layer_kv for x in pair):
                raise ValueError('Collection cache must not retain a training graph')
            cache[key]=repeat_scene(value,1);stats['misses']+=1
        else:stats['hits']+=1
        return repeat_scene(cache[key],len(material['context']))
    object.__setattr__(actor,'_prepare',prepare)
    try:yield stats
    finally:
        object.__setattr__(actor,'_prepare',original);cache.clear()
        if versions!=[p._version for p in parameters]:
            raise ValueError('Optimizer mutation occurred inside collection cache')
