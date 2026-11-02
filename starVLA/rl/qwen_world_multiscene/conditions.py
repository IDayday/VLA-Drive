"""Pack genuine per-scene frozen observations without re-running frozen Qwen.

Selected visual and text features are gathered BEFORE padding. Padding remains
masked within each scene; no source token/attention can cross a scene boundary.
"""
import torch
from starVLA.rl.qwen_world_postcot.rl_actor import RAW_CONTEXT_SCHEMA

SCHEMA = 'world_multiscene_scope_frozen_observations_v1'


def _pad(rows, fill=0):
    if not rows or any(x.shape[0] != 1 for x in rows):
        raise ValueError('Pack only genuine B1 frozen observations')
    size=max(x.shape[1] for x in rows)
    out=rows[0].new_full((len(rows),size,*rows[0].shape[2:]),fill)
    for i,x in enumerate(rows):out[i,:x.shape[1]]=x[0]
    return out


def _gather(value, index):
    return value.gather(1,index[...,None].expand(-1,-1,value.shape[-1]))


def pack_frozen_features(features, selected_layers):
    """Lossless feature indexing plus masked zero padding, not a new encoder."""
    selected={}
    for layer in selected_layers:
        values=[]
        for f in features:
            mapping=f.get('selected_hidden_states')
            hidden=(mapping[layer] if layer in mapping else mapping[str(layer)]) if mapping is not None else f['hidden_states'][layer]
            values.append(_gather(hidden,f['visual_indices']))
        selected[layer]=_pad(values)
    visual_mask=_pad([f['visual_mask'] for f in features],False)
    result=dict(selected_hidden_states=selected,
        visual_indices=torch.arange(visual_mask.shape[1],device=visual_mask.device)[None].expand(len(features),-1),
        visual_mask=visual_mask,camera_ids=_pad([f['camera_ids'] for f in features],-1),
        visual_positions=_pad([f['visual_positions'] for f in features]),
        text_features=_pad([f['text_features'] if 'text_features' in f else _gather(f['final_hidden'],f['text_indices']) for f in features]),
        text_mask=_pad([f['text_mask'] for f in features],False))
    if all('merged_visual_features' in f for f in features):
        result['merged_visual_features']=_pad([f['merged_visual_features'] for f in features])
    elif any('merged_visual_features' in f for f in features):
        raise ValueError('Same-forward merged visual evidence must be present for every scene')
    return result


class PackedConditions:
    """Ephemeral forward-only adapter; saved rollouts keep original descriptors."""
    def __init__(self, conditions, *, selected_layers):
        self.conditions=conditions
        self.selected_layers=tuple(selected_layers)

    def pack(self, descriptors, device):
        count=len(descriptors)
        if count not in (2,4):raise ValueError('Prototype supports exactly two or four genuine scenes')
        tokens=[d['token'] for d in descriptors]
        if len(set(tokens))!=count:raise ValueError('Repeated one-scene carriers are not a multi-scene batch')
        groups=[d['repeats'] for d in descriptors]
        if len(set(groups))!=1 or groups[0] not in (8,16,32):raise ValueError('Aligned real candidate groups required')
        raw=[self.conditions.materialize(d,device) for d in descriptors]
        if any(x['schema']!=RAW_CONTEXT_SCHEMA or x['base_memory'].shape[0]!=1 for x in raw):
            raise ValueError('Expected original B1 memory providers')
        index=torch.arange(count,device=device).repeat_interleave(groups[0])
        base={key:torch.cat([x[key] for x in raw],0).detach() for key in ('base_memory','base_valid','base_ego')}
        return dict(schema=SCHEMA,scene_tokens=tokens,groups=groups[0],scene_index=index,
            frozen_features=pack_frozen_features([x['frozen_features'] for x in raw],self.selected_layers),**base,
            context=base['base_memory'].index_select(0,index),valid=base['base_valid'].index_select(0,index),ego=base['base_ego'].index_select(0,index))
