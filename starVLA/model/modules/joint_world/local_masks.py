"""Separate actor eligibility, task masks, supervision masks and stable identity noise."""
import hashlib
import torch


def task_masks(graph, generator, mode='mask'):
    graph.validate()
    eligible=graph.active_actor_mask & graph.predictable_actor_mask
    batch,actors=eligible.shape
    draw=torch.rand(batch,generator=generator,device='cpu')
    nominal=torch.where(draw<.5,0,torch.where(draw<.75,1,2))
    if mode=='all':nominal.zero_()
    elif mode!='mask':raise ValueError('Unknown local masking objective')
    hidden=torch.zeros_like(eligible);actual=nominal.clone();fallback=torch.zeros(batch,dtype=torch.bool)
    # Draw irrespective of eligibility count, so no label or validity can resample a target.
    neighbor_draw=torch.rand(batch,generator=generator,device='cpu')
    for b in range(batch):
        if nominal[b]==0:hidden[b]=eligible[b]
        elif nominal[b]==1:hidden[b,0]=True
        else:
            ids=torch.where(eligible[b])[0];ids=ids[ids!=0]
            if len(ids):hidden[b,ids[min(int(neighbor_draw[b]*len(ids)),len(ids)-1)]]=True
            else:hidden[b]=eligible[b];actual[b]=0;fallback[b]=True
    return hidden,{'nominal':nominal,'actual':actual,'fallback':fallback}


def stable_noise(tokens, source_slot_ids, steps, seed, sample_indices=None, device=None, source_capacity=64):
    """One full source-slot bank per scene/sample, then gather. CPU float32 protocol v2."""
    if len(tokens)!=len(source_slot_ids):raise ValueError('Noise token batch mismatch')
    if sample_indices is None:sample_indices=[0]*len(tokens)
    if len(sample_indices)!=len(tokens):raise ValueError('Noise sample-index mismatch')
    rows=[]
    for token,ids,sample in zip(tokens,source_slot_ids.cpu(),sample_indices):
        if ((ids < -2)|(ids>=source_capacity)).any():raise ValueError('Invalid stable source slot')
        value=int.from_bytes(hashlib.sha256(f'local_v2:{seed}:{token}:{sample}'.encode()).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator(device='cpu').manual_seed(value)
        bank=torch.randn(source_capacity+1,steps,2,generator=generator)
        rows.append(torch.where((ids!=-2)[:,None,None],bank[(ids+1).clamp_min(0)],0.))
    return torch.stack(rows).to(device if device is not None else source_slot_ids.device)
