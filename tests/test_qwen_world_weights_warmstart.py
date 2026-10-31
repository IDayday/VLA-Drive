import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms.warmstart import restore_current


def test_weights_only_initialization_preserves_frozen_reference_and_requires_full_current_closure():
    actor=torch.nn.Module();actor.current=torch.nn.Linear(2,2)
    actor.reference=torch.nn.Linear(2,2).requires_grad_(False)
    reference={n:p.clone() for n,p in actor.reference.named_parameters()}
    names=[n for n,p in actor.named_parameters() if p.requires_grad]
    state={n:torch.full_like(p,3.5) for n,p in actor.named_parameters() if p.requires_grad}
    restore_current(actor,state,names)
    for n,p in actor.named_parameters():
        if p.requires_grad:assert torch.equal(p,state[n])
    for n,p in actor.reference.named_parameters():assert torch.equal(p,reference[n])
    incomplete=dict(state);incomplete.pop(names[0])
    with pytest.raises(ValueError,match='complete current'):restore_current(actor,incomplete,names)
    wrong=dict(state);wrong[names[0]]=wrong[names[0]].bfloat16()
    with pytest.raises(ValueError,match='shape/dtype'):restore_current(actor,wrong,names)
