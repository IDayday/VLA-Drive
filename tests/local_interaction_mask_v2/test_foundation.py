import torch
from tools.local_interaction_mask_v2.foundation import epoch_batches


def test_sharded_epoch_is_exactly_once_and_resumes_same_suffix():
    length,batch,world=7284,32,8
    shards=[epoch_batches(length,batch,r,world,42,0) for r in range(world)]
    visited=[]
    for b in range(len(shards[0])):
        visited.extend(sum([shards[r][b] for r in range(world)],[]))
    assert len(visited)==len(set(visited))==length
    for rank in range(world):assert epoch_batches(length,batch,rank,world,42,0,64)==shards[rank][2:]


def test_unequal_valid_counts_global_gradient_not_rank_mean():
    # Same global normalization used by FoundationObjective, including empty-label rank.
    p=torch.tensor(2.,requires_grad=True)
    values=[torch.tensor([1.,3.]),torch.tensor([]),torch.tensor([4.])]
    numerator=[((p-v)**2).sum() for v in values]
    count=sum(len(v) for v in values)
    gradients=[torch.autograd.grad(len(values)*n/count,p,retain_graph=True)[0] for n in numerator]
    actual=torch.stack(gradients).mean()
    expected=torch.autograd.grad(sum(numerator)/count,p)[0]
    assert torch.allclose(actual,expected)
