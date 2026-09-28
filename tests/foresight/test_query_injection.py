import pytest
import torch
from starVLA.model.modules.foresight.tokens import replace_query_embeddings


def test_batched_queries_with_different_padding_preserve_values_and_gradients():
    torch.manual_seed(12)
    x=torch.randn(4,12,8,requires_grad=True);w=torch.randn(3,8,requires_grad=True)
    positions=torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
    y=replace_query_embeddings(x,positions,w)
    reference=x.clone();reference[torch.arange(4)[:,None],positions]=w[None]
    assert torch.equal(y,reference)
    weights=torch.randn_like(y)
    got=torch.autograd.grad((y*weights).sum(),(x,w))
    expected=torch.autograd.grad((reference*weights).sum(),(x,w))
    for left,right in zip(got,expected):assert torch.equal(left,right)
    bad=positions.clone();bad[0,1]=bad[0,0]
    with pytest.raises(ValueError):replace_query_embeddings(x,bad,w)


def test_batch_specific_history_values_do_not_mix_scenes():
    x=torch.zeros(3,5,4);positions=torch.tensor([[1],[2],[3]])
    values=torch.arange(12).reshape(3,1,4).float()
    y=replace_query_embeddings(x,positions,values)
    assert torch.equal(y[torch.arange(3),positions[:,0]],values[:,0])
    assert y.sum()==values.sum()
